import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from networks.ntsr_cnn import NTSR_CNN
from networks.sscnn import SSCNN, sscnn_loss
from networks.common.utils import angle_between

import MinkowskiEngine as ME

import lightning.pytorch as pl

class NTSR_SSCNN_Chain(pl.LightningModule):
    def __init__(self,
                 ntsr_cfg = {},
                 sscnn_cfg = {},
                 batch_size=128, 
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.padding = nn.ZeroPad2d((0, 3, 0, 31))
        self.ntsr_cnn = NTSR_CNN(**ntsr_cfg,
                                batch_size=batch_size,
                                lr=lr,
                                lr_schedule=lr_schedule,
                                weight_decay=weight_decay)
        self.SSCNN = SSCNN(**sscnn_cfg,
                                batch_size=batch_size,
                                lr=lr,
                                lr_schedule=lr_schedule,
                                weight_decay=weight_decay)
        
        self.mode = sscnn_cfg['mode']
        ntsr_checkpoint = torch.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/ckpts/ntsr_poisson_latents_epoch29.ckpt')
        self.ntsr_cnn.load_state_dict(ntsr_checkpoint['state_dict'])
        self.ntsr_cnn.eval()
        # freeze weights in ntsr_cnn
        for param in self.ntsr_cnn.parameters():
            param.requires_grad = False
        
        self.geo = torch.from_numpy(np.loadtxt('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/prometheus/resources/geofiles/ice_ortho_det_7_2x_strings.geo', skiprows=3))
        self.geo[:,2] += 2000.
        
        self.validation_step_outputs = []
        self.validation_step_labels = []

        self.test_step_outputs = []
        self.test_step_labels = []
        self.test_results = {}
        
    def forward(self, img):
        upscaled_img = self.ntsr_cnn(img)
        # unpad upscaled_img
        upscaled_img = upscaled_img[:, :, :-31, :-3]
        
        # inputs_mask = (img[:, 3, :-31, :-3] > 0) # input pixels
        # scores_mask = (torch.sigmoid(upscaled_img[:, 0, :, :]) > 0.5) # high scoring pixels from NTSR
        # mask = inputs_mask | scores_mask
        
        mask = (torch.round(torch.exp(upscaled_img[:, 0, :, :])) - 1) >= 1
        
        coords, feats = convert_img_to_sscnn_input(upscaled_img, mask)
        real_coords = extract_real_coords(coords[:, 1:].to(self.device), self.geo.int().to(self.device)) # convert from string sensor coords to real coords
        coords = torch.cat((coords[:, 0].unsqueeze(1), real_coords), dim=1)
        
        sscnn_inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords.int(), device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        
        return self.SSCNN(sscnn_inputs)
    
    def training_step(self, batch, batch_idx):
        masked_imgs, imgs, labels = batch
        labels = labels.squeeze()
        imgs = imgs.float()
        input_img = masked_imgs.float()
        input_img = self.padding(input_img.permute(0, 3, 1, 2))
        
        sscnn_outputs = self(input_img)
        
        loss = sscnn_loss(sscnn_outputs, labels, self.mode)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        masked_imgs, imgs, labels = batch
        labels = labels.squeeze()
        imgs = imgs.float()
        input_img = masked_imgs.float()
        input_img = self.padding(input_img.permute(0, 3, 1, 2))
        
        sscnn_outputs = self(input_img)
        
        loss = sscnn_loss(sscnn_outputs, labels, self.mode)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.validation_step_outputs.append(sscnn_outputs[0].F.detach().cpu().numpy())
        self.validation_step_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        if self.mode == "angular_reco":
            preds = np.concatenate(self.validation_step_outputs, axis=0)
            truth = np.concatenate(self.validation_step_labels, axis=0)[:,1:4]
            angle_diff = []
            for i in range(preds.shape[0]):
                angle_diff.append(angle_between(preds[i], truth[i]))
            angle_diff = np.array(angle_diff) * (180/np.pi)
            self.log("median_angle_diff", np.median(angle_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_angle_diff", angle_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
        elif self.mode == "energy_reco":
            preds = np.concatenate(self.validation_step_outputs, axis=0).flatten()
            truth = np.concatenate(self.validation_step_labels, axis=0)[:,0]
            abs_diff = np.abs(preds - truth)
            self.log("median_abs_energy_diff", np.median(abs_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_abs_energy_diff", abs_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
        else: # both
            preds_A = np.concatenate(self.validation_step_outputs, axis=0)[:,1:4]
            truth_A = np.concatenate(self.validation_step_labels, axis=0)[:,1:4]
            angle_diff = []
            for i in range(preds_A.shape[0]):
                angle_diff.append(angle_between(preds_A[i], truth_A[i]))
            angle_diff = np.array(angle_diff) * (180/np.pi)
            self.log("median_angle_diff", np.median(angle_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_angle_diff", angle_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
            preds_E = np.concatenate(self.validation_step_outputs, axis=0)
            truth_E = np.concatenate(self.validation_step_labels, axis=0)[:,0]
            abs_diff = np.abs(preds_E - truth_E)
            self.log("median_abs_energy_diff", np.median(abs_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_abs_energy_diff", abs_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()
    
    def test_step(self, batch, batch_idx):
        masked_imgs, imgs, labels = batch
        labels = labels.squeeze()
        imgs = imgs.float()
        input_img = masked_imgs.float()
        input_img = self.padding(input_img.permute(0, 3, 1, 2))
        
        sscnn_outputs = self(input_img)
        self.test_step_outputs.append(sscnn_outputs[0].F.detach().cpu().numpy())
        self.test_step_labels.append(labels.cpu().numpy())

    def on_test_epoch_end(self):
        if self.mode == "angular_reco":
            preds = np.concatenate(self.test_step_outputs, axis=0)
            truth = np.concatenate(self.test_step_labels, axis=0)[:,1:4]
            angle_diff = []
            for i in range(preds.shape[0]):
                angle_diff.append(angle_between(preds[i], truth[i]))
            angle_diff = np.array(angle_diff) * (180/np.pi)
            self.log("median_angle_diff", np.median(angle_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_angle_diff", angle_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
            self.test_results['angle_diff'] = angle_diff
            self.test_results['true_e'] = np.concatenate(self.test_step_labels, axis=0)[:,0]
        elif self.mode == "energy_reco":
            preds = np.concatenate(self.test_step_outputs, axis=0)
            truth = np.concatenate(self.test_step_labels, axis=0)[:,0]
            # abs_diff = np.abs(preds - truth)
            # self.log("median_abs_energy_diff", np.median(abs_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            # self.log("mean_abs_energy_diff", abs_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
            self.test_results['preds'] = preds
            self.test_results['truth'] = truth
        else: # both
            preds_A = np.concatenate(self.test_step_outputs, axis=0)[:,1:4]
            truth_A = np.concatenate(self.test_step_labels, axis=0)[:,1:4]
            angle_diff = []
            for i in range(preds_A.shape[0]):
                angle_diff.append(angle_between(preds_A[i], truth_A[i]))
            angle_diff = np.array(angle_diff) * (180/np.pi)
            self.log("median_angle_diff", np.median(angle_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_angle_diff", angle_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
            preds_E = np.concatenate(self.test_step_outputs, axis=0)
            truth_E = np.concatenate(self.test_step_labels, axis=0)[:,0]
            abs_diff = np.abs(preds_E - truth_E)
            self.log("median_abs_energy_diff", np.median(abs_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_abs_energy_diff", abs_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
        self.test_step_outputs.clear()
        self.test_step_labels.clear()
        np.save("/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/results/" + self.logger.name + "_" + self.logger.version + "_results.npy", self.test_results)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]

@torch.compile
def convert_img_to_sscnn_input(images, mask):    
    # Get the indices of nonzero elements in the mask
    nonzero_indices = mask.nonzero(as_tuple=False)
    
    # Get the coordinates
    coords = nonzero_indices.clone()

    # Get the batch, height, and width indices from nonzero elements
    batch_indices = nonzero_indices[:, 0]
    height_indices = nonzero_indices[:, 1]
    width_indices = nonzero_indices[:, 2]
    
    # Get the features corresponding to the nonzero mask locations
    features = images[batch_indices, :, height_indices, width_indices]
    
    return coords, features

@torch.compile
def extract_real_coords(string_sensor_coords, geo):
    last_two_cols_N = geo[:, 3:]
    matches = (last_two_cols_N.unsqueeze(1) == string_sensor_coords.unsqueeze(0)).all(dim=2)
    match_indices = matches.nonzero(as_tuple=False)
    ordered_indices = match_indices[match_indices[:, 1].argsort()][:, 0]
    result = geo[ordered_indices][:, :3]
    return result