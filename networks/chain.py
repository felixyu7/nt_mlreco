import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import lightning.pytorch as pl
from networks.generative_uresnet import ntsr_preprocess, uresnet_loss
from networks.sscnn import sscnn_loss, SSCNN
from networks.generative_uresnet import Generative_UResNet
from networks.common.utils import angle_between
import numpy as np

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
        self.ntsr = Generative_UResNet(**ntsr_cfg,
                                batch_size=batch_size,
                                lr=lr,
                                lr_schedule=lr_schedule,
                                weight_decay=weight_decay)
        self.sscnn = SSCNN(**sscnn_cfg,
                                batch_size=batch_size,
                                lr=lr,
                                lr_schedule=lr_schedule,
                                weight_decay=weight_decay)
        
        self.test_step_outputs = []
        self.test_step_labels = []
        self.test_results = {}
        
    def forward(self, ntsr_inputs, sscnn_inputs):
        prob_pred, timing_pred, new_sparse_tensor = self.ntsr(ntsr_inputs)
        sscnn_input_tensor = ME.SparseTensor(sscnn_inputs[0].reshape(-1, 1), sscnn_inputs[1].int(), device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True,
                                coordinate_manager=new_sparse_tensor.coordinate_manager)
        sscnn_input_tensor = sscnn_input_tensor + new_sparse_tensor
        return prob_pred, timing_pred, self.sscnn(sscnn_input_tensor)
    
    def training_step(self, batch, batch_idx):
        coords, feats, labels = batch
        ntsr_inputs, output_geo_mask = ntsr_preprocess(self.ntsr, coords, feats)
        sscnn_inputs = [feats, coords]
        prob_pred, timing_pred, sscnn_outputs = self(ntsr_inputs, sscnn_inputs)
        cls_loss, timing_loss = uresnet_loss(prob_pred, timing_pred, batch, self.ntsr.geo, self.ntsr.geo_mask, output_geo_mask)
        angular_loss = sscnn_loss(sscnn_outputs, labels, 'angular_reco')
        loss = cls_loss + timing_loss + angular_loss
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        self.log("sscnn_loss", angular_loss, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords, feats, labels = batch
        ntsr_inputs, output_geo_mask = ntsr_preprocess(self.ntsr, coords, feats)
        sscnn_inputs = [feats, coords]
        prob_pred, timing_pred, sscnn_outputs = self(ntsr_inputs, sscnn_inputs)
        cls_loss, timing_loss = uresnet_loss(prob_pred, timing_pred, batch, self.ntsr.geo, self.ntsr.geo_mask, output_geo_mask)
        angular_loss = sscnn_loss(sscnn_outputs, labels, 'angular_reco')
        loss = cls_loss + timing_loss + angular_loss
        self.log("val_train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("val_cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("val_timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        self.log("val_sscnn_loss", angular_loss, batch_size=self.hparams.batch_size)
        return loss
        
    # def on_validation_epoch_end(self):
    #     self.sscnn.on_validation_epoch_end()
        
    def test_step(self, batch, batch_idx):
        coords, feats, labels = batch
        ntsr_inputs, _ = ntsr_preprocess(self.ntsr, coords, feats)
        sscnn_inputs = [feats, coords]
        prob_pred, timing_pred, sscnn_outputs = self(ntsr_inputs, sscnn_inputs)
        
        self.test_step_outputs.append(sscnn_outputs[0].F.detach().cpu().numpy())
        self.test_step_labels.append(labels.cpu().numpy())
        
    def on_test_epoch_end(self):
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
        
        self.test_step_outputs.clear()
        self.test_step_labels.clear()
        np.save("/n/home10/felixyu/nt_mlreco/results/" + self.logger.name + "_" + self.logger.version + "_results.npy", self.test_results)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]