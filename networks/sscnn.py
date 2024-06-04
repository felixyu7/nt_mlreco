import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.common.resnet_block import ResNetBlock
from networks.common.utils import CombinedAngleEnergyLoss, AngularDistanceLoss, LogCoshLoss, angle_between, generate_geo_mask
import MinkowskiEngine as ME
import lightning.pytorch as pl

class SSCNN(pl.LightningModule):
    def __init__(self, 
                 in_features=1, 
                 reps=2, 
                 depth=8, 
                 first_num_filters=16, 
                 stride=2, 
                 dropout=0., 
                 input_dropout=0., 
                 output_dropout=0., 
                 scaling='linear', 
                 output_layer=True, 
                 mode='both', 
                 D=4, 
                 batch_size=128, 
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.validation_step_outputs = []
        self.validation_step_labels = []

        self.test_step_outputs = []
        self.test_step_labels = []
        self.test_results = {}

        if self.hparams.mode == 'both':
            out_features = 4
        elif self.hparams.mode == 'angular_reco':
            out_features = 3
        elif self.hparams.mode == 'energy_reco':
            out_features = 1
        else:
            print("Unknown reco type! Use angular_reco, energy_reco, or both.")
            exit()

        if self.hparams.scaling == 'exp':
            self.nPlanes = [self.hparams.first_num_filters * (2**i) for i in range(self.hparams.depth)]
        else:
            self.nPlanes = [i * self.hparams.first_num_filters for i in range(1, self.hparams.depth + 1)]

        self.input_block = nn.Sequential(
                ME.MinkowskiConvolution(
                in_channels=self.hparams.in_features,
                out_channels=self.hparams.first_num_filters,
                kernel_size=3, stride=3, dimension=self.hparams.D, dilation=1,
                bias=False),
                ME.MinkowskiMaxPooling(kernel_size=8, stride=8, dimension=self.hparams.D),
                ME.MinkowskiPReLU(),
                ME.MinkowskiDropout(self.hparams.input_dropout))

        self.resnet = []
        for i, planes in enumerate(self.nPlanes):
            m = []
            for _ in range(self.hparams.reps):
                m.append(ResNetBlock(planes, planes, dimension=self.hparams.D, dropout=self.hparams.dropout))
            m = nn.Sequential(*m)
            self.resnet.append(m)
            m = []
            if i < self.hparams.depth - 1:
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.hparams.D,
                    bias=False))
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1], track_running_stats=True))
                m.append(ME.MinkowskiPReLU())
            m = nn.Sequential(*m)
            self.resnet.append(m)
        self.resnet = nn.Sequential(*self.resnet)
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()
        if self.hparams.output_layer:
            self.final = ME.MinkowskiLinear(planes, out_features, bias=True)
            self.dropout = ME.MinkowskiDropout(self.hparams.output_dropout)

    def forward(self, x):
        x = self.input_block(x)
        x = self.resnet(x)
        x = self.glob_pool(x)
        x = self.dropout(x)

        # on cpu, store batch ordering for inference. ME bug?
        if self.hparams.output_layer:
            inds = torch.sort(x.C[:,0])[1]
            x = self.final(x)
            return x, inds
        else:
            return x
        
    def training_step(self, batch, batch_idx):
        coords, feats, labels = batch
        inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords.int(), device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        outputs = self(inputs)
        loss = sscnn_loss(outputs, labels, self.hparams.mode)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords, feats, labels = batch
        inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords.int(), device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        outputs = self(inputs)
        loss = sscnn_loss(outputs, labels, self.hparams.mode)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.validation_step_outputs.append(outputs[0].F.detach().cpu().numpy())
        self.validation_step_labels.append(labels.cpu().numpy())

    def on_validation_epoch_end(self):
        if self.hparams.mode == "angular_reco":
            preds = np.concatenate(self.validation_step_outputs, axis=0)
            truth = np.concatenate(self.validation_step_labels, axis=0)[:,1:4]
            angle_diff = []
            for i in range(preds.shape[0]):
                angle_diff.append(angle_between(preds[i], truth[i]))
            angle_diff = np.array(angle_diff) * (180/np.pi)
            self.log("median_angle_diff", np.median(angle_diff), batch_size=self.hparams.batch_size, sync_dist=True)
            self.log("mean_angle_diff", angle_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
        elif self.hparams.mode == "energy_reco":
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
        coords, feats, labels = batch
        inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords.int(), device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        outputs = self(inputs)
        self.test_step_outputs.append(outputs[0].F.detach().cpu().numpy())
        self.test_step_labels.append(labels.cpu().numpy())

    def on_test_epoch_end(self):
        if self.hparams.mode == "angular_reco":
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
        elif self.hparams.mode == "energy_reco":
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

def sscnn_loss(outputs, labels, mode):
    preds = outputs[0].F
    if mode == 'both':
        return CombinedAngleEnergyLoss(preds, labels[:,:4])
    elif mode == 'angular_reco':
        return AngularDistanceLoss(preds, labels[:,1:4])
    elif mode == 'energy_reco':
        return LogCoshLoss(preds, labels[:,0])
    else:
        print("Unknown reco type! Use angular_reco, energy_reco, or both.")
        exit()