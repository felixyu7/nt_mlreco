import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.common.dense_unet import UNet

import segmentation_models_pytorch as smp

import lightning.pytorch as pl

class NTSR_CNN(pl.LightningModule):
    def __init__(self,
                 in_features=1,
                 first_num_filters=16,
                 input_geo_file="/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7.npy",
                 output_geo_file="/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7_2x.npy",
                 batch_size=128, 
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.training_stats = torch.from_numpy(np.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/scratch/ortho_2x_low_e_tracks_gmm_labels_log_training_stats.npy'))
        self.geo = torch.from_numpy(np.load(self.hparams.output_geo_file)).requires_grad_(True)
        self.input_geo = torch.from_numpy(np.load(self.hparams.input_geo_file)).requires_grad_(True)

        # self.unet = UNet(self.hparams.in_features, 2)
        self.padding = nn.ZeroPad2d((1, 2, 15, 16))
        self.unet = smp.UnetPlusPlus(encoder_name='timm-resnest50d', 
                                        encoder_weights=None,
                                        in_channels=self.hparams.in_features, 
                                        classes=2)
        # self.unet = smp.Unet(encoder_name='mit_b2',
        #                      encoder_weights=None,
        #                     in_channels=self.hparams.in_features,
        #                     classes=2)
        # self.unet = smp.MAnet(encoder_name='mit_b2',
        #                         encoder_weights=None,
        #                         in_channels=self.hparams.in_features,
        #                         classes=2)
        self.iter = 0

    def forward(self, img):
        img = img.permute(0, 3, 1, 2)
        img = self.padding(img)
        return self.unet(img)
    
    def training_step(self, batch, batch_idx):
        masked_imgs, imgs = batch
        imgs = imgs.float()
        input_img = masked_imgs.float()
        
        outputs = self(input_img)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.padding(imgs)
        
        counts_label = imgs[:, -1, :, :]
        cls_label = (counts_label > 0).float()
        weighting_factor = 1 / (cls_label.sum() / cls_label.flatten().shape[0])
        weights = (cls_label * (weighting_factor - 1)) + 1
        cls_loss = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], cls_label, weight=weights)
        counts_loss = F.mse_loss(outputs[:, 1, :, :], torch.log10(counts_label + 1), reduction='none')
        counts_loss = (counts_loss * counts_label).sum() / counts_label.sum()
        loss = cls_loss + counts_loss
        
        self.log("cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("counts_loss", counts_loss, batch_size=self.hparams.batch_size)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        masked_imgs, imgs = batch
        imgs = imgs.float()
        input_img = masked_imgs.float()
        
        outputs = self(input_img)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.padding(imgs)
        
        counts_label = imgs[:, -1, :, :]
        cls_label = (counts_label > 0).float()
        weighting_factor = 1 / (cls_label.sum() / cls_label.flatten().shape[0])
        weights = (cls_label * (weighting_factor - 1)) + 1
        cls_loss = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], cls_label, weight=weights)
        counts_loss = F.mse_loss(outputs[:, 1, :, :], torch.log10(counts_label + 1), reduction='none')
        counts_loss = (counts_loss * counts_label).sum() / counts_label.sum()
        loss = cls_loss + counts_loss
        
        self.log("val_cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("val_counts_loss", counts_loss, batch_size=self.hparams.batch_size)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        masked_imgs, imgs = batch
        imgs = imgs.float()
        input_img = masked_imgs.float()
        
        outputs = self(input_img)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.padding(imgs)
        
        counts_label = imgs[:, -1, :, :]
        cls_label = (counts_label > 0).float()
        weighting_factor = 1 / (cls_label.sum() / cls_label.flatten().shape[0])
        weights = (cls_label * (weighting_factor - 1)) + 1
        cls_loss = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], cls_label, weight=weights)
        counts_loss = F.mse_loss(outputs[:, 1, :, :], torch.log10(counts_label + 1), reduction='none')
        counts_loss = (counts_loss * counts_label).sum() / counts_label.sum()
        loss = cls_loss + counts_loss
        
        import pdb; pdb.set_trace()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]