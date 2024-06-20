import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from networks.common.dense_unet import UNet
import lognorm_loss

import segmentation_models_pytorch as smp

import lightning.pytorch as pl

import time

class NTSR_CNN(pl.LightningModule):
    def __init__(self,
                 in_features=1,
                 first_num_filters=16,
                 output_classes=65,
                 batch_size=128, 
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        # self.unet = UNet(self.hparams.in_features, 2)
        self.padding = nn.ZeroPad2d((0, 3, 0, 31))
        self.input_layer = nn.Conv2d(self.hparams.in_features, self.hparams.first_num_filters, 3, padding=1)
        self.input_dropout = nn.Dropout2d(0.5)
        self.unet = smp.UnetPlusPlus(encoder_name='resnet34', 
                                        encoder_weights=None,
                                        in_channels=self.hparams.first_num_filters, 
                                        classes=self.hparams.output_classes)
        
        self.iter = 0

    def forward(self, img):
        outputs = self.input_layer(img)
        outputs = self.input_dropout(outputs)
        outputs = self.unet(outputs)
        
        # replace with input (masked) information
        # outputs[:, 1:, :, :] = (outputs[:, 1:, :, :] * (~(img[:, 3, :, :] > 0)).unsqueeze(1)) + img[:, 3:, :, :]
        outputs = (outputs * (~(img[:, 3, :, :] > 0)).unsqueeze(1)) + img[:, 3:, :, :]

        return outputs
    
    def training_step(self, batch, batch_idx):
        masked_imgs, imgs = batch
        # masked_imgs, imgs, true_time_series = batch
        imgs = imgs.float()
        input_img = masked_imgs.float()
        input_img = self.padding(input_img.permute(0, 3, 1, 2))
        
        # outputs, time_series = self(input_img)
        outputs = self(input_img)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.padding(imgs)
        # true_time_series = true_time_series.permute(0, 3, 1, 2)
        # true_time_series = self.padding(true_time_series)
        
        # counts_loss, time_pdf_loss = ntsr_loss(outputs, time_series, imgs, true_time_series)
        counts_loss, time_pdf_loss = ntsr_loss(outputs, imgs)
        loss = counts_loss + time_pdf_loss
        
        self.log("counts_loss", counts_loss, batch_size=self.hparams.batch_size)
        self.log("time_pdf_loss", time_pdf_loss, batch_size=self.hparams.batch_size)
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        
        # self.iter += 1
        # if self.iter == 100:
        #     import pdb; pdb.set_trace()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        masked_imgs, imgs = batch
        # masked_imgs, imgs, true_time_series = batch
        imgs = imgs.float()
        input_img = masked_imgs.float()
        input_img = self.padding(input_img.permute(0, 3, 1, 2))
        
        # outputs, time_series = self(input_img)
        outputs = self(input_img)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.padding(imgs)
        # true_time_series = true_time_series.permute(0, 3, 1, 2)
        # true_time_series = self.padding(true_time_series)
        
        # counts_loss, time_pdf_loss = ntsr_loss(outputs, time_series, imgs, true_time_series)
        counts_loss, time_pdf_loss = ntsr_loss(outputs, imgs)
        loss = counts_loss + time_pdf_loss
        
        self.log("val_counts_loss", counts_loss, batch_size=self.hparams.batch_size)
        self.log("val_time_pdf_loss", time_pdf_loss, batch_size=self.hparams.batch_size)
        self.log("val_loss", loss, batch_size=self.hparams.batch_size)
    
    def test_step(self, batch, batch_idx):
        masked_imgs, imgs = batch
        imgs = imgs.float()
        input_img = masked_imgs.float()
        input_img = self.padding(input_img.permute(0, 3, 1, 2))
        
        outputs = self(input_img)
        imgs = imgs.permute(0, 3, 1, 2)
        imgs = self.padding(imgs)
        
        # counts_loss, time_pdf_loss = ntsr_loss(outputs, imgs, true_photons)
        # loss = counts_loss + time_pdf_loss
        
        import pdb; pdb.set_trace()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]

def ntsr_loss(outputs, labels):
    counts_label = labels[:, 3, :, :]
    cls_label = (counts_label > 0).float()
    counts_loss = F.mse_loss(F.relu(outputs[:, 0, :, :]), counts_label)
    
    time_pdf_loss = F.mse_loss(outputs[:, 1:, :, :], labels[:, 4:, :, :], reduction='none')
    time_pdf_loss = (time_pdf_loss * cls_label.unsqueeze(1)).sum() / cls_label.sum()
    time_pdf_loss = time_pdf_loss / outputs.shape[0]
    
    return counts_loss, time_pdf_loss