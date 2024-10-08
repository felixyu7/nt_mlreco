import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import lightning.pytorch as pl

import time

class Time_Series_AE(pl.LightningModule):
    def __init__(self,
                 in_features=5000,
                 latent_dim=128,
                 batch_size=128,
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, latent_dim*32),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*32, latent_dim*16),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*16, latent_dim*8),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*8, latent_dim*4),
            nn.LeakyReLU(0.01, True)
        )
        
        self.fc_mu = nn.Linear(latent_dim*4, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim*4, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*4),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*4, latent_dim*8),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*8, latent_dim*16),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*16, latent_dim*32),
            nn.LeakyReLU(0.01, True),
            nn.Linear(latent_dim*32, in_features),
        )
        self.beta = 0.
        self.iter = 0
        self.total_steps = 98314 * 4

    def encode(self, inputs):
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # eps = 0.
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        outputs = self.decode(z)
        return outputs, mu, logvar

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs, mu, logvar = self(inputs)
        
        cls_labels = (inputs > 0).float()
        reconstruction_loss = F.binary_cross_entropy_with_logits(outputs, cls_labels, reduction='sum') / self.hparams.batch_size
        kl_loss = self.kl_divergence(mu, logvar)
        
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        # cosine annealing for beta term 
        self.beta = 1e-5 * ((np.cos(np.pi * (self.iter / self.total_steps - 1)) + 1) / 2)
        self.iter += 1
        
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("kl_loss", kl_loss, batch_size=self.hparams.batch_size)
        self.log("reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size)
        self.log("beta", self.beta, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs = batch
        outputs, mu, logvar = self(inputs)
        
        cls_labels = (inputs > 0).float()
        reconstruction_loss = F.binary_cross_entropy_with_logits(outputs, cls_labels, reduction='sum') / self.hparams.batch_size
        kl_loss = self.kl_divergence(mu, logvar)
        
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        self.log("val_train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("val_kl_loss", kl_loss, batch_size=self.hparams.batch_size)
        self.log("val_reco_loss", reconstruction_loss, batch_size=self.hparams.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs = batch
        outputs, mu, logvar = self(inputs)
        
        cls_labels = (inputs > 0).float()
        reconstruction_loss = F.binary_cross_entropy_with_logits(outputs, cls_labels, reduction='sum') / self.hparams.batch_size
        kl_loss = self.kl_divergence(mu, logvar)
        loss = reconstruction_loss + (self.beta*kl_loss)
        
        import pdb; pdb.set_trace()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]