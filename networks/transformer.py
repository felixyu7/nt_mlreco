# Originally authored by Philip Weigel (MIT)
# Modified by Felix Yu (Harvard)

import math, sys
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, trunc_normal_

from networks.common.embeddings import *
from networks.common.transformer_blocks import *
from networks.common.utils import AngularDistanceLoss, angle_between

import lightning.pytorch as pl
import numpy as np

class NuModel(pl.LightningModule):
    """
    A model based on the DeepIceModel which allows for a variable number of outputs.
    The `num_outputs` parameter sets the size of the output of the final layer.
    """
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=12,
        use_checkpoint=False,
        head_size=32,
        depth_rel=4,
        n_rel=1,
        num_outputs=1,
        lr=1e-3,
        weight_decay=1e-5,
        lr_schedule=[5, 20],
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.validation_step_outputs = []
        self.validation_step_labels = []
        self.test_step_outputs = []
        self.test_step_labels = []
        self.test_results = {}
        
        self.extractor = Extractor(dim_base, dim)
        self.rel_pos = Rel_ds(head_size)
        self.num_outputs = num_outputs
        
        # The sandwich contains the blocks with the relative bias
        self.sandwich = nn.ModuleList(
            [BEiTv2Block_rel(dim=dim, num_heads=dim // head_size) for i in range(depth_rel)]
        )
        
        # cls token used in the main blocks
        self.cls_token = nn.Linear(dim, 1, bias=False)
        
        # The main blocks do not have the relative bias
        self.blocks = nn.ModuleList(
            [
                BEiTv2Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        
        # Project the output of the last block into reco quantities
        self.proj_out = nn.Linear(dim, self.num_outputs)
        
        self.use_checkpoint = use_checkpoint  # Are we using a checkpoint?
        self.apply(self._init_weights)  # Initialize model weights
        trunc_normal_(self.cls_token.weight, std=0.02) # cls token weights
        self.n_rel = n_rel

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0["mask"]  # Get mask
        Lmax = mask.sum(-1).max() 
        
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        x = self.proj_out(x[:, 0])  # cls token
                
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs['auxiliary'] = torch.zeros(inputs['time'].shape).long().to(inputs['time'].device)
        outputs = self(inputs)
        loss = AngularDistanceLoss(outputs, labels[:, 1:4])
        self.log("train_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs['auxiliary'] = torch.zeros(inputs['time'].shape).long().to(inputs['time'].device)
        outputs = self(inputs)
        loss = AngularDistanceLoss(outputs, labels[:, 1:4])
        self.log("val_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        self.validation_step_outputs.append(outputs.detach().cpu().numpy())
        self.validation_step_labels.append(labels.cpu().numpy())
        return loss
    
    def on_validation_epoch_end(self):
        preds = np.concatenate(self.validation_step_outputs, axis=0)
        truth = np.concatenate(self.validation_step_labels, axis=0)[:,1:4]
        angle_diff = []
        for i in range(preds.shape[0]):
            angle_diff.append(angle_between(preds[i], truth[i]))
        angle_diff = np.array(angle_diff) * (180/np.pi)
        self.log("median_angle_diff", np.median(angle_diff), batch_size=self.hparams.batch_size, sync_dist=True)
        self.log("mean_angle_diff", angle_diff.mean(), batch_size=self.hparams.batch_size, sync_dist=True)
        self.validation_step_outputs.clear()
        self.validation_step_labels.clear()
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs['auxiliary'] = torch.zeros(inputs['time'].shape).long().to(inputs['time'].device)
        outputs = self(inputs)
        self.test_step_outputs.append(outputs.detach().cpu().numpy())
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
        np.save("/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/results/" + self.logger.name + "_" + self.logger.version + "_results.npy", self.test_results)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]