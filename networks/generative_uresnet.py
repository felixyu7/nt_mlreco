import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.common.resnet_block import ResNetBlock
import MinkowskiEngine as ME
import lightning.pytorch as pl

class Generative_UResNet(pl.LightningModule):
    def __init__(self, in_features, reps=2, depth=8, first_num_filters=16, stride=2, dropout=0., 
                 input_dropout=0., output_dropout=0., scaling='linear', D=3, geo='icecube',
                 batch_size=128, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        if geo == 'icecube':
            self.geo = torch.from_numpy(np.load('/n/home10/felixyu/nt_mlreco/scratch/geo.npy')).requires_grad_(True)

        if self.hparams.scaling == 'exp':
            self.nPlanes = [self.hparams.first_num_filters * (2**i) for i in range(self.hparams.depth)]
        else:
            self.nPlanes = [i * self.hparams.first_num_filters for i in range(1, self.hparams.depth + 1)]

        self.input_block = nn.Sequential(
                ME.MinkowskiConvolution(
                in_channels=self.hparams.in_features,
                out_channels=self.hparams.first_num_filters,
                kernel_size=3, stride=1, dimension=self.hparams.D, dilation=1,
                bias=False),
                ME.MinkowskiPReLU(),
                ME.MinkowskiDropout(self.hparams.input_dropout))

        self.encoder = []
        for i, planes in enumerate(self.nPlanes):
            if i < self.hparams.depth - 1:
                m = []
                for _ in range(self.hparams.reps):
                    m.append(ResNetBlock(planes, planes, dimension=self.hparams.D, dropout=self.hparams.dropout))
                # m = nn.Sequential(*m)
                # self.encoder.append(m)
                # m = []
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=self.hparams.stride, stride=self.hparams.stride, dimension=self.hparams.D, bias=False))
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1], track_running_stats=True))
                m.append(ME.MinkowskiPReLU())
                m = nn.Sequential(*m)
                self.encoder.append(m)
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        # reverse order
        for i, planes in reversed(list(enumerate(self.nPlanes))):
            if i > 0:
                m = []
                for _ in range(self.hparams.reps):
                    m.append(ResNetBlock(planes, planes, dimension=self.hparams.D, dropout=self.hparams.dropout))
                # m = nn.Sequential(*m)
                # self.decoder.append(m)
                # m = []
                m.append(ME.MinkowskiConvolutionTranspose(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i-1],
                    kernel_size=self.hparams.stride, stride=self.hparams.stride, dimension=self.hparams.D, bias=False))
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i-1], track_running_stats=True))
                m.append(ME.MinkowskiPReLU())
                m = nn.Sequential(*m)
                self.decoder.append(m)
        self.decoder = nn.Sequential(*self.decoder)

        self.probability_pred = nn.Sequential(
                ME.MinkowskiConvolution(
                in_channels=self.hparams.first_num_filters,
                out_channels=1,
                kernel_size=1, stride=1, dimension=self.hparams.D, bias=False),
                ME.MinkowskiDropout(self.hparams.output_dropout))
        
        self.timing_pred = nn.Sequential(
                ME.MinkowskiConvolution(
                in_channels=self.hparams.first_num_filters,
                out_channels=1,
                kernel_size=1, stride=1, dimension=self.hparams.D, bias=False),
                ME.MinkowskiDropout(self.hparams.output_dropout))

    def forward(self, x):
        x = self.input_block(x)
        enc_feature_maps = [x]
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i < len(self.encoder) - 1:
                enc_feature_maps.insert(0, x)

        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            x = x + enc_feature_maps[i]    
        prob_pred = self.probability_pred(x)
        timing_pred = self.timing_pred(x)
        return prob_pred, timing_pred
    
    def training_step(self, batch, batch_idx):
        coords, feats, labels = batch[0]
        pt_feats, pt_coords = get_probability_and_timing_features(coords[:,:4].to(self.device).float(), 
                                                                  self.geo.to(self.device).float(), 
                                                                  coords[:,4].reshape(-1, 1).to(self.device).float(),
                                                                  prob_fill_value=0.)
        pt_feats[:,1] = torch.log10(pt_feats[:,1] + 1)
        inputs = ME.SparseTensor(pt_feats, pt_coords.int(), device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        prob_pred, timing_pred = self(inputs)
        # confidence_feats = torch.ones(feats.shape) * 5
        # inputs_confidence = ME.SparseTensor(confidence_feats.float().reshape(coords.shape[0], -1), coords, device=self.device,
        #                         minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT, 
        #                         coordinate_manager=outputs.coordinate_manager, requires_grad=True)
        # outputs = outputs + inputs_confidence
        cls_loss, timing_loss = uresnet_loss(prob_pred, timing_pred, batch[1], self.geo)
        loss = cls_loss + timing_loss
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        if self.global_step == 2000:
            import pdb; pdb.set_trace()
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass
        # coords, feats, labels = batch[0]
        # pt_feats, pt_coords = get_probability_and_timing_features(coords[:,:4].to(self.device).float(), 
        #                                                           self.geo.to(self.device).float(), 
        #                                                           coords[:,4].reshape(-1, 1).to(self.device).float())
        # inputs = ME.SparseTensor(pt_feats, pt_coords.int(), device=self.device,
        #                         minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT, requires_grad=True)
        # outputs = self(inputs)
        # cls_scores = torch.sigmoid(outputs.F[:,0])
        # confidence_mask = cls_scores > 0.9
        # predicted_coords = self.geo[confidence_mask]
        # predicted_timings = outputs.F[:,1][confidence_mask]
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [], gamma=0.1)
        return [optimizer], [scheduler]

def get_probability_and_timing_features(coords, geo_coords, timings, prob_fill_value=0.5):
    # Get unique batch indices
    batch_indices = torch.unique(coords[:, 0])
    # Initialize list to store tensors for each batch
    batched_tensors = []
    batched_geo_coords = []
    # Iterate over each batch
    for batch_idx in batch_indices:
       # Extract the Nx3 tensor and timings for this batch
       batch_coords_tensor = coords[coords[:, 0] == batch_idx, 1:]
       batch_timings = timings[coords[:, 0] == batch_idx] 
       # Expand dims for broadcasting
       batch_coords_tensor = batch_coords_tensor.unsqueeze(1)
       batch_geo_coords = geo_coords.unsqueeze(0)
       # Comparisons
       match = torch.all(batch_coords_tensor == batch_geo_coords, dim=-1)
       # Any matches
       any_match = match.any(dim=0)
       # Return tensor with 1 where match, 0.5 where no match
       feature_tensor_1 = torch.where(any_match, torch.tensor(1., device=geo_coords.device), torch.tensor(prob_fill_value, device=geo_coords.device))
       # Get timings for matches, 0 for no match
       match_indices = match.nonzero(as_tuple=True) # Get the indices of matches
       timings_tensor = torch.zeros(batch_geo_coords.shape[1], device=geo_coords.device) # Initialize with zeros
       # Assign timings at matched indices, argsort to maintain coord/timing alignment
       timings_tensor[any_match] = batch_timings[torch.argsort(match_indices[1])].squeeze()
       # Concatenate the two feature tensors
       final_tensor = torch.stack((feature_tensor_1, timings_tensor), dim=-1)
       # Append to list
       batched_tensors.append(final_tensor)
       batch_idxs = torch.full((geo_coords.shape[0], 1), batch_idx, device=geo_coords.device)
       batched_geo_coords.append(torch.hstack((batch_idxs, geo_coords)))
    # Concatenate all batch tensors along the 0th dimension
    final_batched_tensor = torch.cat(batched_tensors, dim=0)
    final_batched_geo_coords = torch.cat(batched_geo_coords, dim=0)
    return final_batched_tensor, final_batched_geo_coords

def uresnet_loss(prob_pred, timing_pred, truth, geo):
    score_feats = prob_pred.F.flatten()
    timing_feats = timing_pred.F.flatten()
    # scores = torch.sigmoid(score_feats)
    truth_feats, _ = get_probability_and_timing_features(truth[0][:,:4].to(score_feats.device).float(), 
                                                                geo.to(score_feats.device).float(), 
                                                                truth[0][:,4].reshape(-1, 1).to(score_feats.device).float(),
                                                                prob_fill_value=0.)
    
    _, counts = torch.unique(prob_pred.C[:,0], return_counts=True)
    cumulative_counts = counts.cumsum(dim=0)
    cls_losses = []
    timing_losses = []
    start = 0
    for end in cumulative_counts:
        batch_score_feats = score_feats[start:end]
        batch_timing_feats = timing_feats[start:end]
        batch_truth_feats = truth_feats[start:end]
        weight_scaling_factor = batch_truth_feats[:,0].shape[0] / batch_truth_feats[:,0].sum()
        # weight = (batch_truth_feats[:,0] * weight_scaling_factor) + 1.
        cls_loss = F.binary_cross_entropy_with_logits(batch_score_feats, batch_truth_feats[:,0], pos_weight=weight_scaling_factor)
        timing_loss = F.smooth_l1_loss(batch_timing_feats, torch.log10(batch_truth_feats[:,1] + 1), reduction='none')
        truth_mask = batch_truth_feats[:,1] > 0
        timing_loss = (timing_loss * truth_mask.float()).sum() / truth_mask.sum()
        cls_losses.append(cls_loss)
        timing_losses.append(timing_loss)
        start = end
    mean_cls_loss = torch.stack(cls_losses).mean()
    mean_timing_loss = torch.stack(timing_losses).mean()
    return mean_cls_loss, mean_timing_loss