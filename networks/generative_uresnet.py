import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.common.resnet_block import ResNetBlock
from networks.common.utils import generate_geo_mask
import MinkowskiEngine as ME
import lightning.pytorch as pl

class Generative_UResNet(pl.LightningModule):
    def __init__(self, 
                 in_features=2, 
                 reps=2, 
                 depth=8, 
                 first_num_filters=16, 
                 stride=2, 
                 dropout=0., 
                 input_dropout=0., 
                 output_dropout=0., 
                 scaling='linear', 
                 D=3, 
                 input_geo_file="/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7.npy",
                 output_geo_file="/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7_2x.npy",
                 batch_size=128, 
                 lr=1e-3, 
                 lr_schedule=[2, 20],
                 weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.test_step_scores = []
        self.test_step_coords = []
        self.test_step_timings = []
        self.test_step_truth = []

        self.geo = torch.from_numpy(np.load(self.hparams.output_geo_file)).requires_grad_(True)
        self.input_geo = torch.from_numpy(np.load(self.hparams.input_geo_file)).requires_grad_(True)
        self.geo_mask = rowwise_diff(self.input_geo, self.geo, return_mask=True)

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

        # self.mask_token = nn.Parameter(torch.zeros(1, self.nPlanes[-1]))
        # nn.init.normal_(self.mask_token, std=.02)

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
        # if self.hparams.chain:
        scores = torch.sigmoid(prob_pred.F.flatten())
        coords = prob_pred.C[scores > 0.5]
        timings = timing_pred.F.flatten()[scores > 0.5]
        timings = (10**timings) - 1
        new_coords = torch.hstack((coords, timings.reshape(-1, 1)))
        new_feats = torch.ones(new_coords.shape[0])
        new_sparse_tensor = ME.SparseTensor(new_feats.reshape(-1, 1), new_coords.int(), device=self.device,
                                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        return prob_pred, timing_pred, new_sparse_tensor
        # return prob_pred, timing_pred
    
    def training_step(self, batch, batch_idx):
        coords, feats, _ = batch
        inputs, output_geo_mask = ntsr_preprocess(self, coords, feats)
        prob_pred, timing_pred, _ = self(inputs)
        cls_loss, timing_loss = uresnet_loss(prob_pred, timing_pred, batch, self.geo, self.geo_mask, output_geo_mask)
        loss = cls_loss + timing_loss
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords, feats, _ = batch
        inputs, output_geo_mask = ntsr_preprocess(self, coords, feats)
        prob_pred, timing_pred, _ = self(inputs)
        cls_loss, timing_loss = uresnet_loss(prob_pred, timing_pred, batch, self.geo, self.geo_mask, output_geo_mask)
        loss = cls_loss + timing_loss
        self.log("val_train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("val_cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("val_timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        return loss
    
    def test_step(self, batch, batch_idx):
        coords, feats, _ = batch
        inputs, _ = ntsr_preprocess(self, coords, feats)
        _, _, new_sparse_tensor = self(inputs)
        return new_sparse_tensor
        
    def on_test_epoch_end(self):
        total_scores = np.concatenate(self.test_step_scores, axis=0)
        total_coords = np.concatenate(self.test_step_coords, axis=0)
        total_timings = np.concatenate(self.test_step_timings, axis=0)
        total_truth = np.concatenate(self.test_step_truth, axis=0)
        test_results = {'scores': total_scores, 'coords': total_coords, 'timings': total_timings, 'truth': total_truth}
        np.save("/n/home10/felixyu/nt_mlreco/results/" + self.logger.name + "_" + self.logger.version + "_results.npy", test_results)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 40], gamma=0.1)
        return [optimizer], [scheduler]

def ntsr_preprocess(net, coords, feats):
    # mask = generate_geo_mask(coords, net.input_geo)
    # coords = coords[mask]
    # feats = feats[mask]
    pt_feats, pt_coords = get_probability_and_timing_features(coords[:,:4].to(net.device).float(), 
                                                                net.geo.to(net.device).float(), 
                                                                coords[:,4].reshape(-1, 1).to(net.device).float(),
                                                                counts=feats.reshape(-1, 1).to(net.device).float(),
                                                                prob_fill_value=0.)
    pt_feats[:,1] = torch.log10(pt_feats[:,1] + 1)
    output_geo_mask = extract_near_points(net.geo.to(net.device).float(), coords.float(), net.input_geo.to(net.device).float(), 100)
    pt_coords = pt_coords[output_geo_mask]
    pt_feats = pt_feats[output_geo_mask]
    inputs = ME.SparseTensor(pt_feats, pt_coords.int(), device=net.device,
                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
    return inputs, output_geo_mask

def uresnet_loss(prob_pred, timing_pred, truth, geo, geo_mask, output_geo_mask):
    score_feats = prob_pred.F.flatten()
    timing_feats = timing_pred.F.flatten()
    # scores = torch.sigmoid(score_feats)
    truth_feats, _ = get_probability_and_timing_features(truth[0][:,:4].to(score_feats.device).float(), 
                                                                geo.to(score_feats.device).float(), 
                                                                truth[0][:,4].reshape(-1, 1).to(score_feats.device).float(),
                                                                prob_fill_value=0.)
    truth_feats = truth_feats[output_geo_mask]
    _, counts = torch.unique(prob_pred.C[:,0], return_counts=True)
    cumulative_counts = counts.cumsum(dim=0)
    cls_losses = []
    timing_losses = []
    start = 0
    for end in cumulative_counts:
        # only compute loss on masked coords
        batch_score_feats = score_feats[start:end]
        batch_timing_feats = timing_feats[start:end]
        batch_truth_feats = truth_feats[start:end]
        # weight_scaling_factor = batch_truth_feats[:,0].shape[0] / ((batch_truth_feats[:,0].sum() * 2) + 1e-8)
        # weight = (batch_truth_feats[:,0] * (weight_scaling_factor - 1)) + 1.
        cls_loss = F.binary_cross_entropy_with_logits(batch_score_feats, batch_truth_feats[:,0])
        true_time = torch.log10(batch_truth_feats[:,1] + 1)
        # true_time = (true_time - true_time.mean()) / (true_time.std() + 1e-8)
        timing_loss = F.smooth_l1_loss(batch_timing_feats, true_time, reduction='none')
        truth_mask = (batch_truth_feats[:,1] > 0).float()
        timing_loss = (timing_loss * truth_mask).sum() / (truth_mask.sum() + 1e-8)
        # truth_mask = (batch_truth_feats[:,1] > 0) * (weight_scaling_factor - 1)
        # truth_mask = truth_mask + 1.
        # timing_loss = (timing_loss * truth_mask.float()).sum() / (truth_mask.sum() + 1e-8)
        cls_losses.append(cls_loss)
        timing_losses.append(timing_loss)
        start = end
    mean_cls_loss = torch.stack(cls_losses).mean()
    mean_timing_loss = torch.stack(timing_losses).mean()
    return mean_cls_loss, mean_timing_loss

def extract_near_points(larger_set, coords, exclusion_set, x):
    batch_size = coords[:,0].unique().shape[0]
    total_mask = []
    for b in range(batch_size):
        smaller_set = coords[torch.where(coords[:,0] == b)][:,1:4]
        # Calculate pairwise distance matrix
        dist_matrix = torch.cdist(larger_set, smaller_set)

        # Find minimum distance from each point in larger_set to smaller_set
        min_distances, _ = torch.min(dist_matrix, dim=1)

        # Get a Boolean tensor indicating which points in larger_set are within distance x of smaller_set
        mask = min_distances <= x

        # Find the points in larger_set that are the same as any point in exclusion_set
        mask_exclusion = torch.stack([torch.any(larger_set[:, i, None] == exclusion_set[:, i], dim=-1) for i in range(larger_set.shape[1])]).all(dim=0)
        
        # Find the points in larger_set that are the same as any point in smaller_set, and include them within the mask
        mask_inclusion = torch.stack([torch.any(larger_set[:, i, None] == smaller_set[:, i], dim=-1) for i in range(larger_set.shape[1])]).all(dim=0)

        # Combine the two masks and return
        combined_mask = mask & ~mask_exclusion
        combined_mask[mask_inclusion] = True
        total_mask.append(combined_mask)
    return torch.hstack(total_mask)

def rowwise_diff(tensor1, tensor2, return_mask=False):
    t2_exp = tensor2.unsqueeze(1)
    t1_exp = tensor1.unsqueeze(0)
    # Compute differences
    diff = t2_exp - t1_exp
    mask = (diff == 0).all(-1)
    diff_mask = ~mask.any(-1)
    # Apply the mask to get the difference
    if return_mask:
        return diff_mask
    else:
        result = tensor2[diff_mask]
        return result

def get_probability_and_timing_features(coords, geo_coords, timings, counts=None, prob_fill_value=0.5):
    # IMPORTANT: FIRST HIT MUST BE TRUE IN CONFIG FOR THIS FUNCTION TO WORK
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
        
        if counts is not None:
            batch_counts = counts[coords[:,0] == batch_idx]
            counts_tensor = torch.zeros(batch_geo_coords.shape[1], device=geo_coords.device) # Initialize with zeros
            # Assign timings at matched indices, argsort to maintain coord/timing alignment
            counts_tensor[any_match] = batch_counts[torch.argsort(match_indices[1])].squeeze()
            # Concatenate the two feature tensors
            final_tensor = torch.hstack((final_tensor, counts_tensor.reshape(-1, 1)))
       
        # Append to list
        batched_tensors.append(final_tensor)
        batch_idxs = torch.full((geo_coords.shape[0], 1), batch_idx, device=geo_coords.device)
        batched_geo_coords.append(torch.hstack((batch_idxs, geo_coords)))
        
    # Concatenate all batch tensors along the 0th dimension
    final_batched_tensor = torch.cat(batched_tensors, dim=0)
    final_batched_geo_coords = torch.cat(batched_geo_coords, dim=0)
    return final_batched_tensor, final_batched_geo_coords