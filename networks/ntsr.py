import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.common.unet import UNet
from networks.common.resnet_block import ResNetBlock 
import MinkowskiEngine as ME
import lightning.pytorch as pl

import torch.distributions as dist
import time

from xformers.components import MultiHeadDispatch, build_attention
from xformers.components.positional_embedding import SinePositionalEmbedding

class NTSR(pl.LightningModule):
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
        self.training_stats = torch.from_numpy(np.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/scratch/ortho_2x_low_e_tracks_gmm_labels_log_training_stats.npy'))

        self.candidate_token = nn.Parameter(torch.zeros(self.hparams.in_features))

        self.unet = UNet(in_features=self.hparams.in_features,
                            reps=self.hparams.reps,
                            depth=self.hparams.depth,
                            first_num_filters=self.hparams.first_num_filters,
                            stride=self.hparams.stride,
                            dropout=self.hparams.dropout,
                            input_dropout=self.hparams.input_dropout,
                            scaling=self.hparams.scaling,
                            D=self.hparams.D)
        
        self.pos_embed = nn.Sequential(nn.Linear(3, 13), 
                                       nn.ReLU(),
                                       nn.Linear(13, 13))
        
        # attn_config = {
        #     "name": 'scaled_dot_product',  # you can easily make this dependent on a file, sweep,..
        #     "dropout": 0.1,
        #     "seq_len": 13725
        # }
        # attention = build_attention(attn_config)
        # self.k = nn.Linear(self.hparams.in_features, 64)
        # self.q = nn.Linear(self.hparams.in_features, 64)
        # self.v = nn.Linear(self.hparams.in_features, 64)
        
        # self.transformer = MultiHeadDispatch(
        #     seq_len=13725,
        #     dim_model=64,
        #     residual_dropout=0.1,
        #     num_heads=2,
        #     attention=attention,
        # )

        # self.probability_pred = nn.Linear(64, 1)

        # self.counts_pred = nn.Linear(64, 1)

        # self.time_pdf_pred = nn.Linear(64, 12)

        self.probability_pred = nn.Sequential(
            ME.MinkowskiLinear(self.hparams.first_num_filters, self.hparams.first_num_filters),
            ME.MinkowskiDropout(self.hparams.output_dropout),
            ME.MinkowskiLinear(self.hparams.first_num_filters, 1))

        self.counts_pred = nn.Sequential(
            ME.MinkowskiLinear(self.hparams.first_num_filters, self.hparams.first_num_filters),
            ME.MinkowskiDropout(self.hparams.output_dropout),
            ME.MinkowskiLinear(self.hparams.first_num_filters, 1))

        self.time_pdf_pred = nn.Sequential(
            ME.MinkowskiLinear(self.hparams.first_num_filters, self.hparams.first_num_filters),
            ME.MinkowskiDropout(self.hparams.output_dropout),
            ME.MinkowskiLinear(self.hparams.first_num_filters, 12))
        
        self.iter = 0

    def forward(self, x, pos, batch_timing_stats, batch_counts_stats, mask=None):
        pos_embeddings = self.pos_embed(pos[:, 1:] / 100.)
        input_feats = x + pos_embeddings
        inputs = ME.SparseTensor(input_feats, pos.int(), device=self.device,
                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        x = self.unet(inputs)
        
        # mask = mask.unsqueeze(1).expand(self.hparams.batch_size, mask.shape[1], mask.shape[1])
        # mask = mask & mask.transpose(1, 2)
        # pos_embeddings = self.pos_embed(pos / 100.)
        # key, query, value = self.k(x), self.q(x), self.v(x)
        # key = key + pos_embeddings
        # query = query + pos_embeddings
        # value = value + pos_embeddings
        
        # x = self.transformer(key, query, value, att_mask=None)
        
        prob_pred = self.probability_pred(x)
        time_pdf_pred = self.time_pdf_pred(x)
        counts_pred = self.counts_pred(x)

        return prob_pred, time_pdf_pred, counts_pred
    
    def training_step(self, batch, batch_idx):
        coords_masked, feats_masked, coords, feats = batch
        
        # feats_masked = torch.hstack((coords_masked[:, 1:] / 100., torch.log10(feats_masked[:,0] + 1).reshape(-1, 1)))
        
        # feats_masked = torch.hstack((feats_masked, coords_masked[:, 1:] / 100.))
        # # feats = torch.hstack((feats, coords[:, 1:]))
        
        self.training_stats = self.training_stats.to(self.device).float()
        # # normalize inputs
        feats_masked[:,0] = torch.log10(feats_masked[:,0] + 1)
        feats_masked[:,1:5] = torch.log10(feats_masked[:,1:5] + 1)
        feats_masked[:,1:5] = (feats_masked[:,1:5] - self.training_stats[:,0][1:5]) / self.training_stats[:,1][1:5]
        feats_masked[:,5:9] = torch.log10(feats_masked[:,5:9] + 1)
        
        feats[:,0] = torch.log10(feats[:,0] + 1)
        feats[:,1:5] = torch.log10(feats[:,1:5] + 1)
        feats[:,1:5] = (feats[:,1:5] - self.training_stats[:,0][1:5]) / self.training_stats[:,1][1:5]
        feats[:,5:9] = torch.log10(feats[:,5:9] + 1)
        
        # TEST: randomly pick out points from unmasked points instead
        # coords_masked, feats_masked, coords, feats = random_sampling(coords, feats, 0.5)
        
        # take the masked input coordinates and extract nearby point candidates
        point_candidates_mask = extract_near_points(self.geo.to(self.device).float(), 
                                                    coords_masked.float(), 
                                                    self.input_geo.to(self.device).float(), 
                                                    100)
        # construct point candidates on virtual strings
        batch_geo_coords = self.geo.repeat(self.hparams.batch_size, 1)
        batch_inds = torch.repeat_interleave(torch.arange(0, self.hparams.batch_size), self.geo.shape[0])
        batch_geo_coords = torch.hstack((batch_inds.reshape(-1, 1), batch_geo_coords)).to(self.device)
        point_candidates = batch_geo_coords[point_candidates_mask]
        point_candidates_feats = torch.zeros((point_candidates.shape[0], feats.shape[1])).to(self.device)
        
        # distribute features to points included in the masked input
        input_feats = distribute_feats(point_candidates, coords_masked, feats_masked, init='token', token=self.candidate_token)
        # input_seq = distribute_feats(batch_geo_coords, coords_masked, feats_masked, init='token', token=self.candidate_token)
        # inputs = input_seq.reshape(self.hparams.batch_size, -1, input_seq.shape[1])
        
        # inputs = ME.SparseTensor(input_feats, point_candidates.int(), device=self.device,
        #                     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        # prob_pred, time_pdf_pred, counts_pred = self(inputs, batch_geo_coords[:, 1:].reshape(self.hparams.batch_size, -1, 3).float().requires_grad_(True), 0, 0, mask=point_candidates_mask.reshape(self.hparams.batch_size, -1))
        prob_pred, time_pdf_pred, counts_pred = self(input_feats, point_candidates.float(), 0, 0)
        # cls_loss, counts_loss, timing_loss = ntsr_loss(prob_pred, time_pdf_pred, counts_pred, coords, feats, batch_geo_coords)
        cls_loss, counts_loss, timing_loss = ntsr_loss(prob_pred, time_pdf_pred, counts_pred, coords, feats, coords_masked, feats_masked, point_candidates)
        
        loss = cls_loss + counts_loss
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("counts_loss", counts_loss, batch_size=self.hparams.batch_size)
        self.log("timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        # self.iter += 1
        # if ((self.iter+1) % 1000) == 0:
        #     import pdb; pdb.set_trace()
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords_masked, feats_masked, coords, feats = batch
        
        # feats_masked = torch.hstack((coords_masked[:, 1:] / 100., feats_masked[:,0].reshape(-1, 1)))
        
        # feats_masked = torch.hstack((feats_masked, coords_masked[:, 1:] / 100.))
        # # feats = torch.hstack((feats, coords[:, 1:]))
        
        self.training_stats = self.training_stats.to(self.device).float()
        # # normalize inputs
        feats_masked[:,0] = torch.log10(feats_masked[:,0] + 1)
        feats_masked[:,1:5] = torch.log10(feats_masked[:,1:5] + 1)
        feats_masked[:,1:5] = (feats_masked[:,1:5] - self.training_stats[:,0][1:5]) / self.training_stats[:,1][1:5]
        feats_masked[:,5:9] = torch.log10(feats_masked[:,5:9] + 1)
        
        feats[:,0] = torch.log10(feats[:,0] + 1)
        feats[:,1:5] = torch.log10(feats[:,1:5] + 1)
        feats[:,1:5] = (feats[:,1:5] - self.training_stats[:,0][1:5]) / self.training_stats[:,1][1:5]
        feats[:,5:9] = torch.log10(feats[:,5:9] + 1)
        
        # take the masked input coordinates and extract nearby point candidates
        point_candidates_mask = extract_near_points(self.geo.to(self.device).float(), 
                                                    coords_masked.float(), 
                                                    self.input_geo.to(self.device).float(), 
                                                    100)
        # construct point candidates on virtual strings
        batch_geo_coords = self.geo.repeat(self.hparams.batch_size, 1)
        batch_inds = torch.repeat_interleave(torch.arange(0, self.hparams.batch_size), self.geo.shape[0])
        batch_geo_coords = torch.hstack((batch_inds.reshape(-1, 1), batch_geo_coords)).to(self.device)
        point_candidates = batch_geo_coords[point_candidates_mask]
        point_candidates_feats = torch.zeros((point_candidates.shape[0], feats.shape[1])).to(self.device)
        
        # distribute features to points included in the masked input
        input_feats = distribute_feats(point_candidates, coords_masked, feats_masked, init='token', token=self.candidate_token)
        # input_seq = distribute_feats(batch_geo_coords, coords_masked, feats_masked, init='token', token=self.candidate_token)
        # inputs = input_seq.reshape(self.hparams.batch_size, -1, input_seq.shape[1])
        
        # inputs = ME.SparseTensor(input_feats, point_candidates.int(), device=self.device,
        #                     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        # prob_pred, time_pdf_pred, counts_pred = self(inputs, batch_geo_coords[:, 1:].reshape(self.hparams.batch_size, -1, 3).float().requires_grad_(True), 0, 0, mask=point_candidates_mask.reshape(self.hparams.batch_size, -1))
        prob_pred, time_pdf_pred, counts_pred = self(input_feats, point_candidates.float(), 0, 0)
        # cls_loss, counts_loss, timing_loss = ntsr_loss(prob_pred, time_pdf_pred, counts_pred, coords, feats, batch_geo_coords)
        cls_loss, counts_loss, timing_loss = ntsr_loss(prob_pred, time_pdf_pred, counts_pred, coords, feats, coords_masked, feats_masked, point_candidates)
        
        loss = cls_loss + counts_loss + timing_loss
        self.log("val_train_loss", loss, batch_size=self.hparams.batch_size)
        self.log("val_cls_loss", cls_loss, batch_size=self.hparams.batch_size)
        self.log("val_counts_loss", counts_loss, batch_size=self.hparams.batch_size)
        self.log("val_timing_loss", timing_loss, batch_size=self.hparams.batch_size)
        import pdb; pdb.set_trace()
        return loss
    
    def test_step(self, batch, batch_idx):
        coords_masked, feats_masked, coords, feats = batch
        
        # feats_masked = torch.hstack((coords_masked[:, 1:] / 100., feats_masked[:,0].reshape(-1, 1)))
        
        # feats_masked = torch.hstack((feats_masked, coords_masked[:, 1:] / 100.))
        # # feats = torch.hstack((feats, coords[:, 1:]))
        
        self.training_stats = self.training_stats.to(self.device).float()
        # # normalize inputs
        feats_masked[:,0] = torch.log10(feats_masked[:,0] + 1)
        feats_masked[:,1:5] = torch.log10(feats_masked[:,1:5] + 1)
        feats_masked[:,1:5] = (feats_masked[:,1:5] - self.training_stats[:,0][1:5]) / self.training_stats[:,1][1:5]
        feats_masked[:,5:9] = torch.log10(feats_masked[:,5:9] + 1)
        
        feats[:,0] = torch.log10(feats[:,0] + 1)
        feats[:,1:5] = torch.log10(feats[:,1:5] + 1)
        feats[:,1:5] = (feats[:,1:5] - self.training_stats[:,0][1:5]) / self.training_stats[:,1][1:5]
        feats[:,5:9] = torch.log10(feats[:,5:9] + 1)
        
        # take the masked input coordinates and extract nearby point candidates
        point_candidates_mask = extract_near_points(self.geo.to(self.device).float(), 
                                                    coords_masked.float(), 
                                                    self.input_geo.to(self.device).float(), 
                                                    100)
        # construct point candidates on virtual strings
        batch_geo_coords = self.geo.repeat(self.hparams.batch_size, 1)
        batch_inds = torch.repeat_interleave(torch.arange(0, self.hparams.batch_size), self.geo.shape[0])
        batch_geo_coords = torch.hstack((batch_inds.reshape(-1, 1), batch_geo_coords)).to(self.device)
        point_candidates = batch_geo_coords[point_candidates_mask]
        point_candidates_feats = torch.zeros((point_candidates.shape[0], feats.shape[1])).to(self.device)
        
        # distribute features to points included in the masked input
        input_feats = distribute_feats(point_candidates, coords_masked, feats_masked, init='token', token=self.candidate_token)
        # input_seq = distribute_feats(batch_geo_coords, coords_masked, feats_masked, init='token', token=self.candidate_token)
        # inputs = input_seq.reshape(self.hparams.batch_size, -1, input_seq.shape[1])
        
        # inputs = ME.SparseTensor(input_feats, point_candidates.int(), device=self.device,
        #                     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        # prob_pred, time_pdf_pred, counts_pred = self(inputs, batch_geo_coords[:, 1:].reshape(self.hparams.batch_size, -1, 3).float().requires_grad_(True), 0, 0, mask=point_candidates_mask.reshape(self.hparams.batch_size, -1))
        prob_pred, time_pdf_pred, counts_pred = self(input_feats, point_candidates.float(), 0, 0)
        import pdb; pdb.set_trace()
        
    def on_test_epoch_end(self):
        total_scores = np.concatenate(self.test_step_scores, axis=0)
        total_coords = np.concatenate(self.test_step_coords, axis=0)
        total_timings = np.concatenate(self.test_step_timings, axis=0)
        total_truth = np.concatenate(self.test_step_truth, axis=0)
        test_results = {'scores': total_scores, 'coords': total_coords, 'timings': total_timings, 'truth': total_truth}
        np.save("/n/home10/felixyu/nt_mlreco/results/" + self.logger.name + "_" + self.logger.version + "_results.npy", test_results)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_schedule, gamma=0.1)
        return [optimizer], [scheduler]

def ntsr_loss(prob_pred, time_pdf_pred, counts_pred, unmasked_coords, unmasked_feats, coords_masked, feats_masked, point_candidates):
    score_feats = prob_pred.F.flatten()
    timing_feats = time_pdf_pred.F
    counts_feats = counts_pred.F.flatten()
    # score_feats = prob_pred.flatten()
    # timing_feats = time_pdf_pred
    # counts_feats = counts_pred.flatten()
    
    # construct point candidates labels using unmasked information
    point_candidates_labels = distribute_feats(point_candidates, unmasked_coords, unmasked_feats, init='zeros')
    given_labels = distribute_feats(point_candidates, coords_masked, feats_masked, init='zeros')
    virtual_inds = torch.where((point_candidates_labels[:,0] > 0) & ~(given_labels[:,0] > 0))[0]
    class_labels = (point_candidates_labels[:, 0] > 0).float()
    
    # # normalize counts labels
    # point_candidates_labels[:, 0] = torch.log10(point_candidates_labels[:, 0] + 1)
    # # normalize means
    # point_candidates_labels[:, 1:5] = torch.log10(point_candidates_labels[:, 1:5] + 1)
    # # normalize variances (minimum: 10)
    # point_candidates_labels[:, 5:9] = torch.log10(point_candidates_labels[:, 5:9] + 1) - 1
    
    # pos_inds = torch.where(class_labels > 0)[0]
    cls_loss = F.binary_cross_entropy_with_logits(score_feats, class_labels)
    counts_loss = F.mse_loss(counts_feats[virtual_inds], point_candidates_labels[:, 0][virtual_inds], reduction='none')
    # counts_loss = F.poisson_nll_loss(counts_feats[pos_inds], point_candidates_labels[:, 0][pos_inds], reduction='none')
    counts_loss = (counts_loss * point_candidates_labels[:, 0][virtual_inds]).sum() / point_candidates_labels[:, 0][virtual_inds].sum()
    # variance_penalty = 1 / counts_feats[pos_inds].var()
    # counts_loss = counts_loss + (0.1 * variance_penalty)
    
    # construct weight vector for counts loss (1 / frequency of label)
    # counts_weight = construct_weight_vector(point_candidates_labels[:, 0][pos_inds])
    # counts_weight = 1 / counts_weight
    # counts_loss = (counts_loss * counts_weight).sum()

    timing_loss = 0.
    
    # # timing_loss = mixture_nll_loss(timing_feats[torch.where(class_labels > 0)[0]], 
    # #                                point_candidates_labels[:, 1:][torch.where(class_labels > 0)[0]])
    # timing_loss = F.mse_loss(timing_feats[pos_inds], 
    #                                point_candidates_labels[:, 1:][pos_inds], reduction='none')
    # # construct weight vector for each feature
    # timing_weight_matrix = []
    # for i in range(timing_feats.shape[1]):
    #     feat = timing_feats[pos_inds][:, i]
    #     feat_weight = construct_weight_vector(feat)
    #     feat_weight = 1 / feat_weight
    #     timing_weight_matrix.append(feat_weight)
    # timing_weight_matrix = torch.vstack(timing_weight_matrix).T
    # timing_loss = (timing_loss * timing_weight_matrix).sum() / timing_feats.shape[1]
    # # timing_loss = torch.sum(timing_loss * weight.reshape(-1, 1)) / (unmasked_feats.shape[1] * torch.sum(weight) + 1e-8)    
    return cls_loss, counts_loss, timing_loss

def mixture_nll_loss(y_pred, y_true):
    """
    Calculate the negative log likelihood loss for a Gaussian Mixture Model using PyTorch's built-in functions.
    Assumes the first 4 columns of y_pred are mus, the next 4 are sigmas, and the last 4 are mixing coefficients (pis).
    """
    N = y_pred.shape[0]

    # Extract mus, sigmas, and pis from predictions
    mu_pred = y_pred[:, :4]
    sigma_pred = torch.exp(y_pred[:, 4:8])  # Ensure sigma is positive
    pi_pred = torch.softmax(y_pred[:, 8:], dim=1)  # Ensure pis sum to 1

    # We only use mus from y_true for calculating the loss
    mu_true = y_true[:, :4].unsqueeze(2)  # Add an extra dimension for broadcasting
    
    # Create a Normal distribution for the predicted mus and sigmas
    normal_dist = dist.Normal(mu_pred.unsqueeze(1), sigma_pred.unsqueeze(1))

    # Calculate log probabilities for each Gaussian
    log_probs = normal_dist.log_prob(mu_true)  # This broadcasts mu_true over the different Gaussians
    
    # Weight log probabilities by the mixing coefficients and sum across the Gaussians
    weighted_log_probs = log_probs + torch.log(pi_pred).unsqueeze(1)
    
    # Use logsumexp to sum across the Gaussians dimension for numerical stability
    logsumexp_probs = torch.logsumexp(weighted_log_probs, dim=2)
    
    # Calculate the mean negative log likelihood loss
    loss = -torch.mean(logsumexp_probs)

    return loss

@torch.compile
def construct_weight_vector(input_tensor):
    # Get unique elements and the inverse mapping to reconstruct the input tensor
    _, inverse_indices = torch.unique(input_tensor, return_inverse=True, sorted=False)
    # Use bincount to count occurrences of each unique element based on inverse indices
    counts = torch.bincount(inverse_indices)
    # Map these counts back to the original tensor's shape using inverse indices
    frequency_map = counts[inverse_indices]
    return frequency_map

@torch.compile
def distribute_feats(candidates, coordinates, feats, init='zero', token=None):
    if init == 'zeros':
        result = torch.zeros(candidates.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
    elif init == 'ones':
        result = torch.ones(candidates.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
    elif init == 'token':
        result = token.repeat(candidates.shape[0], 1)
    else:
        result = torch.rand(candidates.shape[0], feats.shape[1], device=feats.device, dtype=feats.dtype)
    # Get unique batch indices from candidates
    batch_indices = torch.unique(candidates[:, 0])
    for batch_idx in batch_indices:
        # Find the rows for the current batch in candidates and coordinates
        candidates_batch = candidates[candidates[:, 0] == batch_idx]
        coordinates_batch = coordinates[coordinates[:, 0] == batch_idx]

        # Remove the batch index column for comparison
        candidates_batch = candidates_batch[:, 1:]
        coordinates_batch = coordinates_batch[:, 1:]

        # Find the indices of the matching rows within the current batch
        # We expand dimensions and use broadcasting to compare all pairs of rows
        expanded_candidates = candidates_batch.unsqueeze(1)
        expanded_coordinates = coordinates_batch.unsqueeze(0)

        # Compare all pairs of rows between candidates and coordinates
        matches = torch.all(expanded_candidates == expanded_coordinates, dim=2)

        # Get the indices of the matching rows
        matching_indices = matches.nonzero(as_tuple=True)[1]

        # Use these indices to index into feats and fill the result tensor
        if matching_indices.numel() > 0:
            # Map back to the original indices in the full tensor
            original_indices = (candidates[:, 0] == batch_idx).nonzero(as_tuple=True)[0]
            matching_candidates_indices = original_indices[matches.any(dim=1)]

            result[matching_candidates_indices] = feats[matching_indices]
    return result

@torch.compile
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

@torch.compile
def random_sampling(coords, feats, percentage):
    coords_masked_total = []
    feats_masked_total = []
    coords_total = []
    feats_total = []
    for b in range(coords[:,0].unique().shape[0]):
        coords_batch = coords[torch.where(coords[:,0] == b)]
        feats_batch = feats[torch.where(coords[:,0] == b)]
        num_samples = int(percentage * coords_batch.shape[0])
        indices = torch.randperm(coords_batch.shape[0])
        coords_masked_total.append(coords_batch[indices[:num_samples]])
        feats_masked_total.append(feats_batch[indices[:num_samples]])
        coords_total.append(coords_batch[indices])
        feats_total.append(feats_batch[indices])
    return torch.vstack(coords_masked_total), torch.vstack(feats_masked_total), torch.vstack(coords_total), torch.vstack(feats_total)

# from einops import repeat

# def random_indexes(size : int):
#     forward_indexes = np.arange(size)
#     np.random.shuffle(forward_indexes)
#     backward_indexes = np.argsort(forward_indexes)
#     return forward_indexes, backward_indexes

# def random_sampling(coords, feats, ratio : float):
#     B = coords[:,0].unique().shape[0]
#     coords_total = []
#     feats_total = []
#     forward_indexes_total = []
#     backward_indexes_total = []
#     for b in range(B):
#         coords_batch = coords[torch.where(coords[:,0] == b)]
#         feats_batch = feats[torch.where(coords[:,0] == b)]
#         T = coords_batch.shape[0]
#         remain_T = int(T * (1 - ratio))
#         indexes = [random_indexes(T)]
#         forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(coords.device)
#         backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(coords.device)
#         coords_batch = coords_batch[forward_indexes.flatten()]
#         feats_batch = feats_batch[forward_indexes.flatten()]
#         coords_total.append(coords_batch[:remain_T])
#         feats_total.append(feats_batch[:remain_T])
#         forward_indexes_total.append(forward_indexes.flatten())
#         backward_indexes_total.append(backward_indexes.flatten())
        
#     return torch.vstack(coords_total), torch.vstack(feats_total), forward_indexes_total, backward_indexes_total