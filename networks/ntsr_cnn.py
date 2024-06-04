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

def ntsr_loss(outputs, imgs):
    # counts_label = imgs[:, 3, :, :]
    # true_counts = torch.exp(counts_label) - 1
    # cls_label = (counts_label > 0).float()
    # weighting_factor = 1 / (cls_label.sum() / cls_label.flatten().shape[0])
    # weights = (cls_label * (weighting_factor - 1)) + 1
    # cls_loss = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], cls_label, weight=weights)

    # counts_loss = F.mse_loss(outputs[:, 1, :, :], counts_label, reduction='none')
    # counts_loss = (counts_loss * true_counts).sum() / true_counts.sum()
    # counts_loss = counts_loss + cls_loss
    
    counts_label = imgs[:, 3, :, :]
    cls_label = (counts_label > 0).float()
    # cls_loss = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], cls_label)
    # counts_loss = F.mse_loss(outputs[:, 1, :, :], counts_label, reduction='none')
    # counts_loss = (counts_loss * cls_label.unsqueeze(1)).sum() / cls_label.sum()
    # counts_loss = counts_loss + cls_loss
    # counts_loss = F.mse_loss(F.relu(outputs[:, 0, :, :]), counts_label)
    counts_loss = F.poisson_nll_loss(F.relu(outputs[:, 0, :, :]), counts_label, log_input=False, full=False)
    
    # time_pdf_loss = F.mse_loss(outputs[:, 2:, :, :], imgs[:, 4:, :, :], reduction='none')
    time_pdf_loss = F.mse_loss(outputs[:, 1:, :, :], imgs[:, 4:, :, :], reduction='none')
    time_pdf_loss = (time_pdf_loss * cls_label.unsqueeze(1)).sum() / (cls_label.sum() * 64)
    
    return counts_loss, time_pdf_loss
  
# def ntsr_loss(outputs, time_series, imgs, true_time_series):
#     counts_label = imgs[:, 3, :, :]
#     true_counts = torch.exp(counts_label) - 1
#     cls_label = (counts_label > 0).float()
#     weighting_factor = 1 / (cls_label.sum() / cls_label.flatten().shape[0])
#     weights = (cls_label * (weighting_factor - 1)) + 1
#     cls_loss = F.binary_cross_entropy_with_logits(outputs[:, 0, :, :], cls_label, weight=weights)

#     counts_loss = F.mse_loss(outputs[:, 1, :, :], counts_label, reduction='none')
#     counts_loss = (counts_loss * true_counts).sum() / true_counts.sum()
#     counts_loss = counts_loss + cls_loss
#     # counts_loss = F.poisson_nll_loss(F.relu(outputs[:, 0, :, :]), counts_label, log_input=False, full=False)
    
#     # outputs[:, 9:17, :, :] = F.relu(outputs[:, 9:17, :, :]) + 1e-6
#     # outputs[:, 17:25, :, :] = F.softmax(outputs[:, 17:25, :, :], dim=1)
#     # time_pdf_loss = F.mse_loss(outputs[:, 1:, :, :], imgs[:, 4:, :, :], reduction='none')
#     # time_pdf_loss = (time_pdf_loss * counts_label.unsqueeze(1)).sum() / counts_label.sum()
#     # time_pdf_loss = lognorm_mixture_binned_likelihood_loss(outputs, imgs, true_photons)
    
#     # post-process outputs (apply relu and cumsum)
#     # time_pdf_outputs = torch.cumsum(F.relu(outputs[:, 2:, :, :]), dim=1)
#     # time_pdf_outputs = F.relu(outputs[:, 2:, :, :])
#     # time_pdf_loss = F.smooth_l1_loss(outputs[:, 2:, :, :], imgs[:, 4:, :, :], reduction='mean')
#     # time_pdf_mask = (imgs[:, 4:, :, :] >= 0).float()
#     # time_pdf_loss = (time_pdf_loss * time_pdf_mask)
#     # time_pdf_loss = (time_pdf_loss * true_counts.unsqueeze(1)).sum() / true_counts.sum()
    
#     true_time_series = (true_time_series.permute(0, 2, 3, 1).reshape(-1, time_series.shape[1]) > 0).float()
#     time_series_mask = (torch.sum(true_time_series, dim=1) > 0).float().reshape(-1, 1)
#     time_pdf_loss = F.binary_cross_entropy_with_logits(time_series, true_time_series, weight=time_series_mask)
    
#     return counts_loss, time_pdf_loss

@torch.compile
def compute_lognorm_pdf(x, mu, sigma, weight):
    # Ensure dimensions match up to broadcast: [N, C, H, W, X] where X is linspace size
    x = x.view(1, 1, 1, 1, -1)  # Add dimensions to x for broadcasting

    # Ensure mu, sigma, and weight are expanded in the last dimension
    mu = mu.unsqueeze(-1)     # Shape: [batch_size, n_components, height, width, 1]
    sigma = sigma.unsqueeze(-1) # Shape: [batch_size, n_components, height, width, 1]
    weight = weight.unsqueeze(-1) # Shape: [batch_size, n_components, height, width, 1]

    # Compute the log-normal PDF
    factor = 1 / (x * sigma * ((2 * torch.pi) ** 0.5))
    exp_component = torch.exp(-0.5 * ((torch.log(x) - mu) / sigma) ** 2)
    pdf = factor * exp_component

    # Compute the mixture of PDFs
    mixture_pdf = (weight * pdf).sum(dim=1)  # Sum over the component dimension
    
    return mixture_pdf

@torch.compile
def lognorm_mixture_binned_likelihood_loss(outputs, imgs, true_photons_binned):
    # Extract params
    # pred_mus = outputs[:, 1:9, :, :]
    # pred_sigmas = (F.relu(outputs[:, 9:17, :, :]) + 1e-6) ** 0.5 # ensure always positive
    # pred_weights = F.softmax(outputs[:, 17:25, :, :], dim=1)
    
    # get bin centers
    # bin_edges = torch.linspace(0, 5000, 1001).to(outputs.device)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # pred_pdfs = compute_lognorm_pdf(bin_centers, pred_mus, pred_sigmas, pred_weights)
    
    pred_pdfs = F.relu(outputs[:, 1:, :, :])
    
    total_loss = 0.
    for b in range(outputs.shape[0]):
        true_photon_binned = true_photons_binned[true_photons_binned[:,0] == b][:, 1:]
        
        true_photon_binned_pos_x = true_photon_binned[:,0].int()
        true_photon_binned_pos_y = true_photon_binned[:,1].int()
        true_photon_binned = true_photon_binned[:, 2:]
        
        # pred_mu = pred_mus[b][:, true_photon_binned_pos_x, true_photon_binned_pos_y].T
        # pred_sigma = pred_sigmas[b][:, true_photon_binned_pos_x, true_photon_binned_pos_y].T
        # pred_weight = pred_weights[b][:, true_photon_binned_pos_x, true_photon_binned_pos_y].T
        
        # # construct pdf
        # bin_edges = torch.linspace(0, 5000, 501).to(outputs.device)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # pdfs = log_normal_pdf(bin_centers.unsqueeze(0).expand(pred_mu.size(0), -1).unsqueeze(1), 
        #                       pred_mu.unsqueeze(-1).expand(-1, -1, bin_centers.size(0)), 
        #                       pred_sigma.unsqueeze(-1).expand(-1, -1, bin_centers.size(0)))
        # pdfs = pdfs * pred_weight.unsqueeze(-1).expand(-1, -1, bin_centers.size(0))
        # pdfs = pdfs.sum(dim=1)
        
        # # convert to counts space
        # pred_binned_pdfs = pdfs * true_photon_binned.sum(dim=1).reshape(-1, 1)
        
        pred_binned_pdfs = pred_pdfs[b][:, true_photon_binned_pos_x, true_photon_binned_pos_y].T
        pred_binned_pdfs = F.softmax(pred_binned_pdfs, dim=1)
        loss = wasserstein_distance_1d(pred_binned_pdfs, true_photon_binned)
        # weights = true_photon_binned + 1
        # loss = F.poisson_nll_loss(pred_binned_pdfs, true_photon_binned, log_input=False, full=False, reduction='none')
        # loss = (loss * weights).sum() / weights.sum()
        total_loss += loss
        
    return total_loss / outputs.shape[0]
        
@torch.compile
def wasserstein_distance_1d(predicted, target):
    # Ensure the histograms are normalized to be proper probability distributions
    # predicted = predicted / predicted.sum(dim=1, keepdim=True)
    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
    
    # Compute the cumulative distribution functions (CDFs)
    cdf_predicted = torch.cumsum(predicted, dim=1)
    cdf_target = torch.cumsum(target, dim=1)
    
    # Compute the Wasserstein distance for each histogram in the batch
    wasserstein_distance = torch.sum(torch.abs(cdf_predicted - cdf_target), dim=1)
    
    # Return the mean Wasserstein distance over the batch
    return wasserstein_distance.mean()

@torch.compile
def lognorm_mixture_kl_div_loss(outputs, imgs):
    epsilon = 1e-8
    # Extract params
    pred_mus = outputs[:, 1:9, :, :]
    pred_sigmas = (F.relu(outputs[:, 9:17, :, :]) + 1e-6) ** 0.5 # ensure always positive
    pred_weights = F.softmax(outputs[:, 17:25, :, :], dim=1)

    mus = imgs[:, 4:12, :, :]
    sigmas = ((imgs[:, 12:20, :, :]) ** 0.5) + epsilon
    weights = imgs[:, 20:28, :, :]

    hit_inds = (imgs[:, 3, :, :] > 0).float()
    
    x = torch.linspace(epsilon, 2500, 500).to(outputs.device)  # Avoid division by zero at x=0

    true_pdf = compute_lognorm_pdf(x, mus, sigmas, weights)
    pred_pdf = compute_lognorm_pdf(x, pred_mus, pred_sigmas, pred_weights)
    
    # Avoid log(0) issues
    kl_div = true_pdf * torch.log((true_pdf + epsilon) / (pred_pdf + epsilon))
    kl_div = kl_div.sum(dim=-1)  # Sum over linspace

    # Masked mean KL divergence
    kl_div_masked = kl_div * hit_inds
    loss = kl_div_masked.sum() / hit_inds.sum()
    return loss
    
@torch.compile
def log_normal_pdf(x, mu, sigma):
    """Calculate the log-normal probability density function."""
    return torch.exp(-0.5 * ((torch.log(x) - mu) / sigma) ** 2) / (x * sigma * ((2 * torch.pi) ** 0.5))

@torch.compile
def lognorm_mixture_nll_loss(outputs, imgs, true_photons):
    batch_size = outputs.shape[0]
    n_components = 8
    total_loss = 0
    
    for b in range(batch_size):
        output = outputs[b]
        img = imgs[b]
        true_photon = true_photons[true_photons[:, 0] == b][:, 1:]
        hit_inds = img[3, :, :] > 0
        positions = torch.round(img[:3, :, :][:, hit_inds] * 100).T
        
        # Flatten the parameters and reshape for computation
        params = output[1:25, :, :][:, hit_inds].T.reshape(-1, n_components, 3)
        sigmas = F.relu(params[:, :, 1]) + 1e-6  # sigmas will always be positive
        pis = F.softmax(params[:, :, 2], dim=0) # pis is a pdf
        
        event_loss = 0
        randperm = torch.randperm(len(positions))
        if len(positions) > 50:
            randperm = randperm[:50]
        # for pos_idx, pos in enumerate(positions):
        for pos_idx in randperm:
            pos = positions[pos_idx]
            # Find the times that correspond to the current position
            times = true_photon[torch.all(true_photon[:, :3] == pos, dim=1)]
            mu = params[:, :, 0][pos_idx]
            sigma = sigmas[pos_idx]
            pi = pis[pos_idx]  
            
            # Calculate log-normal PDF for each component and data point
            densities = log_normal_pdf(times[:,3][:, None] + 1e-8, mu, sigma)  # Shape (m, n)
            
            # Weighted mixture of PDFs
            mixture_pdf = torch.sum(pi * densities, dim=1)  # Shape (m,)
            
            # Compute weighted log likelihood
            log_likelihood = torch.log(mixture_pdf + 1e-8)  # Shape (m,)
            
            # Calculate weighted negative log likelihood
            weighted_nll = -torch.sum(times[:,4] * log_likelihood)  # Scalar
            event_loss += (weighted_nll / len(times))
        
        if len(positions) > 0:
            event_loss /= len(positions)
        total_loss += event_loss
    return total_loss / batch_size

def asymmetric_gaussian(x, mu, sigma, r):
    N = 2 / (((2 * torch.pi) ** 0.5) * sigma * (r + 1))
    x = x.unsqueeze(-1)  # Shape of x: [1000, 1] if originally [1000]
    return torch.where(x <= mu,
                       N * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2)),
                       N * torch.exp(-(x - mu) ** 2 / (2 * (sigma * r) ** 2)))

def mixture_density(x, mus, sigmas, rs, weights):
    # Broadcast all parameters to the shape of [num_points, n_components]
    # where num_points is the number of elements in x
    densities = weights * asymmetric_gaussian(x, mus, sigmas, rs)
    return torch.sum(densities, dim=-1)  # Sum over the last dimension (components)

def asymmetric_gaussian_mixture_kl_div_loss(outputs, imgs):
    # Extract AGMM params
    pred_mus = outputs[:, 2:6, :, :]
    pred_sigmas = torch.exp(outputs[:, 6:10, :, :]) + 1
    pred_rs = torch.exp(outputs[:, 10:14, :, :]) + 1
    pred_weights = F.softmax(outputs[:, 14:18, :, :], dim=1)

    mus = imgs[:, 4:8, :, :]
    sigmas = torch.exp(imgs[:, 8:12, :, :])
    rs = torch.exp(imgs[:, 12:16, :, :])
    weights = imgs[:, 16:20, :, :]

    hit_inds = imgs[:, 3, :, :] > 0
    hit_inds = hit_inds.unsqueeze(1).expand_as(pred_mus)

    # Using expanded hit indices to filter parameters
    pred_mus = pred_mus[hit_inds].reshape(-1, 4)
    pred_sigmas = pred_sigmas[hit_inds].reshape(-1, 4)
    pred_rs = pred_rs[hit_inds].reshape(-1, 4)
    pred_weights = pred_weights[hit_inds].reshape(-1, 4)

    mus = mus[hit_inds].reshape(-1, 4)
    sigmas = sigmas[hit_inds].reshape(-1, 4)
    rs = rs[hit_inds].reshape(-1, 4)
    weights = weights[hit_inds].reshape(-1, 4)

    # Create range for t
    t_range = torch.linspace(0, 5000, 2500).to(outputs.device).unsqueeze(-1)

    # Compute mixture densities using broadcasting
    pred_y_mixture = mixture_density(t_range, pred_mus, pred_sigmas, pred_rs, pred_weights)
    y_mixture = mixture_density(t_range, mus, sigmas, rs, weights)

    # Compute KL divergence
    kl_div = y_mixture * torch.log(y_mixture / (pred_y_mixture + 1e-8) + 1e-8)
    loss = torch.sum(kl_div, dim=0)
    import pdb; pdb.set_trace()
    return torch.mean(loss)

def asymmetric_gaussian_mixture_nll_loss(outputs, imgs, true_photons):
    """
    Compute the negative log likelihood of the data given the mixture model parameters.
    Vectorized version for efficiency.
    """
    batch_size = outputs.shape[0]
    n_components = 4
    total_loss = 0
    
    for b in range(batch_size):
        output = outputs[b]
        img = imgs[b]
        true_photon = true_photons[true_photons[:, 0] == b][:, 1:]
        hit_inds = img[3, :, :] > 0
        positions = torch.round(img[:3, :, :][:, hit_inds] * 100).T
        
        # Flatten the parameters and reshape for computation
        params = output[2:18, :, :][:, hit_inds].T.reshape(-1, n_components, 4)
        mu = F.relu(params[:, :, 0]) * 1000 # means will always be positive
        sigma = torch.exp(params[:, :, 1]) + 1 # sigmas must be above 1
        r = torch.exp(params[:, :, 2]) + 1 # rs must be above 1
        weight = F.softmax(params[:, :, 3], dim=0) # weight is a pdf
        
        event_loss = 0
        for pos_idx, pos in enumerate(positions):
            # Find the times that correspond to the current position
            times = true_photon[torch.all(true_photon[:, :3] == pos, dim=1), 3].reshape(-1, 1)
            
            # Select the parameters for the current position
            mu_pos = mu[pos_idx]
            sigma_pos = sigma[pos_idx]
            r_pos = r[pos_idx]
            weight_pos = weight[pos_idx]
            
            # Calculate the gaussian mixture components
            N = 2 / (((2 * np.pi) ** 0.5) * sigma_pos * (r_pos + 1))
            gaussian_left = N * torch.exp(-(times - mu_pos) ** 2 / (2 * sigma_pos ** 2))
            gaussian_right = N * torch.exp(-(times - mu_pos) ** 2 / (2 * (sigma_pos * r_pos) ** 2))
            gaussian = torch.where(times <= mu_pos, gaussian_left, gaussian_right)
            
            # Weighted sum of gaussians
            log_likelihood = torch.sum(weight_pos * gaussian, dim=1)
            
            # Compute the loss for the current position
            event_loss += (-torch.sum(torch.log(log_likelihood + 1e-8))) / len(times)
        
        if len(positions) > 0:
            event_loss /= len(positions)
        total_loss += event_loss
    
    return total_loss / batch_size

# @torch.compile
# def asymmetric_gaussian_mixture_nll_loss(outputs, imgs, true_photons):
#     """
#     Compute the negative log likelihood of the data given the mixture model parameters.
#     """
#     total_loss = 0
#     for b in range(outputs.shape[0]):
#         output = outputs[b]
#         img = imgs[b]
#         true_photon = true_photons[true_photons[:,0] == b][:,1:]
#         hit_inds = (img[3, :, :] > 0)
        
#         positions = torch.hstack([img[0, :, :][hit_inds],
#                                   img[1, :, :][hit_inds], 
#                                   img[2, :, :][hit_inds]]) 
#         positions = positions * 1000.
#         event_loss = 0
#         for pos in positions:
#             # get true photon times that happen at pos
#             times = true_photon[torch.all(true_photon[:,:3] == pos, dim=1)][:, 3].reshape(-1, 1)
#             # get predicted params at pos
#             params = torch.vstack([output[2, :, :][hit_inds],
#                                    output[6, :, :][hit_inds],
#                                    output[10, :, :][hit_inds],
#                                    output[14, :, :][hit_inds],
#                                    output[3, :, :][hit_inds],
#                                    output[7, :, :][hit_inds],
#                                    output[11, :, :][hit_inds],
#                                    output[15, :, :][hit_inds],
#                                    output[4, :, :][hit_inds],
#                                    output[8, :, :][hit_inds],
#                                    output[12, :, :][hit_inds],
#                                    output[16, :, :][hit_inds],
#                                    output[5, :, :][hit_inds],
#                                    output[9, :, :][hit_inds],
#                                    output[13, :, :][hit_inds],
#                                    output[17, :, :][hit_inds]])
#             n_components = 4
#             log_likelihood = 0
#             for i in range(n_components):
#                 mu, sigma, r, weight = params[i*4:(i+1)*4]
#                 mu = mu * 1000.
#                 sigma = torch.exp(sigma)
#                 r = torch.exp(r)
#                 N = 2 / (((2 * np.pi)**0.5) * sigma * (r + 1))
#                 gaussian_left = N * torch.exp(-(times - mu) ** 2 / (2 * sigma ** 2))
#                 gaussian_right = N * torch.exp(-(times - mu) ** 2 / (2 * (sigma * r) ** 2))
#                 gaussian = torch.where(times <= mu, gaussian_left, gaussian_right)
#                 log_likelihood += weight * gaussian
#             event_loss += ((-torch.sum(torch.log(log_likelihood + 1e-8))) / len(times))
#         event_loss /= len(positions)
#         total_loss += event_loss
#     return total_loss / outputs.shape[0]
        