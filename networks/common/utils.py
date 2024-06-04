import torch
import numpy as np
import torch.nn.functional as F
import random

def LogCoshLoss(pred, truth):
    """LogCosh loss function. approximated for easier to compute gradients"""
    x = pred - truth
    return (x + torch.nn.functional.softplus(-2.0 * x) - np.log(2.0)).mean()

def AngularDistanceLoss(pred, truth, weights=None, eps=1e-7, reduction="mean"):
    """Angular distance loss function"""
    # clamp prevents invalid input to arccos
    if weights is None:
        x = torch.acos(torch.clamp(F.cosine_similarity(pred, truth), min=-1.+eps, max=1.-eps)) / np.pi
    else:
        x = (torch.acos(torch.clamp(F.cosine_similarity(pred, truth), min=-1.+eps, max=1.-eps)) / np.pi) * weights
    if reduction == "mean":
        return x.mean()
    else:
        return x
    
def CombinedAngleEnergyLoss(pred, truth):
    """Combined loss function for both angular and energy reco"""
    angles_pred = pred[:, 1:]
    angles_truth = truth[:, 1:]
    energy_pred = pred[:, 0]
    energy_truth = truth[:, 0]
    # 0.5 weighting on the energy loss since it tends to be larger
    loss = AngularDistanceLoss(angles_pred, angles_truth) + (0.5 * LogCoshLoss(energy_pred, energy_truth))
    return loss

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # clip prevents invalid input to arccos
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0+1e-7, 1.0-1e-7))

@torch.compile
def generate_geo_mask(coords, geo_coords):
    if coords.shape[1] == 5:
        mask = (coords[:, 1:4] == geo_coords[:, None]).all(dim=2).any(dim=0)
    else:
        mask = (coords[:, :3] == geo_coords[:, None]).all(dim=2).any(dim=0)
    return mask

def generate_geo_mask_cuda(coords, geo_coords, cuda=True, chunk_size=-1, mode='batch'):
    if not cuda:
        geo_coords = geo_coords.to(coords.device)
        if mode == 'batch':
            mask = (coords[:, 1:4] == geo_coords[:, None]).all(dim=2).any(dim=0)
        else:
            mask = (coords[:, :3] == geo_coords[:, None]).all(dim=2).any(dim=0)
        return mask
    else:
        if ((chunk_size == -1) or (chunk_size > coords.shape[0])):
            chunk_size = coords.shape[0]
        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device='cpu')
        geo_coords_gpu = geo_coords.cuda()
        for i in range(0, coords.shape[0], chunk_size):
            end = min(i+chunk_size, coords.shape[0])
            coords_chunk = coords[i:end].cuda()
            if mode == 'batch':
                mask_chunk = (coords_chunk[:, 1:4] == geo_coords_gpu[:, None]).all(dim=2).any(dim=0)
            else:
                mask_chunk = (coords_chunk[:, :3] == geo_coords_gpu[:, None]).all(dim=2).any(dim=0)
            mask[i:end] = mask_chunk.cpu()
        return mask

@torch.compile
def tensor_row_match_mask(n_by_5_tensor, m_by_5_tensor):
    # Expand the Nx5 tensor to Nx1x5 and the Mx5 tensor to 1xMx5
    # This will allow us to compare each row of the first tensor against all rows of the second tensor
    n_expanded = n_by_5_tensor.unsqueeze(1)
    m_expanded = m_by_5_tensor.unsqueeze(0)
    
    # Compare the expanded tensors to get a NxMx5 tensor of True/False values
    comparison = n_expanded == m_expanded
    
    # Reduce along the last dimension (5) to get a NxM tensor of True/False values
    # A row is True only if all elements in that row are True (i.e., the row matches completely)
    matches = comparison.all(dim=2)
    
    # Reduce along the M dimension to check if each row in Nx5 tensor has a match in Mx5 tensor
    # This will result in an Nx1 tensor of True/False values
    mask = matches.any(dim=1, keepdim=True)
    
    return mask

@torch.compile
def mask_out_input_strings(Nx5_tensor, Mx5_tensor):
    # Extract the 2nd and 3rd columns from both tensors
    Nx5_columns = Nx5_tensor[:, 1:3].unsqueeze(1)  # Add an extra dimension to Nx5 tensor for broadcasting
    Mx5_columns = Mx5_tensor[:, 1:3].unsqueeze(0)  # Add an extra dimension to Mx5 tensor for broadcasting
    
    # Compare the 2nd and 3rd columns of Nx5 tensor with those of Mx5 tensor using broadcasting
    # The result will be a tensor of shape (N, M, 2) where each element is True if the corresponding elements match
    comparison = Nx5_columns == Mx5_columns
    
    # We need to check if both columns match, so we take the logical AND along the last dimension
    # This will give us a tensor of shape (N, M) where True indicates a match for both columns
    matches = comparison.all(dim=2)
    
    # Now we check if there are any matches across the M dimension for each N
    # This will give us a tensor of shape (N,) where True indicates that there is at least one match in Mx5 tensor
    any_matches = matches.any(dim=1)
    
    # The mask should be True where there are no matches, so we invert the any_matches tensor
    mask = ~any_matches
    
    return mask

@torch.compile
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

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

def get_p_of_bins(metric, es, bins, p):
    """For a given metric, bin by energy and return the percentile p of the metric for each bin"""
    indices = np.digitize(es, bins)
    ps = []
    for i in range(1, bins.shape[0] + 1):
        ps.append(np.percentile(metric[np.where(indices == i)], p))
    return np.array(ps)

def get_mean_of_bins(metric, es, bins):
    """For a given metric, bin by energy and return the mean of the metric for each bin"""
    indices = np.digitize(es, bins)
    ps = []
    for i in range(1, bins.shape[0] + 1):
        ps.append(np.mean(metric[np.where(indices == i)]))
    return np.array(ps)