import torch
import numpy as np
import torch.nn.functional as F

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