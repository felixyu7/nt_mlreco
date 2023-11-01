# preliminary version: intersection point proposal network

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.common.resnet_block import ResNetBlock
from networks.sscnn import SSCNN
from networks.common.utils import AngularDistanceLoss, angle_between
import MinkowskiEngine as ME
import lightning.pytorch as pl
from scipy.spatial.distance import cdist

class IPPN(pl.LightningModule):
    def __init__(self, in_features, reps=2, depth=8, first_num_filters=16, stride=2, 
                 expand=False, dropout=0., input_dropout=0., output_dropout=0., scaling='linear', 
                 output_layer=True, mode='both', n=1000, r=700, D=4, batch_size=128, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.validation_step_outputs = []
        self.validation_step_labels = []

        self.test_step_outputs = []
        self.test_step_labels = []
        self.test_results = {}

        self.encoder = SSCNN(1, reps=reps, 
                        depth=depth, 
                        first_num_filters=first_num_filters, 
                        stride=stride, 
                        dropout=dropout,
                        input_dropout=input_dropout,
                        output_dropout=output_dropout,
                        mode=mode,
                        D=D,
                        output_layer=False,
                        batch_size=batch_size, 
                        lr=lr, 
                        weight_decay=weight_decay)
        
        num_features = first_num_filters * (depth - 1)
        self.num_points = n
        self.radius = r
        self.entry_anchors, self.exit_anchors = fibonacci_sphere_points(self.num_points, self.radius)
        self.entry_point_classifier = ME.MinkowskiConvolution(num_features, self.entry_anchors.shape[0], kernel_size=1, stride=1, dimension=4, bias=True)
        self.exit_point_classifier = ME.MinkowskiConvolution(num_features, self.exit_anchors.shape[0], kernel_size=1, stride=1, dimension=4, bias=True)

    def forward(self, x):
        feature_map = self.encoder(x)
        entry_point_logits = self.entry_point_classifier(feature_map)
        exit_point_logits = self.exit_point_classifier(feature_map)
        return entry_point_logits, exit_point_logits

    def training_step(self, batch, batch_idx):
        coords, feats, labels = batch
        inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords, device=self.device,
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        entry_point_logits, exit_point_logits = self(inputs)
        loss = ippn_cls_loss(entry_point_logits, exit_point_logits, self.entry_anchors, self.exit_anchors, labels)
        self.log("cls_loss", loss, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss

def ippn_cls_loss(entry_point_logits, exit_point_logits, entry_anchors, exit_anchors, labels, sigma=100.):
    labels_cpu = labels.cpu().detach().numpy()
    start_pts = labels[:, 4:7]
    dir_vec = labels[:, 1:4]

    entry_distances = cdist(labels_cpu[:, 0, :], entry_anchors.cpu().numpy())
    exit_distances = cdist(labels_cpu[:, 1, :], exit_anchors.cpu().numpy())

    label_entry = torch.from_numpy(generate_gaussian_weights(entry_distances, sigma=sigma)).cuda()
    label_exit = torch.from_numpy(generate_gaussian_weights(exit_distances, sigma=sigma)).cuda()

    cls_loss = F.binary_cross_entropy_with_logits(entry_point_logits.F, label_entry) + \
    F.binary_cross_entropy_with_logits(exit_point_logits.F, label_exit)

    return cls_loss

def sphere_line_intersection(start_points, direction_vectors, r):
    # Calculate the coefficients of the quadratic equation
    a = np.sum(direction_vectors**2, axis=1)
    b = 2 * np.sum(start_points * direction_vectors, axis=1)
    c = np.sum(start_points**2, axis=1) - r**2

    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Calculate the two roots of the quadratic equation
    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)

    # Calculate the intersection points
    intersection_points1 = start_points + np.expand_dims(t1, axis=1) * direction_vectors
    intersection_points2 = start_points + np.expand_dims(t2, axis=1) * direction_vectors

    # Combine the intersection points into a single array
    result = np.stack((intersection_points1, intersection_points2), axis=1)

    # Replace the intersection points of lines that do not intersect the sphere with NaNs
    result[discriminant < 0] = np.nan

    return result

def fibonacci_sphere_points(n, r):
    points = []
    offset = 2.0 / n
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2)
        r_ = np.sqrt(1 - y*y)
        phi = ((i + 1) % n) * increment
        x = np.cos(phi) * r_
        z = np.sin(phi) * r_
        points.append([x, y, z])
    points = np.array(points) * r
    # FOR DOWNGOING EVENTS: ENTRY IS ON TOP, EXIT ON BOTTOM
    # Sort the coordinates based on z values
    z_sorted_indices = np.argsort(points[:, 2])[::-1]  # sort in descending order
    sorted_coords = points[z_sorted_indices]

    # Split the sorted coordinates into two arrays
    # n1 = points.shape[0] // 2
    entry_anchor_points = np.copy(sorted_coords)
    exit_anchor_points = np.copy(sorted_coords)
    return torch.from_numpy(entry_anchor_points), torch.from_numpy(exit_anchor_points)

def generate_gaussian_weights(distances, sigma=1.0):
    """
    Generates weights based on distances from a given index using a Gaussian-like falloff.

    Parameters:
    distances (ndarray): 2D array of distances between set1 and set2, with shape (batch_size, num_distances).
    sigma (float): Standard deviation to use for the Gaussian-like falloff.

    Returns:
    ndarray: Resultant 2D weight array of shape (batch_size, set1_size).
    """
    batch_size, num_distances = distances.shape
    
    # Compute the Gaussian-like falloff based on distances
    weights = np.exp(-distances**2 / (2*sigma**2))
    
    # Normalize weights to be between 0 and 1 for each batch
    weights = (weights - weights.min(axis=1, keepdims=True)) / (weights.max(axis=1, keepdims=True) - weights.min(axis=1, keepdims=True))
    
    return weights