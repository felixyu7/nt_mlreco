import numpy as np
import awkward as ak
import glob
import time
from collections import defaultdict
import sys

import torch
import torch.nn as nn
    
class Time_Series_Encoder(nn.Module):
    def __init__(self, in_features=3000, latent_dim=64):
        super(Time_Series_Encoder, self).__init__()
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
        
    def encode(self, inputs):
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        eps = 0.
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ntsr_sims/ice_ortho_det_7_2x_basecut_split_5k_test/'
output_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ntsr_sims/ortho_2x_tracks_total_vae_v3_latents64_test/'
# indices = [25, 26, 27, 37, 50, 53, 54, 55, 56, 57, 58, 59]
files_list = sorted(glob.glob(dir + '*.parquet'))
pos_offset = np.array([0, 0, 2000])

file = files_list[int(sys.argv[1])]

print("Processing file: {}".format(file))
data = ak.from_parquet(file)

checkpoint = torch.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/ckpts/time_series_vae_64_v3_epoch4.ckpt',  
                        map_location=torch.device('cpu'))
encoder_weights = {k: v for k, v in checkpoint['state_dict'].items() if (k.startswith('encoder') or k.startswith('fc_mu') or k.startswith('fc_logvar'))}

# load in only the encoder weights to net
net = Time_Series_Encoder()
net.load_state_dict(encoder_weights)
net.eval()

data_labels = []
for event in data:
    stime = time.time()
    pos_t = np.array([event.photons.sensor_pos_x.to_numpy(),
                event.photons.sensor_pos_y.to_numpy(),
                event.photons.sensor_pos_z.to_numpy(),
                event.photons.string_id.to_numpy(),
                event.photons.sensor_id.to_numpy(),
                event.photons.t.to_numpy() - event.photons.t.to_numpy().min()]).T
    unique_coords_dict = defaultdict(list)
    for i, coord in enumerate(pos_t[:, :5]):
        unique_coords_dict[tuple(coord)].append(i)
    event_labels = {'string_sensor_pos': [], 'pos': [], 'num_hits': [], 'latents': [], 'first_time_hit': []}
    for coord, indices in unique_coords_dict.items():
        event_labels['pos'].append((np.array(coord)[:3] + pos_offset).tolist())
        event_labels['string_sensor_pos'].append((np.array(coord)[3:5]).tolist())
        mask = np.zeros(pos_t.shape[0], dtype=bool)
        mask[indices] = True
        event_labels['num_hits'].append(mask.sum())
        
        dom_times = pos_t[:,-1][mask]
        first_dom_hit = dom_times.min()
        
        # bin hits on individual sensors
        num_bins = 3000
        max_time = 3000
        bin_edges = np.linspace(0, max_time, num_bins + 1, endpoint=True)
        
        dom_times = dom_times - first_dom_hit # shift by first hit time
        
        # do not consider hits with time > max_time
        max_time_mask = (dom_times < max_time)
        
        binned_times = np.digitize(dom_times[max_time_mask], bin_edges, right=True)
        binned_time_counts = np.histogram(binned_times, bins=bin_edges)[0]
        
        # put hits with time > max_time in the last bin
        binned_time_counts[-1] += np.sum(~max_time_mask)
        
        input_vector = torch.from_numpy(binned_time_counts).float()
        input_vector = torch.log(input_vector + 1)
        latent_vector = net(input_vector)
        event_labels['latents'].append(latent_vector.tolist())
        event_labels['first_time_hit'].append(first_dom_hit)
    
    data_labels.append(event_labels)
    print('Time taken: {}'.format(time.time() - stime))
new_data = ak.with_field(data, data_labels, where='latents')
ak.to_parquet(new_data, output_dir + file.split('ice_ortho_det_7_2x_basecut_split_5k_test/')[1])
        