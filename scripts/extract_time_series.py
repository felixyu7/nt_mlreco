import numpy as np
import awkward as ak
import glob
import time
from collections import defaultdict
import sys
import random

dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ntsr_sims/ice_ortho_det_7_2x_basecut_split_5k_test/'
output_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ntsr_sims/ortho_2x_tracks_total_ft_2kbins_test/'
# indices = [25, 26, 27, 37, 50, 53, 54, 55, 56, 57, 58, 59]
files_list = sorted(glob.glob(dir + '*.parquet'))
pos_offset = np.array([0, 0, 2000])

file = files_list[int(sys.argv[1])]

print("Processing file: {}".format(file))
data = ak.from_parquet(file)
data_labels = []
chunk_size = 250000
file_id = 0
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
    for coord, indices in unique_coords_dict.items():
        mask = np.zeros(pos_t.shape[0], dtype=bool)
        mask[indices] = True
        
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
        
        dom_data = {'first_hit_time': first_dom_hit, 'binned_time_counts': binned_time_counts.astype(np.int32).tolist()}
        data_labels.append(dom_data)
        
        # whenever data_labels hits chunk_size, save to parquet and reset
        if len(data_labels) == chunk_size:
            ak.to_parquet(data_labels, output_dir + str(int(sys.argv[1])) + '_chunk_' + str(file_id) + '.parquet')
            del data_labels
            data_labels = []
            file_id += 1
            print('chunk done!')

ak.to_parquet(data_labels, output_dir + str(int(sys.argv[1])) + '_chunk_' + str(file_id) + '.parquet')