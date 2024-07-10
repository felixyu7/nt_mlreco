import numpy as np
import awkward as ak
import glob
import time
from collections import defaultdict
import sys

from scipy.interpolate import LinearNDInterpolator

dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ntsr_sims/ortho_2x_tracks_total_vae_latents64_test/'
output_dir = '/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ntsr_sims/ortho_2x_tracks_total_interpolated_latents_test/'
# indices = [25, 26, 27, 37, 50, 53, 54, 55, 56, 57, 58, 59]
files_list = sorted(glob.glob(dir + '*.parquet'))
pos_offset = np.array([0, 0, 2000])

geo_x = np.linspace(-420, 420, 15)
geo_y = np.linspace(-420, 420, 15)
geo_z = np.linspace(-(60*15)/2, (60*15)/2, 61)
# Create the meshgrid for all combinations
X, Y, Z = np.meshgrid(geo_x, geo_y, geo_z, indexing='ij')
# Flatten the meshgrid arrays and stack them as columns
coordinates = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
sparse_geo_x = np.linspace(-420, 420, 8)
sparse_geo_y = np.linspace(-420, 420, 8)
# Create the meshgrid for all combinations
X, Y, Z = np.meshgrid(sparse_geo_x, sparse_geo_y, geo_z, indexing='ij')
# Flatten the meshgrid arrays and stack them as columns
sparse_coordinates = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
real_sensors = sparse_coordinates
mask = np.array([row in sparse_coordinates.tolist() for row in coordinates.tolist()])
virtual_sensors = coordinates[~mask]

file = files_list[int(sys.argv[1])]

print("Processing file: {}".format(file))
data = ak.from_parquet(file)

data_labels = []
interpolated_labels_total = []
for event in data:
    stime = time.time()
    
    # xs = event.photons.sensor_pos_x.to_numpy()
    # ys = event.photons.sensor_pos_y.to_numpy()
    # zs = event.photons.sensor_pos_z.to_numpy() + 2000
    # ts = event.photons.t.to_numpy() - event.photons.t.to_numpy().min()

    # pos_t = np.array([
    #     xs,
    #     ys,
    #     zs,
    #     ts
    # ]).T
    
    # spos_t = pos_t[np.argsort(pos_t[:,-1])]
    # _, indices, feats = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
    # pos_t = spos_t[indices]
    # pos_t = np.trunc(pos_t)
    # counts = feats.reshape(-1, 1).astype(np.float32)
    # feats = np.hstack([counts.reshape(-1, 1), pos_t[:,-1].reshape(-1, 1)])
    # pts = pos_t[:,:3]
    
    pts = event.latents.pos.to_numpy()
    counts = event.latents.num_hits.to_numpy()
    latents = event.latents.latents.to_numpy()
    feats = np.hstack([counts.reshape(-1, 1), latents])
    
    # remove coordinate (row) from pts if it is contained in virtual_sensors
    mask = np.array([row in virtual_sensors.tolist() for row in pts.tolist()])
    pts = pts[~mask]
    feats = feats[~mask]
    
    pts += np.random.normal(0, 1e-3, pts.shape)
    
    interpolater = LinearNDInterpolator(pts, feats, fill_value=0, rescale=False)
    interpolate_pts = interpolater(virtual_sensors)
    
    new_feats = interpolate_pts[np.where(interpolate_pts[:,0] > 0)]
    new_coords = virtual_sensors[np.where(interpolate_pts[:,0] > 0)]
    
    event_labels = {'pos': [], 'num_hits': [], 'latents': []}
    event_labels['pos'] = np.round(pts)
    event_labels['num_hits'] = counts[~mask]
    # event_labels['first_time_hit'] = pos_t[:,-1][~mask]
    event_labels['latents'] = latents[~mask]
    
    interpolated_labels = {'pos': [], 'num_hits': [], 'latents': []}
    interpolated_labels['pos'] = np.round(new_coords)
    interpolated_labels['num_hits'] = new_feats[:,0]
    # interpolated_labels['first_time_hit'] = new_feats[:,1]
    interpolated_labels['latents'] = new_feats[:,1:]
    
    data_labels.append(event_labels)
    interpolated_labels_total.append(interpolated_labels)
    print('Time taken: {}'.format(time.time() - stime))
    
# for event in data:
#     stime = time.time()
#     pts = event.latents.pos.to_numpy()
#     feats = np.vstack((event.latents.num_hits.to_numpy(), event.latents.first_time_hit.to_numpy())).T
    
#     # # check if pts contains all the same value along one column
#     # if (len(np.unique(pts[:,0])) == 1) or (len(np.unique(pts[:,1])) == 1) or (len(np.unique(pts[:,2])) == 1):
#     # add small random noise
#     pts += np.random.normal(0, 1e-6, pts.shape)
    
#     interpolater = LinearNDInterpolator(pts, feats, fill_value=0, rescale=False)
#     interpolate_pts = interpolater(virtual_sensors)
    
#     new_feats = interpolate_pts[np.where(interpolate_pts[:,0] > 0)]
#     new_coords = virtual_sensors[np.where(interpolate_pts[:,0] > 0)]
    
#     event_labels = {'pos': [], 'num_hits': [], 'first_time_hit': []}
#     event_labels['pos'] = np.vstack((event.latents.pos.to_numpy(), new_coords))
#     event_labels['num_hits'] = np.hstack((event.latents.num_hits.to_numpy(), new_feats[:,0]))
#     event_labels['first_time_hit'] = np.hstack((event.latents.first_time_hit.to_numpy(), new_feats[:,1]))
    
#     data_labels.append(event_labels)
#     print('Time taken: {}'.format(time.time() - stime))
    
new_data = ak.with_field(data, data_labels, where='unique_oms')
new_data = ak.with_field(new_data, interpolated_labels_total, where='interpolated')
ak.to_parquet(new_data, output_dir + file.split('ortho_2x_tracks_total_vae_latents64_test/')[1])
    