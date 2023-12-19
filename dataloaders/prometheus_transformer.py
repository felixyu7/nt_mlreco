import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import MinkowskiEngine as ME
import os, gc
import time
from collections import defaultdict
import pyarrow.parquet as pq
import polars
from collections import OrderedDict
from bisect import bisect_right
import glob
from networks.common.utils import generate_geo_mask_cuda
from dataloaders.lazy_prometheus import PrometheusCollator, ParquetFileSampler

class PrometheusTransformerDataModule(pl.LightningDataModule):
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if self.cfg['training']:
            train_files = sorted(glob.glob(self.cfg['data_options']['train_data_file'] + '*.parquet'))
            self.train_dataset = PrometheusTransformerDataset(train_files,
                                                             self.cfg['data_options']['scale_factor'],
                                                             self.cfg['data_options']['offset'],
                                                             self.cfg['data_options']['max_length'])
            
        valid_files = sorted(glob.glob(self.cfg['data_options']['valid_data_file'] + '*.parquet'))
        self.valid_dataset = PrometheusTransformerDataset(valid_files,
                                                            self.cfg['data_options']['scale_factor'],
                                                            self.cfg['data_options']['offset'],
                                                            self.cfg['data_options']['max_length'])
            
    def train_dataloader(self):
        collate_fn = PrometheusTransformerCollator(masking=self.cfg['data_options']['masking'],
                                                    input_geo_file=self.cfg['data_options']['input_geo_file'])
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            collate_fn=prometheus_transformer_collate_fn,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        collate_fn = PrometheusTransformerCollator(masking=self.cfg['data_options']['masking'],
                                                    input_geo_file=self.cfg['data_options']['input_geo_file'])
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            collate_fn=prometheus_transformer_collate_fn,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        # return torch.utils.data.DataLoader(self.valid_dataset, 
        #                                     batch_size = self.cfg['training_options']['batch_size'], 
        #                                     shuffle=False,
        #                                     collate_fn=prometheus_collate_fn,
        #                                     num_workers=len(os.sched_getaffinity(0)))

    def test_dataloader(self):
        collate_fn = PrometheusTransformerCollator(masking=self.cfg['data_options']['masking'],
                                                    input_geo_file=self.cfg['data_options']['input_geo_file'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=prometheus_transformer_collate_fn,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
         
class PrometheusTransformerDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        files,
        scale_factor,
        offset,
        max_length):

        self.files = files
        self.scale_factor = scale_factor
        self.offset = offset
        self.max_length = max_length
        
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        num_events = np.array(num_events)
        self.cumulative_lengths = np.cumsum(num_events)
        self.dataset_size = self.cumulative_lengths[-1]
        
        self.current_file = ''
        self.current_data = None
        
        # self.cache = OrderedDict()
        # self.cache_size = 20

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        if i < 0 or i >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_lengths, i+1)
        # file_index = bisect_right(self.cumulative_lengths, i)
        true_idx = i - (self.cumulative_lengths[file_index-1] if file_index > 0 else 0)

        # file = self.files[file_index]
        # if file not in list(self.cache.keys()):
        #     data = ak.from_parquet(file)
        #     self.cache[file] = data
        #     if len(self.cache) > self.cache_size:
        #         del self.cache[list(self.cache.keys())[0]]
                
        # event = self.cache[file][true_idx]
        if self.current_file != self.files[file_index]:
            self.current_file = self.files[file_index]
            self.current_data = ak.from_parquet(self.files[file_index])
        
        # data = ak.from_parquet(self.files[file_index], row_groups=true_idx)
        # event = data[0]
        event = self.current_data[true_idx]

        zenith = event.mc_truth.initial_state_zenith
        azimuth = event.mc_truth.initial_state_azimuth
        dir_x = np.cos(azimuth) * np.sin(zenith)
        dir_y = np.sin(azimuth) * np.sin(zenith)
        dir_z = np.cos(zenith)
        
        # [energy, dir_x, dir_y, dir_z]
        label = [event.mc_truth.initial_state_energy, 
                 dir_x, 
                 dir_y, 
                 dir_z]

        xs = (event.photons.sensor_pos_x.to_numpy() + self.offset[0]) * self.scale_factor
        ys = (event.photons.sensor_pos_y.to_numpy() + self.offset[1]) * self.scale_factor
        zs = (event.photons.sensor_pos_z.to_numpy() + self.offset[2]) * self.scale_factor
        ts = event.photons.t.to_numpy() - event.photons.t.to_numpy().min()
        
        pos_t = np.array([
            xs,
            ys,
            zs,
            ts
        ]).T.astype(np.float32)
        
        # bin by nanoseconds in time
        # pos_t = np.trunc(pos_t)
        # pos_t, feats = np.unique(pos_t, return_counts=True, axis=0)
        
        # first hit only
        spos_t = pos_t[np.argsort(pos_t[:,-1])]
        _, indices, feats = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
        pos_t = spos_t[indices]
        pos_t = np.trunc(pos_t)
        
        feats = feats.astype(np.float32)
                
        sort_indices = np.argsort(pos_t[:,3])
        xs = pos_t[:,0][sort_indices]
        ys = pos_t[:,1][sort_indices]
        zs = pos_t[:,2][sort_indices]
        ts = pos_t[:,3][sort_indices]

        L0 = len(ts)
        
        if L0 < self.max_length:
            xs = np.pad(xs, (0, max(0, self.max_length - L0)))
            ys = np.pad(ys, (0, max(0, self.max_length - L0)))
            zs = np.pad(zs, (0, max(0, self.max_length - L0)))
            ts = np.pad(ts, (0, max(0, self.max_length - L0)))
            charge = np.pad(feats, (0, max(0, self.max_length- L0)))
            L = L0
        else:
            # rand_sample = np.random.choice(range(len(xs)), size=self.max_length, replace=False)
            # xs = xs[rand_sample]
            # ys = ys[rand_sample]
            # zs = zs[rand_sample]
            # ts = ts[rand_sample]
            # charge = feats[rand_sample]
            xs = xs[:self.max_length]
            ys = ys[:self.max_length]
            zs = zs[:self.max_length]
            ts = ts[:self.max_length]
            charge = feats[:self.max_length]
            L = self.max_length
            
        attn_mask = torch.zeros(self.max_length, dtype=torch.bool)
        attn_mask[:L] = True
        pos = torch.from_numpy(np.array([xs, ys, zs])).T
        ts = torch.from_numpy(ts)
        charge = torch.from_numpy(charge)
        label = torch.from_numpy(np.array([label]))
        
        inputs = {
            "pos": pos,
            "time": ts,
            "charge": charge,
            "mask": attn_mask,
            "L0": torch.tensor(L0),
        }
        return inputs, label

class PrometheusTransformerCollator(object):
    
    def __init__(self, masking=False, input_geo_file=''):
        self.masking = masking
        self.input_geo_file = input_geo_file
        if self.input_geo_file != '':
            self.input_geo = torch.from_numpy(np.load(self.input_geo_file))
            
    def __call__(self, data_labels):
        # Unzip the list of tuples into two lists
        data_list, label_list = zip(*data_labels)

        # Initialize two empty dictionaries to hold the stacked data and labels
        data_batch = {}

        # For each key in the data dictionary...
        for key in data_list[0].keys():
            # Stack the corresponding values from each data dictionary along a new dimension
            data_batch[key] = torch.stack([data[key] for data in data_list])
            
        labels_batch = torch.stack(label_list).squeeze()
        
        if self.masking:
            mask = generate_geo_mask_cuda(data_batch['pos'], 
                                          self.input_geo, 
                                          cuda=True, 
                                          chunk_size=100000)
            data_batch['pos'] = data_batch['pos'][mask]
            data_batch['time'] = data_batch['time'][mask]
            data_batch['charge'] = data_batch['charge'][mask]
            data_batch['mask'] = data_batch['mask'][mask]
        
        return data_batch, labels_batch

def prometheus_transformer_collate_fn(data_labels):
    # Unzip the list of tuples into two lists
    data_list, label_list = zip(*data_labels)

    # Initialize two empty dictionaries to hold the stacked data and labels
    data_batch = {}

    # For each key in the data dictionary...
    for key in data_list[0].keys():
        # Stack the corresponding values from each data dictionary along a new dimension
        data_batch[key] = torch.stack([data[key] for data in data_list])
        
    labels_batch = torch.stack(label_list).squeeze()
    return data_batch, labels_batch