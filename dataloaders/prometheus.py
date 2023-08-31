import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import MinkowskiEngine as ME
import os
import time

class PrometheusDataModule(pl.LightningDataModule):
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field

    def prepare_data(self):
        if self.cfg['training']:
            t_photons_data, t_nu_data = prometheus_data_prep(self.cfg['data_options']['train_data_file'])
            if self.cfg['data_options']['train_event_list'] != '':
                event_list = np.loadtxt(self.cfg['data_options']['train_event_list']).astype(np.int32)
                t_photons_data = t_photons_data[event_list]
                t_nu_data = t_nu_data[event_list]
            self.train_photons = t_photons_data
            self.train_nu = t_nu_data
            if self.cfg['data_options']['double_dataset']:
                t_photons_data2, t_nu_data2 = prometheus_data_prep(self.cfg['data_options']['train_data_file2'])
                if self.cfg['data_options']['train_event_list2'] != '':
                    event_list = np.loadtxt(self.cfg['data_options']['train_event_list2']).astype(np.int32)
                    t_photons_data2 = t_photons_data2[event_list]
                    t_nu_data2 = t_nu_data2[event_list]
                self.train_photons2 = t_photons_data2
                self.train_nu2 = t_nu_data2

        v_photons_data, v_nu_data = prometheus_data_prep(self.cfg['data_options']['valid_data_file'])
        if self.cfg['data_options']['valid_event_list'] != '':
            event_list = np.loadtxt(self.cfg['data_options']['valid_event_list']).astype(np.int32)
            v_photons_data = v_photons_data[event_list]
            v_nu_data = v_nu_data[event_list]
        
        self.valid_photons = v_photons_data
        self.valid_nu = v_nu_data

    def setup(self, stage=None):
        if self.cfg['training']:
            self.train_dataset = SparsePrometheusDataset(self.train_photons, 
                                                         self.train_nu, self.cfg['data_options']['scale_factor'], self.cfg['data_options']['first_hit'])
            if self.cfg['data_options']['double_dataset']:
                train_dataset2 = SparsePrometheusDataset(self.train_photons2, 
                                                         self.train_nu2, self.cfg['data_options']['scale_factor'], self.cfg['data_options']['first_hit'])
                self.train_dataset = CombinedDataset(self.train_dataset, train_dataset2)
                
        self.valid_dataset = SparsePrometheusDataset(self.valid_photons, 
                                                     self.valid_nu, self.cfg['data_options']['scale_factor'], self.cfg['data_options']['first_hit'])

    def train_dataloader(self):
        if self.cfg['data_options']['double_dataset']:
            collate_fn = double_prometheus_collate_fn
        else:
            collate_fn = prometheus_collate_fn
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=True,
                                            collate_fn=collate_fn,
                                            num_workers=0,
                                            pin_memory=True)
        # if self.cfg['data_options']['double_dataset']:
        #     dataloader_2 = torch.utils.data.DataLoader(self.train_dataset2, 
        #                                     batch_size = self.cfg['training_options']['batch_size'], 
        #                                     shuffle=True,
        #                                     collate_fn=prometheus_collate_fn,
        #                                     num_workers=len(os.sched_getaffinity(0)),
        #                                     pin_memory=True)
        #     # dataloader = zip(dataloader, dataloader_2)
        #     return [dataloader, dataloader_2]
        return dataloader
            
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=prometheus_collate_fn,
                                            num_workers=0,
                                            pin_memory=True)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=prometheus_collate_fn,
                                            num_workers=0,
                                            pin_memory=True)

class SparsePrometheusDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        photons_data,
        nu_data,
        scale_factor,
        first_hit):

        self.data = photons_data
        self.nu_data = nu_data
        self.dataset_size = len(self.data)
        self.scale_factor = scale_factor
        self.first_hit = first_hit
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        
        # [energy, dir_x, dir_y, dir_z, x, y, z]
        label = [self.nu_data[i][0], self.nu_data[i][1], self.nu_data[i][2], self.nu_data[i][3],
                 self.nu_data[i][4], self.nu_data[i][5], self.nu_data[i][6]]

        xs = (self.data[i].photons.sensor_pos_x.to_numpy() - 5.87082946) * self.scale_factor
        ys = (self.data[i].photons.sensor_pos_y.to_numpy() + 2.51860853)  * self.scale_factor
        zs = (self.data[i].photons.sensor_pos_z.to_numpy() + 1971.9757655) * self.scale_factor
        ts = self.data[i].photons.t.to_numpy() - self.data[i].photons.t.to_numpy().min()

        pos_t = np.array([
            xs,
            ys,
            zs,
            ts
        ]).T

        # only use first photon hit time per dom
        if self.first_hit:
            spos_t = pos_t[np.argsort(pos_t[:,-1])]
            _, indices, feats = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
            pos_t = spos_t[indices]
            pos_t = np.trunc(pos_t)
        else:
            pos_t = np.trunc(pos_t)
            pos_t, feats = np.unique(pos_t, return_counts=True, axis=0)

        feats = feats.reshape(-1, 1).astype(np.float64)

        return torch.from_numpy(pos_t), torch.from_numpy(feats).view(-1, 1), torch.from_numpy(np.array([label]))

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
       self.dataset1 = dataset1
       self.dataset2 = dataset2

    def __getitem__(self, index):
       x1 = self.dataset1[index]
       x2 = self.dataset2[index]
       return {'input': x1, 'truth': x2}

    def __len__(self):
       # Assumes both datasets are of same length, add check if not sure
       return len(self.dataset1)

def prometheus_data_prep(data_file, field='mc_truth'):
    tsime = time.time()
    photons_data = ak.from_parquet(data_file, columns=[field, "photons"])
    print("total time:", time.time() - tsime)

    # converting read data to inputs
    es = np.array(photons_data[field]['initial_state_energy'])
    zenith = np.array(photons_data[field]['initial_state_zenith'])
    azimuth = np.array(photons_data[field]['initial_state_azimuth'])
    x = np.array(photons_data[field]['initial_state_x'])
    y = np.array(photons_data[field]['initial_state_y'])
    z = np.array(photons_data[field]['initial_state_z'])
    
    # energy transforms/normalization
    es = np.log10(es)
    es_transformed = es
    # es_transformed = (es - es.mean()) / (es.std() + 1e-8)

    dir_x = np.cos(azimuth) * np.sin(zenith)
    dir_y = np.sin(azimuth) * np.sin(zenith)
    dir_z = np.cos(zenith)

    nu_data = np.dstack((es_transformed, dir_x, dir_y, dir_z, x, y, z)).reshape(-1, 7)
    return photons_data, nu_data

def prometheus_collate_fn(data_labels):
    """collate function for data input, creates batched data to be fed into network"""
    coords, feats, labels = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()
    
    return bcoords, feats_batch, labels_batch

def double_prometheus_collate_fn(data_labels):
    """collate function for data input, creates batched data to be fed into network for double dataset option"""
    
    # reverse dictionary/list ordering for collate fn
    from collections import defaultdict
    # Initialize a default dictionary with empty lists
    result = defaultdict(list)
    # Loop over each dictionary in the list
    for dct in data_labels:
        # Loop over each key-value pair in the dictionary
        for key, value in dct.items():
            # Append the value to the appropriate list in the result dictionary
            result[key].append(value)
    
    bcoords1, feats_batch1, labels_batch1 = prometheus_collate_fn(result['input'])
    bcoords2, feats_batch2, labels_batch2 = prometheus_collate_fn(result['truth'])
    return [bcoords1, feats_batch1, labels_batch1], [bcoords2, feats_batch2, labels_batch2]