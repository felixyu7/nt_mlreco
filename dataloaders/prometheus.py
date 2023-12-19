import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import MinkowskiEngine as ME
import os
import time
from collections import defaultdict
from networks.common.utils import generate_geo_mask

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
                                                         self.train_nu, 
                                                         self.cfg['data_options']['scale_factor'], 
                                                         self.cfg['data_options']['offset'],
                                                         self.cfg['data_options']['first_hit'])
                
        self.valid_dataset = SparsePrometheusDataset(self.valid_photons, 
                                                     self.valid_nu, 
                                                     self.cfg['data_options']['scale_factor'], 
                                                     self.cfg['data_options']['offset'],
                                                     self.cfg['data_options']['first_hit'])

    def train_dataloader(self):
        collate_fn = PrometheusCollator(masking=self.cfg['data_options']['masking'], input_geo_file=self.cfg['data_options']['input_geo_file'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=True,
                                            collate_fn=collate_fn,
                                            num_workers=self.cfg['training_options']['num_workers'],
                                            pin_memory=True)
        return dataloader
            
    def val_dataloader(self):
        collate_fn = PrometheusCollator(masking=self.cfg['data_options']['masking'], input_geo_file=self.cfg['data_options']['input_geo_file'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=collate_fn,
                                            num_workers=self.cfg['training_options']['num_workers'],
                                            pin_memory=True)
    def test_dataloader(self):
        collate_fn = PrometheusCollator(masking=self.cfg['data_options']['masking'], input_geo_file=self.cfg['data_options']['input_geo_file'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=collate_fn,
                                            num_workers=self.cfg['training_options']['num_workers'],
                                            pin_memory=True)

class SparsePrometheusDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        photons_data,
        nu_data,
        scale_factor,
        offset,
        first_hit):

        self.data = photons_data
        self.nu_data = nu_data
        self.dataset_size = len(self.data)
        self.scale_factor = scale_factor
        self.offset = offset
        self.first_hit = first_hit
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        
        # [energy, dir_x, dir_y, dir_z, x, y, z]
        label = [self.nu_data[i][0], self.nu_data[i][1], self.nu_data[i][2], self.nu_data[i][3],
                 self.nu_data[i][4], self.nu_data[i][5], self.nu_data[i][6]]

        xs = (self.data[i].photons.sensor_pos_x.to_numpy() + self.offset[0]) * self.scale_factor
        ys = (self.data[i].photons.sensor_pos_y.to_numpy() + self.offset[1])  * self.scale_factor
        zs = (self.data[i].photons.sensor_pos_z.to_numpy() + self.offset[2]) * self.scale_factor
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
    # check if data_file is a parquet file or directory. If directory, loop over all contained parquet files
    photons_data_total = []
    nu_data_total = []
    if os.path.isdir(data_file):
        data_files = os.listdir(data_file)
        data_files = sorted([os.path.join(data_file, f) for f in data_files if f.endswith('.parquet')])
    else:
        data_files = [data_file]
        
    for file in data_files:
        photons_data = ak.from_parquet(file, columns=[field, "photons"])
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
        photons_data = photons_data[[x for x in ak.fields(photons_data) if x != field]]
        # photons_data = reduce_photons_data_precision(photons_data)
        photons_data_total.append(photons_data)
        nu_data_total.append(nu_data)
    # concatenate events from all files
    photons_data = ak.concatenate(photons_data_total, axis=0)
    nu_data = np.concatenate(nu_data_total, axis=0)
    print("total time:", time.time() - tsime)
    return photons_data, nu_data

def reduce_photons_data_precision(photons_data):
    converted_dict = {}
    photons = photons_data.photons
    for field in photons.fields:
        if photons[field].type.type.type == ak.types.PrimitiveType('float64'):
            converted_dict[field] = ak.values_astype(photons[field], 'float32')
        elif photons[field].type.type.type == ak.types.PrimitiveType('int64'):
            converted_dict[field] = ak.values_astype(photons[field], 'int32')
        else:
            converted_dict[field] = photons[field]

    new_photons_data = ak.Array({'photons': converted_dict})
    return new_photons_data

class PrometheusCollator(object):
    
    def __init__(self, masking=False, input_geo_file=''):
        self.masking = masking
        self.input_geo_file = input_geo_file
        if self.input_geo_file != '':
            self.input_geo = torch.from_numpy(np.load(self.input_geo_file))
            
    def __call__(self, data_labels):
        """collate function for data input, creates batched data to be fed into network"""
        coords, feats, labels = list(zip(*data_labels))

        # Create batched coordinates for the SparseTensor input
        bcoords = ME.utils.batched_coordinates(coords)

        # Concatenate all lists
        feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
        labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()

        if self.masking:
            mask = generate_geo_mask(bcoords, self.input_geo)
            bcoords_masked = bcoords[mask]
            feats_batch_masked = feats_batch[mask]
            return bcoords_masked, feats_batch_masked, bcoords, feats_batch, labels_batch

        return bcoords, feats_batch, labels_batch