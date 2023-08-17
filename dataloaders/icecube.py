import torch
import numpy as np
import lightning.pytorch as pl
import MinkowskiEngine as ME
import glob
import os
import time
from icecube import dataio, dataclasses

class IceCubeDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    def prepare_data(self):
        if self.cfg['training']:
            train_files_list = sorted(glob.glob(self.cfg['data_options']['train_data_file']))
            total_train_hits = []
            total_train_labels = []
            for f in train_files_list:
                hits, labels = load_i3_data(f)
                total_train_hits.extend(hits)
                total_train_labels.extend(labels)
            self.total_train_hits = total_train_hits
            self.total_train_labels = total_train_labels

        val_files_list = sorted(glob.glob(self.cfg['data_options']['valid_data_file']))[:25]
        total_val_hits = []
        total_val_labels = []
        for f in val_files_list:
            hits, labels = load_i3_data(f)
            total_val_hits.extend(hits)
            total_val_labels.extend(labels)
        self.total_val_hits = total_val_hits
        self.total_val_labels = total_val_labels

    def setup(self, stage=None):
        if self.cfg['training']:
            self.train_dataset = SparseICDataset(self.total_train_hits, self.total_train_labels, self.cfg['data_options']['first_hit'])
        self.valid_dataset = SparseICDataset(self.total_val_hits, self.total_val_labels, self.cfg['data_options']['first_hit'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=True,
                                            collate_fn=icecube_collate_fn,
                                            num_workers=len(os.sched_getaffinity(0)),
                                            pin_memory=True)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=icecube_collate_fn,
                                            num_workers=len(os.sched_getaffinity(0)),
                                            pin_memory=True)
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            collate_fn=icecube_collate_fn,
                                            num_workers=len(os.sched_getaffinity(0)),
                                            pin_memory=True)

class SparseICDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        data,
        labels,
        first_hit):

        self.data = data
        self.labels = labels
        self.dataset_size = len(self.data)
        self.first_hit = first_hit
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        
        label = [np.log10(self.labels[i][0]), self.labels[i][1], self.labels[i][2], self.labels[i][3]]

        xs = self.data[i][:,0] * 4.566
        ys = self.data[i][:,1] * 4.566
        zs = self.data[i][:,2] * 4.566
        ts = self.data[i][:,3] - self.data[i][:,3].min()

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
            # feats = (feats - feats.mean()) / (feats.std() + 1e-8)
        else:
            pos_t = np.trunc(pos_t)
            unique_pos_t, feats = np.unique(pos_t, return_counts=True, axis=0)

        return torch.from_numpy(unique_pos_t), torch.from_numpy(feats).view(-1, 1), torch.from_numpy(np.array([label]))

def load_i3_data(filename):
    # Create empty lists to store the data
    total_hits = []
    labels = []
    geo = np.load('/n/home10/felixyu/nt_mlreco/scratch/geo_dict.npy', allow_pickle=True).item()
    # Open the data file
    with dataio.I3File(filename) as f:
        # Loop over all frames in the file
        counter = 0
        while f.more():
            if counter % 1 == 0:
                frame = f.pop_physics()
                pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'InIceDSTPulses')
                om_keys = pulse_map.keys()
                pulses = pulse_map.values()
                pulse_times = [[obj.time for obj in inner_list] for inner_list in pulses]
                pulse_charges = [[obj.charge for obj in inner_list] for inner_list in pulses]

                om_pos = []
                for omkey in om_keys:
                    om_pos.append(geo[omkey])
                om_pos = np.array(om_pos)

                # convert nested list to flat array
                pulse_times_arr = np.concatenate(pulse_times)
                pulse_charges_arr = np.concatenate(pulse_charges)

                # repeat arr1 for each element in lst
                om_pos = np.repeat(om_pos, [len(l) for l in pulse_times], axis=0)

                # concatenate arr1_repeated and arr2
                hits = np.concatenate((om_pos, pulse_times_arr.reshape((-1, 1)), pulse_charges_arr.reshape((-1, 1))), axis=1)
                total_hits.append(hits)

                mc_primary = frame['I3MCTree'].primaries[0]
                # mc_primary = frame['I3MCTree_preMuonProp'].primaries[0]
                label = [mc_primary.energy, mc_primary.dir.x, mc_primary.dir.y, mc_primary.dir.z]
                labels.append(label)
                counter += 1
            else:
                frame = f.pop_physics()
                counter += 1
                continue
    return total_hits, labels

def icecube_collate_fn(data_labels):
    """collate function for data input, creates batched data to be fed into network"""
    coords, feats, labels = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).float()
    
    return bcoords, feats_batch, labels_batch