import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import pyarrow.parquet as pq
import glob
from networks.common.utils import generate_geo_mask_cuda

class PrometheusNTSRDataModule(pl.LightningDataModule):
    def __init__(self, cfg, field='mc_truth'):
        super().__init__()
        self.cfg = cfg
        self.field = field
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if self.cfg['training']:
            train_files = sorted(glob.glob(self.cfg['data_options']['train_data_file'] + '*.parquet'))
            self.train_dataset = PrometheusNTSRDataset(train_files,
                                                             self.cfg['data_options']['scale_factor'],
                                                             self.cfg['data_options']['offset'],
                                                             self.cfg['data_options']['first_hit'])
            
        valid_files = sorted(glob.glob(self.cfg['data_options']['valid_data_file'] + '*.parquet'))
        self.valid_dataset = PrometheusNTSRDataset(valid_files,
                                                            self.cfg['data_options']['scale_factor'],
                                                            self.cfg['data_options']['offset'],
                                                            self.cfg['data_options']['first_hit'])
            
    def train_dataloader(self):
        # dataloader = torch.utils.data.DataLoader(self.train_dataset, 
        #                                     batch_size = self.cfg['training_options']['batch_size'], 
        #                                     shuffle=True,
        #                                     collate_fn=collate_fn,
        #                                     num_workers=len(os.sched_getaffinity(0)))
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        # return torch.utils.data.DataLoader(self.valid_dataset, 
        #                                     batch_size = self.cfg['training_options']['batch_size'], 
        #                                     shuffle=False,
        #                                     collate_fn=prometheus_collate_fn,
        #                                     num_workers=len(os.sched_getaffinity(0)))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
         
class PrometheusNTSRDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        files,
        scale_factor,
        offset,
        first_hit):

        self.files = files
        self.scale_factor = scale_factor
        self.offset = offset
        self.first_hit = first_hit
        
        num_events = []
        for file in self.files:
            data = pq.ParquetFile(file)
            num_events.append(data.metadata.num_rows)
        num_events = np.array(num_events)
        self.cumulative_lengths = np.cumsum(num_events)
        self.dataset_size = self.cumulative_lengths[-1]
        
        self.current_file = ''
        self.current_data = None
        
        self.string_mask = np.load('/n/holylfs05/LABS/arguelles_delgado_lab/Users/felixyu/nt_mlreco/scratch/225_to_64_string_mask.npy')
        
        # self.cache = OrderedDict()
        # self.cache_size = 20

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        if i < 0 or i >= self.cumulative_lengths[-1]:
            raise IndexError("Index out of range")
        file_index = np.searchsorted(self.cumulative_lengths, i+1)
        true_idx = i - (self.cumulative_lengths[file_index-1] if file_index > 0 else 0)
                
        if self.current_file != self.files[file_index]:
            self.current_file = self.files[file_index]
            self.current_data = ak.from_parquet(self.files[file_index])
        
        event = self.current_data[true_idx]

        string_id = event.photons.string_id.to_numpy()
        sensor_id = event.photons.sensor_id.to_numpy()
        pos = np.vstack([string_id, sensor_id]).T
        
        sensor_pos = np.vstack([event.photons.sensor_pos_x.to_numpy(),
                                event.photons.sensor_pos_y.to_numpy(),
                                event.photons.sensor_pos_z.to_numpy()]).T
        
        # get unique pos row-wise, and count number of times they appear in pos
        unique_pos, inds, counts = np.unique(pos, axis=0, return_index=True, return_counts=True)
        unique_sensor_pos = sensor_pos[inds]
        
        feats = np.hstack([unique_sensor_pos / 100., counts.reshape(-1, 1)])
        
        # efficiently project positions onto image of size 225x61, with counts as pixel values
        image = np.zeros((225, 61, feats.shape[1]))
        image[unique_pos[:, 0], unique_pos[:, 1]] = feats
        
        # mask out on first (string) dimension with string mask
        masked_image = image * np.expand_dims(self.string_mask, (1, 2))
        
        return torch.from_numpy(masked_image), torch.from_numpy(image)

class ParquetFileSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, parquet_file_idxs, batch_size):
       self.data_source = data_source
       self.parquet_file_idxs = parquet_file_idxs
       self.batch_size = batch_size

    def __iter__(self):
        # Determine the number of batches in each parquet file
        num_entries_per_file = [end - start for start, end in zip(self.parquet_file_idxs[:-1], self.parquet_file_idxs[1:])]
      
        # Create an array of file indices, repeated by the number of entries in each file
        file_indices = np.repeat(np.arange(len(num_entries_per_file)), num_entries_per_file)
      
        # Shuffle the file indices
        np.random.shuffle(file_indices)

        for file_index in file_indices:
            start_idx, end_idx = self.parquet_file_idxs[file_index], self.parquet_file_idxs[file_index + 1]
            indices = np.random.permutation(np.arange(start_idx, end_idx))
            
            # Yield batches of indices ensuring all entries are seen
            for i in range(0, len(indices), self.batch_size):
                yield from indices[i:i+self.batch_size].tolist()

    def __len__(self):
       return len(self.data_source)