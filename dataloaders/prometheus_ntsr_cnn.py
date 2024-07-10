import torch
import numpy as np
import awkward as ak
import lightning.pytorch as pl
import pyarrow.parquet as pq
import glob
from collections import defaultdict

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
                                                             self.cfg['data_options']['first_hit'],
                                                             self.cfg['data_options']['labels'])
            
        valid_files = sorted(glob.glob(self.cfg['data_options']['valid_data_file'] + '*.parquet'))
        self.valid_dataset = PrometheusNTSRDataset(valid_files,
                                                            self.cfg['data_options']['scale_factor'],
                                                            self.cfg['data_options']['offset'],
                                                            self.cfg['data_options']['first_hit'],
                                                            self.cfg['data_options']['labels'])
            
    def train_dataloader(self):
        collate_fn = PrometheusNTSRCollator()
        sampler = ParquetFileSampler(self.train_dataset, self.train_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            # collate_fn=collate_fn,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        return dataloader
    
    def val_dataloader(self):
        collate_fn = PrometheusNTSRCollator()
        sampler = ParquetFileSampler(self.valid_dataset, self.valid_dataset.cumulative_lengths, self.cfg['training_options']['batch_size'])
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            # shuffle=True,
                                            sampler=sampler,
                                            # collate_fn=collate_fn,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
        # return torch.utils.data.DataLoader(self.valid_dataset, 
        #                                     batch_size = self.cfg['training_options']['batch_size'], 
        #                                     shuffle=False,
        #                                     collate_fn=prometheus_collate_fn,
        #                                     num_workers=len(os.sched_getaffinity(0)))

    def test_dataloader(self):
        collate_fn = PrometheusNTSRCollator()
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                            batch_size = self.cfg['training_options']['batch_size'], 
                                            shuffle=False,
                                            # collate_fn=collate_fn,
                                            pin_memory=True,
                                            persistent_workers=True,
                                            num_workers=self.cfg['training_options']['num_workers'])
         
class PrometheusNTSRDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        files,
        scale_factor,
        offset,
        first_hit,
        labels):

        self.files = files
        self.scale_factor = scale_factor
        self.offset = offset
        self.first_hit = first_hit
        self.labels = labels
        
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
        
        # latents
        unique_pos = event.latents.string_sensor_pos.to_numpy().astype(np.int32)
        unique_sensor_pos = event.latents.pos.to_numpy()
        counts = event.latents.num_hits.to_numpy()
        latents = event.latents.latents.to_numpy()
        
        # scale normalize positions
        unique_sensor_pos = (unique_sensor_pos / 100.).astype(np.float32)
        # log normalize counts
        counts = np.log(counts + 1).astype(np.float32)
        
        feats = np.hstack([unique_sensor_pos, counts.reshape(-1, 1), latents])
        
        # efficiently project positions onto image of size 225x61, with counts as pixel values
        image = np.zeros((225, 61, feats.shape[1]), dtype=np.float32)
        image[unique_pos[:, 0], unique_pos[:, 1]] = feats
        
        # mask out on first (string) dimension with string mask
        masked_image = image * np.expand_dims(self.string_mask, (1, 2)).astype(np.float32)
        
        if self.labels:
            zenith = event.mc_truth.initial_state_zenith
            azimuth = event.mc_truth.initial_state_azimuth
            dir_x = np.cos(azimuth) * np.sin(zenith)
            dir_y = np.sin(azimuth) * np.sin(zenith)
            dir_z = np.cos(zenith)
            
            # [energy, dir_x, dir_y, dir_z, x, y, z]
            label = [np.log10(event.mc_truth.initial_state_energy), 
                    dir_x, 
                    dir_y, 
                    dir_z,
                    event.mc_truth.initial_state_x, 
                    event.mc_truth.initial_state_y, 
                    event.mc_truth.initial_state_z]
            return torch.from_numpy(masked_image), torch.from_numpy(image), torch.from_numpy(np.array([label]))
        
        return torch.from_numpy(masked_image), torch.from_numpy(image)
        # return torch.from_numpy(masked_image), torch.from_numpy(image), torch.from_numpy(full_time_series_image)

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
   
class PrometheusNTSRCollator(object):
    
    def __init__(self):
        pass
            
    def __call__(self, data_labels):
        """collate function for data input, creates batched data to be fed into network"""
        masked_image, image, true_photons = list(zip(*data_labels))

        # concatenate masked_image and image along batch dimension
        masked_image_batch = torch.stack(masked_image)
        image_batch = torch.stack(image)
        
        true_photons_batch = batched_coords(true_photons)
            
        return masked_image_batch, image_batch, true_photons_batch

@torch.compile
def batched_coords(coords_list):
    # Initialize a list to hold the tensors with their batch indices
    combined_list = []
    
    # Iterate through the list of tensors with the enumeration to get the batch index
    for idx, coords in enumerate(coords_list):
        # Generate a tensor of batch indices for the current tensor
        batch_indices = torch.full((coords.size(0), 1), idx, dtype=coords.dtype)
        
        # Concatenate the batch index tensor with the original tensor along the columns
        combined_tensor = torch.cat([batch_indices, coords], dim=1)
        
        # Append the combined tensor to the list
        combined_list.append(combined_tensor)
    
    # Concatenate all tensors in the list vertically to form a large 2D tensor
    final_coords = torch.cat(combined_list, dim=0)
    
    return final_coords