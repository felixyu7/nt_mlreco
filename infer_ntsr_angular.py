import torch
import numpy as np
import lightning.pytorch as pl
import os
import MinkowskiEngine as ME
import yaml

from dataloaders.lazy_prometheus import LazyPrometheusDataModule

from networks.sscnn import SSCNN
from networks.generative_uresnet import Generative_UResNet
from networks.common.utils import angle_between, generate_geo_mask

with open('/n/home10/felixyu/nt_mlreco/infer_ntsr_angular.cfg', 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

dm = LazyPrometheusDataModule(cfg)

sscnn = SSCNN(1,
                reps=cfg['sscnn_model_options']['reps'], 
                depth=cfg['sscnn_model_options']['depth'], 
                first_num_filters=cfg['sscnn_model_options']['num_filters'], 
                stride=cfg['sscnn_model_options']['stride'], 
                dropout=0., 
                mode=cfg['sscnn_model_options']['reco_type'],
                D=4).to(torch.device(cfg['device']))

ntsr = Generative_UResNet(in_features=3,
    reps=1,
    depth=9,
    first_num_filters=32,
    stride=2,
    dropout=0.,
    input_dropout=0.5,
    output_dropout=0.1,
    scaling='linear',
    D=3,
    input_geo_file="/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7.npy",
    output_geo_file="/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7_2x.npy").to(torch.device(cfg['device']))
                          
# checkpoint = torch.load('/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_ckpts/sscnn_angular_trigger_final_v3/epoch_9.ckpt', map_location=torch.device(cfg['device']))
# sscnn.load_state_dict(checkpoint['model_state_dict'])
sscnn = sscnn.load_from_checkpoint('./wandb/sscnn_epoch9.ckpt').to(torch.device(cfg['device']))
ntsr = ntsr.load_from_checkpoint('./wandb/ntsr_epoch9.ckpt').to(torch.device(cfg['device']))

dm.setup()
test_loader = dm.test_dataloader()
preds = []
preds_w_ntsr = []
preds_truth = []
truth = []
for i, batch in enumerate(test_loader):
    # batch[0] = batch[0].to(torch.device(cfg['device']))
    # batch[1] = batch[1].to(torch.device(cfg['device']))
    # batch[2] = batch[2].to(torch.device(cfg['device']))
    inputs = ME.SparseTensor(batch[1].to(torch.device(cfg['device'])).float().reshape(batch[0].shape[0], -1), batch[0].to(torch.device(cfg['device'])), 
                             device=cfg['device'],
                             minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
    
    new_sparse_tensor = ntsr.test_step((batch[0].to(torch.device(cfg['device'])), 
                                        batch[1].to(torch.device(cfg['device'])),
                                        batch[2].to(torch.device(cfg['device']))), i)

    input_geo = torch.from_numpy(np.load('/n/home10/felixyu/nt_mlreco/scratch/ice_ortho_7.npy'))
    mask = generate_geo_mask(inputs.C, input_geo)
    inputs_masked = ME.SparseTensor(inputs.F[mask], inputs.C[mask], device=cfg['device'],
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, 
                                requires_grad=True, 
                                coordinate_manager=new_sparse_tensor.coordinate_manager)
    
    sscnn_inputs = inputs_masked + new_sparse_tensor

    outputs = sscnn(inputs_masked)
    preds.append(outputs[0].F.detach().cpu().numpy())
    outputs = sscnn(sscnn_inputs)
    preds_w_ntsr.append(outputs[0].F.detach().cpu().numpy())
    outputs = sscnn(inputs)
    preds_truth.append(outputs[0].F.detach().cpu().numpy())
    
    truth.append(batch[2].cpu().numpy())
    
    if i == 100:
        break

preds = np.concatenate(preds, axis=0)
preds_w_ntsr = np.concatenate(preds_w_ntsr, axis=0)
preds_truth = np.concatenate(preds_truth, axis=0)
true_angle = np.concatenate(truth, axis=0)[:,1:4]
true_energy = np.concatenate(truth, axis=0)[:,0]

angle_diff = []
for i in range(preds.shape[0]):
    angle_diff.append(angle_between(preds[i], true_angle[i]))
angle_diff = np.array(angle_diff)

angle_diff_w_ntsr = []
for i in range(preds.shape[0]):
    angle_diff_w_ntsr.append(angle_between(preds_w_ntsr[i], true_angle[i]))
angle_diff_w_ntsr = np.array(angle_diff_w_ntsr)

angle_diff_truth = []
for i in range(preds_truth.shape[0]):
    angle_diff_truth.append(angle_between(preds_truth[i], true_angle[i]))
angle_diff_truth = np.array(angle_diff_truth)

# np.save('/n/home10/felixyu/nt_mlreco/results/sscnn_input_masked_trained_on_input_masked.npy', angle_diff)
np.save('/n/home10/felixyu/nt_mlreco/results/sscnn_input_masked_trained_on_input_ntsr.npy', angle_diff_w_ntsr)
# np.save('/n/home10/felixyu/nt_mlreco/results/sscnn_input_masked_trained_on_input_true.npy', angle_diff_truth)
# np.save('/n/home10/felixyu/nt_mlreco/results/sscnn_input_masked_trained_true_e.npy', true_energy)



