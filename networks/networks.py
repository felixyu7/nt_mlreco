from networks.sscnn import SSCNN
from networks.generative_uresnet import Generative_UResNet
from networks.chain import NTSR_SSCNN_Chain
import lightning.pytorch as pl
import torch.nn as nn

def get_network(name, cfg):
    if name == 'sscnn':
        return SSCNN(**cfg['sscnn_model_options'],
                    batch_size=cfg['training_options']['batch_size'],
                    lr=cfg['training_options']['lr'],
                    lr_schedule=cfg['training_options']['lr_schedule'],
                    weight_decay=cfg['training_options']['weight_decay'])
    elif name == 'ntsr':
        return Generative_UResNet(**cfg['ntsr_model_options'],
                                batch_size=cfg['training_options']['batch_size'],
                                lr=cfg['training_options']['lr'],
                                lr_schedule=cfg['training_options']['lr_schedule'],
                                weight_decay=cfg['training_options']['weight_decay'])
    elif name == 'ntsr_sscnn_chain':
        return NTSR_SSCNN_Chain(ntsr_cfg=cfg['ntsr_model_options'],
                                sscnn_cfg=cfg['sscnn_model_options'],
                                batch_size=cfg['training_options']['batch_size'],
                                lr=cfg['training_options']['lr'],
                                lr_schedule=cfg['training_options']['lr_schedule'],
                                weight_decay=cfg['training_options']['weight_decay'])
    else:
        raise Exception("Unknown network name: {}".format(name))
        