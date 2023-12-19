from networks.sscnn import SSCNN
from networks.generative_uresnet import Generative_UResNet
from networks.ntsr import NTSR
from networks.chain import NTSR_SSCNN_Chain
from networks.transformer import NuModel
import lightning.pytorch as pl
import torch.nn as nn

def get_network(name, cfg):
    if name == 'sscnn':
        if cfg['checkpoint'] != '':
            print("Loading checkpoint: ", cfg['checkpoint'])
            return SSCNN.load_from_checkpoint(cfg['checkpoint'])
        return SSCNN(**cfg['sscnn_model_options'],
                    batch_size=cfg['training_options']['batch_size'],
                    lr=cfg['training_options']['lr'],
                    lr_schedule=cfg['training_options']['lr_schedule'],
                    weight_decay=cfg['training_options']['weight_decay'])
    elif name == 'transformer':
        if cfg['checkpoint'] != '':
            print("Loading checkpoint: ", cfg['checkpoint'])
            return NuModel.load_from_checkpoint(cfg['checkpoint'])
        return NuModel(**cfg['transformer_model_options'],
                    batch_size=cfg['training_options']['batch_size'],
                    lr=cfg['training_options']['lr'],
                    lr_schedule=cfg['training_options']['lr_schedule'],
                    weight_decay=cfg['training_options']['weight_decay'])
    elif name == 'ntsr':
        if cfg['checkpoint'] != '':
            print("Loading checkpoint: ", cfg['checkpoint'])
            return Generative_UResNet.load_from_checkpoint(cfg['checkpoint'])
        return Generative_UResNet(**cfg['ntsr_model_options'],
                                batch_size=cfg['training_options']['batch_size'],
                                lr=cfg['training_options']['lr'],
                                lr_schedule=cfg['training_options']['lr_schedule'],
                                weight_decay=cfg['training_options']['weight_decay'])
        # if cfg['checkpoint'] != '':
        #     print("Loading checkpoint: ", cfg['checkpoint'])
        #     return NTSR.load_from_checkpoint(cfg['checkpoint'])
        # return NTSR()
    elif name == 'ntsr_sscnn_chain':
        if cfg['checkpoint'] != '':
            print("Loading checkpoint: ", cfg['checkpoint'])
            return NTSR_SSCNN_Chain.load_from_checkpoint(cfg['checkpoint'])
        return NTSR_SSCNN_Chain(ntsr_cfg=cfg['ntsr_model_options'],
                                sscnn_cfg=cfg['sscnn_model_options'],
                                batch_size=cfg['training_options']['batch_size'],
                                lr=cfg['training_options']['lr'],
                                lr_schedule=cfg['training_options']['lr_schedule'],
                                weight_decay=cfg['training_options']['weight_decay'])
    else:
        raise Exception("Unknown network name: {}".format(name))
        