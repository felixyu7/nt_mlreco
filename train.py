import torch
import numpy as np
import lightning.pytorch as pl
import wandb
import os
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from networks.networks import get_network

import yaml

def initialize_args():
    import argparse
    parser =  argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        dest="cfg_file",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args

if __name__=="__main__":

    torch.set_float32_matmul_precision('medium')

    args = initialize_args()

    with open(args.cfg_file, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # initialize dataloaders
    if cfg['dataloader'] == 'prometheus':
        from dataloaders.prometheus import PrometheusDataModule
        from dataloaders.lazy_prometheus import LazyPrometheusDataModule
        # dm = PrometheusDataModule(cfg)
        dm = LazyPrometheusDataModule(cfg)
    elif cfg['dataloader'] == 'icecube':
        from dataloaders.icecube import IceCubeDataModule
        dm = IceCubeDataModule(cfg)
    else:
        print("Unknown dataloader!")
        exit()

    # initialize models
    if isinstance(cfg['model'], list):
        # chain of networks
        chain = []
        for name in cfg['model']:
            chain.append(get_network(name, cfg))
        chain = torch.nn.Sequential(*chain)
    else:
        net = get_network(cfg['model'], cfg)

    if cfg['checkpoint'] != "":
        net = net.load_from_checkpoint(cfg['checkpoint'])
        print("Loaded checkpoint", cfg['checkpoint'])

    if cfg['training']:
        # initialise the wandb logger and name your wandb project
        os.environ["WANDB_DIR"] = os.path.abspath("/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/felixyu/nt_mlreco_projects/wandb")
        wandb_logger = WandbLogger(project=cfg['project_name'], save_dir=cfg['project_save_dir'], log_model='all')

        # add your batch size to the wandb config
        wandb_logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']

        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(dirpath=cfg['project_save_dir'] + '/' + cfg['project_name'] + '/' + wandb_logger.version + '/checkpoints',
                                              filename='model-{epoch:02d}-{val_loss:.2f}.ckpt', 
                                              every_n_epochs=cfg['training_options']['save_epochs'],
                                              save_on_train_epoch_end=True)
        trainer = pl.Trainer(accelerator=cfg['accelerator'], 
                             strategy='auto', 
                             devices=cfg['num_devices'], 
                             num_nodes=1, 
                             max_epochs=cfg['training_options']['epochs'],                    
                             log_every_n_steps=1, 
                             logger=wandb_logger, 
                             callbacks=[lr_monitor, checkpoint_callback])
        trainer.fit(model=net, datamodule=dm)
    else:
        logger = WandbLogger(project=cfg['project_name'], save_dir=cfg['project_save_dir'])
        logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']
        trainer = pl.Trainer(accelerator=cfg['accelerator'], 
                             profiler='simple', 
                             logger=logger)
        trainer.test(model=net, datamodule=dm)

