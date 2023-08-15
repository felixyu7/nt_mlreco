import torch
import numpy as np
import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from dataloaders.prometheus import PrometheusDataModule
from networks.sscnn import SSCNN

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

    args = initialize_args()

    with open(args.cfg_file, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # initialize dataloaders
    # train_dataloader, valid_dataloader = create_train_dataloaders(cfg)
    dm = PrometheusDataModule(cfg)
    net = SSCNN(1, reps=cfg['model_options']['reps'], 
                        depth=cfg['model_options']['depth'], 
                        first_num_filters=cfg['model_options']['num_filters'], 
                        stride=cfg['model_options']['stride'], 
                        dropout=cfg['model_options']['dropout'],
                        input_dropout=cfg['model_options']['input_dropout'],
                        output_dropout=cfg['model_options']['output_dropout'],
                        mode=cfg['model_options']['reco_type'],
                        D=4,
                        batch_size=cfg['training_options']['batch_size'], 
                        lr=cfg['training_options']['lr'], 
                        weight_decay=cfg['training_options']['weight_decay'])

    if cfg['checkpoint'] != "":
        net = net.load_from_checkpoint(cfg['checkpoint'])
        print("Loaded checkpoint", cfg['checkpoint'])

    if cfg['training']:
        # initialise the wandb logger and name your wandb project
        wandb_logger = WandbLogger(project='nt_mlreco')

        # add your batch size to the wandb config
        wandb_logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']

        lr_monitor = LearningRateMonitor(logging_interval='step')
        trainer = pl.Trainer(accelerator='gpu', max_epochs=5, log_every_n_steps=10, default_root_dir='./logs', 
                            logger=wandb_logger, callbacks=[lr_monitor])
        trainer.fit(model=net, datamodule=dm)
    else:
        trainer = pl.Trainer(accelerator='gpu', profiler='simple')
        val_outs = trainer.validate(model=net, datamodule=dm)
        import pdb; pdb.set_trace()

