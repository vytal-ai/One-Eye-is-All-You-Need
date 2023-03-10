import cv2
import json
import os, time
import shutil
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from lit_tiny_resnet import lit_gazetrack_model

from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger


import argparse
parser = argparse.ArgumentParser(description='Train GazeTracker')
parser.add_argument('--dataset_dir', default='../dataset/', help='Path to converted dataset')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
parser.add_argument('--save_dir', default='../model/google_tiny_resnet_2_SOUP_1/', help='Path store checkpoints')
parser.add_argument('--comet_name', default='google-tiny-resnet-2-SOUP-1', help='Path store checkpoints')
parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs to use')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
parser.add_argument('--checkpoint', default=None, help='Path to load pre trained weights')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    print("Saving to " + args.save_dir)
    proj_name = args.comet_name
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, filename='{epoch}-{val_loss:.3f}-{train_loss:.3f}', save_top_k=-1)
    logger = CometLogger(
        api_key="REPLACE WITH API KEY",
        project_name=proj_name,
    )
    
    model = lit_gazetrack_model(args.dataset_dir, args.save_dir, args.batch_size, logger)
    if(args.checkpoint):
        if(args.gpus==0):
            w = torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            w = torch.load(args.checkpoint)['state_dict']
        model.load_state_dict(w)
        print("Loaded checkpoint")
        
    trainer = pl.Trainer(gpus=args.gpus, logger=logger, accelerator="gpu", max_epochs=args.epochs, default_root_dir=args.save_dir, auto_lr_find=True, auto_scale_batch_size=True, callbacks=[checkpoint_callback])
    trainer.fit(model)
    print("DONE")