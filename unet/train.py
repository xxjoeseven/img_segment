from utility.seed import set_seed

set_seed(77)

import os
import time
from glob import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_data.drive_data import DriveDataset
from model.model import build_unet
from utility.loss import DiceBCELoss, DiceLoss
from utility.utils import make_dir, epoch_time

# Hyperparameters - Create config file separately
HEIGHT = 512
WEIGHT = 512
SIZE = (HEIGHT, WEIGHT)
BATCH_SIZE = 2
EPOCH = 50
LEARNING_RATE = 1e-4

# Paths to save weights
checkpoint_path = "weights/checkpoint.pth"

# Train Validate Data Split
train_x = ''
train_y = ''
valid_x = ''
valid_y = ''

# Load Dataset
train_dataset = DriveDataset(train_x, train_y)
valid_dataset = DriveDataset(valid_x, valid_y)
