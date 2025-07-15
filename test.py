import os
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import torchvision
import sys
import logging
import warnings
import time
import pdb
import tempfile
import config
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from torch.nn.modules.loss import CrossEntropyLoss
from torch import distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dataloader import get_loader
from utils.utils import init_logging_test, TimeLogging, get_config, setup_seed
from inference import validation
from network import get_model


def test(args):
    #Config
    os.makedirs(os.path.join(args.ckpt_path, args.expt_name), exist_ok=True)
    local_rank = 0
    rank = 0

    log_root = init_logging_test(0, os.path.join(args.ckpt_path, args.expt_name))
    log_root.info("---"*15)
    for arg, value in vars(args).items():
        log_root.info(f"{arg}: {value}")
    log_root.info("--"*15)    

    test_dataloaders, test_names = get_loader(args, args.dataset, "test", args.train_bs, args.val_bs, args.test_bs, 0, args.seed, args.num_workers)
    log_root.info("Data Loaded")

    #Model Backbone
    model = get_model(args.backbone, args.input_resolution, args.model).cuda()
    model.eval()
    log_root.info("Model Loaded")

    weights_path = args.model_path
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    validation(args, model, test_dataloaders, test_names, log_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="ckpts")
    parser.add_argument("--expt_name", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_resolution", type=int)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--val_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--num_classes", type=int, default=11)
    args = parser.parse_args()  
    args.ckpt_path = os.path.join(os.getenv('LOG_PATH'), args.ckpt_path)
    test(args)