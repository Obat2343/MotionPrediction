import os
import sys
import time
import datetime
import argparse
import torch
import yaml
import shutil
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
sys.path.append('../')

from pycode.dataset import RLBench_dataset, Softargmax_dataset, imageaug_full_transform, train_val_split, pose_aug
from pycode.config import _C as cfg
from pycode.model.Hourglass import stacked_hourglass_model
from pycode.loss.mp_loss import Train_Loss_sequence_hourglass
from pycode.misc import save_outputs, build_model_MP, build_dataset_MP, build_optimizer, str2bool, save_args, save_checkpoint, load_checkpoint

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
# args = parser.parse_args(args=['--checkpoint_path','output/2020-04-02_18:28:18.736004/model_log/checkpoint_epoch9_iter11'])
args = parser.parse_args()

# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

cfg.freeze()

# set seed and cuda
torch.manual_seed(cfg.BASIC.SEED)
cuda = torch.cuda.is_available()
device = torch.device(cfg.BASIC.DEVICE)

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.BASIC.SEED)

# set dataset
train_dataset = build_dataset_MP(cfg, save_dataset=False, mode='train')
val_dataset = build_dataset_MP(cfg, save_dataset=False, mode='val')

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)


start_epoch, start_iter = 0, 1

# start train
tic = time.time()
end = time.time()
trained_time = 0
max_iter = cfg.BASIC.MAX_EPOCH * len(train_dataloader)
for epoch in range(start_epoch, cfg.BASIC.MAX_EPOCH):
    for iteration, inputs in enumerate(train_dataloader, 1):
        total_iteration = len(train_dataloader) * epoch + iteration
        print(iteration)
        print(inputs['input_rotation'].shape)
        print('===> Iter: {:06d}/{:06d}, Cost: {:.2f}s, Eta: {}'.format(total_iteration, 
                max_iter, time.time() - tic, str(datetime.timedelta(seconds=eta_seconds))))
        # print(inputs)
        # skip until start iter
        if iteration < start_iter:
            continue

        # time setting
        trained_time += time.time() - end
        end = time.time() 
            
                
    train_dataset.update_seed()
    print("seed: {}".format(train_dataset.seed))
    start_iter = 1