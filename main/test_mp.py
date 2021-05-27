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

from pycode.dataset import RLBench_dataset_test, Softargmax_dataset_test, RLBench_dataset3_test
from pycode.config import _C as cfg
from pycode.model.Hourglass import sequence_hourglass
from pycode.loss.mp_loss import Test_Loss_sequence_hourglass
from pycode.misc import load_hourglass, save_outputs, build_model_MP, build_dataset_MP, str2bool, save_args, save_checkpoint, load_checkpoint

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, help='path to config file')
parser.add_argument('--checkpoint_path', type=str, help='')
parser.add_argument('--video_checkpoint_path','-v', type=str, default='', help='e.g. output/RLdata/VP_pcf_dis_random/model_log/checkpoint_epoch0_iter100000/')
parser.add_argument('--log2wandb', type=str2bool, default=False)
parser.add_argument('--wandb_group', type=str, default='')
parser.add_argument('--blas_num_threads', type=str, default="4", help='set this not to cause openblas error')
# args = parser.parse_args(args=['--checkpoint_path','output/2020-04-02_18:28:18.736004/model_log/checkpoint_epoch9_iter11'])
args = parser.parse_args()

# os.environ["OPENBLAS_NUM_THREADS"] = args.blas_num_threads
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)
cfg.freeze()

# define save model path
save_path = os.path.join(args.checkpoint_path, 'test')

# make output dir
os.makedirs(save_path, exist_ok=True)

# set seed and cuda
torch.manual_seed(cfg.BASIC.SEED)
cuda = torch.cuda.is_available()
device = torch.device(cfg.BASIC.DEVICE)

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.BASIC.SEED)

# set wandb TODO
# with open(args.config_file) as file:
#     obj = yaml.safe_load(file)

# if args.log2wandb:
#     import wandb
#     wandb.login()
#     if args.wandb_group == '':
#         group = None
#     else:
#         group = args.wandb_group
#     run = wandb.init(project='MotionPrediction-{}-test'.format(cfg.DATASET.NAME), entity='tendon',
#                     config=obj, save_code=True, name=args.output_dirname, dir=os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME),
#                     group=group)

# set dataset
if cfg.DATASET.NAME == 'HMD':
    test_dataset = Softargmax_dataset_test(cfg, save_dataset=False)
elif cfg.DATASET.NAME == 'RLBench':
    test_dataset = RLBench_dataset_test(cfg, save_dataset=True)
elif cfg.DATASET.NAME == 'RLBench3':
    test_dataset = RLBench_dataset3_test(cfg, save_dataset=True)
else:
    raise ValueError("Invalid dataset name")

# set dataloader
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.BASIC.WORKERS)

# set model
if cfg.DATASET.NAME == 'HMD':
    output_dim = 21
elif (cfg.DATASET.NAME == 'RLBench') or (cfg.DATASET.NAME == 'RLBench3'):
    output_dim = 1

model = sequence_hourglass(cfg, output_dim=output_dim)
model_path = os.path.join(args.checkpoint_path, 'mp.pth')
if cfg.MP_MODEL_NAME == 'hourglass':
    model.hour_glass, _, _, _, _ = load_checkpoint(model.hour_glass, model_path, fix_parallel=True)
elif (cfg.MP_MODEL_NAME == 'sequence_hourglass') and (args.video_checkpoint_path == ''):
    model.hour_glass = load_hourglass(model.hour_glass, model_path)
elif cfg.MP_MODEL_NAME == 'sequence_hourglass':
    print('load mp model')
    model.hour_glass = load_hourglass(model.hour_glass, model_path)
    video_model_path = os.path.join(args.video_checkpoint_path, 'vp.pth')
    print('load vp model')
    model.video_pred_model, _, _, _, _ = load_checkpoint(model.video_pred_model, video_model_path, fix_parallel=True)

model = model.to(device)

# set loss
test_loss_mp = Test_Loss_sequence_hourglass(cfg, device)
# test_loss_mp2 = Test_Loss_sequence_hourglass(cfg, device)

# start train
for iteration, inputs in enumerate(test_dataloader, 1):
    
    with torch.no_grad():
        if args.video_checkpoint_path == '':
            outputs = model(inputs,mp_mode=1)
            _ = test_loss_mp(inputs, outputs)
        # save_outputs(inputs, outputs, save_path, iteration, cfg, mode='test')
        
        if args.video_checkpoint_path != '':
            outputs = model(inputs,mp_mode=2)
            _ = test_loss_mp(inputs, outputs)

    # save_outputs(inputs, outputs, save_path, iteration, cfg, mode='test')
    print("{} / {}".format(iteration, len(test_dataloader)))

# save and print log
# log = test_loss_mp1.get_log()
# print('MP mode1: 1frame pred')
# for key in log.keys():
#     print('key:{}  Value:{}'.format(key, log[key]))

log = test_loss_mp.get_log()
# print('\nMP mode2: end2end pred')
keys = list(log.keys())
keys.sort()
for key in keys:
    print('key:{}  Value:{}'.format(key, log[key]))

# TODO add two logs
# if args.log2wandb:
#     wandb.log(log)