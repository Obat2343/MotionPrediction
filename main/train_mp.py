import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import argparse
import sys
import time
import datetime
import torch
import yaml
import shutil
import threading
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
from pycode.misc import save_outputs, build_model_MP, build_dataset_MP, build_optimizer, str2bool, save_args, save_checkpoint, load_checkpoint, Timer, Time_dict

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
parser.add_argument('--log_step', type=int, default=100, help='')
parser.add_argument('--save_step', type=int, default=10000, help='')
parser.add_argument('--eval_step', type=int, default=5000, help='')
parser.add_argument('--output_dirname', type=str, default='', help='')
parser.add_argument('--checkpoint_path', type=str, default=None, help='')
parser.add_argument('--vp_path', type=str, default='')
parser.add_argument('--hourglass_path', type=str, default='')
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--wandb_group', type=str, default='') # e.g. compare_input
parser.add_argument('--save_dataset', type=str2bool, default=False)
# args = parser.parse_args(args=['--checkpoint_path','output/2020-04-02_18:28:18.736004/model_log/checkpoint_epoch9_iter11'])
args = parser.parse_args()

# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

if cfg.SKIP_LEN == 0:
    print("##############################")
    print("INFO: cfg.SKIP_LEN is 0. This means that RLBench_grasp dataset is used.")

# define output dirname
if len(args.output_dirname) == 0:
    dt_now = datetime.datetime.now()
    output_dirname = str(dt_now.date()) + '_' + str(dt_now.time())
else:
    output_dirname = args.output_dirname

# TODO change TASK_LIST[0]
output_dirname = os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_LIST[0], output_dirname)
if os.path.exists(output_dirname):
    while 1:
        ans = input('The specified output dir is already exists. Overwrite? y or n: ')
        if ans == 'y':
            break
        elif ans == 'n':
            raise ValueError("Please specify correct output dir")
        else:
            print('please type y or n')

# define save model path
model_path = os.path.join(output_dirname, 'model_log')

# make output dir
os.makedirs(output_dirname, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# copy config file
if len(args.config_file) > 0:
    shutil.copy(args.config_file,output_dirname)

# save args
argsfile_path = os.path.join(output_dirname, "args.txt")
save_args(args,argsfile_path)

# set seed and cuda
torch.manual_seed(cfg.BASIC.SEED)
cuda = torch.cuda.is_available()
device = torch.device(cfg.BASIC.DEVICE)

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.BASIC.SEED)

# set wandb
with open(args.config_file) as file:
    obj = yaml.safe_load(file)

if args.log2wandb:
    import wandb
    wandb.login()
    if args.wandb_group == '':
        group = None
    else:
        group = args.wandb_group
    run = wandb.init(project='MotionPrediction-{}-{}'.format(cfg.DATASET.NAME, cfg.DATASET.RLBENCH.TASK_LIST[0]), entity='tendon',
                    config=obj, save_code=True, name=args.output_dirname, dir=os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME),
                    group=group)

# set dataset
train_dataset = build_dataset_MP(cfg, save_dataset=args.save_dataset, mode='train')
val_dataset = build_dataset_MP(cfg, save_dataset=args.save_dataset, mode='val')

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)

# set model
model = build_model_MP(cfg, args)

# set optimizer
optimizer = build_optimizer(cfg, model, 'mp')
scheduler = StepLR(optimizer, step_size=cfg.SCHEDULER.STEPLR.STEP_SIZE, gamma=cfg.SCHEDULER.STEPLR.GAMMA)

model = torch.nn.DataParallel(model, device_ids = list(range(cfg.BASIC.NUM_GPU)))
model = model.to(device)

# set loss
train_loss = Train_Loss_sequence_hourglass(cfg, device)
val_loss = Train_Loss_sequence_hourglass(cfg, device)

# load checkpoint
if args.checkpoint_path != None:
    checkpoint_path = os.path.join(args.checkpoint_path, 'mp.pth')
    
    if cfg.LOAD_MODEL == 'all':
        model, optimizer, start_epoch, start_iter, scheduler = load_checkpoint(model, checkpoint_path, optimizer=optimizer, scheduler=scheduler)
    elif cfg.LOAD_MODEL == 'model_only':
        # dose tukawan kara nokosu. keshitemoiiyo
        model, _, _, _, _ = load_checkpoint(model, checkpoint_path)
        start_epoch, start_iter = 0, 1
else:
    start_epoch, start_iter = 0, 1

# start train
tic = time.time()
end = time.time()
trained_time = 0
# max_iter = cfg.BASIC.MAX_EPOCH * len(train_dataloader)
max_iter = cfg.BASIC.MAX_ITER
time_dict = Time_dict()
load_start = time.time()
# torch.autograd.set_detect_anomaly(True)

cfg.freeze()

for epoch in range(start_epoch, cfg.BASIC.MAX_EPOCH):
    for iteration, inputs in enumerate(train_dataloader, 1):
        time_dict.load_data += time.time() - load_start
        total_iteration = len(train_dataloader) * epoch + iteration
            
        # skip until start iter
        if total_iteration < start_iter:
            continue
            
        # optimize generator
        optimizer.zero_grad()
        
        with Timer() as t:
            outputs = model(inputs)
        time_dict.forward += t.secs

        with Timer() as t:
            loss = train_loss(inputs, outputs)
        time_dict.loss += t.secs

        with Timer() as t:
            loss.backward()
            optimizer.step()
            scheduler.step()
        time_dict.backward += t.secs
        
        # time setting
        trained_time += time.time() - end
        end = time.time() 
        
        # save and print log
        if total_iteration % args.log_step == 0:
            log = train_loss.get_log()
            eta_seconds = int((trained_time / total_iteration) * (max_iter - total_iteration))
            
            if (args.log2wandb) and (total_iteration % (args.log_step * 10)):
                wandb.log(log,step=total_iteration)
            
            # print(threading.active_count())
            print('===> Iter: {:06d}/{:06d}, LR: {:.5f}, Cost: {:.2f}s, Load: {:.2f}, Forward: {:.2f}, Backward: {:.2f}, Loss: {:.6f}'.format(total_iteration, 
                max_iter, optimizer.param_groups[0]['lr'], time.time() - tic, 
                time_dict.load_data, time_dict.forward, time_dict.backward, log['train/weight_loss']))
            
            train_loss.reset_log()
            tic = time.time()
            time_dict.reset()
        
        # save checkpoint
        if total_iteration % args.save_step == 0:
            checkpoint_dir = os.path.join(model_path,'checkpoint_iter{}'.format(total_iteration))
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp_path = os.path.join(checkpoint_dir, 'mp.pth')
            save_checkpoint(model, optimizer, epoch, iteration, cp_path, scheduler)
            
            # save output image
            for i, inputs in enumerate(train_dataloader, 1):
                with torch.inference_mode():
                    outputs = model(inputs)
                    save_outputs(inputs, outputs, checkpoint_dir, i, cfg, mode='train')
                    
                if i >= 5:
                    break
            
            for i, inputs in enumerate(val_dataloader, 1):
                with torch.inference_mode():
                    outputs = model(inputs)
                    save_outputs(inputs, outputs, checkpoint_dir, i, cfg, mode='val')
                    
                if i >= 5:
                    break

        # validation
        if total_iteration % args.eval_step == 0:
            print('validation start')
            for iteration, inputs in enumerate(val_dataloader, 1):
                with torch.inference_mode():
                    outputs = model(inputs)
                    _ = val_loss(inputs, outputs, mode='val')
                if iteration >= 1000:
                    break
            
            val_log = val_loss.get_log()
            if args.log2wandb:
                wandb.log(val_log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, VAL Loss: {:.6f}'.format(total_iteration, max_iter, val_log['val/weight_loss']))
            print('')
            val_loss.reset_log()        

        load_start = time.time()

        if total_iteration == cfg.BASIC.MAX_ITER:
            sys.exit()

    train_dataset.update_seed()
    print("seed: {}".format(train_dataset.seed))
    start_iter = 1