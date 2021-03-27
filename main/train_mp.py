import os
import sys
import time
import datetime
import argparse
import tensorboardX
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
parser.add_argument('--log_step', type=int, default=50, help='')
parser.add_argument('--save_step', type=int, default=5000, help='')
parser.add_argument('--eval_step', type=int, default=5000, help='')
parser.add_argument('--output_dirname', type=str, default='', help='')
parser.add_argument('--checkpoint_path', type=str, default=None, help='')
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--wandb_group', type=str, default='')
# args = parser.parse_args(args=['--checkpoint_path','output/2020-04-02_18:28:18.736004/model_log/checkpoint_epoch9_iter11'])
args = parser.parse_args()

# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

# define output dirname
if len(args.output_dirname) == 0:
    dt_now = datetime.datetime.now()
    output_dirname = str(dt_now.date()) + '_' + str(dt_now.time())
else:
    output_dirname = args.output_dirname
    
output_dirname = os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME, output_dirname)
cfg.freeze()

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
    run = wandb.init(project='MotionPrediction-{}'.format(cfg.DATASET.NAME), entity='tendon',
                    config=obj, save_code=True, name=args.output_dirname, dir=os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME),
                    group=group)

# set dataset
train_dataset = build_dataset_MP(cfg, save_dataset=False, mode='train')
val_dataset = build_dataset_MP(cfg, save_dataset=False, mode='val')

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=False, num_workers=cfg.BASIC.WORKERS)

# set model
model = build_model_MP(cfg)
model = torch.nn.DataParallel(model, device_ids = list(range(cfg.BASIC.NUM_GPU)))
model = model.to(device)

# set loss
train_loss = Train_Loss_sequence_hourglass(cfg, device)
val_loss = Train_Loss_sequence_hourglass(cfg, device)

# set optimizer
optimizer = build_optimizer(cfg, model, 'mp')
scheduler = StepLR(optimizer, step_size=cfg.SCHEDULER.STEPLR.STEP_SIZE, gamma=cfg.SCHEDULER.STEPLR.GAMMA)

# load checkpoint
if args.checkpoint_path != None:
    checkpoint_path = os.path.join(args.checkpoint_path, 'mp.pth')
    
    if cfg.LOAD_MODEL == 'all':
        model, optimizer, start_epoch, start_iter, scheduler = load_checkpoint(model, checkpoint_path, optimizer=optimizer, scheduler=scheduler)
    elif cfg.LOAD_MODEL == 'model_only':
        model, _, _, _, _ = load_checkpoint(model, checkpoint_path)
        start_epoch, start_iter = 0, 1
else:
    start_epoch, start_iter = 0, 1

# start train
tic = time.time()
end = time.time()
trained_time = 0
max_iter = cfg.BASIC.MAX_EPOCH * len(train_dataloader)
for epoch in range(start_epoch, cfg.BASIC.MAX_EPOCH):
    for iteration, inputs in enumerate(train_dataloader, 1):
        total_iteration = len(train_dataloader) * epoch + iteration
            
        # skip until start iter
        if iteration < start_iter:
            continue
            
        # optimize generator
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = train_loss(inputs, outputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # time setting
        trained_time += time.time() - end
        end = time.time() 
        
        # save and print log
        if total_iteration % args.log_step == 0:
            log = train_loss.get_log()
            eta_seconds = int((trained_time / total_iteration) * (max_iter - total_iteration))
            
            if args.log2wandb:
                wandb.log(log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Loss: {:.6f}'.format(total_iteration, 
                max_iter, optimizer.param_groups[0]['lr'], time.time() - tic, 
                str(datetime.timedelta(seconds=eta_seconds)), log['train/weight_loss']))
            
            train_loss.reset_log()
            tic = time.time()
            
        # validation
        if total_iteration % args.eval_step == 0:
            print('validation start')
            for iteration, inputs in enumerate(val_dataloader, 1):
                with torch.no_grad():
                    outputs = model(inputs)
                    _ = val_loss(inputs, outputs, mode='val')
            
            val_log = val_loss.get_log()
            if args.log2wandb:
                wandb.log(val_log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, VAL Loss: {:.6f}'.format(total_iteration, max_iter, val_log['val/weight_loss']))
            print('')
            val_loss.reset_log()
        
        # save checkpoint
        if total_iteration % args.save_step == 0:
            checkpoint_dir = os.path.join(model_path,'checkpoint_epoch{}_iter{}'.format(epoch,iteration))
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp_path = os.path.join(checkpoint_dir, 'mp.pth')
            save_checkpoint(model, optimizer, epoch, iteration, cp_path, scheduler)
            
            # save output image
            for i, inputs in enumerate(train_dataloader, 1):
                with torch.no_grad():
                    outputs = model(inputs)
                    save_outputs(inputs, outputs, checkpoint_dir, i, cfg, mode='train')
                    
                if i >= 5:
                    break
            
            for i, inputs in enumerate(val_dataloader, 1):
                with torch.no_grad():
                    outputs = model(inputs)
                    save_outputs(inputs, outputs, checkpoint_dir, i, cfg, mode='val')
                    
                if i >= 5:
                    break        
                
    train_dataset.update_seed()
    print("seed: {}".format(train_dataset.seed))
    start_iter = 1