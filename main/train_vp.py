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
from pycode.model.VideoPrediction import VIDEO_HOURGLASS, Discriminator
from pycode.loss.vp_loss import Train_Loss_Video, D_Loss
from pycode.misc import build_dataset_VP, build_optimizer, str2bool, save_args, save_checkpoint, load_checkpoint, save_outputs_vp

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
parser.add_argument('--log_step', type=int, default=100, help='')
parser.add_argument('--save_step', type=int, default=5000, help='')
parser.add_argument('--eval_step', type=int, default=5000, help='')
parser.add_argument('--output_dirname', type=str, default='', help='')
parser.add_argument('--checkpoint_path', type=str, default=None, help='')
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--wandb_group', type=str, default='')
parser.add_argument('--save_dataset', type=str2bool, default=False)
parser.add_argument('--blas_num_threads', type=str, default="4", help='set this not to cause openblas error')
# args = parser.parse_args(args=['--checkpoint_path','output/2020-04-02_18:28:18.736004/model_log/checkpoint_epoch9_iter11'])
args = parser.parse_args()

os.environ["OPENBLAS_NUM_THREADS"] = args.blas_num_threads

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
if os.path.exists(output_dirname):
    while 1:
        ans = input('The specified output dir is already exists. Overwrite? y or n: ')
        if ans == 'y':
            break
        elif ans == 'n':
            raise ValueError("Please specify correct output dir")
        else:
            print('please type y or n')

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
    run = wandb.init(project='VideoPrediction-{}'.format(cfg.DATASET.NAME), entity='tendon',
                    config=obj, save_code=True, name=args.output_dirname, dir=os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME),
                    group=group)

# set dataset
train_dataset = build_dataset_VP(cfg, save_dataset=args.save_dataset, mode='train')
val_dataset = build_dataset_VP(cfg, save_dataset=args.save_dataset, mode='val')

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)

# set model
model = VIDEO_HOURGLASS(cfg)
model = torch.nn.DataParallel(model, device_ids = list(range(cfg.BASIC.NUM_GPU)))
model = model.to(device)

discriminator = Discriminator(cfg, train_dataset.size)
discriminator = torch.nn.DataParallel(discriminator, device_ids = list(range(cfg.BASIC.NUM_GPU)))
discriminator = discriminator.to(device)

# set loss
train_loss_d = D_Loss(cfg, device)
train_loss_vp = Train_Loss_Video(cfg, device)
val_loss_vp = Train_Loss_Video(cfg, device)

# set optimizer
optimizer_vp = build_optimizer(cfg, model, 'vp')
scheduler_vp = StepLR(optimizer_vp, step_size=cfg.SCHEDULER.STEPLR.STEP_SIZE, gamma=cfg.SCHEDULER.STEPLR.GAMMA)

optimizer_d = build_optimizer(cfg, discriminator, 'dis')
scheduler_d = StepLR(optimizer_d, step_size=cfg.SCHEDULER.STEPLR.STEP_SIZE, gamma=cfg.SCHEDULER.STEPLR.GAMMA)

# load checkpoint
if args.checkpoint_path != None:
    checkpoint_path = os.path.join(args.checkpoint_path, 'vp.pth')
    
    if cfg.LOAD_MODEL == 'all':
        model, optimizer_vp, start_epoch, start_iter, scheduler_vp = load_checkpoint(model, checkpoint_path, optimizer=optimizer_vp, scheduler=scheduler_vp)
    elif cfg.LOAD_MODEL == 'model_only':
        model, _, _, _, _ = load_checkpoint(model, checkpoint_path)
        start_epoch, start_iter = 0, 1
    
    if discriminator != None:
        d_checkpoint_path = os.path.join(args.checkpoint_path, 'dis.pth')
        if os.path.exists(d_checkpoint_path):
            if cfg.LOAD_MODEL == 'all':
                discriminator, optimizer_d, _, _, scheduler_d = load_checkpoint(discriminator, d_checkpoint_path, optimizer=optimizer_d, scheduler=scheduler_d)
            elif cfg.LOAD_MODEL == 'model_only':
                discriminator, _, _, _, _ = load_checkpoint(discriminator, d_checkpoint_path)
        else:
            raise ValueError("no discriminator checkpoint path")
else:
    start_epoch, start_iter = 0, 1

def make_videomodel_input(inputs, device, sequence_id=0):
    '''
    output:
    dictionary{
    rgb => torch.Tensor shape=(B,S,C,H,W),
    pose => torch.Tensor shape=(B,S,C,H,W)}
    '''
    if cfg.VIDEO_HOUR.MODE == 'pcf':
        index_list = [sequence_id, sequence_id+1, sequence_id+3]
        rgb = inputs['rgb'][:,index_list].to(device)
        pose_heatmap = inputs['pose'][:,:4].to(device)
        pose_xyz = inputs['pose_xyz'][:,:4].to(device)
        rotation_matrix = inputs['rotation_matrix'][:,:4].to(device)
        grasp = inputs['grasp'][:,:4].to(device)
    elif cfg.VIDEO_HOUR.MODE == 'pc':
        index_list = [sequence_id, sequence_id+1]
        rgb = inputs['rgb'][:,index_list].to(device)
        pose_heatmap = inputs['pose'][:,:3].to(device)
        pose_xyz = inputs['pose_xyz'][:,:3].to(device)
        rotation_matrix = inputs['rotation_matrix'][:,:3].to(device)
        grasp = inputs['grasp'][:,:3].to(device)
    elif cfg.VIDEO_HOUR.MODE == 'c':
        rgb = inputs['rgb'][:,1].to(device)
        pose_heatmap = inputs['pose'][:,1:3].to(device)
        pose_xyz = inputs['pose_xyz'][:,1:3].to(device)
        rotation_matrix = inputs['rotation_matrix'][:,1:3].to(device)
        grasp = inputs['grasp'][:,1:3].to(device)
    return {'rgb':rgb, 'pose':pose_heatmap, 'pose_xyz':pose_xyz, 'rotation_matrix':rotation_matrix, 'grasp':grasp}

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
        
        # optimize discriminator
        with torch.no_grad():
            output = model(make_videomodel_input(inputs,device))
            output_image = output['rgb'].detach()

        optimizer_d.zero_grad()
        d_loss = train_loss_d(inputs, output_image, discriminator)
        d_loss.backward()
        optimizer_d.step()
        scheduler_d.step()
            
        # optimize generator
        outputs = model(make_videomodel_input(inputs,device))
        loss = train_loss_vp(inputs, outputs) + train_loss_d(inputs, outputs['rgb'], discriminator, train_d=False)
        
        optimizer_vp.zero_grad()
        loss.backward()
        optimizer_vp.step()
        scheduler_vp.step()
        
        # time setting
        trained_time += time.time() - end
        end = time.time() 
        
        # save and print log
        if total_iteration % args.log_step == 0:
            log = train_loss_vp.get_log()
            eta_seconds = int((trained_time / total_iteration) * (max_iter - total_iteration))
            
            if args.log2wandb:
                wandb.log(log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Loss: {:.6f}'.format(total_iteration, 
                max_iter, optimizer_vp.param_groups[0]['lr'], time.time() - tic, 
                str(datetime.timedelta(seconds=eta_seconds)), log['train/weight_loss']))
            
            train_loss_vp.reset_log()
            train_loss_d.reset_log()
            tic = time.time()
            
        # validation
        if total_iteration % args.eval_step == 0:
            print('validation start')
            for iteration, inputs in enumerate(val_dataloader, 1):
                with torch.no_grad():
                    outputs = model(make_videomodel_input(inputs,device))
                    _ = val_loss_vp(inputs, outputs, mode='val')
                if iteration >= 1000:
                    break
            
            val_log = val_loss_vp.get_log()
            if args.log2wandb:
                wandb.log(val_log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, VAL Loss: {:.6f}'.format(total_iteration, max_iter, val_log['val/weight_loss']))
            print('')
            val_loss_vp.reset_log()
        
        # save checkpoint
        if total_iteration % args.save_step == 0:
            checkpoint_dir = os.path.join(model_path,'checkpoint_iter{}'.format(total_iteration))
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp_path_vp = os.path.join(checkpoint_dir, 'vp.pth')
            cp_path_d = os.path.join(checkpoint_dir, 'dis.pth')
            save_checkpoint(model, optimizer_vp, epoch, iteration, cp_path_vp, scheduler_vp)
            save_checkpoint(discriminator, optimizer_d, epoch, iteration, cp_path_d, scheduler_d)
            
            # save output image
            for i, inputs in enumerate(train_dataloader, 1):
                with torch.no_grad():
                    outputs = model(make_videomodel_input(inputs,device))
                    save_outputs_vp(inputs, outputs, checkpoint_dir, i, cfg, mode='train')
                    
                if i >= 5:
                    break
            
            for i, inputs in enumerate(val_dataloader, 1):
                with torch.no_grad():
                    outputs = model(make_videomodel_input(inputs,device))
                    save_outputs_vp(inputs, outputs, checkpoint_dir, i, cfg, mode='train')
                    
                if i >= 5:
                    break 
                
    train_dataset.update_seed()
    print("seed: {}".format(train_dataset.seed))
    start_iter = 1