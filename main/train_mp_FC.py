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
from pycode.model.VideoPrediction import VIDEO_HOURGLASS
from pycode.loss.mp_loss import Train_Loss_sequence_hourglass
from pycode.misc import save_outputs, build_model_MP, build_dataset_MP, build_optimizer, str2bool, save_args, save_checkpoint, load_checkpoint, Timer, Time_dict, make_differentiable_pos_image, make_pos_image

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
parser.add_argument('--log_step', type=int, default=100, help='')
parser.add_argument('--save_step', type=int, default=10000, help='')
parser.add_argument('--eval_step', type=int, default=5000, help='')
parser.add_argument('--output_dirname', type=str, default='', help='')
parser.add_argument('--checkpoint_path', type=str, default=None, help='')
parser.add_argument('--vp_path', type=str, default='')
parser.add_argument('--mp_path', type=str, default='')
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--wandb_group', type=str, default='') # e.g. compare_input
parser.add_argument('--save_dataset', type=str2bool, default=False)
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
mp_model = stacked_hourglass_model(cfg, output_dim=1)
vp_model = VIDEO_HOURGLASS(cfg) 

# set optimizer
mp_optimizer = build_optimizer(cfg, mp_model, 'mp')
mp_scheduler = StepLR(mp_optimizer, step_size=cfg.SCHEDULER.STEPLR.STEP_SIZE, gamma=cfg.SCHEDULER.STEPLR.GAMMA)

vp_optimizer = build_optimizer(cfg, vp_model, 'vp')
vp_scheduler = StepLR(vp_optimizer, step_size=cfg.SCHEDULER.STEPLR.STEP_SIZE, gamma=cfg.SCHEDULER.STEPLR.GAMMA)

mp_model = torch.nn.DataParallel(mp_model, device_ids = list(range(cfg.BASIC.NUM_GPU)))
mp_model = mp_model.to(device)

vp_model = torch.nn.DataParallel(vp_model, device_ids = list(range(cfg.BASIC.NUM_GPU)))
vp_model = vp_model.to(device)

# set loss
train_loss = Train_Loss_sequence_hourglass(cfg, device)
MSE = torch.nn.MSELoss()
val_loss = Train_Loss_sequence_hourglass(cfg, device)

mp_checkpoint_path = os.path.join(args.mp_path, 'mp.pth')
vp_checkpoint_path = os.path.join(args.vp_path, 'vp.pth')

mp_model, _, _, _, _ = load_checkpoint(mp_model, mp_checkpoint_path)
vp_model, _, _, _, _ = load_checkpoint(vp_model, vp_checkpoint_path)

# load checkpoint
if args.checkpoint_path != None:
    mp_checkpoint_path = os.path.join(args.checkpoint_path, 'mp.pth')
    vp_checkpoint_path = os.path.join(args.checkpoint_path, 'vp.pth')

    if cfg.LOAD_MODEL == 'all':
        mp_model, mp_optimizer, start_epoch, start_iter, mp_scheduler = load_checkpoint(mp_model, mp_checkpoint_path, optimizer=mp_optimizer, scheduler=mp_scheduler)
        vp_model, vp_optimizer, start_epoch, start_iter, vp_scheduler = load_checkpoint(vp_model, vp_checkpoint_path, optimizer=vp_optimizer, scheduler=vp_scheduler)
    elif cfg.LOAD_MODEL == 'model_only':
        # dose tukawan kara nokosu. keshitemoiiyo
        mp_model, _, _, _, _ = load_checkpoint(mp_model, mp_checkpoint_path)
        vp_model, _, _, _, _ = load_checkpoint(vp_model, vp_checkpoint_path)
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

def make_videomodel_input(inputs, outputs, action='pred'):
    '''
    output:
    dictionary{
    rgb => torch.Tensor shape=(B,S,C,H,W),
    pose => torch.Tensor shape=(B,S,C,H,W)}

    mode1: input output heatmap
    mode2: input dataset heatmap
    '''
    data_dict = {}
    device = outputs['pose'].device

    if action == 'pred':
        t1_heatmap = outputs['heatmap'][:,-1:,-1]
        t1_pose = outputs['pose'][:,-1:,-1, 0]
        t1_rotation = outputs['rotation'][:,-1:,-1]
        t1_grasp = outputs['grasp'][:,-1:,-1, 0]
    elif action == 'gt':
        t1_heatmap = inputs['heatmap'][:,2:3].to(device)
        t1_pose = inputs['pose_xyz'][:,2:3].to(device)
        t1_rotation = inputs['rotation_matrix'][:,2:3].to(device)
        t1_grasp = inputs['grasp'][:,2:3].to(device)
    else:
        ValueError("invalid action")

    data_dict['rgb'] = inputs['rgb'][:,:2].to(device)
    
    pose_heatmap = inputs['heatmap'][:,:2].to(device)
    data_dict['heatmap'] = torch.cat((pose_heatmap, t1_heatmap), 1)
    
    pose_xyz = inputs['pose_xyz'][:,:2].to(device)
    data_dict['pose_xyz'] = torch.cat((pose_xyz, t1_pose), 1)
    
    rotation_matrix = inputs['rotation_matrix'][:,:2].to(device)
    data_dict['rotation_matrix'] = torch.cat((rotation_matrix, t1_rotation), 1)
    
    grasp = inputs['grasp'][:,:2].to(device)
    data_dict['grasp'] = torch.cat((grasp, t1_grasp), 1)
            
    return data_dict

for epoch in range(start_epoch, cfg.BASIC.MAX_EPOCH):
    for iteration, inputs in enumerate(train_dataloader, 1):
        time_dict.load_data += time.time() - load_start
        total_iteration = len(train_dataloader) * epoch + iteration
            
        # skip until start iter
        if total_iteration < start_iter:
            continue
            
        # optimize generator
        mp_optimizer.zero_grad()
        vp_optimizer.zero_grad()
        
        with Timer() as t:
            outputs = mp_model(inputs)
            # pos_image = make_differentiable_pos_image((256,256), outputs['uv'][:,0,-1].to('cpu'), inputs['uv_mask'][:,1])
            # pos_image = torch.unsqueeze(pos_image, 1)
            video_inputs = make_videomodel_input(inputs, outputs)
            video_outputs = vp_model(video_inputs)
            pred_image1 = video_outputs['rgb']

            video_inputs = make_videomodel_input(inputs, outputs, action='gt')
            video_outputs = vp_model(video_inputs)
            pred_image2 = video_outputs['rgb']
        time_dict.forward += t.secs

        with Timer() as t:
            loss = train_loss(inputs, outputs)
            loss += MSE(inputs['rgb'][:,2].to(device), pred_image1)
            loss += MSE(inputs['rgb'][:,2].to(device), pred_image2)
        time_dict.loss += t.secs

        with Timer() as t:
            loss.backward()
            mp_optimizer.step()
            mp_scheduler.step()
            vp_optimizer.step()
            vp_scheduler.step()
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
                max_iter, mp_optimizer.param_groups[0]['lr'], time.time() - tic, 
                time_dict.load_data, time_dict.forward, time_dict.backward, log['train/weight_loss']))
            
            train_loss.reset_log()
            tic = time.time()
            time_dict.reset()
        
        # save checkpoint
        if total_iteration % args.save_step == 0:
            checkpoint_dir = os.path.join(model_path,'checkpoint_iter{}'.format(total_iteration))
            os.makedirs(checkpoint_dir, exist_ok=True)
            cp_path = os.path.join(checkpoint_dir, 'mp.pth')
            save_checkpoint(mp_model, mp_optimizer, epoch, iteration, cp_path, mp_scheduler)

            cp_path = os.path.join(checkpoint_dir, 'vp.pth')
            save_checkpoint(vp_model, vp_optimizer, epoch, iteration, cp_path, vp_scheduler)
            
            # save output image
            for i, inputs in enumerate(train_dataloader, 1):
                with torch.inference_mode():
                    outputs = mp_model(inputs)
                    save_outputs(inputs, outputs, checkpoint_dir, i, cfg, mode='train')
                    
                if i >= 5:
                    break
            
            for i, inputs in enumerate(val_dataloader, 1):
                with torch.inference_mode():
                    outputs = mp_model(inputs)
                    save_outputs(inputs, outputs, checkpoint_dir, i, cfg, mode='val')
                    
                if i >= 5:
                    break

        # validation
        if total_iteration % args.eval_step == 0:
            print('validation start')
            for iteration, inputs in enumerate(val_dataloader, 1):
                with torch.inference_mode():
                    outputs = mp_model(inputs)
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