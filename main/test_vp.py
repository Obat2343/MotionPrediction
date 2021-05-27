import sys
import os
import time
import datetime
import argparse
import torch
import shutil
import torchvision
from tqdm import tqdm
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
sys.path.append('../')

from pycode.dataset import RLBench_dataset_VP, imageaug_full_transform, train_val_split
from pycode.config import _C as cfg
from pycode.model.VideoPrediction import VIDEO_HOURGLASS, Discriminator
from pycode.loss.vp_loss import Test_Loss_Video
from pycode.misc import build_dataset_VP, str2bool, save_args, save_checkpoint, load_checkpoint

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='if not specified, use chepoint_path')
parser.add_argument('--output_dirname', type=str, default='', help='if not specified, use checkpoint_path')
parser.add_argument('--checkpoint_path','-c', type=str, help='e.g. output/RLdata/VP_pcf_dis_random/model_log/checkpoint_epoch0_iter100000/')
args = parser.parse_args()

# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)
else:
    config_dir = os.path.abspath(os.path.join(args.checkpoint_path, '../../'))
    file_list = os.listdir(config_dir)
    print(file_list)
    yaml_file_name = [x for x in file_list if '.yaml' in x]
    print(yaml_file_name)
    config_file_path = os.path.join(config_dir, yaml_file_name[0])
    print('Loaded configration file {}'.format(config_file_path))
    cfg.merge_from_file(config_file_path)

# define output dirname
if len(args.output_dirname) > 0:
    output_dirname = args.output_dirname
    output_dirname = os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME, output_dirname)
else:
    output_dirname = os.path.abspath(os.path.join(args.checkpoint_path,"../../"))

cfg.PRED_LEN = 1
cfg.BASIC.NUM_GPU = 1
cfg.BASIC.BATCH_SIZE = 1
cfg.freeze()

# make output dir
os.makedirs(cfg.BASIC.OUTPUT_DIR, exist_ok=True)

# set seed and cuda
torch.manual_seed(cfg.BASIC.SEED)
cuda = torch.cuda.is_available()
device = torch.device(cfg.BASIC.DEVICE)

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(cfg.BASIC.SEED)

# set dataset
test_dataset = build_dataset_VP(cfg, mode='test')

# set dataloader
test_dataloader = DataLoader(test_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=False, num_workers=cfg.BASIC.WORKERS)

# set model
model = VIDEO_HOURGLASS(cfg)
video_checkpoint_path = os.path.join(args.checkpoint_path, 'vp.pth')
model, _, _, _, _ = load_checkpoint(model, video_checkpoint_path, fix_parallel=True)

model = torch.nn.DataParallel(model, device_ids = list(range(cfg.BASIC.NUM_GPU)))
model = model.to(device)

eval_loss = Test_Loss_Video(cfg,device)

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
        if cfg.VIDEO_HOUR.INPUT_DEPTH:
            depth = inputs['depth'][:,index_list].to(device)
    elif cfg.VIDEO_HOUR.MODE == 'pc':
        index_list = [sequence_id, sequence_id+1]
        rgb = inputs['rgb'][:,index_list].to(device)
        pose_heatmap = inputs['pose'][:,:3].to(device)
        pose_xyz = inputs['pose_xyz'][:,:3].to(device)
        rotation_matrix = inputs['rotation_matrix'][:,:3].to(device)
        grasp = inputs['grasp'][:,:3].to(device)
        if cfg.VIDEO_HOUR.INPUT_DEPTH:
            depth = inputs['depth'][:,index_list].to(device)
    elif cfg.VIDEO_HOUR.MODE == 'c':
        rgb = inputs['rgb'][:,1].to(device)
        pose_heatmap = inputs['pose'][:,1:3].to(device)
        pose_xyz = inputs['pose_xyz'][:,1:3].to(device)
        rotation_matrix = inputs['rotation_matrix'][:,1:3].to(device)
        grasp = inputs['grasp'][:,1:3].to(device)
        if cfg.VIDEO_HOUR.INPUT_DEPTH:
            depth = inputs['depth'][:,1].to(device)
    
    input_dict = {}
    input_dict['rgb'] = rgb
    input_dict['pose'] = pose_heatmap
    input_dict['pose_xyz'] = pose_xyz
    input_dict['rotation_matrix'] = rotation_matrix
    input_dict['grasp'] = grasp
    if cfg.VIDEO_HOUR.INPUT_DEPTH:
        input_dict['depth'] = depth

    return input_dict

for inputs in tqdm(test_dataloader):
    with torch.no_grad():
        outputs = model(make_videomodel_input(inputs,device))
        _ = eval_loss(inputs, outputs)

eval_log = eval_loss.get_log()
keys = list(eval_log.keys())
keys.sort()
for key in keys:
    print('key:{} value:{}'.format(key, eval_log[key]))
    
eval_loss.reset_log()