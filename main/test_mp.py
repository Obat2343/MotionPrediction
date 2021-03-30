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

from pycode.dataset import RLBench_dataset_test, Softargmax_dataset_test
from pycode.config import _C as cfg
from pycode.model.Hourglass import sequence_hourglass
from pycode.loss.mp_loss import Test_Loss_sequence_hourglass
from pycode.misc import save_outputs, build_model_MP, build_dataset_MP, str2bool, save_args, save_checkpoint, load_checkpoint

# parser
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
parser.add_argument('--output_dirname', type=str, default='', help='')
parser.add_argument('--checkpoint_path', type=str, default=None, help='')
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--wandb_group', type=str, default='')
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
cfg.freeze()

# define save model path
save_path = os.path.join(args.checkpoint_path, 'test')

# make output dir
os.makedirs(output_dirname, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

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
    run = wandb.init(project='MotionPrediction-{}-test'.format(cfg.DATASET.NAME), entity='tendon',
                    config=obj, save_code=True, name=args.output_dirname, dir=os.path.join(cfg.BASIC.OUTPUT_DIR, cfg.DATASET.NAME),
                    group=group)

# set dataset
if cfg.DATASET.NAME == 'HMD':
    test_dataset = Softargmax_dataset_test(cfg, save_dataset=False)
elif cfg.DATASET.NAME == 'RLBench':
    test_dataset = RLBench_dataset_test(cfg, save_dataset=False)
else:
    raise ValueError("Invalid dataset name")

# set dataloader
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.BASIC.WORKERS)

# set model
if cfg.DATASET.NAME == 'HMD':
    output_dim = 21
elif cfg.DATASET.NAME == 'RLBench':
    output_dim = 1

model = sequence_hourglass(cfg, output_dim=output_dim)
model_path = os.path.join(args.checkpoint_path, 'mp.pth')
if cfg.MP_MODEL_NAME == 'hourglass':
    model.hour_glass, _, _, _, _ = load_checkpoint(model.hour_glass, model_path, fix_parallel=True)
elif cfg.MP_MODEL_NAME == 'sequence_hourglass':
    model, _, _, _, _ = load_checkpoint(model, model_path, fix_parallel=True)

model = model.to(device)

# set loss
test_loss = Test_Loss_sequence_hourglass(cfg, device)

# start train
for iteration, inputs in enumerate(test_dataloader, 1):
    
    with torch.no_grad():
        outputs = model(inputs)
        loss = test_loss(inputs, outputs)
    
    # save_outputs(inputs, outputs, save_path, iteration, cfg, mode='test')
    print("{} / {}".format(iteration, len(test_dataloader)))

# save and print log
log = train_loss.get_log()
for key in log.keys():
    print('key:{}  Value:{}'.format(key, log[key]))

if args.log2wandb:
    wandb.log(log,step=total_iteration)