import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os

import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import datetime
import shutil
import yaml
import sys
import time
sys.path.append("../git/future-image-similarity")

import utils

sys.path.append('../')
from pycode.dataset import RLBench_dataset3
from pycode.config import _C as cfg
from pycode.misc import save_outputs, build_model_MP, build_dataset_MP, build_optimizer, str2bool, save_args, save_checkpoint, load_checkpoint, Timer, Time_dict

import models.model_predictor as model
from models.model_predictor import gaussian_lstm as lstm_model
from models.model_value import ModelValue
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save trained models')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='lab_pose', help='predictor training data: lab_pose or gaz_pose')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
parser.add_argument('--output_dirname', type=str, default='', help='')
parser.add_argument('--log_step', type=int, default=100, help='')
parser.add_argument('--save_step', type=int, default=10000, help='')
parser.add_argument('--eval_step', type=int, default=5000, help='')
parser.add_argument('--log2wandb', type=str2bool, default=True)
parser.add_argument('--wandb_group', type=str, default='') # e.g. compare_input
parser.add_argument('--save_dataset', type=str2bool, default=False)
parser.add_argument('--checkpoint_path', type=str, default=None, help='')
parser.add_argument('--vp_checkpoint_path', type=str)

args = parser.parse_args()

# get cfg data
if len(args.config_file) > 0:
    print('Loaded configration file {}'.format(args.config_file))
    cfg.merge_from_file(args.config_file)

if cfg.PAST_LEN == 1:
    input_past = True
elif cfg.PAST_LEN == 0:
    input_past = False
else:
    raise ValueError("wrong length")
    
# define output dirname
if len(args.output_dirname) == 0:
    dt_now = datetime.datetime.now()
    output_dirname = str(dt_now.date()) + '_' + str(dt_now.time())
else:
    output_dirname = args.output_dirname

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

cfg.PAST_LEN = 0
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
random.seed(cfg.BASIC.SEED)
torch.manual_seed(cfg.BASIC.SEED)
torch.cuda.manual_seed_all(cfg.BASIC.SEED)
cuda = torch.cuda.is_available()
device = torch.device(cfg.BASIC.DEVICE)
dtype = torch.cuda.FloatTensor

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

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

vp_checkpoint_path = args.vp_checkpoint_path

prior_path = os.path.join(vp_checkpoint_path, "prior.pth")
encoder_path = os.path.join(vp_checkpoint_path, "encoder.pth")
decoder_path = os.path.join(vp_checkpoint_path, "decoder.pth")
pose_network_path = os.path.join(vp_checkpoint_path, "pose_network.pth")
conv_network_path = os.path.join(vp_checkpoint_path, "conv_network.pth")

lstm_input_size = int(((args.image_width / 16) - 2)**2 * args.g_dim)
lstm_output_size = int(((args.image_width / 16) - 2)**2 * 16)

if args.model_dir != '':
    prior = saved_model['prior']
else:
    prior = lstm_model(lstm_input_size, lstm_output_size, args.rnn_size, args.prior_rnn_layers, cfg.BASIC.BATCH_SIZE)
    prior.apply(utils.init_weights)

if args.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder_conv(args.g_dim, args.channels)
    decoder = model.decoder_conv(args.g_dim, args.channels, height=(args.image_width / 16) - 2, width=(args.image_width / 16) - 2)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

pose_network = model.pose_network(16, 14, 14, 13)
conv_network = model.conv_network(16+args.g_dim+int(args.z_dim/4), args.g_dim)
pose_network.apply(utils.init_weights)
conv_network.apply(utils.init_weights)

prior, _, _, _, _ = load_checkpoint(prior, prior_path)
encoder, _, _, _, _ = load_checkpoint(encoder, encoder_path)
decoder, _, _, _, _ = load_checkpoint(decoder, decoder_path)
pose_network, _, _, _, _ = load_checkpoint(pose_network, pose_network_path)
conv_network, _, _, _, _ = load_checkpoint(conv_network, conv_network_path)
prior, encoder, decoder, pose_network, conv_network, prior.cuda(), encoder.cuda(), decoder.cuda(), pose_network.cuda(), conv_network.cuda()

value_network = ModelValue(input_past=input_past)
value_network.apply(utils.init_weights)
value_network = value_network.cuda()

# ---------------- optimizers ----------------
args.optimizer = optim.Adam
value_network_optimizer = args.optimizer(value_network.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# --------- loss functions -------------------
image_criterion = nn.SmoothL1Loss()

class Loss(nn.Module):
    def __init__(self, cfg, device, mode):
        super(Loss, self).__init__()
        self.loss_dict = {}
        self.count = 0
        self.device = device
        self.l1_loss = nn.SmoothL1Loss()
        self.mode = mode
        
        self.loss_dict["Model based BC {}/l1 loss value net".format(self.mode)] = 0
        self.loss_dict["Model based BC {}/loss value net".format(self.mode)] = 0
        
    def get_log(self):
        for key in self.loss_dict.keys():
            self.loss_dict[key] /= self.count
        return self.loss_dict
    
    def reset_log(self):
        self.count = 0
        for key in self.loss_dict.keys():
            self.loss_dict[key] = 0
    
    def l1_criterion(self, pred_x, gt_x):
        l1_loss = self.l1_loss(pred_x, gt_x * 256 * 256)
        self.loss_dict["Model based BC {}/l1 loss value net".format(self.mode)] += l1_loss.item()
        self.loss_dict["Model based BC {}/loss value net".format(self.mode)] += l1_loss.item()
        return l1_loss

train_loss = Loss(cfg, 'cuda', 'train')
val_loss = Loss(cfg, 'cuda', 'val')

# set dataset
train_dataset = build_dataset_MP(cfg, save_dataset=args.save_dataset, mode='train')
val_dataset = build_dataset_MP(cfg, save_dataset=args.save_dataset, mode='val')

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS, drop_last=True)

def make_action(pose,rotation,grasp):
    B,S,_ = pose.shape
    rotation = rotation.view(B,S,-1)
    grasp = torch.unsqueeze(grasp, 2)
    
    action = torch.cat([pose,rotation,grasp],2)
    return action

def make_random_action(pose, rotation, grasp):
    B, _ = pose.shape
    random_pose = [random.uniform(-0.1,0.1) for _ in range(3 * B)]
    random_pose = torch.tensor(random_pose, dtype=torch.float32).view(B,3)
    random_pose = pose + random_pose.cuda()
    
    random_angles = [[random.uniform(-30,30) for _ in range(3)] for _ in range(B)]
    r = R.from_euler('zyx', random_angles, degrees=True)
    random_matrix = torch.tensor(r.as_matrix(), dtype=torch.float32).cuda()
    random_matrix = torch.bmm(random_matrix, rotation)
    random_matrix = random_matrix.view(B,-1)
    
    random_grasp = [random.randint(0,1) for _ in range(B)]
    random_grasp = torch.tensor(random_grasp, dtype=torch.float32).view(B,-1).cuda()
    
    random_action = torch.cat((random_pose, random_matrix, random_grasp), 1)
    return random_action

# start train
tic = time.time()
end = time.time()
trained_time = 0
# max_iter = cfg.BASIC.MAX_EPOCH * len(train_dataloader)
max_iter = cfg.BASIC.MAX_ITER
time_dict = Time_dict()
load_start = time.time()

start_epoch = 0
start_iter = 0

value_network.train()
prior.eval()
pose_network.eval()
conv_network.eval()
encoder.eval()
decoder.eval()
    
for epoch in range(start_epoch, cfg.BASIC.MAX_EPOCH):
    for iteration, inputs in enumerate(train_dataloader,1):
        time_dict.load_data += time.time() - load_start
        total_iteration = len(train_dataloader) * epoch + iteration
            
        # skip until start iter
        if iteration < start_iter:
            continue
            
        x = inputs["rgb"].cuda()
        pose = inputs["pose_xyz"].cuda()
        rotation = inputs["rotation_matrix"].cuda()
        grasp = inputs["grasp"].cuda()

        action = make_action(pose, rotation, grasp)
        B,S,C,H,W = x.shape

        # initialize the hidden state.
        prior.hidden = prior.init_hidden()

        reward_loss = 0.0
        running_loss = 0.0

        # forward
        value_network.zero_grad()
        
        with Timer() as t:
            with torch.no_grad():
                h_conv = encoder(x[:,0])
                h_conv, skip = h_conv
                B,C,H,W = h_conv.shape
                h = h_conv.view(B, H*W*args.g_dim)

                z_t, _, _ = prior(h)
                z_t = z_t.view(cfg.BASIC.BATCH_SIZE, -1, 14, 14)

                z_d_exp = pose_network(action[:,1].cuda()).detach()
                h_pred_exp = conv_network(torch.cat([h_conv, z_t, z_d_exp], 1)).detach()
                x_pred_exp = decoder([h_pred_exp, skip]).detach()

            num_cand = 15
            
            for j in range(num_cand):
                if j == 0:
                    input_action = action[:,1].cuda()
                else:
                    input_action = make_random_action(pose[:,0], rotation[:,0], grasp[:,0])
                
                with torch.no_grad():
                    z_d_rand = pose_network(input_action.cuda())
                    h_pred_rand = conv_network(torch.cat([h_conv, z_t, z_d_rand], 1))
                    x_pred_rand = decoder([h_pred_rand, skip])
                
                if input_past:
                    x_pred_rand = torch.cat((x_pred_rand, x[:,0]),1)
                x_value_rand = value_network(x_pred_rand)

                reward_label = []
                for batch_idx in range(cfg.BASIC.BATCH_SIZE):
                    reward_label.append(image_criterion(x_pred_rand[batch_idx,:3], x_pred_exp[batch_idx].detach()).data)
#                     print(reward_label[-1])
                    
                reward_label = torch.unsqueeze(torch.stack(reward_label),1)
                reward_label = Variable(reward_label, requires_grad=False)

                reward_loss += train_loss.l1_criterion(x_value_rand, reward_label)
        loss = reward_loss
        train_loss.count += 1
        
        time_dict.forward += t.secs
        
        with Timer() as t:
            loss.backward()
        time_dict.backward += t.secs
            
        value_network_optimizer.step()
        
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
            print('===> Iter: {:06d}/{:06d}, Cost: {:.2f}s, Load: {:.2f}, Forward: {:.2f}, Backward: {:.2f}, Loss: {:.6f}'.format(total_iteration, 
                max_iter,  time.time() - tic, 
                time_dict.load_data, time_dict.forward, time_dict.backward, log["Model based BC train/loss value net"]))
            
            train_loss.reset_log()
            tic = time.time()
            time_dict.reset()
        
        # save checkpoint
        if total_iteration % args.save_step == 0:
            checkpoint_dir = os.path.join(model_path,'checkpoint_iter{}'.format(total_iteration))
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            value_network_path = os.path.join(checkpoint_dir, 'value_network.pth')
            save_checkpoint(value_network, value_network_optimizer, epoch, iteration, value_network_path)
            
        # validation
        if total_iteration % args.eval_step == 0:
            print('validation start')
            for iteration, inputs in enumerate(val_dataloader, 1):
                with torch.inference_mode():
                    x = inputs["rgb"].cuda()
                    pose = inputs["pose_xyz"].cuda()
                    rotation = inputs["rotation_matrix"].cuda()
                    grasp = inputs["grasp"].cuda()

                    action = make_action(pose, rotation, grasp)
                    B,S,C,H,W = x.shape

                    # initialize the hidden state.
                    prior.hidden = prior.init_hidden()
                    
                    h_conv = encoder(x[:,0])
                    h_conv, skip = h_conv
                    B,C,H,W = h_conv.shape
                    h = h_conv.view(B, H*W*args.g_dim)

                    z_t, _, _ = prior(h)
                    z_t = z_t.view(cfg.BASIC.BATCH_SIZE, -1, 14, 14)

                    z_d_exp = pose_network(action[:,1].cuda()).detach()
                    h_pred_exp = conv_network(torch.cat([h_conv, z_t, z_d_exp], 1)).detach()
                    x_pred_exp = decoder([h_pred_exp, skip]).detach()
            
                    for j in range(num_cand):
                        if j == 0:
                            input_action = action[:,1].cuda()
                        else:
                            input_action = make_random_action(pose[:,0], rotation[:,0], grasp[:,0])
                        value_network.zero_grad()
                        z_d_rand = pose_network(input_action.cuda())
                        h_pred_rand = conv_network(torch.cat([h_conv, z_t, z_d_rand], 1))
                        x_pred_rand = decoder([h_pred_rand, skip])
                        if input_past:
                            x_pred_rand = torch.cat((x_pred_rand, x[:,0]),1)
                        x_value_rand = value_network(x_pred_rand)

                        reward_label = []
                        for batch_idx in range(cfg.BASIC.BATCH_SIZE):
                            reward_label.append(image_criterion(x_pred_rand[batch_idx,:3], x_pred_exp[batch_idx].detach()).data)
                            
                        reward_label = torch.unsqueeze(torch.stack(reward_label),1)
                        reward_label = Variable(reward_label, requires_grad=False)

                        reward_loss += val_loss.l1_criterion(x_value_rand, reward_label)
                    val_loss.count += 1

                    if iteration >= 100:
                        break
            
            val_log = val_loss.get_log()
            if args.log2wandb:
                wandb.log(val_log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, VAL Loss: {:.6f}'.format(total_iteration, max_iter, val_log['Model based BC val/loss value net']))
            print('')
            val_loss.reset_log()        

        load_start = time.time()

        if total_iteration == cfg.BASIC.MAX_ITER:
            sys.exit()

    train_dataset.update_seed()
    print("seed: {}".format(train_dataset.seed))
    start_iter = 1