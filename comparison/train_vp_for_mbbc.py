import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os

import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import itertools
import progressbar
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
from pycode.misc import save_outputs, build_dataset_VP, build_optimizer, str2bool, save_args, save_checkpoint, load_checkpoint, Timer, Time_dict

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
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

# set dataset
train_dataset = build_dataset_VP(cfg, save_dataset=args.save_dataset, mode='train')
val_dataset = build_dataset_VP(cfg, save_dataset=args.save_dataset, mode='val')

# set dataloader
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BASIC.BATCH_SIZE, shuffle=True, num_workers=cfg.BASIC.WORKERS)

from models.model_predictor import gaussian_lstm as lstm_model

lstm_input_size = int(((args.image_width / 16) - 2)**2 * args.g_dim)
lstm_output_size = int(((args.image_width / 16) - 2)**2 * 16)

if args.model_dir != '':
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    posterior = lstm_model(lstm_input_size, lstm_output_size, args.rnn_size, args.posterior_rnn_layers, cfg.BASIC.BATCH_SIZE)
    prior = lstm_model(lstm_input_size, lstm_output_size, args.rnn_size, args.prior_rnn_layers, cfg.BASIC.BATCH_SIZE)

    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

import models.model_predictor as model
       
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

args.optimizer = optim.Adam

posterior_optimizer = args.optimizer(posterior.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
prior_optimizer = args.optimizer(prior.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
encoder_optimizer = args.optimizer(encoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
decoder_optimizer = args.optimizer(decoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
pose_network_optimizer = args.optimizer(pose_network.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
conv_network_optimizer = args.optimizer(conv_network.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# --------- loss functions ------------------------------------
class Loss(nn.Module):
    def __init__(self, cfg, device, mode, beta):
        super(Loss, self).__init__()
        self.loss_dict = {}
        self.count = 0
        self.device = device
        self.l1_loss = nn.SmoothL1Loss()
        self.batch_size = cfg.BASIC.BATCH_SIZE
        self.mode = mode
        self.beta = beta
        
        self.loss_dict["Model based BC {}/kl loss".format(self.mode)] = 0
        self.loss_dict["Model based BC {}/l1 loss".format(self.mode)] = 0
        self.loss_dict["Model based BC {}/loss".format(self.mode)] = 0
        
    def get_log(self):
        for key in self.loss_dict.keys():
            self.loss_dict[key] /= self.count
        return self.loss_dict
    
    def reset_log(self):
        self.count = 0
        for key in self.loss_dict.keys():
            self.loss_dict[key] = 0
    
    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        kld = kld.sum() / self.batch_size
        self.loss_dict["Model based BC {}/kl loss".format(self.mode)] += kld.item()
        
        kld = self.beta * kld
        self.loss_dict["Model based BC {}/loss".format(self.mode)] += kld.item()
        return kld
    
    def l1_criterion(self, pred_x, gt_x):
        l1_loss = self.l1_loss(pred_x, gt_x)
        self.loss_dict["Model based BC {}/l1 loss".format(self.mode)] += l1_loss.item()
        self.loss_dict["Model based BC {}/loss".format(self.mode)] += l1_loss.item()
        return l1_loss

train_loss = Loss(cfg, 'cuda', 'train', args.beta)
val_loss = Loss(cfg, 'cuda', 'val', args.beta)

# --------- transfer to gpu ------------------------------------
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
pose_network.cuda()
conv_network.cuda()

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

def make_action(pose,rotation,grasp):
    B,S,_ = pose.shape
    rotation = rotation.view(B,S,-1)
    grasp = torch.unsqueeze(grasp, 2)
    
    action = torch.cat([pose,rotation,grasp],2)
    return action

def plot(x, action, val_iter, index, mode):
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
    gen_seq.append(x[:,0])
    for i in range(1, cfg.PRED_LEN + cfg.PAST_LEN):
        h_conv = encoder(x[:,i-1])
        h_target = encoder(x[:,i])[0]
        
        if args.last_frame_skip or i < cfg.PAST_LEN:	
            h_conv, skip = h_conv
        else:
            h_conv = h_conv[0]
        
        B,C,H,W = h_conv.shape
        h = h_conv.view(B, H*W*args.g_dim)

        h_conv = h_conv.detach()
        h_target = h_target.detach()
        z_t, _, _= posterior(h_target)
        z_t = z_t.view(cfg.BASIC.BATCH_SIZE, -1, 14, 14)

        if i < cfg.PAST_LEN:
            gen_seq.append(x[:,i])
        else:
            z_d = pose_network(Variable(action[:,i-1].cuda())).detach()
            h_pred = conv_network(torch.cat([h_conv, z_t, z_d], 1)).detach()
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)

    to_plot = []
    nrow = min(cfg.BASIC.BATCH_SIZE, 10)
    for i in range(nrow):
        row = []
        for t in range(cfg.PRED_LEN + cfg.PAST_LEN):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/rec_%d_%d_%s.png' % (checkpoint_dir, val_iter, index, mode) 
    utils.save_tensors_image(fname, to_plot)

# start train
tic = time.time()
end = time.time()
trained_time = 0
# max_iter = cfg.BASIC.MAX_EPOCH * len(train_dataloader)
max_iter = cfg.BASIC.MAX_ITER
time_dict = Time_dict()
load_start = time.time()

for epoch in range(start_epoch, cfg.BASIC.MAX_EPOCH):
    for iteration, inputs in enumerate(train_dataloader, 1):
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

        posterior.zero_grad()
        prior.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()
        pose_network.zero_grad()
        conv_network.zero_grad()

        # initialize the hidden state.
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

        l1 = 0
        kld = 0
        
        # forward
        with Timer() as t:
            for sequence_index in range(1, S-1):
                h_conv = encoder(x[:,sequence_index-1])
                h_target = encoder(x[:,sequence_index])[0]
                if args.last_frame_skip or sequence_index < cfg.PAST_LEN:	
                    h_conv, skip = h_conv
                else:
                    h_conv = h_conv[0]

                B,C,H,W = h_conv.shape
                h = h_conv.view(B, H*W*args.g_dim)

                z_t, mu, logvar = posterior(h_target)
                z_t = z_t.view(cfg.BASIC.BATCH_SIZE, -1, 14, 14)
                _, mu_p, logvar_p = prior(h)

                z_d = pose_network(action[:,sequence_index])
                h_pred = conv_network(torch.cat([h_conv, z_t, z_d], 1))

                x_pred = decoder([h_pred, skip])

                l1 += train_loss.l1_criterion(x_pred, x[:,sequence_index])
                kld += train_loss.kl_criterion(mu, logvar, mu_p, logvar_p)
                train_loss.count += 1
                
            loss = l1 + kld*args.beta
        time_dict.forward += t.secs
        
        # backward
        with Timer() as t:
            loss.backward()
            posterior_optimizer.step()
            prior_optimizer.step()
            encoder_optimizer.step()
            decoder_optimizer.step()
            pose_network_optimizer.step()
            conv_network_optimizer.step()
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
            print('===> Iter: {:06d}/{:06d}, Cost: {:.2f}s, Load: {:.2f}, Forward: {:.2f}, Backward: {:.2f}, Loss: {:.6f}'.format(total_iteration, 
                max_iter,  time.time() - tic, 
                time_dict.load_data, time_dict.forward, time_dict.backward, log["Model based BC train/loss"]))
            
            train_loss.reset_log()
            tic = time.time()
            time_dict.reset()
        
        # save checkpoint
        if total_iteration % args.save_step == 0:
            checkpoint_dir = os.path.join(model_path,'checkpoint_iter{}'.format(total_iteration))
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            posterior_path = os.path.join(checkpoint_dir, 'posterior.pth')
            save_checkpoint(posterior, posterior_optimizer, epoch, iteration, posterior_path)
            
            prior_path = os.path.join(checkpoint_dir, 'prior.pth')
            save_checkpoint(prior, prior_optimizer, epoch, iteration, prior_path)
            
            encoder_path = os.path.join(checkpoint_dir, 'encoder.pth')
            save_checkpoint(encoder, encoder_optimizer, epoch, iteration, encoder_path)
            
            decoder_path = os.path.join(checkpoint_dir, 'decoder.pth')
            save_checkpoint(decoder, decoder_optimizer, epoch, iteration, decoder_path)
            
            pose_network_path = os.path.join(checkpoint_dir, 'pose_network.pth')
            save_checkpoint(pose_network, pose_network_optimizer, epoch, iteration, pose_network_path)
            
            conv_network_path = os.path.join(checkpoint_dir, 'conv_network.pth')
            save_checkpoint(conv_network, conv_network_optimizer, epoch, iteration, conv_network_path)
            
            # save output image
            for i, inputs in enumerate(train_dataloader, 1):
                with torch.inference_mode():
                    x = inputs["rgb"].cuda()
                    pose = inputs["pose_xyz"].cuda()
                    rotation = inputs["rotation_matrix"].cuda()
                    grasp = inputs["grasp"].cuda()

                    action = make_action(pose, rotation, grasp)
                    plot(x, action, total_iteration, i, "train")
                    
                if i >= 5:
                    break
            
            for i, inputs in enumerate(val_dataloader, 1):
                with torch.inference_mode():
                    x = inputs["rgb"].cuda()
                    pose = inputs["pose_xyz"].cuda()
                    rotation = inputs["rotation_matrix"].cuda()
                    grasp = inputs["grasp"].cuda()

                    action = make_action(pose, rotation, grasp)
                    plot(x, action, total_iteration, i, "val")
                    
                if i >= 5:
                    break

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
                    posterior.hidden = posterior.init_hidden()
                    prior.hidden = prior.init_hidden()

                    l1 = 0
                    kld = 0

                    # forward
                    for sequence_index in range(1, S-1):
                        h_conv = encoder(x[:,sequence_index-1])
                        h_target = encoder(x[:,sequence_index])[0]
                        if args.last_frame_skip or sequence_index < cfg.PAST_LEN:	
                            h_conv, skip = h_conv
                        else:
                            h_conv = h_conv[0]

                        B,C,H,W = h_conv.shape
                        h = h_conv.view(B, H*W*args.g_dim)

                        z_t, mu, logvar = posterior(h_target)
                        z_t = z_t.view(cfg.BASIC.BATCH_SIZE, -1, 14, 14)
                        _, mu_p, logvar_p = prior(h)

                        z_d = pose_network(action[:,sequence_index])
                        h_pred = conv_network(torch.cat([h_conv, z_t, z_d], 1))

                        x_pred = decoder([h_pred, skip])

                        l1 += val_loss.l1_criterion(x_pred, x[:,sequence_index])
                        kld += val_loss.kl_criterion(mu, logvar, mu_p, logvar_p)
                        val_loss.count += 1

                    if iteration >= 100:
                        break
            
            val_log = val_loss.get_log()
            if args.log2wandb:
                wandb.log(val_log,step=total_iteration)
            
            print('===> Iter: {:06d}/{:06d}, VAL Loss: {:.6f}'.format(total_iteration, max_iter, val_log['Model based BC val/loss']))
            print('')
            val_loss.reset_log()        

        load_start = time.time()

        if total_iteration == cfg.BASIC.MAX_ITER:
            sys.exit()

    train_dataset.update_seed()
    print("seed: {}".format(train_dataset.seed))
    start_iter = 1