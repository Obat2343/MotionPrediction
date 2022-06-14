import sys
import json
import os
import cv2
import numpy as np
from numpy.core.numeric import zeros_like
import torch
import torchvision
import torch_optimizer as new_optim
from collections import OrderedDict
from torchvision import datasets, models, transforms
from PIL import Image, ImageDraw, ImageOps
from .dataset import Softargmax_dataset, Softargmax_dataset_VP, RLBench_dataset, RLBench_dataset_VP, RLBench_dataset3, RLBench_dataset3_VP, RLBench_dataset_skip, RLBench_dataset_VP_skip
import time
from fastdtw import fastdtw # https://github.com/slaypni/fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R
from einops import rearrange, reduce, repeat

def build_dataset_MP(cfg, save_dataset=False, mode='train'):
    if cfg.DATASET.NAME == 'HMD':
        if mode == 'train':
            dataset = Softargmax_dataset(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = Softargmax_dataset(cfg,save_dataset=save_dataset, mode=mode)
    elif cfg.DATASET.NAME == 'RLBench':
        if mode == 'train':
            dataset = RLBench_dataset(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = RLBench_dataset(cfg, save_dataset=save_dataset, mode=mode)
    elif (cfg.DATASET.NAME == 'RLBench3') or (cfg.DATASET.NAME == 'RLBench4') or (cfg.DATASET.NAME == 'RLBench4-sawyer'):
        if mode == 'train':
            dataset = RLBench_dataset_skip(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = RLBench_dataset_skip(cfg, save_dataset=save_dataset, mode=mode, num_sequence=100)

    return dataset

def build_dataset_VP(cfg, save_dataset=False, mode='train'):
    if cfg.DATASET.NAME == 'HMD':
        if mode == 'train':
            dataset = Softargmax_dataset_VP(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = Softargmax_dataset_VP(cfg,save_dataset=save_dataset, mode=mode)
        elif mode == 'test':
            dataset = Softargmax_dataset_VP(cfg,save_dataset=save_dataset, mode=mode, random_len=1)
    elif cfg.DATASET.NAME == 'RLBench':
        if mode == 'train':
            dataset = RLBench_dataset_VP(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = RLBench_dataset_VP(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'test':
            dataset = RLBench_dataset_VP(cfg, save_dataset=save_dataset, mode='val', random_len=1)
    elif (cfg.DATASET.NAME == 'RLBench3') or (cfg.DATASET.NAME == 'RLBench4') or (cfg.DATASET.NAME == 'RLBench4-sawyer'):
        if mode == 'train':
            dataset = RLBench_dataset_VP_skip(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = RLBench_dataset_VP_skip(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'test':
            dataset = RLBench_dataset_VP_skip(cfg, save_dataset=save_dataset, mode='val', random_len=0)
    
    return dataset

def build_model_MP(cfg, args):
    from .model.Hourglass import stacked_hourglass_model, sequence_hourglass
    if cfg.DATASET.NAME == 'HMD':
        output_dim = 21
    elif 'RLBench' in cfg.DATASET.NAME:
        output_dim = 1

    if cfg.MP_MODEL_NAME == 'hourglass':
        print('use hourglass')
        model = stacked_hourglass_model(cfg, output_dim=output_dim)
    elif cfg.MP_MODEL_NAME == 'sequence_hourglass':
        print('use sequence hourglass')
        model = sequence_hourglass(cfg, output_dim=output_dim)
        if (args.vp_path != "") and (cfg.SEQUENCE_HOUR.USE_VIDEOMODEL):
            print("load vp")
            vp_path = os.path.join(args.vp_path, 'vp.pth')
            model.video_pred_model, _, _, _, _ = load_checkpoint(model.video_pred_model, vp_path, fix_parallel=True)

        if (len(args.hourglass_path) != 0) and cfg.MP_MODEL_NAME == 'sequence_hourglass':
            print("load hourglass")
            model.hour_glass, _, _, _, _ = load_checkpoint(model.hour_glass, args.hourglass_path, fix_parallel=True)
        
        for param in model.video_pred_model.parameters():
            param.requires_grad = False
    return model

def build_optimizer(cfg, model, model_type='vp'):
    if model_type == 'vp':
        lr = cfg.OPTIM.VPLR
        name = cfg.OPTIM.VPNAME
    elif model_type == 'dis':
        lr = cfg.OPTIM.DLR
        name = cfg.OPTIM.DNAME
    elif model_type == 'mp':
        lr = cfg.OPTIM.MPLR
        name = cfg.OPTIM.MPNAME

    if name == "sgd":
        optimizer = torch.optim.SGD(
            filter(lambda p:p.requires_grad, model.parameters()),
            lr = lr,
            momentum = cfg.OPTIM.SGD.MOMENTUM,
            dampening = cfg.OPTIM.SGD.DAMPENING,
            weight_decay = cfg.OPTIM.SGD.WEIGHT_DECAY,
            nesterov = cfg.OPTIM.SGD.NESTEROV,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p:p.requires_grad, model.parameters()),
            lr= lr,
            betas=(cfg.OPTIM.RADAM.BETA1, cfg.OPTIM.RADAM.BETA2),
            eps=cfg.OPTIM.RADAM.EPS,
            weight_decay=cfg.OPTIM.RADAM.WEIGHT_DECAY,
        )
    elif name == "radam":
        optimizer = new_optim.RAdam(
            filter(lambda p:p.requires_grad, model.parameters()),
            lr= lr,
            betas=(cfg.OPTIM.RADAM.BETA1, cfg.OPTIM.RADAM.BETA2),
            eps=cfg.OPTIM.RADAM.EPS,
            weight_decay=cfg.OPTIM.RADAM.WEIGHT_DECAY,
        )

    return optimizer

def str2bool(s):
    return s.lower() in ('true', '1')

def save_args(args,file_path="args_data.json"):
    with open(file_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def save_checkpoint(model, optimizer, epoch, iteration, file_path, scheduler=False):
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch'] = epoch
    checkpoint['iteration'] = iteration
    if scheduler != False:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint,file_path)

def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None,fix_parallel=False):
    checkpoint = torch.load(checkpoint_path)
    if fix_parallel:
        print('fix parallel')
        model.load_state_dict(fix_model_state_dict(checkpoint['model']), strict=True)
    else:
        model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']

    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler != None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return model, optimizer, epoch, iteration, scheduler

def load_hourglass(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    def hourglass_fix_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.hour_glass.'):
                name = name[18:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
        return new_state_dict
    
    model.load_state_dict(hourglass_fix_state_dict(checkpoint['model']), strict=True)
    return model

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        if 'norm' in name:
            start_index = name.find('norm')
            name = name[:start_index] + 'pose' + name[start_index+4:]
        new_state_dict[name] = v
    return new_state_dict

class debug_set(object):
    def __init__(self, output_dir, dprint=False, dwrite=False):
        self.output_file = os.path.join(output_dir,'debug.txt')
        self.dprint = dprint
        self.dwrite = dwrite
        if self.dwrite == True:
            with open(self.output_file, mode='w') as f:
                f.write('debug file\n')

    def do_all(self,sentence):
        self.print(sentence)
        self.write(sentence)

    def print(self, sentence):
        if self.dprint == True:
            print(sentence)
    
    def write(self, sentence):
        if self.dwrite == True:
            with open(self.output_file, mode='a') as f:
                f.write('{}\n'.format(sentence))

def save_outputs(inputs, outputs, path, index, cfg, mode='train'):
    uv_list, heatmap_list, pose_list = outputs['uv'], outputs['heatmap'], outputs['pose']
    Batch, Seq, Block, C, H, W = heatmap_list.shape
    heatmaps = convert_heatmaps(heatmap_list[:,:,-1])
    for sequence_id in range(Seq):
        heatmap = heatmaps[:,sequence_id]
        #gt_current_pose_image = torch.unsqueeze(torch.unsqueeze(torch.sum(inputs['heatmap'],dim=1),1),1)
        #gt_future_pose_image = torch.unsqueeze(torch.sum(inputs['future_pose_image'],dim=2),2)

        # save rgb image
        save_image_path = os.path.join(path,'{}_image_{}_{}.jpg'.format(mode, index, sequence_id))
        torchvision.utils.save_image(inputs['rgb'][:,sequence_id+1],save_image_path,nrow=cfg.BASIC.BATCH_SIZE)

        # save heatmap
        save_heatmap_path = os.path.join(path,'{}_heatmap_{}_{}.jpg'.format(mode, index, sequence_id))
        torchvision.utils.save_image(torch.unsqueeze(heatmap.view(-1,H,W),1),save_heatmap_path,nrow=cfg.BASIC.BATCH_SIZE)

        # save rgb image with heatmap
        overlay_image = make_overlay_image_and_heatmap(inputs['rgb'][:,sequence_id+1], heatmap.to('cpu'))
        save_heatmap_overlay_path = os.path.join(path,'{}_image_heatmap_{}_{}.jpg'.format(mode, index, sequence_id))
        torchvision.utils.save_image(overlay_image.view(-1,3,H,W),save_heatmap_overlay_path)

        # save uv cordinate
        pos_image = make_pos_image((W,H),uv_list[:,sequence_id,-1].to('cpu'),inputs['uv_mask'][:,sequence_id+2])
        overlay_image = make_overlay_image(inputs['rgb'][:,sequence_id+1], pos_image)
        pos_image_gt = make_pos_image((W,H),inputs['uv'][:,2+sequence_id],inputs['uv_mask'][:,2+sequence_id])
        overlay_image_gt = make_overlay_image(inputs['rgb'][:,sequence_id+1], pos_image_gt)
        uv_images = torch.cat((overlay_image, overlay_image_gt), 0)
        save_uv_path = os.path.join(path,'{}_image_uv_{}_{}.jpg'.format(mode, index, sequence_id))
        torchvision.utils.save_image(uv_images, save_uv_path, nrow=cfg.BASIC.BATCH_SIZE)

        # save output image
        if cfg.HOURGLASS.PRED_RGB:
            sequence_image = torch.cat((outputs['rgb'][:,sequence_id,-1].to('cpu'),inputs['rgb'][:,sequence_id+2]),0)
            save_image_path = os.path.join(model_path,'checkpoint_epoch{}_iter{}'.format(epoch,iteration),'{}_image_output_{}_{}.jpg'.format(mode, index, sequence_id))
            torchvision.utils.save_image(sequence_image, save_image_path,nrow=cfg.BASIC.BATCH_SIZE)
        
        # save trajectory
        if cfg.HOURGLASS.PRED_TRAJECTORY:
            trajectory_image = outputs['trajectory'][:,sequence_id,-1].to('cpu')
            trajectory_gt = inputs['trajectory'][:,sequence_id+1]
            trajectory_images = torch.cat((trajectory_image, trajectory_gt), 0)
            save_trajectory_path = os.path.join(path,'{}_trajectory_{}_{}.jpg'.format(mode, index, sequence_id))
            torchvision.utils.save_image(trajectory_images, save_trajectory_path, nrow=cfg.BASIC.BATCH_SIZE)
    

    # save rotation image
    if cfg.HOURGLASS.PRED_ROTATION:
        rotation_image = make_rotation_image(inputs['rgb'][:,2:], outputs['rotation'][:,:,-1], outputs['pose'][:,:,-1,0], inputs['mtx'])
        rotation_gt_image = make_rotation_image(inputs['rgb'][:,2:], inputs['rotation_matrix'][:,2:], inputs['pose_xyz'][:,2:], inputs['mtx'])
        rotation_images = torch.cat((rotation_image, rotation_gt_image), 0)
        rotation_images = rotation_images.view(-1, 3, H, W)
        save_rotation_path = os.path.join(path,'{}_rotation_{}.jpg'.format(mode, index))
        torchvision.utils.save_image(rotation_images, save_rotation_path, nrow=Seq)

    if Seq > 2:
        rotation_gt_image = make_rotation_image(inputs['rgb'][:,2:], inputs['rotation_matrix'][:,2:], inputs['pose_xyz'][:,2:], inputs['mtx'])
        rotation_image = make_rotation_image(outputs["rgb"][:,:,-1].to('cpu'), outputs['rotation'][:,:,-1], outputs['pose'][:,:,-1,0], inputs['mtx'])
        pose_overlay_images = make_overlay_image_and_heatmaps(outputs["rgb"][:,:,-1].to('cpu'),outputs["heatmap"][:,:,-1].to('cpu'))
        images_tensor = torch.cat([inputs["rgb"][:,2:], rotation_gt_image.to('cpu'), outputs["rgb"][:,:,-1].to('cpu'), rotation_image.to('cpu'), pose_overlay_images.to('cpu')], 1)
        images_tensor = images_tensor.view(-1, 3, H, W)
        save_images_path = os.path.join(path,'{}_seq_image_{}.jpg'.format(mode, index))
        torchvision.utils.save_image(images_tensor, save_images_path, nrow=Seq)

def save_outputs_vp(inputs, outputs, path, index, cfg, mode='train'):
    pred_image = outputs['rgb'].cpu()
    B, C, H, W = pred_image.shape

    # save images
    gt_images = inputs['rgb'][:,2]
    diff_images = save_diff_heatmap_overlay(pred_image, gt_images)
    save_images = torch.cat((gt_images, pred_image, diff_images), 0)
    save_images_path = os.path.join(path,'{}_image_{}.jpg'.format(mode,index))
    torchvision.utils.save_image(save_images, save_images_path, nrow=cfg.BASIC.BATCH_SIZE)

    # save gt rgb image with uv cordinate
    pos_image = make_pos_image((W,H),inputs['uv'][:,2],inputs['uv_mask'][:,2])
    overlay_gt_image = make_overlay_image(inputs['rgb'][:,2], pos_image)
    overlay_pred_image = make_overlay_image(pred_image, pos_image)
    overlay_diff_image = make_overlay_image(diff_images, pos_image)
    overlay_images = torch.cat((overlay_gt_image, overlay_pred_image, diff_images), 0)
    save_over_path = os.path.join(path,'{}_image_uv_{}.jpg'.format(mode,index))
    torchvision.utils.save_image(overlay_images,save_over_path,nrow=cfg.BASIC.BATCH_SIZE)

def convert_heatmap(heatmap):
    B, C, H, W = heatmap.shape
    max_value = torch.max(heatmap.view(B,C,-1),2)[0]
    max_value = torch.unsqueeze(max_value,2)
    max_value = max_value.expand(B,C,H*W)
    max_value = max_value.view(B, C, H, W)
    heatmap = heatmap / max_value
    return heatmap

def convert_heatmaps(heatmaps):
    B, S, C, H, W = heatmaps.shape
    for s in range(S):
        heatmap = heatmaps[:,s]
        max_value = torch.max(heatmap.view(B,C,-1),2)[0]
        max_value = torch.unsqueeze(max_value,2)
        max_value = max_value.expand(B,C,H*W)
        max_value = max_value.view(B, C, H, W)
        heatmap = heatmap / max_value
        heatmaps[:,s] = heatmap
    return heatmaps

def draw_matrix(image, rotation_matrix, pos_vector, intrinsic_matrix):
    """
    image: PIL.Image
    pose_matrix: np.array (4X4)
        pose is position and orientation in the camera coordinate.
    intrinsic_matrix: np.array(4X4)
    """
    pose_matrix = np.append(rotation_matrix, pos_vector.T, 1)
    pose_matrix = np.append(pose_matrix, np.array([[0,0,0,1]]),0)
    
    cordinate_vector_array = np.array([[0,0,0,1],[0,0,0.1,1],[0,0.1,0,1],[0.1,0,0,1]]).T
    cordinate_matrix = np.dot(pose_matrix, cordinate_vector_array)
    
    draw = ImageDraw.Draw(image)
    color_list = [(255,0,0), (0,255,0), (0,0,255)]
    
    base_cordinate = cordinate_matrix.T[0]
    cordinates = cordinate_matrix.T[1:]

    base_cordinate = base_cordinate[:3] / base_cordinate[2]
    base_uv = np.dot(intrinsic_matrix, base_cordinate)
    base_u, base_v = base_uv[0], base_uv[1]
    
    for i in range(len(cordinates)):
        cordinate = cordinates[i]
        cordinate = cordinate[:3] / cordinate[2]
        uv = np.dot(intrinsic_matrix, cordinate)
        u, v = uv[0], uv[1]
        
        draw.line((base_u, base_v, u, v), fill=color_list[i], width=3)
    
    return image

def make_rotation_image(rgb, rotation_matrix, pose_vec, intrinsic_matrix):
    """
    input
    rgb: tensor (B, S, C, H, W)
    rotation_matrix: tensor
    intrinsic_matrix: tensor
    
    output
    image_sequence: tensor (BS, C, H ,W)
    """

    B, S, C, H, W = rgb.shape
    image_sequence = torch.zeros(B, S, C, H, W)
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    for b in range(B):
        for s in range(S):
            # make pil image with rotation 
            image_pil = topil(torch.clamp(rgb[b,s], 0, 1))
            rotation_np = rotation_matrix[b,s].cpu().numpy()

            pos_vec_np = pose_vec[b,s].cpu().numpy()
            pos_vec_np = np.expand_dims(pos_vec_np, 0)

            intrinsic_matrix_np = intrinsic_matrix[b].cpu().numpy()
            image_pil = draw_matrix(image_pil, rotation_np, pos_vec_np, intrinsic_matrix_np)
            # to tensor batch
            image_tensor = totensor(image_pil)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            
            image_sequence[b,s] = image_tensor
            # if (s == 0) and (b == 0):
            #     image_sequence = image_tensor
            # else:
            #     image_sequence = torch.cat((image_sequence, image_tensor),0)
    
    return image_sequence

def make_overlay_image(rgb, pose_image):
    B, C, H, W = pose_image.shape
    red_image = torch.zeros(B, 3, H, W)
    red_image[:,0:1] = torch.clamp(pose_image, 0, 1)

    pose_image_rgb = torch.cat((pose_image, pose_image, pose_image), 1)
    over_image = torch.where(pose_image_rgb.cpu() >= 1, red_image, rgb.cpu())
    return over_image

def make_overlay_images(rgb, pose_image):
    B, S, C, H, W = pose_image.shape
    red_image = torch.zeros(B, S, 3, H, W)
    red_image[:,:,0:1] = torch.clamp(pose_image, 0, 1)

    pose_image_rgb = torch.cat((pose_image, pose_image, pose_image), 2)
    over_image = torch.where(pose_image_rgb.cpu() >= 1, red_image, rgb.cpu())
    return over_image

def make_overlay_image_and_heatmap(rgb, heatmap):
    B, C, H, W = heatmap.shape
    for i in range(C):
        red_image = torch.zeros(B, 3, H, W)
        red_image[:,0] = torch.clamp(heatmap[:,i], 0, 1)

        over_image = 0.3*rgb + 0.7*red_image
        if i == 0:
            over_image_batch = torch.unsqueeze(over_image, 1)
        else:
            over_image_batch = torch.cat((over_image_batch, torch.unsqueeze(over_image, 1)), 1)
    return over_image_batch

def make_overlay_image_and_heatmaps(rgbs, heatmaps):
    B, S, C, H, W = heatmaps.shape
    for i in range(C):
        red_image = torch.zeros(B, S, 3, H, W)
        red_image[:,:,0] = torch.clamp(heatmaps[:,:,i], 0, 1)

        over_image = 0.3*rgbs + 0.7*red_image
        if i == 0:
            over_image_batch = torch.unsqueeze(over_image, 2)
        else:
            over_image_batch = torch.cat((over_image_batch, torch.unsqueeze(over_image, 2)), 2)
    if C != 1:
        raise ValueError("TODO change")
    return over_image_batch[:,:,0] # if C is larger than 1. please change

def make_pos_image(size,uv_data,uv_mask,r=3):
    B, C, _ = uv_data.shape
    totensor = transforms.ToTensor()
    uv_data_np = uv_data.detach().numpy()
    for b in range(B):
        pos_image = Image.new('L',size)
        draw = ImageDraw.Draw(pos_image)
        for uv,mask_uv in zip(uv_data_np[b],uv_mask[b]):
            u,v = int(uv[0]), int(uv[1])
            u_mask, v_mask = mask_uv[0], mask_uv[1]
            if (u_mask == 0) and (v_mask == 0):
                continue
            draw.ellipse((u-r, v-r, u+r, v+r), fill=(255), outline=(0))
    
        pos_image = totensor(pos_image)
        if b == 0:
            pos_image_batch = torch.unsqueeze(pos_image, 0)
        else:
            pos_image_batch = torch.cat((pos_image_batch, torch.unsqueeze(pos_image, 0)), 0)
    
    return pos_image_batch

def make_differentiable_pos_image(size, uv_data, uv_mask):
    B, C, _ = uv_data.shape
    if C != 1:
        raise ValueError("TODO")

    W, H = size[0], size[1]
    device = uv_data.device

    probmap_list = []
    for b in range(B):
        m = torch.distributions.multivariate_normal.MultivariateNormal(uv_data[b, 0], torch.eye(2) * 2)
        base_prob = torch.exp(m.log_prob(uv_data[b, 0]))
        
        xx_ones = torch.ones([1, 1, W], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, H], dtype=torch.int32)

        xx_range = torch.arange(W, dtype=torch.int32)
        yy_range = torch.arange(H, dtype=torch.int32)
        xx_range = xx_range[None, :, None]
        yy_range = yy_range[None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        xx_channel = xx_channel.permute(0, 2, 1)

        xx_channel = xx_channel.repeat(1, 1, 1)
        yy_channel = yy_channel.repeat(1, 1, 1)

        xx_channel = xx_channel.to(device)
        yy_channel = yy_channel.to(device)
        coords_map = torch.cat([xx_channel, yy_channel], dim=0)
        
        flat_coords = rearrange(coords_map, "C H W -> (H W) C")
        prob_map = torch.exp(m.log_prob(flat_coords)) / base_prob
        prob_map = rearrange(prob_map, "(H W) -> H W", H=W, W=W)
        prob_map = torch.unsqueeze(prob_map, 0)
        probmap_list.append(prob_map)

    prob_map = torch.stack(probmap_list, 0)
    return prob_map

def save_diff_heatmap_overlay(pred, gt):
    totensor = torchvision.transforms.ToTensor()
    pred = pred.cpu()
    heatmap = torch.abs(pred - gt).squeeze(1)

    for i in range(heatmap.shape[0]):
        image_ins = pred[i].numpy().transpose((1, 2, 0))*255
        image_ins = image_ins.clip(0, 255).astype(np.uint8)
        
        heatmap_ins = torch.mean(heatmap[i],dim=0).cpu().numpy()*255
        heatmap_ins = heatmap_ins.clip(0, 255)
        heatmap_ins  = 255 - heatmap_ins.astype(np.uint8)

        heatmap_ins = cv2.applyColorMap(heatmap_ins, cv2.COLORMAP_JET)
        overlayed_image = cv2.addWeighted(heatmap_ins, 0.6, image_ins, 0.4, 0)
        overlayed_image = totensor(overlayed_image.transpose((0,1,2)))

        if i == 0:
            overlayed_image_batch = torch.unsqueeze(overlayed_image,0)
        else:
            overlayed_image_batch = torch.cat((overlayed_image_batch, torch.unsqueeze(overlayed_image,0)), 0)
    return overlayed_image_batch

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs

class Time_dict(object):
    def __init__(self):
        self.forward = 0
        self.load_data = 0
        self.backward = 0
        self.loss = 0

    def reset(self):
        self.forward = 0
        self.load_data = 0
        self.backward = 0
        self.loss = 0

### For evaluation

def calculate_dtw_pos(pred_action, gt_action):
    pred_xyz = np.array(pred_action)[:,:3]
    gt_xyz = np.array(gt_action)[:,:3]

    print("calculate dtw pose")
    dtw_error_xyz, path_xyz = fastdtw(pred_xyz, gt_xyz, dist=euclidean)
    print(dtw_error_xyz)
    print(path_xyz)
    error_xyz_list = error_divide_time(pred_xyz, gt_xyz, euclidean, path_xyz)
    mean_dtw_xyz = dtw_error_xyz / len(path_xyz)

    dtw_error_x, path_x = fastdtw(pred_xyz[:,0], gt_xyz[:,0], dist=euclidean)
    error_x_list = error_divide_time(pred_xyz[:,0], gt_xyz[:,0], euclidean, path_x)
    mean_dtw_x = dtw_error_x / len(path_x)

    dtw_error_y, path_y = fastdtw(pred_xyz[:,1], gt_xyz[:,1], dist=euclidean)
    error_y_list = error_divide_time(pred_xyz[:,1], gt_xyz[:,1], euclidean, path_y)
    mean_dtw_y = dtw_error_y / len(path_y)

    dtw_error_z, path_z = fastdtw(pred_xyz[:,2], gt_xyz[:,2], dist=euclidean)
    error_z_list = error_divide_time(pred_xyz[:,2], gt_xyz[:,2], euclidean, path_z)
    mean_dtw_z = dtw_error_z / len(path_z)

    return mean_dtw_xyz, mean_dtw_x, mean_dtw_y, mean_dtw_z, error_xyz_list, error_x_list, error_y_list, error_z_list

def calculate_dtw_angle(pred_action, gt_action):
    pred_quat = np.array(pred_action)[:,3:7]
    gt_quat = np.array(gt_action)[:,3:7]

    r = R.from_quat(pred_quat)
    pred_eular = r.as_euler('xyz')

    r = R.from_quat(gt_quat)
    gt_eular = r.as_euler('xyz')

    def angle_euclidean(angle1, angle2):
        diff_eular = angle1 - angle2
        diff_eular = np.where(abs(diff_eular) > np.pi, (2*np.pi) - abs(diff_eular), abs(diff_eular))
        return np.linalg.norm(diff_eular)

    print("calculate dtw angle")
    dtw_error_xyz, path_xyz = fastdtw(pred_eular, gt_eular, dist=angle_euclidean)
    error_xyz_list = error_divide_time(pred_eular, gt_eular, angle_euclidean, path_xyz)
    mean_dtw_xyz = dtw_error_xyz / len(path_xyz)

    dtw_error_x, path_x = fastdtw(pred_eular[:,0], gt_eular[:,0], dist=angle_euclidean)
    error_x_list = error_divide_time(pred_eular[:,0], gt_eular[:,0], angle_euclidean, path_x)
    mean_dtw_x = dtw_error_x / len(path_x)

    dtw_error_y, path_y = fastdtw(pred_eular[:,1], gt_eular[:,1], dist=angle_euclidean)
    error_y_list = error_divide_time(pred_eular[:,1], gt_eular[:,1], angle_euclidean, path_y)
    mean_dtw_y = dtw_error_y / len(path_y)

    dtw_error_z, path_z = fastdtw(pred_eular[:,2], gt_eular[:,2], dist=angle_euclidean)
    error_z_list = error_divide_time(pred_eular[:,2], gt_eular[:,2], angle_euclidean, path_z)
    mean_dtw_z = dtw_error_z / len(path_z)

    return mean_dtw_xyz, mean_dtw_x, mean_dtw_y, mean_dtw_z, error_xyz_list, error_x_list, error_y_list, error_z_list

def error_divide_time(pred, gt, dist, path):
    # error_list = []
    # for i,j in path:
    #     error = dist(pred[i], gt[j])
    #     error_list.append(error)
    
    error_list = [0] * len(gt)
    for i,j in path:
        error = dist(pred[i],gt[j])
        error_list[j] = error
    return error_list