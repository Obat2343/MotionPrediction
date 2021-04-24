import sys
import json
import os
import cv2
import numpy as np
import torch
import torchvision
import torch_optimizer as new_optim
from collections import OrderedDict
from torchvision import datasets, models, transforms
from PIL import Image, ImageDraw, ImageOps
from .dataset import Softargmax_dataset, Softargmax_dataset_VP, Softargmax_dataset_test, RLBench_dataset, RLBench_dataset_VP, RLBench_dataset_test, RLBench_dataset2, RLBench_dataset2_VP
from .model.Hourglass import stacked_hourglass_model, sequence_hourglass

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
    elif cfg.DATASET.NAME == 'RLBench2':
        if mode == 'train':
            dataset = RLBench_dataset2(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = RLBench_dataset2(cfg, save_dataset=save_dataset, mode=mode)

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
    elif cfg.DATASET.NAME == 'RLBench2':
        if mode == 'train':
            dataset = RLBench_dataset2_VP(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'val':
            dataset = RLBench_dataset2_VP(cfg, save_dataset=save_dataset, mode=mode)
        elif mode == 'test':
            dataset = RLBench_dataset2_VP(cfg, save_dataset=save_dataset, mode='val', random_len=1)
    
    return dataset

def build_model_MP(cfg):
    if cfg.DATASET.NAME == 'HMD':
        output_dim = 21
    elif (cfg.DATASET.NAME == 'RLBench') or (cfg.DATASET.NAME == 'RLBench2'):
        output_dim = 1

    if cfg.MP_MODEL_NAME == 'hourglass':
        print('use hourglass')
        model = stacked_hourglass_model(cfg, output_dim=output_dim)
    elif cfg.MP_MODEL_NAME == 'sequence_hourglass':
        print('use sequence hourglass')
        model = sequence_hourglass(cfg, output_dim=output_dim)
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
            model.parameters(),
            lr = lr,
            momentum = cfg.OPTIM.SGD.MOMENTUM,
            dampening = cfg.OPTIM.SGD.DAMPENING,
            weight_decay = cfg.OPTIM.SGD.WEIGHT_DECAY,
            nesterov = cfg.OPTIM.SGD.NESTEROV,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr= lr,
            betas=(cfg.OPTIM.RADAM.BETA1, cfg.OPTIM.RADAM.BETA2),
            eps=cfg.OPTIM.RADAM.EPS,
            weight_decay=cfg.OPTIM.RADAM.WEIGHT_DECAY,
        )
    elif name == "radam":
        optimizer = new_optim.RAdam(
            model.parameters(),
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
    for sequence_id in range(len(uv_list)):
        heatmap = convert_heatmap(heatmap_list[sequence_id][-1])
        B, C, H, W = heatmap.shape

        #gt_current_pose_image = torch.unsqueeze(torch.unsqueeze(torch.sum(inputs['pose'],dim=1),1),1)
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
        pos_image = make_pos_image((W,H),uv_list[sequence_id][-1].to('cpu'),inputs['uv_mask'][:,sequence_id+2])
        overlay_image = make_overlay_image(inputs['rgb'][:,sequence_id+1], pos_image)
        pos_image_gt = make_pos_image((W,H),inputs['uv'][:,2+sequence_id],inputs['uv_mask'][:,2+sequence_id])
        overlay_image_gt = make_overlay_image(inputs['rgb'][:,sequence_id+1], pos_image)
        uv_images = torch.cat((overlay_image, overlay_image_gt), 0)
        save_uv_path = os.path.join(path,'{}_image_uv_{}_{}.jpg'.format(mode, index, sequence_id))
        torchvision.utils.save_image(uv_images, save_uv_path, nrow=cfg.BASIC.BATCH_SIZE)
        
        # save output image
        if cfg.HOURGLASS.PRED_RGB:
            sequence_image = torch.cat((outputs['rgb'][sequence_id][-1].to('cpu'),inputs['rgb'][:,sequence_id+2]),0)
            save_image_path = os.path.join(model_path,'checkpoint_epoch{}_iter{}'.format(epoch,iteration),'{}_image_output_{}_{}.jpg'.format(mode, index, sequence_id))
            torchvision.utils.save_image(sequence_image, save_image_path,nrow=cfg.BASIC.BATCH_SIZE)

        # save rotation image
        if cfg.HOURGLASS.PRED_ROTATION:
            rotation_image = make_rotation_image(inputs['rgb'][:,sequence_id+1:sequence_id+2], torch.unsqueeze(outputs['rotation'][sequence_id][-1],1), outputs['pose'][sequence_id][-1], inputs['mtx'])
            rotation_gt_image = make_rotation_image(inputs['rgb'][:,sequence_id+1:sequence_id+2], inputs['rotation_matrix'][:,sequence_id+1:sequence_id+2], inputs['pose_xyz'][:,sequence_id+1:sequence_id+2], inputs['mtx'])
            rotation_images = torch.cat((rotation_image, rotation_gt_image), 0)
            save_rotation_path = os.path.join(path,'{}_rotation_{}_{}.jpg'.format(mode, index, sequence_id))
            torchvision.utils.save_image(rotation_images, save_rotation_path, nrow=cfg.BASIC.BATCH_SIZE)

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
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    for b in range(B):
        for s in range(S):
            # make pil image with rotation 
            image_pil = topil(rgb[b,s])
            rotation_np = rotation_matrix[b,s].cpu().numpy()

            pos_vec_np = pose_vec[b,s].cpu().numpy()
            pos_vec_np = np.expand_dims(pos_vec_np, 0)

            intrinsic_matrix_np = intrinsic_matrix[b].cpu().numpy()
            image_pil = draw_matrix(image_pil, rotation_np, pos_vec_np, intrinsic_matrix_np)
            # to tensor batch
            image_tensor = totensor(image_pil)
            image_tensor = torch.unsqueeze(image_tensor, 0)

            if (s == 0) and (b == 0):
                image_sequence = image_tensor
            else:
                image_sequence = torch.cat((image_sequence, image_tensor),0)
    
    return image_sequence

def make_overlay_image(rgb, pose_image):
    B, C, H, W = pose_image.shape
    red_image = torch.zeros(B, 3, H, W)
    red_image[:,0:1] = torch.clamp(pose_image, 0, 1)

    pose_image_rgb = torch.cat((pose_image, pose_image, pose_image), 1)
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

def make_pos_image(size,uv_data,uv_mask,r=3):
    B, C, _ = uv_data.shape
    totensor = transforms.ToTensor()
    uv_data = uv_data.numpy()
    for b in range(B):
        pos_image = Image.new('L',size)
        draw = ImageDraw.Draw(pos_image)
        for uv,mask_uv in zip(uv_data[b],uv_mask[b]):
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