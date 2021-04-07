import sys
import torch
import kornia
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from .discriminator_loss import Gradient_Penalty, W_GAN_Loss

class Train_Loss_Video(nn.Module):
    def __init__(self,cfg,device):
        super(Train_Loss_Video, self).__init__()
        self.loss_list = []
        self.loss_dict = {}
        self.count = 0
        self.device = device

        self.loss_list.append(Mse_sequence_loss(device))
        self.loss_list.append(Mse_sequence_loss_gt_diff(device))

    def forward(self,inputs,outputs,mode='train'):
        total_loss = torch.tensor(0.,).to(self.device)

        for loss_func in self.loss_list:
            loss, loss_dict = loss_func(inputs,outputs,mode)
            for key in loss_dict.keys():
                if key not in self.loss_dict.keys():
                    self.loss_dict[key] = loss_dict[key]
                else:
                    self.loss_dict[key] += loss_dict[key]
            
            total_loss += loss

        self.count += 1
        return total_loss
    
    def get_log(self):
        for key in self.loss_dict.keys():
            self.loss_dict[key] /= self.count
        return self.loss_dict
    
    def reset_log(self):
        self.count = 0
        for key in self.loss_dict.keys():
            self.loss_dict[key] = 0

class Mse_sequence_loss(nn.Module):
    def __init__(self,device):
        super(Mse_sequence_loss, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.weight = 1.0 # TODO

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        output_image_sequence_list = outputs['rgb']
        if type(output_image_sequence_list) != list:
            output_image_sequence_list = [[output_image_sequence_list]]

        loss_list = []
        for sequence_id, pred_rgb_list in enumerate(output_image_sequence_list):
            gt_image = inputs['rgb'][:,2+sequence_id].to(self.device)
            
            for intermidiate_id, pred_rgb in enumerate(pred_rgb_list):
                inverse_intermidiate_id = len(pred_rgb_list) - intermidiate_id - 1
                loss = self.loss(pred_rgb, gt_image)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/mse_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/mse'.format(mode)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_mse'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss'.format(mode)] = loss.item()

        return loss, loss_dict
    
class Mse_sequence_loss_gt_diff(nn.Module):
    def __init__(self,device):
        super(Mse_sequence_loss_gt_diff, self).__init__()
        self.loss = torch.nn.MSELoss()
        self.device = device

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        output_image_sequence_list = outputs['rgb']
        if type(output_image_sequence_list) != list:
            output_image_sequence_list = [[output_image_sequence_list]]

        loss_list = []
        for sequence_id, pred_rgb_list in enumerate(output_image_sequence_list):
            gt_image = inputs['rgb'][:,2+sequence_id].to(self.device)
            ref_image = inputs['rgb'][:,1+sequence_id].to(self.device)

            for intermidiate_id, pred_rgb in enumerate(pred_rgb_list):
                inverse_intermidiate_id = len(pred_rgb_list) - intermidiate_id - 1
                loss = self.loss(pred_rgb, gt_image) - self.loss(ref_image,gt_image)
                loss_list.append(loss)
                loss_dict['additional_info_{}/mse_t{}_moduel_index_{}_gtdiff'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/mse_gt_diff'.format(mode)] = loss.item()

        return torch.tensor(0).to(self.device), loss_dict

class D_Loss(nn.Module):
    def __init__(self,cfg,device):
        super(D_Loss, self).__init__()
        self.loss_list = []
        self.loss_dict = {}
        self.count = 0
        self.device = device

        self.loss_list.append(W_GAN_Loss(cfg,device))
        self.gp = Gradient_Penalty(cfg, device)

        if len(self.loss_list) == 0:
            raise ValueError('d_loss is empty')
    
    def forward(self, inputs, generated_image, discriminator, mode='train', train_d=True):
        
        fake_input = generated_image
        real_input = inputs['rgb'][:,2].to(self.device)

        fake_prediction = discriminator(fake_input)
        real_prediction = discriminator(real_input)

        total_loss = torch.tensor(0.,).to(self.device)

        for loss_func in self.loss_list:
            if train_d:
                loss, loss_dict = loss_func(fake_prediction,real_prediction,mode)
                for key in loss_dict.keys():
                    if key not in self.loss_dict.keys():
                        self.loss_dict[key] = loss_dict[key]
                    else:
                        self.loss_dict[key] += loss_dict[key]
            else:
                loss, loss_dict = loss_func(real_prediction,fake_prediction,mode)

            total_loss += loss
        
        if train_d:
            loss, loss_dict = self.gp(fake_input, real_input, discriminator, mode)
            for key in loss_dict.keys():
                if key not in self.loss_dict.keys():
                    self.loss_dict[key] = loss_dict[key]
                else:
                    self.loss_dict[key] += loss_dict[key]

            self.count += 1
        
        return total_loss
    
    def get_log(self):
        for key in self.loss_dict.keys():
            self.loss_dict[key] /= self.count
        return self.loss_dict
    
    def reset_log(self):
        self.count = 0
        for key in self.loss_dict.keys():
            self.loss_dict[key] = 0

class Test_Loss_Video(nn.Module):
    def __init__(self,cfg,device):
        super(Test_Loss_Video, self).__init__()
        self.loss_list = []
        self.loss_dict = {}
        self.count = 0
        self.device = device

        self.loss_list.append(PSNR(device))
        self.loss_list.append(L1(device))

    def forward(self,inputs,outputs,mode='test'):
        total_loss = torch.tensor(0.,).to(self.device)

        for loss_func in self.loss_list:
            _, loss_dict = loss_func(inputs,outputs,mode)
            for key in loss_dict.keys():
                if key not in self.loss_dict.keys():
                    self.loss_dict[key] = [loss_dict[key]]
                else:
                    self.loss_dict[key].append(loss_dict[key])
    
    def get_log(self):
        return_dict = {}
        for key in self.loss_dict.keys():
            return_dict[key] = sum(self.loss_dict[key]) / len(self.loss_dict[key])
        return return_dict
    
    def reset_log(self):
        self.count = 0
        for key in self.loss_dict.keys():
            self.loss_dict[key] = []

class PSNR(nn.Module):
    def __init__(self,device):
        super(PSNR, self).__init__()
        self.loss = kornia.losses.psnr_loss
        self.device = device

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}

        output_image = outputs['rgb']
        gt_image = inputs['rgb'][:,2].to(self.device)
        B, _, _, _ = output_image.shape
        if B != 1:
            raise ValueError("not implemented")

        action = inputs['action_name'][0]

        PSNR = self.loss(torch.clamp(output_image*255,0,255),gt_image*255, 255)
        loss_dict['{}/psnr_{}'.format(mode, action)] = PSNR
        loss_dict['{}/psnr_mean'.format(mode)] = PSNR
        # if C == 4:
        #     depth_loss = self.loss(output_image[:,3],DEPTH[:,2])
        # else:
        #     depth_loss = torch.tensor(0).to(self.device)
        # loss_dict['{}/depth_mse'.format(mode)] = depth_loss.item()

        return torch.tensor(0).to(self.device), loss_dict

class L1(nn.Module):
    def __init__(self,device):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss()
        self.device = device

    def forward(self,inputs,outputs,mode='train',frame=1):
        loss_dict = {}

        output_image = torch.clamp(outputs['rgb']*255, 0, 255)
        gt_image = inputs['rgb'][:,2].to(self.device) * 255
        B, _, _, _ = output_image.shape
        if B != 1:
            raise ValueError("not implemented")

        action = inputs['action_name'][0]

        loss = self.loss(output_image, gt_image)
        loss_dict['{}/l1_{}'.format(mode,action)] = loss
        loss_dict['{}/l1_mean'.format(mode)] = loss
        # if C == 4:
        #     depth_loss = self.loss(output_image[:,3],DEPTH[:,2])
        # else:
        #     depth_loss = torch.tensor(0).to(self.device)
        # loss_dict['{}/depth_mse'.format(mode)] = depth_loss.item()

        return torch.tensor(0).to(self.device), loss_dict
