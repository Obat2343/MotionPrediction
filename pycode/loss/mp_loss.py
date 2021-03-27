import sys
import torch
import kornia
import torch.nn as nn
from torchvision.utils import make_grid

class Train_Loss_sequence_hourglass(nn.Module):
    def __init__(self,cfg,device):
        super(Train_Loss_sequence_hourglass, self).__init__()
        self.loss_list = []
        self.loss_dict = {}
        self.count = 0
        self.device = device
        # self.use_future_past = cfg.USE_FUTURE_PAST_FRAME

        self.loss_list.append(argmax_sequence_loss(cfg,device))
        self.loss_list.append(pose_sequence_loss(cfg,device))
        if cfg.HOURGLASS.PRED_ROTATION:
            self.loss_list.append(rotation_sequence_loss(cfg,device))
            self.loss_list.append(grasp_sequence_loss(cfg,device))
        if cfg.LOSS.RGB:
            self.loss_list.append(Mse_sequence_loss(device))
        
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

class argmax_sequence_loss(nn.Module):
    def __init__(self,cfg,device):
        super(argmax_sequence_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.L1Loss(reduction='none')
        self.pred_len = cfg.PRED_LEN
        self.weight = cfg.LOSS.ARGMAX.WEIGHT

    def forward(self,inputs,outputs,mode='train'):
        pred_uv_sequence_list = outputs['uv'] #list-list-tensor　Num_sequence Num_intermidiate Pose_tensor 
        loss_list = []
        loss_dict = {}

        for sequence_id, pred_uv_list in enumerate(pred_uv_sequence_list):
            gt_uv = inputs['uv'][:,2+sequence_id].float()
            gt_uv_mask = inputs['uv_mask'][:,2+sequence_id].float()
            B,P,_ = gt_uv.shape # Batch, Num pose, uv_dim(=2)
            for intermidiate_id, pred_uv in enumerate(pred_uv_list):
                inverse_intermidiate_id = len(pred_uv_list) - intermidiate_id - 1
                loss = self.loss(pred_uv, gt_uv.to(self.device)) * gt_uv_mask.to(self.device)
                loss = torch.mean(loss)
                loss_list.append(loss)
                # if (mode == 'train') or (mode == 'val'):
                loss_dict['additional_info_{}/l1_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_uv_sequence_list) - 1):
                    loss_dict['test_info/l1_uv_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 
        loss_dict['{}/l1_t2'.format(mode)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_l1_t2'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

class pose_sequence_loss(nn.Module):
    def __init__(self,cfg,device):
        super(pose_sequence_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.L1Loss(reduction='none')
        self.pred_len = cfg.PRED_LEN
        self.weight = cfg.LOSS.POSE.WEIGHT

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        pred_xyz_sequence_list = outputs['pose']
        B, _, _ = pred_xyz_sequence_list[0][0].shape

        loss_list = []
        for sequence_id, pred_xyz_list in enumerate(pred_xyz_sequence_list):
            gt_xyz = inputs['pose_xyz'][:,2+sequence_id].float()
            gt_xyz = gt_xyz.view(B, -1, 3)
            gt_xyz_mask = inputs['pose_xyz_mask'][:,2+sequence_id]
            gt_xyz_mask = gt_xyz_mask.view(B, -1, 3)
            for intermidiate_id, pred_xyz in enumerate(pred_xyz_list):
                inverse_intermidiate_id = len(pred_xyz_list) - intermidiate_id - 1
                pred_xyz = pred_xyz.view(B, -1, 3)
                loss = self.loss(pred_xyz, gt_xyz.to(self.device)) * gt_xyz_mask.to(self.device)
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/xyz_l1_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_xyz_sequence_list) - 1):
                    loss_dict['test_info/l1_xyz_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/xyz_l1_t2'.format(mode)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_xyz_l1_t2'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

class rotation_sequence_loss(nn.Module):
    def __init__(self,cfg,device):
        super(rotation_sequence_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.L1Loss(reduction='none')
        self.pred_len = cfg.PRED_LEN
        self.weight = cfg.LOSS.ROTATION.WEIGHT

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        pred_rotation_sequence_list = outputs['rotation']
        B, _, _ = pred_rotation_sequence_list[0][0].shape

        loss_list = []
        for sequence_id, pred_rotation_list in enumerate(pred_rotation_sequence_list):
            gt_rotation = inputs['rotation_matrix'][:,2+sequence_id].float()
            gt_rotation = gt_rotation.view(B, -1, 3)
            for intermidiate_id, pred_rotation in enumerate(pred_rotation_list):
                inverse_intermidiate_id = len(pred_rotation_list) - intermidiate_id - 1
                pred_rotation = pred_rotation.view(B, -1, 3)
                loss = self.loss(pred_rotation, gt_rotation.to(self.device))
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/rotation_l1_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_rotation_sequence_list) - 1):
                    loss_dict['test_info/l1_rotation_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/rotation_l1_t2'.format(mode)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_rotation_l1_t2'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

class grasp_sequence_loss(nn.Module):
    def __init__(self,cfg,device):
        super(grasp_sequence_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='none')
        self.pred_len = cfg.PRED_LEN
        self.weight = cfg.LOSS.GRASP.WEIGHT

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        pred_grasp_sequence_list = outputs['grasp']
        B, _ = pred_grasp_sequence_list[0][0].shape

        loss_list = []
        for sequence_id, pred_grasp_list in enumerate(pred_grasp_sequence_list):
            gt_grasp = inputs['grasp'][:,2+sequence_id].float()
            gt_grasp = gt_grasp.view(B,1)
            for intermidiate_id, pred_grasp in enumerate(pred_grasp_list):
                inverse_intermidiate_id = len(pred_grasp_list) - intermidiate_id - 1
                loss = self.loss(pred_grasp, gt_grasp.to(self.device))
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/grasp_bce_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_grasp_sequence_list) - 1):
                    loss_dict['test_info/grasp_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/grasp_bce'.format(mode)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        loss *= self.weight
        loss_dict['{}/wegiht_grasp_bce'.format(mode)] = loss.item()
        loss_dict['{}/weight_loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

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

class Test_Loss_sequence_hourglass(nn.Module):
    def __init__(self,cfg,device):
        super(Test_Loss_sequence_hourglass, self).__init__()
        self.loss_list = []
        self.loss_dict = {}
        self.count = 0
        self.device = device
        # self.use_future_past = cfg.USE_FUTURE_PAST_FRAME

        # self.loss_list.append(argmax_loss(device))
        self.loss_list.append(argmax_sequence_test_loss(cfg,device))
        # self.loss_list.append(pose_loss(device))
        self.loss_list.append(pose_sequence_test_loss(cfg,device))
        if cfg.HOURGLASS.PRED_ROTATION:
            self.loss_list.append(rotation_sequence_loss(cfg,device))
            self.loss_list.append(grasp_sequence_loss(cfg,device))
        if cfg.LOSS.RGB:
            self.loss_list.append(Mse_sequence_loss(device))
        
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

class argmax_sequence_test_loss(nn.Module):
    def __init__(self,cfg,device):
        super(argmax_sequence_test_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.L1Loss(reduction='none')
        self.pred_len = cfg.PRED_LEN

    def forward(self,inputs,outputs,mode='test'):
        pred_uv_sequence_list = outputs['uv'] #list-list-tensor　Num_sequence Num_intermidiate Pose_tensor 
        loss_list = []
        loss_dict = {}
        action = inputs['action_name'][0]

        for sequence_id, pred_uv_list in enumerate(pred_uv_sequence_list):
            gt_uv = inputs['uv'][:,2+sequence_id].float()
            gt_uv_mask = inputs['uv_mask'][:,2+sequence_id].float()
            B,P,_ = gt_uv.shape # Batch, Num pose, uv_dim(=2)
            for intermidiate_id, pred_uv in enumerate(pred_uv_list):
                inverse_intermidiate_id = len(pred_uv_list) - intermidiate_id - 1
                loss = self.loss(pred_uv, gt_uv.to(self.device)) * gt_uv_mask.to(self.device)
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/l1_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_uv_sequence_list) - 1):
                    loss_dict['test_info/l1_uv_last'] = loss.item()

        if B != 1:
            raise ValueError("Sorry not implemented")

        loss = sum(loss_list) / len(loss_list) 
        loss_dict['{}/l1_mean'.format(mode)] = loss.item()
        loss_dict['{}/l1_{}'.format(mode, action)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

class pose_sequence_test_loss(nn.Module):
    def __init__(self,cfg,device):
        super(pose_sequence_test_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.L1Loss(reduction='none')

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        pred_xyz_sequence_list = outputs['pose']
        B, _, _ = pred_xyz_sequence_list[0][0].shape
        action = inputs['action_name'][0]

        loss_list = []
        for sequence_id, pred_xyz_list in enumerate(pred_xyz_sequence_list):
            gt_xyz = inputs['pose_xyz'][:,2+sequence_id].float()
            gt_xyz = gt_xyz.view(B, -1, 3)
            gt_xyz_mask = inputs['pose_xyz_mask'][:,2+sequence_id]
            gt_xyz_mask = gt_xyz_mask.view(B, -1, 3)
            for intermidiate_id, pred_xyz in enumerate(pred_xyz_list):
                inverse_intermidiate_id = len(pred_xyz_list) - intermidiate_id - 1
                pred_xyz = pred_xyz.view(B, -1, 3)
                loss = self.loss(pred_xyz, gt_xyz.to(self.device)) * gt_xyz_mask.to(self.device)
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/xyz_l1_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_xyz_sequence_list) - 1):
                    loss_dict['test_info/l1_xyz_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/xyz_l1_mean'.format(mode)] = loss.item()
        loss_dict['{}/xyz_l1_{}'.format(mode,action)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

class rotation_sequence_test_loss(nn.Module):
    def __init__(self,cfg,device):
        super(rotation_sequence_test_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.L1Loss(reduction='none')
        self.pred_len = cfg.PRED_LEN
        self.weight = 10

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        pred_rotation_sequence_list = outputs['rotation']
        B, _, _ = pred_rotation_sequence_list[0][0].shape
        action = inputs['action_name'][0]

        loss_list = []
        for sequence_id, pred_rotation_list in enumerate(pred_rotation_sequence_list):
            gt_rotation = inputs['rotation_matrix'][:,2+sequence_id].float()
            gt_rotation = gt_rotation.view(B, -1, 3)
            for intermidiate_id, pred_rotation in enumerate(pred_rotation_list):
                inverse_intermidiate_id = len(pred_rotation_list) - intermidiate_id - 1
                pred_rotation = pred_rotation.view(B, -1, 3)
                loss = self.loss(pred_rotation, gt_rotation.to(self.device))
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/rotation_l1_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_rotation_sequence_list) - 1):
                    loss_dict['test_info/l1_rotation_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/rotation_l1_mean'.format(mode)] = loss.item()
        loss_dict['{}/rotation_l1_{}'.format(mode, action)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict

class grasp_sequence_test_loss(nn.Module):
    def __init__(self,cfg,device):
        super(grasp_sequence_test_loss, self).__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='none')
        self.pred_len = cfg.PRED_LEN
        self.weight = 1

    def forward(self,inputs,outputs,mode='train'):
        loss_dict = {}
        pred_grasp_sequence_list = outputs['grasp']
        B, _ = pred_grasp_sequence_list[0][0].shape
        action = inputs['action_name'][0]

        loss_list = []
        for sequence_id, pred_grasp_list in enumerate(pred_grasp_sequence_list):
            gt_grasp = inputs['grasp'][:,2+sequence_id].float()
            gt_grasp = gt_grasp.view(B,1)
            for intermidiate_id, pred_grasp in enumerate(pred_grasp_list):
                inverse_intermidiate_id = len(pred_grasp_list) - intermidiate_id - 1
                loss = self.loss(pred_grasp, gt_grasp.to(self.device))
                loss = torch.mean(loss)
                loss_list.append(loss)
                if (mode == 'train') or (mode == 'val'):
                    loss_dict['additional_info_{}/grasp_bce_t{}_moduel_index_{}'.format(mode, 2 + sequence_id, inverse_intermidiate_id)] = loss.item()
                if (mode == 'test') and (inverse_intermidiate_id == 0) and (sequence_id == len(pred_grasp_sequence_list) - 1):
                    loss_dict['test_info/grasp_last'] = loss.item()

        loss = sum(loss_list) / len(loss_list) 

        loss_dict['{}/grasp_bce_mean'.format(mode)] = loss.item()
        loss_dict['{}/grasp_bce_{}'.format(mode,action)] = loss.item()
        loss_dict['{}/loss'.format(mode)] = loss.item()

        # loss_dict['weight/l1-norm-reg'] = self.weight
        return loss, loss_dict
