import torch
import torch.nn as nn
import numpy as np
from .base_networks import ConvBlock, SoftArgmax2D, SigmoidArgmax2D, hourglass_module, ResidualBlock
from .tools import compute_rotation_matrix_from_ortho6d as compute_rm
from .VideoPrediction import VIDEO_HOURGLASS

class stacked_hourglass_model(torch.nn.Module):
    def __init__(self, cfg, input_dim=3, output_dim=1, pred_len='none', device='cuda'):
        super(stacked_hourglass_model, self).__init__()

        # option
        self.device = device
        self.num_hourglass = cfg.HOURGLASS.NUM_BLOCK
        self.num_downscale = cfg.HOURGLASS.NUM_DOWNSCALE
        activation = cfg.HOURGLASS.ACTIVATION
        norm = cfg.HOURGLASS.NORM
        base_filter = cfg.HOURGLASS.BASE_FILTER

        # input_option
        self.input_pose = cfg.HOURGLASS.INPUT_POSE
        self.input_z = cfg.HOURGLASS.INPUT_Z
        self.input_rotation = cfg.HOURGLASS.INPUT_ROTATION
        self.input_grasp = cfg.HOURGLASS.INPUT_GRASP

        self.input_depth = cfg.HOURGLASS.INPUT_DEPTH # TODO
        self.use_past = cfg.HOURGLASS.INPUT_PAST
        if self.use_past:
            input_dim *= 2
        
        self.use_past_pose = cfg.HOURGLASS.INPUT_PAST
        if self.use_past_pose:
            input_pose_dim = (output_dim + (6 * self.input_rotation) + self.input_grasp) * 2
        else:
            input_pose_dim = (output_dim + (6 * self.input_rotation) + self.input_grasp)

        # output_option
        if pred_len == 'none':
            self.pred_len = cfg.PRED_LEN
        else:
            self.pred_len = pred_len
        output_dim = output_dim * self.pred_len

        if cfg.HOURGLASS.SINGLE_DEPTH:
            self.depth_channel = 1
        else:
            self.depth_channel = output_dim

        rotation_channel = 6
        grasp_channel = 1
        self.rgb_pred = cfg.HOURGLASS.PRED_RGB
        self.rotation_pred = cfg.HOURGLASS.PRED_ROTATION
        self.grasp_pred = cfg.HOURGLASS.PRED_GRASP

        # loss_option
        self.intermediate_loss = cfg.HOURGLASS.INTERMEDIATE_LOSS 

        self.initial_conv = torch.nn.ModuleList([
            ConvBlock(input_dim, int(base_filter / 2), 3, 1, 1, activation=activation, norm=norm),
            ResidualBlock(int(base_filter / 2), base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm)
        ])

        self.hourglass_list = torch.nn.ModuleList([hourglass_module(base_filter, self.num_downscale, activation=activation, norm=norm) for _ in range(self.num_hourglass)])
        
        self.heatmap_layer_list = torch.nn.ModuleList([ConvBlock(base_filter, output_dim, activation='none', norm='none') for _ in range(self.num_hourglass)]) # use same parameter?
        self.depth_layer_list = torch.nn.ModuleList([ConvBlock(base_filter, self.depth_channel, activation='none', norm='none') for _ in range(self.num_hourglass)])

        self.merge_heatmap_list = torch.nn.ModuleList([ConvBlock(output_dim, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])
        self.merge_depth_list = torch.nn.ModuleList([ConvBlock(self.depth_channel, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])
        self.merge_feature_list = torch.nn.ModuleList([ConvBlock(base_filter, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])

        if self.rotation_pred:
            self.rotation_layer_list = torch.nn.ModuleList([ConvBlock(base_filter, rotation_channel, activation='none', norm='none') for _ in range(self.num_hourglass)])
            self.merge_rotation_list = torch.nn.ModuleList([ConvBlock(rotation_channel, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])

        if self.grasp_pred:
            self.grasp_layer_list = torch.nn.ModuleList([ConvBlock(base_filter, grasp_channel, activation='none', norm='none') for _ in range(self.num_hourglass)])
            self.merge_grasp_list = torch.nn.ModuleList([ConvBlock(grasp_channel, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])

        if self.rgb_pred:
            self.rgb_layer_list = torch.nn.ModuleList([ConvBlock(base_filter, 3, activation='none', norm='none') for _ in range(self.num_hourglass)])
            self.merge_rgb_list = torch.nn.ModuleList([ConvBlock(3, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])

        if self.input_pose:
            self.initial_pose_conv = torch.nn.ModuleList([
            ConvBlock(input_pose_dim, int(base_filter / 2), 3, 1, 1, activation=activation, norm=norm),
            ResidualBlock(int(base_filter / 2), base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm)
        ])

        # self.vector_u_map = ConvBlock(base_filter, 21, 3, 1, 1, activation=None, norm=None)
        # self.vector_v_map = ConvBlock(base_filter, 21, 3, 1, 1, activation=None, norm=None)

        if cfg.HOURGLASS.ARGMAX == 'softargmax':
            self.softargmax = SoftArgmax2D(device=self.device)
        elif cfg.HOURGLASS.ARGMAX == 'sigmoidargmax':
            self.softargmax = SigmoidArgmax2D(device=self.device)
        else:
            raise ValueError("undefined argmax")

    def forward(self, inputs):
        '''
        inputs -> dict

        outputs -> dict
            dict['pose'] = list(sequence)-list(intermidiate output)-tensor(output)
        '''
        # get rgb image
        if self.use_past:
            B, _, C, H, W = inputs['rgb'].shape
            x = inputs['rgb'][:,:2].to(self.device)
            x = x.view(B, -1, H, W)
        else:
            x = inputs['rgb'][:,1].to(self.device)
        
        # get pose image (or pose_depth image)
        if self.input_pose:

            if self.use_past_pose:
                pose_heatmap = inputs['pose'][:,:2].to(self.device) #B,2,C,H,W
                B,S,C,H,W = pose_heatmap.shape
                pose_heatmap = pose_heatmap.view(B,-1,H,W)
                C = S * C
                start_index = 0
            else:
                pose_heatmap = inputs['pose'][:,1].to(self.device) #B,C,H,W
                B,C,H,W = pose_heatmap.shape
                S = 1
                start_index = 1

            pose_map = []
            if self.input_z:
                input_z = inputs['pose_xyz'][:,start_index:2,2::3]# B,S,C
                input_z = input_z.view(B,-1)
                input_z = torch.unsqueeze(input_z,2)
                input_z = torch.unsqueeze(input_z,3)
                pose_map.append(pose_heatmap * input_z.expand(B,C,H,W))
            if self.input_rotation:
                input_rotation = inputs['rotation_matrix'][:,start_index:2,:2]
                input_rotation = input_rotation.view(B,-1,6)
                input_rotation = input_rotation.contiguous().view(B,-1)
                input_rotation = torch.unsqueeze(input_rotation, 2)
                input_rotation = torch.unsqueeze(input_rotation, 3)
                input_rotation = input_rotation.expand(B,6*(2-start_index),H,W)

                heatmap_for_rotation = torch.cat([pose_heatmap[:,s:s+1].expand(B,6,H,W) for s in range(S)], 1)
                pose_map.append(heatmap_for_rotation * input_rotation)
            if self.input_grasp:
                input_grasp = inputs['grasp'][:,start_index:2]
                input_grasp = input_grasp.view(B,-1)
                input_grasp = torch.unsqueeze(input_grasp, 2)
                input_grasp = torch.unsqueeze(input_grasp, 3)
                pose_map.append(pose_heatmap * input_grasp.expand(B,2-start_index,H,W))
            
            if len(pose_map) > 0:
                pose_heatmap = torch.cat(pose_map, 1)

        # get inverse intrinsic camera parameter matrix
        mtx = inputs['inv_mtx'].float().to(self.device)

        # list for output
        uv_list = []
        pose_list = []
        heatmap_list = []
        rotation_list = []
        rgb_list = []
        grasp_list = []
        
        # initial conv
        for i in range(len(self.initial_conv)):
            x = self.initial_conv[i](x)

        if self.input_pose:
            for i in range(len(self.initial_pose_conv)):
                pose_heatmap = self.initial_pose_conv[i](pose_heatmap)

            x = x + pose_heatmap

        # hourglass
        for i in range(self.num_hourglass):
            x = self.hourglass_list[i](x)

            # pred heatmap (uv)
            heatmap = self.heatmap_layer_list[i](x)
            uv, argmax, heatmap_softmax = self.softargmax(heatmap)

            # pred depth
            depth = self.depth_layer_list[i](x)

            # convert to xyz
            z = heatmap_softmax * depth
            B,C,H,W = z.shape
            z = torch.unsqueeze(torch.sum(z.view(B,C,-1),2),2)
            # z = torch.unsqueeze(inputs['pose_xyz'][:,1,[i + 2 for i in range(2, 65, 3)]],2).to(self.device)
            XYZ = self.uv2xyz(uv, mtx, z)

            # pred rotation
            if self.rotation_pred:
                rotation_map = self.rotation_layer_list[i](x)
                rotation_map = heatmap_softmax * rotation_map # TODO change here
                B,C,H,W = rotation_map.shape
                rotation = torch.sum(rotation_map.view(B,C,-1),2)
                rotation = compute_rm(rotation)

            # pred grasp
            if self.grasp_pred:
                grasp_map = self.grasp_layer_list[i](x)
                grasp_map = heatmap_softmax * grasp_map
                B,C,H,W = grasp_map.shape
                grasp = torch.sum(grasp_map.view(B,C,-1),2)
                grasp = torch.sigmoid(grasp)

            # pred rgb
            if self.rgb_pred:
                rgb = self.rgb_layer_list[i](x)

            # feedback output into feature
            if i < self.num_hourglass - 1:                
                heatmap_feature = self.merge_heatmap_list[i](heatmap)
                depth_feature = self.merge_depth_list[i](depth)
                merge_feature = self.merge_feature_list[i](x)

                if self.rgb_pred:
                    rgb_feature = self.merge_rgb_list[i](rgb)
                    x = x + merge_feature + heatmap_feature + depth_feature + rgb_feature
                else:
                    x = x + merge_feature + heatmap_feature + depth_feature

            # convert data for list data
            if self.pred_len >= 2:
                B, _, _ = uv.shape
                uv = uv.view(B, self.pred_len, -1, 2)
                uv = [uv[:,sequence_id] for sequence_id in range(self.pred_len)]
                B,C,H,W = heatmap_softmax.shape
                heatmap_softmax = heatmap_softmax.view(B, self.pred_len, -1, H, W)
                heatmap_softmax = [heatmap_softmax[:,sequence_id] for sequence_id in range(self.pred_len)]

                XYZ = XYZ.view(B, self.pred_len, -1, 3)
                XYZ = [XYZ[:,sequence_id] for sequence_id in range(self.pred_len)]

            # append data
            uv_list.append(uv)
            heatmap_list.append(heatmap_softmax)
            pose_list.append(XYZ)
            if self.rotation_pred:
                rotation_list.append(rotation)
            if self.grasp_pred:
                grasp_list.append(grasp)
            if self.rgb_pred:
                rgb_list.append(rgb)

        if not self.intermediate_loss:
            uv_list, heatmap_list, pose_list = uv_list[-1:], heatmap_list[-1:], pose_list[-1:]
            if self.rgb_pred:
                rgb_list = rgb_list[-1:]
            if self.rotation_pred:
                rotation_list = rotation_list[-1:]
            if self.grasp_pred:
                grasp_list = grasp_list[-1:]
        
        if self.pred_len >= 2:
            uv_list = [[*hoge] for hoge in zip(*uv_list)]
            heatmap_list = [[*hoge] for hoge in zip(*heatmap_list)]
            pose_list = [[*hoge] for hoge in zip(*pose_list)]
        else:
            uv_list = [uv_list]
            heatmap_list = [heatmap_list]
            pose_list = [pose_list]
            if self.rgb_pred:
                rgb_list = [rgb_list]
            if self.rotation_pred:
                rotation_list = [rotation_list]
            if self.grasp_pred:
                grasp_list = [grasp_list]

        # make output dict
        output_dict = {'uv':uv_list ,'heatmap':heatmap_list, 'pose':pose_list}
        if self.rgb_pred:
            output_dict['rgb'] = rgb_list
        if self.rotation_pred:
            output_dict['rotation'] = rotation_list
        if self.grasp_pred:
            output_dict['grasp'] = grasp_list
        return output_dict

    def uv2xyz(self, uv, inv_mtx, z):
        # uv_cordinate -> torch.tensor
        # mtx -> tensor (B, 3, 3)
        # z -> tensor
        xy = uv * z
        small_xyz = torch.cat((xy,z),2).float()
        large_XYZ = torch.matmul(inv_mtx,small_xyz.permute(0,2,1)).permute(0,2,1)
        return large_XYZ

class sequence_hourglass(torch.nn.Module):
    def __init__(self, cfg, input_dim=3, output_dim=21, argmax='softargmax', single_depth=False, device='cuda'):
        super(sequence_hourglass, self).__init__()
        '''
        mode1: pretrain hourglass with bc
        mode2: end2end learning
        '''
        self.use_video_pred_model = cfg.SEQUENCE_HOUR.USE_VIDEOMODEL
        self.mode = cfg.SEQUENCE_HOUR.MODE 
        self.pred_len = cfg.PRED_LEN
        if self.pred_len == 1:
            raise ValueError("pred len must be larger than 1 for sequence hourglass")
        self.pred_rotation = cfg.HOURGLASS.PRED_ROTATION
        self.device = device

        self.hour_glass = stacked_hourglass_model(cfg, input_dim=input_dim, output_dim=output_dim, pred_len=1, device=device)
        if self.use_video_pred_model == True:
            self.video_pred_model = VIDEO_HOURGLASS(cfg, pose_dim=output_dim)

    def forward(self, inputs, mp_mode=None, vp_mode=None):
        if mp_mode == None:
            mp_mode = self.mode
        
        if vp_mode == None:
            vp_mode = 1

        B,S,C,H,W = inputs['rgb'].shape

        outputs_history_dict = {}
        for sequence_id in range(S-2):
            model_inputs = self.make_hourglass_input(inputs, outputs_history_dict, sequence_id, mp_mode)
            outputs = self.hour_glass(model_inputs)

            if sequence_id < S-3:
                if self.mode == 1: # input dataset image
                    output_image = inputs['rgb'][:,sequence_id+2].to(self.device) # rgb loss will be 0
                if self.mode == 2: # input output image
                    if self.use_video_pred_model:
                        video_inputs = self.make_videomodel_input(inputs, outputs, outputs_history_dict, sequence_id, vp_mode)
                        video_outputs = self.video_pred_model(video_inputs)
                        output_image = [video_outputs['rgb']]
                    else:
                        raise ValueError('not implement rotation')

            if sequence_id == 0:
                outputs_history_dict['uv'] = [outputs['uv'][0]]
                outputs_history_dict['heatmap'] = [outputs['heatmap'][0]]
                outputs_history_dict['pose'] = [outputs['pose'][0]]
                outputs_history_dict['rgb'] = [output_image]
                if self.pred_rotation:
                    outputs_history_dict['rotation'] = [outputs['rotation'][0]]
                    outputs_history_dict['grasp'] = [outputs['grasp'][0]]
            else:
                outputs_history_dict['uv'].append(outputs['uv'][0])
                outputs_history_dict['heatmap'].append(outputs['heatmap'][0])
                outputs_history_dict['pose'].append(outputs['pose'][0])
                outputs_history_dict['rgb'].append(output_image)
                if self.pred_rotation:
                    outputs_history_dict['rotation'].append(outputs['rotation'][0])
                    outputs_history_dict['grasp'].append(outputs['grasp'][0])
        
        return outputs_history_dict

    def make_hourglass_input(self, inputs, outputs_history_dict, sequence_id, mode=1):
        """
        mode
        1: BC
        2: use predicted image end2end manner
        3: use predicted image w/o gradient chain
        """
        # get rgb image
        if mode == 1:
            rgb = inputs['rgb'][:,sequence_id:sequence_id+2].to(self.device)
            pose_heatmap = inputs['pose'][:,sequence_id:sequence_id+2].to(self.device)
            xyz = inputs['pose_xyz'][:,sequence_id:sequence_id+2]
        elif mode == 2:
            if sequence_id == 0:
                rgb = inputs['rgb'][:,sequence_id:sequence_id+2].to(self.device)
                pose_heatmap = inputs['pose'][:,sequence_id:sequence_id+2].to(self.device)
                xyz = inputs['pose_xyz'][:,sequence_id:sequence_id+2].to(self.device)
            elif sequence_id == 1:
                rgb = torch.cat((inputs['rgb'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['rgb'][-1][-1], 1)), 1)
                pose_heatmap = torch.cat((inputs['pose'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['heatmap'][-1][-1], 1)), 1)
                
                dataset_xyz = inputs['pose_xyz'][:,sequence_id:sequence_id+1].to(self.device) # B, 1, Dim_pose
                output_xyz = outputs_history_dict['pose'][-1][-1]
                B, _, _ = output_xyz.shape # Batch, Num_sequence, Num_Pose(21), 3
                output_xyz = torch.unsqueeze(output_xyz.contiguous().view(B, -1),1) 
                xyz = torch.cat((dataset_xyz, output_xyz), 1) # TODO change output xyz shape into (3,1,63)
            else:
                rgb = torch.cat([torch.unsqueeze(intermidiate_img[-1],1) for intermidiate_img in outputs_history_dict['rgb'][-2:]], 1)
                pose_heatmap = torch.cat([torch.unsqueeze(intermidiate_heatmap[-1],1) for intermidiate_heatmap in outputs_history_dict['heatmap'][-2:]], 1)
                B, _, _ = outputs_history_dict['pose'][-1][-1].shape
                xyz = torch.cat([torch.unsqueeze(intermidiate_pose[-1].contiguous().view(B, -1),1) for intermidiate_pose in outputs_history_dict['pose'][-2:]], 1)
        elif mode == 3:
            if sequence_id == 0:
                rgb = inputs['rgb'][:,sequence_id:sequence_id+2].to(self.device)
                pose_heatmap = inputs['pose'][:,sequence_id:sequence_id+2].to(self.device)
                xyz = inputs['pose_xyz'][:,sequence_id:sequence_id+2].to(self.device)
            elif sequence_id == 1:
                with torch.no_grad():
                    rgb = torch.cat((inputs['rgb'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['rgb'][-1][-1], 1)), 1)
                    pose_heatmap = torch.cat((inputs['pose'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['heatmap'][-1][-1], 1)), 1)
                    
                    dataset_xyz = inputs['pose_xyz'][:,sequence_id:sequence_id+1].to(self.device) # B, 1, Dim_pose
                    output_xyz = outputs_history_dict['pose'][-1][-1]
                    B, _, _ = output_xyz.shape # Batch, Num_sequence, Num_Pose(21), 3
                    output_xyz = torch.unsqueeze(output_xyz.contiguous().view(B, -1),1) 
                    xyz = torch.cat((dataset_xyz, output_xyz), 1) # TODO change output xyz shape into (3,1,63)
            else:
                with torch.no_grad():
                    rgb = torch.cat([torch.unsqueeze(intermidiate_img[-1],1) for intermidiate_img in outputs_history_dict['rgb'][-2:]], 1)
                    pose_heatmap = torch.cat([torch.unsqueeze(intermidiate_heatmap[-1],1) for intermidiate_heatmap in outputs_history_dict['heatmap'][-2:]], 1)
                    B, _, _ = outputs_history_dict['pose'][-1][-1].shape
                    xyz = torch.cat([torch.unsqueeze(intermidiate_pose[-1].contiguous().view(B, -1),1) for intermidiate_pose in outputs_history_dict['pose'][-2:]], 1)

        # get inverse intrinsic camera parameter matrix
        mtx = inputs['inv_mtx'].float().to(self.device)
        return {'rgb':rgb, 'pose':pose_heatmap, 'pose_xyz':xyz, 'inv_mtx':mtx}
    
    def make_videomodel_input(self, inputs, outputs, outputs_history_dict, sequence_id, mode=1):
        '''
        output:
        dictionary{
        rgb => torch.Tensor shape=(B,S,C,H,W),
        pose => torch.Tensor shape=(B,S,C,H,W)}

        mode1: input output heatmap
        mode2: input dataset heatmap
        '''
        index_list = [sequence_id, sequence_id+1, sequence_id+3]
        rgb = inputs['rgb'][:,index_list].to(self.device)
        
        if mode == 1:
            t1_heatmap = torch.unsqueeze(outputs['heatmap'][0][-1],1)
        elif mode == 2:
            print('mode == 2')
            t1_heatmap = inputs['pose'][:,sequence_id+2:sequence_id+3]

        pose_heatmap = inputs['pose'][:,index_list].to(self.device)
        pose_heatmap = torch.cat((pose_heatmap[:,:2], t1_heatmap, pose_heatmap[:,2:]),1)

        return {'rgb':rgb, 'pose':pose_heatmap}