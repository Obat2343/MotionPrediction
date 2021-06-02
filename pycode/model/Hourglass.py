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

        if self.input_depth:
            input_depth_dim = 1 + int(self.use_past)

        # output_option
        if pred_len == 'none':
            self.pred_len = cfg.PRED_LEN
        else:
            self.pred_len = pred_len
        output_dim = output_dim * self.pred_len

        if cfg.HOURGLASS.SINGLE_DEPTH:
            self.depth_channel = self.pred_len
        else:
            self.depth_channel = output_dim

        rotation_channel = 6 * self.pred_len
        grasp_channel = 1 * self.pred_len
        trajectory_channel = 1 # TODO multiply num pose?

        self.rgb_pred = cfg.HOURGLASS.PRED_RGB
        self.rotation_pred = cfg.HOURGLASS.PRED_ROTATION
        self.grasp_pred = cfg.HOURGLASS.PRED_GRASP
        self.trajectory_pred = cfg.HOURGLASS.PRED_TRAJECTORY

        # loss_option
        self.intermediate_loss = cfg.HOURGLASS.INTERMEDIATE_LOSS 

        self.initial_conv = torch.nn.ModuleList([
            ConvBlock(input_dim, int(base_filter / 2), 3, 1, 1, activation=activation, norm=norm),
            ResidualBlock(int(base_filter / 2), base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm)
        ])

        if self.input_pose:
            self.initial_pose_conv = torch.nn.ModuleList([
            ConvBlock(input_pose_dim, int(base_filter / 2), 3, 1, 1, activation=activation, norm=norm),
            ResidualBlock(int(base_filter / 2), base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm),
            ResidualBlock(base_filter, base_filter, activation=activation, norm=norm)
        ])

        if self.input_depth:
            self.initial_depth_conv = torch.nn.ModuleList([
            ConvBlock(input_depth_dim, int(base_filter / 2), 3, 1, 1, activation=activation, norm=norm),
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

        if self.trajectory_pred:
            self.trajectory_layer_list = torch.nn.ModuleList([ConvBlock(base_filter, trajectory_channel, activation='none', norm='none') for _ in range(self.num_hourglass)])
            self.merge_trajectory_list = torch.nn.ModuleList([ConvBlock(trajectory_channel, base_filter, 1, 1, 0, activation='none', norm='none') for _ in range(self.num_hourglass - 1)])

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
                input_z = inputs['pose_xyz'][:,start_index:2,2::3].to(self.device) # B,S,C
                input_z = input_z.view(B,-1)
                input_z = torch.unsqueeze(input_z,2)
                input_z = torch.unsqueeze(input_z,3)
                pose_map.append(pose_heatmap * input_z.expand(B,C,H,W))
            if self.input_rotation:
                input_rotation = inputs['rotation_matrix'][:,start_index:2,:2].to(self.device)
                input_rotation = input_rotation.view(B,-1,6)
                input_rotation = input_rotation.contiguous().view(B,-1)
                input_rotation = torch.unsqueeze(input_rotation, 2)
                input_rotation = torch.unsqueeze(input_rotation, 3)
                input_rotation = input_rotation.expand(B,6*(2-start_index),H,W)

                heatmap_for_rotation = torch.cat([pose_heatmap[:,s:s+1].expand(B,6,H,W) for s in range(S)], 1)
                pose_map.append(heatmap_for_rotation * input_rotation)
            if self.input_grasp:
                input_grasp = inputs['grasp'][:,start_index:2].to(self.device)
                input_grasp = input_grasp.view(B,-1)
                input_grasp = torch.unsqueeze(input_grasp, 2)
                input_grasp = torch.unsqueeze(input_grasp, 3)
                pose_map.append(pose_heatmap * input_grasp.expand(B,2-start_index,H,W))
            
            if len(pose_map) > 0:
                pose_heatmap = torch.cat(pose_map, 1)

        if self.input_depth:
            if self.use_past:
                depth = inputs['depth'][:,:2].to(self.device)
                depth = depth.view(B, -1, H, W)
            else:
                depth = inputs['depth'][:,1].to(self.device)

        # get inverse intrinsic camera parameter matrix
        mtx = inputs['inv_mtx'].float().to(self.device)

        # list for output
        uv_list = []
        pose_list = []
        heatmap_list = []
        rotation_list = []
        rgb_list = []
        grasp_list = []
        trajectory_list = []
        
        # initial conv
        for i in range(len(self.initial_conv)):
            x = self.initial_conv[i](x)

        if self.input_pose:
            for i in range(len(self.initial_pose_conv)):
                pose_heatmap = self.initial_pose_conv[i](pose_heatmap)

            x = x + pose_heatmap

        if self.input_depth:
            for i in range(len(self.initial_depth_conv)):
                depth = self.initial_depth_conv[i](depth)
            
            x = x + depth

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
            S = self.pred_len
            C = int(C / S)
            if self.rotation_pred:
                rotation = []
                rotation_feature = self.rotation_layer_list[i](x)
                for pred_index in range(self.pred_len):
                    rotation_map = heatmap_softmax[:,C*pred_index:C*(pred_index+1)] * rotation_feature[:,6*C*pred_index:6*C*(pred_index+1)] # TODO change here
                    rotation_vec = torch.sum(rotation_map.view(B,6,-1),2)
                    rotation_mat = compute_rm(rotation_vec)
                    rotation.append(rotation_mat)

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

            if self.trajectory_pred:
                trajectory_map = self.trajectory_layer_list[i](x)
                trajectory_map = torch.sigmoid(trajectory_map)

            # feedback output into feature
            if i < self.num_hourglass - 1:               
                heatmap_feature = self.merge_heatmap_list[i](heatmap)
                depth_feature = self.merge_depth_list[i](depth)
                merge_feature = self.merge_feature_list[i](x)
                feature_list = [heatmap_feature, depth_feature, merge_feature] 

                if self.rotation_pred:
                    rotation_feature = self.merge_rotation_list[i](rotation_map)
                    feature_list.append(rotation_feature)

                if self.grasp_pred:
                    grasp_feature = self.merge_grasp_list[i](grasp_map)
                    feature_list.append(grasp_feature)

                if self.trajectory_pred:
                    trajectory_feature = self.merge_trajectory_list[i](trajectory_map)
                    feature_list.append(trajectory_feature)

                if self.rgb_pred:
                    rgb_feature = self.merge_rgb_list[i](rgb)
                    feature_list.append(rgb_feature)
                
                x =  x + sum(feature_list)

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

                if self.grasp_pred:
                    grasp = [grasp[:,sequence_id:sequence_id+1] for sequence_id in range(self.pred_len)]

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
            if self.trajectory_pred:
                trajectory_list.append(trajectory_map)

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
            if self.rgb_pred:
                rgb_list = [[*hoge] for hoge in zip(*rgb_list)]
            if self.rotation_pred:
                rotation_list = [[*hoge] for hoge in zip(*rotation_list)]
            if self.grasp_pred:
                grasp_list = [[*hoge] for hoge in zip(*grasp_list)]
        else:
            uv_list = [uv_list]
            heatmap_list = [heatmap_list]
            pose_list = [pose_list]
            if self.rgb_pred:
                rgb_list = [rgb_list]
            if self.rotation_pred:
                rotation_list = [[rot[0] for rot in rotation_list]]
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
        if self.trajectory_pred:
            output_dict['trajectory'] = [trajectory_list]
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
        self.pred_rotation = cfg.HOURGLASS.PRED_ROTATION
        self.pred_grasp = cfg.HOURGLASS.PRED_GRASP
        self.pred_trajectory = cfg.HOURGLASS.PRED_TRAJECTORY
        self.input_depth = cfg.HOURGLASS.INPUT_DEPTH
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
                if mp_mode == 1: # input dataset image
                    output_image = inputs['rgb'][:,sequence_id+2].to(self.device) # rgb loss will be 0
                    if self.input_depth:
                        output_depth = inputs['depth'][:,sequence_id+2].to(self.device)
                if mp_mode == 2: # input output image
                    if self.use_video_pred_model:
                        video_inputs = self.make_videomodel_input(inputs, outputs, sequence_id, vp_mode)
                        video_outputs = self.video_pred_model(video_inputs)
                        output_image = [video_outputs['rgb']]
                        if self.input_depth:
                            output_depth = [video_outputs['depth']]
                    else:
                        raise ValueError('not implement rotation')

            if sequence_id == 0:
                outputs_history_dict['uv'] = [outputs['uv'][0]]
                outputs_history_dict['heatmap'] = [outputs['heatmap'][0]]
                outputs_history_dict['pose'] = [outputs['pose'][0]]
                outputs_history_dict['rgb'] = [output_image]
                if self.input_depth:
                    outputs_history_dict['depth'] = [output_depth]
                if self.pred_rotation:
                    outputs_history_dict['rotation'] = [outputs['rotation'][0]]
                if self.pred_grasp:
                    outputs_history_dict['grasp'] = [outputs['grasp'][0]]
                if self.pred_trajectory:
                    outputs_history_dict['trajectory'] = [outputs['trajectory'][0]]
            else:
                outputs_history_dict['uv'].append(outputs['uv'][0])
                outputs_history_dict['heatmap'].append(outputs['heatmap'][0])
                outputs_history_dict['pose'].append(outputs['pose'][0])
                outputs_history_dict['rgb'].append(output_image)
                if self.input_depth:
                    outputs_history_dict['depth'].append(output_depth)
                if self.pred_rotation:
                    outputs_history_dict['rotation'].append(outputs['rotation'][0])
                if self.pred_grasp:
                    outputs_history_dict['grasp'].append(outputs['grasp'][0])
                if self.pred_trajectory:
                    outputs_history_dict['trajectory'].append(outputs['trajectory'][0])
        
        return outputs_history_dict

    def make_hourglass_input(self, inputs, outputs_history_dict, sequence_id, mode=1):
        """
        mode
        1: use training data
        2: use predicted image
        """
        data_dict = {}
        if mode == 1:
            data_dict['rgb'] = inputs['rgb'][:,sequence_id:sequence_id+2].to(self.device)
            data_dict['pose'] = inputs['pose'][:,sequence_id:sequence_id+2].to(self.device)
            data_dict['pose_xyz'] = inputs['pose_xyz'][:,sequence_id:sequence_id+2].to(self.device)
            if self.input_depth:
                data_dict['depth'] = inputs['depth'][:,sequence_id:sequence_id+2].to(self.device)
            if self.pred_rotation:
                data_dict['rotation_matrix'] = inputs['rotation_matrix'][:,sequence_id:sequence_id+2].to(self.device)
            if self.pred_grasp:
                data_dict['grasp'] = inputs['grasp'][:,sequence_id:sequence_id+2].to(self.device)
        elif mode == 2:
            if sequence_id == 0:
                data_dict['rgb'] = inputs['rgb'][:,sequence_id:sequence_id+2].to(self.device)
                data_dict['pose'] = inputs['pose'][:,sequence_id:sequence_id+2].to(self.device)
                data_dict['pose_xyz'] = inputs['pose_xyz'][:,sequence_id:sequence_id+2].to(self.device)
                if self.input_depth:
                    data_dict['depth'] = inputs['depth'][:,sequence_id:sequence_id+2].to(self.device)
                if self.pred_rotation:
                    data_dict['rotation_matrix'] = inputs['rotation_matrix'][:,sequence_id:sequence_id+2].to(self.device)
                if self.pred_grasp:
                    data_dict['grasp'] = inputs['grasp'][:,sequence_id:sequence_id+2].to(self.device)
            elif sequence_id == 1:
                data_dict['rgb'] = torch.cat((inputs['rgb'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['rgb'][-1][-1], 1)), 1)
                data_dict['pose'] = torch.cat((inputs['pose'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['heatmap'][-1][-1], 1)), 1)

                dataset_xyz = inputs['pose_xyz'][:,sequence_id:sequence_id+1].to(self.device) # B, 1, Dim_pose
                output_xyz = outputs_history_dict['pose'][-1][-1]
                B, _, _ = output_xyz.shape # Batch, Num_sequence, Num_Pose(21), 3
                output_xyz = torch.unsqueeze(output_xyz.contiguous().view(B, -1),1) 
                data_dict['pose_xyz'] = torch.cat((dataset_xyz, output_xyz), 1)
                if self.input_depth:
                    data_dict['depth'] = torch.cat((inputs['depth'][:,sequence_id:sequence_id+1].to(self.device), torch.unsqueeze(outputs_history_dict['depth'][-1][-1], 1)), 1)
                if self.pred_rotation:
                    data_dict['rotation_matrix'] = torch.cat((inputs['rotation_matrix'][:,sequence_id:sequence_id+1].to(self.device),torch.unsqueeze(outputs_history_dict['rotation'][-1][-1], 1)), 1)
                if self.pred_grasp:
                    data_dict['grasp'] = torch.cat((inputs['grasp'][:,sequence_id:sequence_id+1].to(self.device), outputs_history_dict['grasp'][-1][-1]), 1)
            else:
                data_dict['rgb'] = torch.cat([torch.unsqueeze(intermidiate_img[-1],1) for intermidiate_img in outputs_history_dict['rgb'][-2:]], 1)
                data_dict['pose'] = torch.cat([torch.unsqueeze(intermidiate_heatmap[-1],1) for intermidiate_heatmap in outputs_history_dict['heatmap'][-2:]], 1)
                B, _, _ = outputs_history_dict['pose'][-1][-1].shape
                data_dict['pose_xyz'] = torch.cat([torch.unsqueeze(intermidiate_pose[-1].contiguous().view(B, -1),1) for intermidiate_pose in outputs_history_dict['pose'][-2:]], 1)
                if self.input_depth:
                    data_dict['depth'] = torch.cat([torch.unsqueeze(intermidiate_heatmap[-1],1) for intermidiate_heatmap in outputs_history_dict['depth'][-2:]], 1)
                if self.pred_rotation:
                    data_dict['rotation_matrix'] = torch.cat([torch.unsqueeze(intermidiate_rotation[-1],1) for intermidiate_rotation in outputs_history_dict['rotation'][-2:]], 1)
                if self.pred_grasp:
                    data_dict['grasp'] = torch.cat([torch.unsqueeze(intermidiate_grasp[-1],1) for intermidiate_grasp in outputs_history_dict['grasp'][-2:]], 1)

        # get inverse intrinsic camera parameter matrix
        data_dict['inv_mtx'] = inputs['inv_mtx'].float().to(self.device)
        return data_dict
    
    def make_videomodel_input(self, inputs, outputs, sequence_id, mode=1):
        '''
        output:
        dictionary{
        rgb => torch.Tensor shape=(B,S,C,H,W),
        pose => torch.Tensor shape=(B,S,C,H,W)}

        mode1: input output heatmap
        mode2: input dataset heatmap
        '''
        data_dict = {}

        if mode == 1:
            t1_heatmap = torch.unsqueeze(outputs['heatmap'][0][-1],1)
        elif mode == 2:
            t1_heatmap = inputs['pose'][:,sequence_id+2:sequence_id+3].to(self.device)

        if self.video_pred_model.mode == 'pcf':
            index_list = [sequence_id, sequence_id+1, sequence_id+3]
            data_dict['rgb'] = inputs['rgb'][:,index_list].to(self.device)
            if self.input_depth:
                data_dict['depth'] = inputs['depth'][:,index_list].to(self.device)
            pose_heatmap = inputs['pose'][:,:4].to(self.device)
            data_dict['pose'] = torch.cat((pose_heatmap[:,:2], t1_heatmap, pose_heatmap[:,3:]), 1)
            data_dict['pose_xyz'] = inputs['pose_xyz'][:,:4].to(self.device)
            data_dict['rotation_matrix'] = inputs['rotation_matrix'][:,:4].to(self.device)
            data_dict['grasp'] = inputs['grasp'][:,:4].to(self.device)
        elif self.video_pred_model.mode == 'pc':
            index_list = [sequence_id, sequence_id+1]
            data_dict['rgb'] = inputs['rgb'][:,index_list].to(self.device)
            if self.input_depth:
                data_dict['depth'] = inputs['depth'][:,index_list].to(self.device)
            pose_heatmap = inputs['pose'][:,:3].to(self.device)
            data_dict['pose'] = torch.cat((pose_heatmap[:,:2], t1_heatmap), 1)
            data_dict['pose_xyz'] = inputs['pose_xyz'][:,:3].to(self.device)
            data_dict['rotation_matrix'] = inputs['rotation_matrix'][:,:3].to(self.device)
            data_dict['grasp'] = inputs['grasp'][:,:3].to(self.device)
        elif self.video_pred_model.mode == 'c':
            data_dict['rgb'] = inputs['rgb'][:,1].to(self.device)
            if self.input_depth:
                data_dict['depth'] = inputs['depth'][:,1].to(self.device)
            pose_heatmap = inputs['pose'][:,1:3].to(self.device)
            data_dict['pose'] = torch.cat((pose_heatmap[:,:1], t1_heatmap), 1)
            data_dict['pose_xyz'] = inputs['pose_xyz'][:,1:3].to(self.device)
            data_dict['rotation_matrix'] = inputs['rotation_matrix'][:,1:3].to(self.device)
            data_dict['grasp'] = inputs['grasp'][:,1:3].to(self.device)
            
        return data_dict