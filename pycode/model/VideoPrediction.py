import torch
import torch.nn as nn
import numpy as np
from .base_networks import ConvBlock, ConvOffset2D, ResidualBlock

class ResSubsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, activation='relu'):
        super(ResSubsampleBlock, self).__init__()
        self.shortcut = nn.Sequential(nn.AvgPool2d(2), ConvBlock(in_channels, out_channels, 1, 1, 0, activation=activation))
        self.conv1 = ConvBlock(in_channels, out_channels, filter_size, 1, 1, activation=activation)
        self.conv2 = nn.Sequential(ConvBlock(out_channels, out_channels, filter_size, 1, 1, activation=activation), nn.AvgPool2d(2))

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return shortcut + out

class ResUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, activation='relu', norm='none'):
        super(ResUpsampleBlock, self).__init__()
        self.shortcut = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), ConvBlock(in_channels, out_channels, filter_size, 1, 1, activation='none'))
        self.conv1 = ConvBlock(in_channels, out_channels, filter_size, 1, 1, activation=activation, norm=norm)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = ConvBlock(out_channels, out_channels, filter_size, 1, 1, activation=activation, norm=norm)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(x)
        out = self.upsample(out)
        out = self.conv2(out)
        return shortcut + out

class Encoder(nn.Module):

    def __init__(self, input_dim=3, filter_size=3, min_filter_num=64, max_filter_num=512, downscale_num=4, activation='relu', norm='none'):
        super(Encoder, self).__init__()
        self.downscale_num = downscale_num

        res_blocks = []
        current_filter_num = min_filter_num
        for i in range(self.downscale_num):
            next_filter_num = min(current_filter_num * 2, max_filter_num)
            res_blocks.append(ResSubsampleBlock(current_filter_num, next_filter_num, filter_size, activation=activation))
            current_filter_num = next_filter_num

        self.res_blocks = nn.ModuleList(res_blocks)
        self.conv = ConvBlock(input_dim, min_filter_num, filter_size, 1, 1, activation=activation, norm=norm)
        # self.conv = nn.Conv2d(input_dim, min_filter_num, filter_size, 1, 1)
        # self.conv_z = nn.Conv2d(current_filter_num, z_dim, 1, 1, 0)

    def forward(self, x):
        out = [self.conv(x)]
        for res_block in self.res_blocks:
            out.append(res_block(out[-1]))
        # z = self.conv_z(out)
        return out

class Decoder(nn.Module):

    def __init__(self, input_dim=3, filter_size=3, min_filter_num=64, max_filter_num_image=256, max_filter_num_pose=128, last_activation='none', deformable=False, activation='relu', norm='none', downscale_num = 4):
        super(Decoder, self).__init__()
        self.downscale_num = downscale_num

        res_blocks = []
        compressor = []
        filters = []
        current_filter_num = min_filter_num
        compress_filter_num = min_filter_num
        for i in range(self.downscale_num - 1):
            next_filter_num_image = min(current_filter_num * 2, max_filter_num_image)
            next_filter_num_pose = min(current_filter_num * 2, max_filter_num_pose)

            max_filter_num = max(max_filter_num_image, max_filter_num_pose)
            concat_filter_num = next_filter_num_image + next_filter_num_pose
            filters.append([max_filter_num, concat_filter_num, compress_filter_num, 0])
            compress_filter_num = max_filter_num
            current_filter_num *= 2

        for i in range(self.downscale_num - 1):
            compressor.append(ConvBlock(filters[i][1], filters[i][0], 3, 1, 1, activation=activation, norm=norm))
            res_blocks.append(ResUpsampleBlock(filters[i][0] * 2, filters[i][2], 3, activation=activation, norm=norm))
        res_blocks.reverse()
        compressor.reverse()

        self.res_blocks = nn.ModuleList(res_blocks)
        self.compressor = nn.ModuleList(compressor)
        self.first_upsample = ResUpsampleBlock(concat_filter_num, max_filter_num, 3, activation=activation, norm=norm) # TODO change this

        # self.fc = nn.Linear(z_dim, self.max_filter_num * self.height * self.width)
        self.relu = nn.ReLU()
        self.conv = ConvBlock(min_filter_num, input_dim, filter_size, 1, 1, activation='none', norm='none')
        
        if deformable == True:
            self.conv2d_offset = ConvOffset2D(min_filter_num)
        else:
            self.conv2d_offset = None

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.rgb_mask = nn.Conv2d(min_filter_num, 1, filter_size, 1, 1)
        self.depth_mask = nn.Conv2d(min_filter_num, 1, filter_size, 1, 1)
        self.last_activation = last_activation

    def forward(self, outs):
        h0 = self.first_upsample(torch.cat([out.pop() for out in outs], 1))
        for i in range(len(self.res_blocks)):
            concat_compress_data = self.compressor[i](torch.cat([out.pop() for out in outs], 1))
            h0 = torch.cat([concat_compress_data, h0], 1)
            h0 = self.res_blocks[i](h0)
        
        if self.conv2d_offset != None:
            h0 = self.conv2d_offset(h0)

        rgb_mask = self.sigmoid(self.rgb_mask(h0))
        depth_mask = self.sigmoid(self.depth_mask(h0))

        x = self.conv(h0)
        
        if self.last_activation == 'tanh':
            x = self.tanh(x)
        elif self.last_activation == 'relu':
            x = self.relu(x)
        
        return x, rgb_mask, depth_mask

class VIDEO_HOURGLASS(nn.Module):

    def __init__(self, cfg, pose_dim='none', filter_size=3, device='cuda'):
        super(VIDEO_HOURGLASS, self).__init__()
        self.device = device
        self.mode = cfg.VIDEO_HOUR.MODE #'pcf','pc','c'
        self.use_depth = cfg.VIDEO_HOUR.DEPTH # TODO
        self.input_z = cfg.VIDEO_HOUR.INPUT_Z
        self.input_rotation = cfg.VIDEO_HOUR.INPUT_ROTATION
        self.input_grasp = cfg.VIDEO_HOUR.INPUT_GRASP

        basic_dim = 3 + int(self.use_depth)

        if pose_dim == 'none':
            if cfg.DATASET.NAME == 'HMD':
                pose_dim = 21
            elif (cfg.DATASET.NAME == 'RLBench') or (cfg.DATASET.NAME == 'RLBench2'):
                pose_dim = 1
            else:
                raise ValueError("invalid pose dim")

        pose_dim = (pose_dim + (6 * self.input_rotation) + self.input_grasp)

        if self.mode == 'pcf':
            encoder_input_dim = basic_dim * 3
            pose_encoder_dim = pose_dim * 4
        elif self.mode == 'pc':
            encoder_input_dim = basic_dim * 2
            pose_encoder_dim = pose_dim * 3
        else:
            encoder_input_dim = basic_dim
            pose_encoder_dim = pose_dim * 2
        
        decoder_output_dim = basic_dim

        min_filter_num = int(cfg.VIDEO_HOUR.MIN_FILTER_NUM)
        max_filter_num = int(cfg.VIDEO_HOUR.MAX_FILTER_NUM)

        self.img_encoder = Encoder(input_dim=encoder_input_dim, filter_size=filter_size, min_filter_num=min_filter_num, max_filter_num=max_filter_num, downscale_num=cfg.VIDEO_HOUR.NUM_DOWN)
        self.pose_encoder = Encoder(input_dim=pose_encoder_dim, filter_size=filter_size, min_filter_num=min_filter_num, max_filter_num=256, downscale_num=cfg.VIDEO_HOUR.NUM_DOWN)
        self.decoder = Decoder(input_dim=decoder_output_dim, min_filter_num=min_filter_num, max_filter_num_image=256, max_filter_num_pose=256, last_activation='none', downscale_num=cfg.VIDEO_HOUR.NUM_DOWN)

        self.last = cfg.VIDEO_HOUR.LAST_LAYER

    def forward(self, inputs):
        RGB, POSE = inputs['rgb'], inputs['pose']
        B,S,C,H,W = POSE.shape
        output_dict = {}

        if (self.mode == 'pcf') and (self.mode == 'pc'):
            BASE_IMAGE = RGB[:,1]
        else:
            BASE_IMAGE = RGB[:,0]

        RGB = RGB.view(B, -1, H, W)
        POSE = POSE.view(B, -1, H, W)

        pose_map = []
        if self.input_z:
            input_z = inputs['pose_xyz'][:,:,2::3] # B,S,C
            input_z = input_z.view(B,-1) # B, S
            input_z = torch.unsqueeze(input_z,2)
            input_z = torch.unsqueeze(input_z,3)
            pose_map.append(POSE * input_z.expand(B,S,H,W))
        if self.input_rotation:
            input_rotation = inputs['rotation_matrix'][:,:,:2]
            input_rotation = input_rotation.view(B,-1,6)
            input_rotation = input_rotation.contiguous().view(B,-1)
            input_rotation = torch.unsqueeze(input_rotation, 2)
            input_rotation = torch.unsqueeze(input_rotation, 3)
            input_rotation = input_rotation.expand(B,6*S,H,W)

            heatmap_for_rotation = torch.cat([POSE[:,s:s+1].expand(B,6,H,W) for s in range(S)], 1)
            pose_map.append(heatmap_for_rotation * input_rotation)
        if self.input_grasp:
            input_grasp = inputs['grasp'].view(B,-1)
            input_grasp = torch.unsqueeze(input_grasp, 2)
            input_grasp = torch.unsqueeze(input_grasp, 3)
            pose_map.append(POSE * input_grasp.expand(B,S,H,W))

        if len(pose_map) > 0:
            POSE = torch.cat(pose_map, 1)

        u = self.img_encoder(RGB.view(B, -1, H, W))
        u0 = self.pose_encoder(POSE.view(B, -1, H, W))
        # d = n0 - n1
        out, rgb_mask, depth_mask = self.decoder([u, u0])
        rgb_mask = rgb_mask.repeat(1, 3, 1, 1)

        if self.use_depth:
            m = torch.cat((rgb_mask, depth_mask), 1)
        else:
            m = rgb_mask

        if self.last == 'heatmap':
            output_dict['rgb'] = out * m + BASE_IMAGE * (1 - m)
        elif self.last == 'residual':
            output_dict['rgb'] = BASE_IMAGE - out
        elif self.last == 'normal':
            output_dict['rgb'] = out

        output_dict['heatmap'] = rgb_mask
        output_dict['depth_heatmap'] = depth_mask
        output_dict['diff_img'] = out
        
        return output_dict
        # return out * m + x * (1 - m), w, out

class Discriminator(nn.Module):

    def __init__(self, cfg, input_size, activation='lrelu', norm='none'):
        super(Discriminator, self).__init__()
        filter_size = 3
        min_filter_num = cfg.DISCRIMINATOR.MIN_FILTER_NUM
        max_filter_num = cfg.DISCRIMINATOR.MAX_FILTER_NUM

        self.block_num, self.height, self.width = self._compute_layer_config(input_size)

        res_blocks = []
        current_filter_num = min_filter_num
        for i in range(self.block_num):
            next_filter_num = min(current_filter_num * 2, max_filter_num)
            res_blocks.append(ResSubsampleBlock(current_filter_num, next_filter_num, filter_size, activation=activation))
            current_filter_num = next_filter_num

        self.res_blocks = nn.ModuleList(res_blocks)
        self.conv = ConvBlock(3, min_filter_num, 3, 1, 1, activation=activation, norm=norm)
        self.fc_dim = current_filter_num * self.height * self.width
        self.fc = nn.Linear(self.fc_dim, 1)

    def forward(self, x):
        out = self.conv(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = out.view(-1, self.fc_dim)
        out = self.fc(out)
        return out
    
    def _compute_layer_config(self,img_size):
        h, w = img_size
        min_size = min(img_size)
        block_num = 0
        while min_size >= 16:
            min_size /= 2
            h /= 2
            w /= 2
            block_num += 1
        return int(block_num), int(h), int(w)