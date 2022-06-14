import torch
import cv2
import numpy as np
from torchvision import datasets, models, transforms
from collections import deque
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from eval_util import get_pil_image, get_concat_h, draw_matrix

class data_process(object):
    def __init__(self, max_len=5):
        self.ToTensor = transforms.ToTensor()
        self.obs_dict = {}
        self.obs_dict['rgb'] = None
        self.obs_dict['depth'] = None
        self.obs_dict['pose_xyz'] = None
        self.obs_dict['heatmap'] = None
        self.obs_dict['rotation_matrix'] = None
        self.obs_dict['grasp'] = None
        self.max_len=max_len
        self.image_size = None
        
    def make_inputs(self,obs):
        image_tensor = self.get_rgb(obs)
        depth_tensor = torch.tensor(np.array(obs.front_depth), dtype=torch.float)
        
        # get camera info
        camera_intrinsic = obs.misc["front_camera_intrinsics"]
        camera_extrinsic = obs.misc["front_camera_extrinsics"] # world2camera

        # get gripper info
        gripper_pos, gripper_matrix, gripper_open = self.get_gripper(obs)

        # get uv cordinate and pose image
        pose_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, self.image_size)
        
        gripper_pos = torch.tensor(gripper_pos, dtype=torch.float32)
        
        if self.obs_dict['rgb'] == None:
            self.obs_dict['rgb'] = deque([torch.unsqueeze(image_tensor, 0) for _ in range(self.max_len)], self.max_len)
            image_tensor_seq = torch.cat(list(self.obs_dict['rgb']), 0)
            rgb_batch = torch.unsqueeze(image_tensor_seq, 0)
            
            self.obs_dict['depth'] = deque([torch.unsqueeze(depth_tensor, 0) for _ in range(self.max_len)], self.max_len)
            depth_tensor_seq = torch.cat(list(self.obs_dict['depth']), 0)
            depth_batch = torch.unsqueeze(depth_tensor_seq, 0)
            
            self.obs_dict['pose_xyz'] = deque([torch.unsqueeze(gripper_pos, 0) for _ in range(self.max_len)], self.max_len)
            gripper_pos_seq = torch.cat(list(self.obs_dict['pose_xyz']),0)
            pose_batch = torch.unsqueeze(gripper_pos_seq, 0)
            
            self.obs_dict['heatmap'] = deque([torch.unsqueeze(pose_image, 0) for _ in range(self.max_len)], self.max_len)
            pose_image_seq = torch.cat(list(self.obs_dict['heatmap']), 0)
            pose_image_batch = torch.unsqueeze(pose_image_seq, 0)
            
            self.obs_dict['rotation_matrix'] = deque([torch.unsqueeze(gripper_matrix, 0) for _ in range(self.max_len)], self.max_len)
            gripper_rot_seq = torch.cat(list(self.obs_dict['rotation_matrix']), 0)
            gripper_rot_batch = torch.unsqueeze(gripper_rot_seq, 0)
            
            self.obs_dict['grasp'] = deque([torch.unsqueeze(gripper_open, 0) for _ in range(self.max_len)], self.max_len)
            grasp_seq = torch.cat(list(self.obs_dict['grasp']), 0)
            grasp_batch = torch.unsqueeze(grasp_seq, 0)
        else:
            self.obs_dict['rgb'].popleft()
            self.obs_dict['rgb'].append(torch.unsqueeze(image_tensor, 0))
            image_tensor_seq = torch.cat(list(self.obs_dict['rgb']), 0)
            rgb_batch = torch.unsqueeze(image_tensor_seq, 0)
            
            self.obs_dict['depth'].popleft()
            self.obs_dict['depth'].append(torch.unsqueeze(depth_tensor, 0))
            depth_tensor_seq = torch.cat(list(self.obs_dict['depth']), 0)
            depth_batch = torch.unsqueeze(depth_tensor_seq, 0)
            
            self.obs_dict['pose_xyz'].popleft()
            self.obs_dict['pose_xyz'].append(torch.unsqueeze(gripper_pos, 0))
            gripper_pos_seq = torch.cat(list(self.obs_dict['pose_xyz']),0)
            pose_batch = torch.unsqueeze(gripper_pos_seq, 0)
            
            self.obs_dict['heatmap'].popleft()
            self.obs_dict['heatmap'].append(torch.unsqueeze(pose_image, 0))
            pose_image_seq = torch.cat(list(self.obs_dict['heatmap']), 0)
            pose_image_batch = torch.unsqueeze(pose_image_seq, 0)
            
            self.obs_dict['rotation_matrix'].popleft()
            self.obs_dict['rotation_matrix'].append(torch.unsqueeze(gripper_matrix, 0))
            gripper_rot_seq = torch.cat(list(self.obs_dict['rotation_matrix']), 0)
            gripper_rot_batch = torch.unsqueeze(gripper_rot_seq, 0)
            
            self.obs_dict['grasp'].popleft()
            self.obs_dict['grasp'].append(torch.unsqueeze(gripper_open, 0))
            grasp_seq = torch.cat(list(self.obs_dict['grasp']), 0)
            grasp_batch = torch.unsqueeze(grasp_seq, 0)
            
        input_dict = {}
        input_dict['rgb'] = rgb_batch.to('cuda')
        input_dict['depth'] = depth_batch.to('cuda')
        input_dict['heatmap'] = pose_image_batch.to('cuda')
        input_dict['pose_xyz'] = pose_batch.to('cuda')
        input_dict['rotation_matrix'] = gripper_rot_batch.to('cuda')
        input_dict['grasp'] = grasp_batch.to('cuda')
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic)).to('cuda')
        
        return input_dict
            
    def get_rgb(self,obs):
        # prepare rgb image
        rgb_image_np = obs.front_rgb
        rgb_image_pil = Image.fromarray(rgb_image_np.astype(np.uint8))
        self.image_size = rgb_image_pil.size
        return self.ToTensor(rgb_image_pil)
    
    def get_gripper(self,obs):
        gripper_pos_WorldCor = np.append(obs.gripper_pose[:3], 1)
        gripper_matrix_WorldCor = obs.gripper_matrix
        gripper_open = obs.gripper_open
        
        world2camera_matrix = obs.misc["front_camera_extrinsics"]
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.dot(camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.dot(camera2world_matrix, gripper_matrix_WorldCor)
        
        return torch.tensor(gripper_pose_CamCor[:3], dtype=torch.float), torch.tensor(gripper_matrix_CamCor[:3,:3], dtype=torch.float), torch.tensor(gripper_open, dtype=torch.float)
    
    def transform_pos2image(self, pos, intrinsic, image_size):
        uv = self.get_uv(pos, intrinsic)
        pos_image = self.make_pos_image(image_size,uv)
        pos_image = self.ToTensor(pos_image)
        uv = torch.tensor(uv)[:,:2]
        uv_mask = torch.where(torch.isnan(uv), 0., 1.)
        uv = torch.where(torch.isnan(uv), 0., uv)
        return pos_image, uv, uv_mask

    def make_pos_image(self,size,uv_data,r=3):
        poslist = []
        for u,v, _ in uv_data:
            pos_image = Image.new('L',size)
            draw = ImageDraw.Draw(pos_image)
            draw.ellipse((u-r, v-r, u+r, v+r), fill=(255), outline=(0))
            poslist.append(np.array(pos_image))
        return np.array(poslist).transpose(1,2,0)

    def get_uv(self,pos_data, intrinsic_matrix):
        # transfer position data(based on motive coordinate) to camera coordinate
        pos_data = np.array(pos_data) # x,y,z
        pos_data = pos_data / pos_data[2] # u,v,1
        uv_result = np.dot(intrinsic_matrix, pos_data)
        return [uv_result]

    def outputs2action(self, outputs, obs):
        world2camera_matrix = obs.misc["front_camera_extrinsics"]
        
        gripper_rotation = outputs['rotation'][0,0,-1].cpu().detach().numpy()
        gripper_pos = outputs['pose'][0,0,-1].cpu().detach().numpy().T
        gripper_matrix = np.append(gripper_rotation, gripper_pos,1)
        gripper_matrix = np.append(gripper_matrix, np.array([[0,0,0,1]]),0)
        gripper_matrix = np.dot(world2camera_matrix, gripper_matrix)
        r = R.from_matrix(gripper_matrix[:3,:3])
        
        quat = r.as_quat()
        gripper_pose = np.append(gripper_matrix[:3,3], quat)
        gripper_open = outputs['grasp'][0,0,-1].cpu().detach().numpy()

        action = np.append(gripper_pose, gripper_open)
        return action
    
    def reset(self):
        self.obs_dict = {}
        self.obs_dict['rgb'] = None
        self.obs_dict['depth'] = None
        self.obs_dict['pose_xyz'] = None
        self.obs_dict['heatmap'] = None
        self.obs_dict['rotation_matrix'] = None
        self.obs_dict['grasp'] = None

def get_various_image(inputs, outputs, obs, iteration, gt_matrix_list, gt_image_path_list, save_gt_image):
    # return concatenated image
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    
    pil_image = get_pil_image(obs)
    torch_image = totensor(pil_image)
    torch_image = torch.unsqueeze(torch_image, 0)
    torch_image = torch.unsqueeze(torch_image, 0)
    
    rotation_image = make_rotation_image(torch_image, torch.unsqueeze(outputs['rotation'][:,0,-1],1), outputs['pose'][:,0,-1],
                                          torch.unsqueeze(torch.tensor(obs.misc["front_camera_intrinsics"]), 0))
    merge_image = get_concat_h(pil_image, topil(rotation_image[0][0]))
    
    if "trajectory" in outputs.keys():
        trajectory_image = topil(outputs["trajectory"][0,0,-1])
        merge_image = get_concat_h(merge_image, trajectory_image)
        
        overlay_image = make_overlay_image(pil_image, trajectory_image)
        merge_image = get_concat_h(merge_image, overlay_image)
        
    if save_gt_image:
        if iteration < len(gt_image_path_list):
            gt_image_path = gt_image_path_list[iteration]
            gt_matrix = gt_matrix_list[iteration]
        else:
            gt_image_path = gt_image_path_list[-1]
            gt_matrix = gt_matrix_list[-1]
        
        # get gt image
        gt_image = Image.open(gt_image_path)
#         merge_image = get_concat_h(merge_image, gt_image)
        
        # get gt pose
        world2camera_matrix = obs.misc["front_camera_extrinsics"]
        camera2world_matrix = np.linalg.inv(world2camera_matrix)        
        gt_matrix = np.dot(camera2world_matrix, gt_matrix)
        
        # get pred pose image and overlay gt pose
        rotation_image_with_gt = draw_matrix(topil(rotation_image[0][0]), gt_matrix, obs.misc["front_camera_intrinsics"])
        merge_image = get_concat_h(merge_image, rotation_image_with_gt)
        
        # get gt pose image
        gt_rotation_image = draw_matrix(gt_image, gt_matrix, obs.misc["front_camera_intrinsics"])
        merge_image = get_concat_h(merge_image, gt_rotation_image)
        
    return merge_image

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
            pose_matrix = np.append(rotation_np, pos_vec_np.T, 1)
            pose_matrix = np.append(pose_matrix, np.array([[0,0,0,1]]),0)
            image_pil = draw_matrix(image_pil, pose_matrix, intrinsic_matrix_np)
            # to tensor batch
            image_tensor = totensor(image_pil)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            
            image_sequence[b,s] = image_tensor
    
    return image_sequence

def make_overlay_image(base_image, over_image):
    # image2 is base image
    image1 = np.array(base_image)
    image2 = np.array(over_image)
    image2 = np.repeat(image2[:,:,np.newaxis], 3, axis=2)
    image = cv2.addWeighted(image2, 0.7, image1, 0.3, 0)
    return Image.fromarray(image)