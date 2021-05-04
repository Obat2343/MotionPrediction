import torch
import os
import sys
import json
import math
import random
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from pprint import pprint
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
import pandas as pd
import pickle

python_version = sys.version
if float(python_version[:3]) > 3.6:
    import kornia
else:
    print('kornia requires version >= 3.6. your version {}'.format(float(python_version[:3])))

class imageaug_full_transform(object):

    def __init__(self, cfg):
        self.rgb_list = []
        if cfg.AUGMENTATION.GAUSSIAN_NOISE > 0:
            self.rgb_list.append(iaa.AdditiveGaussianNoise(scale=(0.,cfg.AUGMENTATION.GAUSSIAN_NOISE*255), per_channel=True))
        if cfg.AUGMENTATION.GAUSSIAN_BLUR > 0:
            self.rgb_list.append(iaa.GaussianBlur(sigma=(0.,cfg.AUGMENTATION.GAUSSIAN_BLUR)))
        if cfg.AUGMENTATION.CHANGE_BRIGHTNESS == True:
            mul_range = cfg.AUGMENTATION.BRIGHTNESS_MUL_RANGE
            add_range = cfg.AUGMENTATION.BRIGHTNESS_ADD_RANGE
            self.rgb_list.append(iaa.MultiplyAndAddToBrightness(mul=mul_range, add=add_range))
        if cfg.AUGMENTATION.GAMMA_CONTRAST != 1:
            self.rgb_list.append(iaa.GammaContrast((1. / cfg.AUGMENTATION.GAMMA_CONTRAST, cfg.AUGMENTATION.GAMMA_CONTRAST)))
        
        self.rgb_aug = iaa.Sequential(self.rgb_list)

    def __call__(self,img,seed):
        # https://imgaug.readthedocs.io/en/latest/source/augmenters.html#additivegaussiannoise
        ia.seed(seed)
        img = np.array(img) # H,W,C
        img = self.rgb_aug.augment_image(img)
        img = Image.fromarray(img)
        return img

class pose_aug(object):
    
    def __init__(self, cfg, seed=1):
        random.seed(seed)
        self.max_noise = cfg.AUGMENTATION.MAX_NOISE
        self.max_hand_dropout = cfg.AUGMENTATION.HAND_DROPOUT_MAX
        self.max_knife_dropout = cfg.AUGMENTATION.KNIFE_DROPOUT_MAX
        
    def __call__(self, pose):
        # define num dropout
        hand_dropout_num = random.randint(0,self.max_hand_dropout)
        knife_dropout_num = random.randint(0,self.max_knife_dropout)

        # get original pose data
        hand_pose = pose[2:47]
        knife_pose = pose[47:]

        ##### for hand #####
        # count nan in the original pose data
        hand_not_nan_list = []
        hand_nan_count = 0
        for i in range(0, len(hand_pose)):
            if math.isnan(hand_pose[i]):
                hand_nan_count += 1
            else:
                noise = ((random.random() * 2) - 1) * self.max_noise
                hand_pose[i] += noise
                if i % 3 == 0:
                    hand_not_nan_list.append(i)

        # apply dropout
        if hand_dropout_num - hand_nan_count > 0:
            hand_nan_index = random.sample(hand_not_nan_list, hand_dropout_num - hand_nan_count)
            for i in hand_nan_index:
                hand_pose[i:i+3] = [np.nan] * 3
        else:
            hand_nan_index = []

        ##### for knife #####
        # count nan in the original pose data
        knife_not_nan_list = []
        knife_nan_count = 0
        for i in range(0, len(knife_pose)):
            if math.isnan(knife_pose[i]):
                knife_nan_count += 1
            else:
                noise = ((random.random() * 2) - 1) * self.max_noise
                knife_pose[i] += noise
                if i % 3 == 0:
                    knife_not_nan_list.append(i)

        # apply dropout
        if knife_dropout_num - knife_nan_count > 0:
            knife_nan_index = random.sample(knife_not_nan_list, knife_dropout_num - knife_nan_count)
            for i in knife_nan_index:
                knife_pose[i:i+3] = [np.nan] * 3
        else:
            knife_nan_index = []

        pose[2:47] = hand_pose
        pose[47:] = knife_pose
        return pose

class depth_aug(object):
    
    def __init__(self, cfg, seed=1):
        self.depth_blur = cfg.AUGMENTATION.DEPTH_BLUR # gauss, box, median, median_double, none
        self.kernel_size = cfg.AUGMENTATION.DEPTH_BLUR_KERNEL_SIZE
        
    def __call__(self, depth):
        if self.depth_blur != 'none':
            depth = torch.unsqueeze(depth, 0)
            
            if self.depth_blur == 'median':
                depth = kornia.median_blur(depth,(self.kernel_size,self.kernel_size))
            elif self.depth_blur == 'median_double':
                depth = kornia.median_blur(depth,(self.kernel_size,self.kernel_size))
                depth = kornia.median_blur(depth,(self.kernel_size,self.kernel_size))
            elif self.depth_blur == 'box':
                depth = kornia.box_blur(depth,(self.kernel_size,self.kernel_size))
            elif self.depth_blur == 'gauss':
                depth = kornia.gaussian_blur2d(depth,(self.kernel_size,self.kernel_size),(1.5, 1.5))
        
            depth = torch.squeeze(depth, 0)
        return depth

def train_val_split(root_dir, num_val, num_test=0, seed=1):
    # num_val -> validation data for each class
    train_list = []
    val_list = []
    test_list = []

    for actionname in os.listdir(root_dir):
        folder_list = []
        if actionname in ['else', 'templete']:
            continue
        elif os.path.isfile(os.path.join(root_dir, actionname)):
            continue
        
        for foldername in os.listdir(os.path.join(root_dir, actionname)):
            folder_list.append(os.path.join(root_dir, actionname, foldername))
    
        random.seed(seed)
        val_sample = random.sample(folder_list, num_val)
    
        for i in val_sample:
            folder_list.remove(i)

        if num_test > 0:
            test_sample = random.sample(folder_list, num_test)

            for i in test_sample:
                folder_list.remove(i)
            
            test_list.extend(test_sample)

        train_list.extend(folder_list)
        val_list.extend(val_sample)
    
    train_list.sort()
    val_list.sort()
    test_list.sort()

    return train_list, val_list, test_list

class Dataset_Template(torch.utils.data.Dataset):

    def transform_rgb(self, rgb_image, idx):
        if self.img_trans != None:
            rgb_image = self.img_trans(rgb_image, self.seed + idx)
        if self.ToTensor != None:
            rgb_image = self.ToTensor(rgb_image)
        return rgb_image
    
    def transform_pos(self, pose):
        mask = np.where(np.isnan(pose), 0, 1)
        pose = np.where(np.isnan(pose), 0, pose)
        if self.ToTensor != None:
            pose = torch.tensor(pose, dtype=torch.float32)
            mask = torch.tensor(mask)
        return pose, mask

    def transform_pos2image(self, pos, json_data, image_size):
        uv = self.get_uv(pos, json_data['dist'], json_data['mtx'])
        pos_image = self.make_pos_image(image_size,uv)
        if self.ToTensor != None:
            pos_image = self.ToTensor(pos_image)
        uv = torch.tensor(uv)[:,:2]
        uv_mask = torch.where(torch.isnan(uv), 0., 1.)
        uv = torch.where(torch.isnan(uv), 0., uv)
        return pos_image, uv, uv_mask

    def load_json(self, json_path):
        # load json data
        with open(json_path) as f:
            data = f.read()
        data = json.loads(data)
        return data

    def make_pos_image(self,size,uv_data,r=3,one_channel=False):
        if not one_channel:
            poslist = []
            for u,v, _ in uv_data:
                pos_image_list = []
                pos_image = Image.new('L',size)
                draw = ImageDraw.Draw(pos_image)
                draw.ellipse((u-r, v-r, u+r, v+r), fill=(255), outline=(0))
                poslist.append(np.array(pos_image))
            return np.array(poslist).transpose(1,2,0)
        else:
            pos_image = Image.new('L',size)
            draw = ImageDraw.Draw(pos_image)
            for u,v, _ in uv_data:
                draw.ellipse((u-r, v-r, u+r, v+r), fill=(255), outline=(0))
            return pos_image

    def update_seed(self):
        # change seed. augumentation will be changed.
        self.seed += 1
    
    def view_calib_result(self, pil_image, uv_result):
        draw = ImageDraw.Draw(pil_image)
        # グラフの描画
        r = 3
        for u,v, _ in uv_result:
            draw.ellipse((u - r, v -r , u + r, v + r), fill=(255, 0, 0), outline=(0, 0, 0))
            # plt.plot(u, v, color='r', marker='o', markersize=2)
        
        return draw

class Softargmax_dataset(Dataset_Template):

    def __init__(self,cfg,save_dataset=False,mode='train'):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = cfg.DATASET.HMD.PATH

        self.get_future_image = True
        self.use_future_past_frame = True
        self.skip_until_move = True
        self.frame_interval = cfg.DATASET.HMD.FRAME_INTERVAL
        self.class_setting = cfg.DATASET.HMD.ACTION_CLASS
        self.size = None
        self.data_dict = None
        self.label_list = None
        self.seed = 0
        
        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None

        self.pred_len = cfg.PRED_LEN
        print('length of future is {} frame'.format(self.pred_len))
        
        if len(cfg.DATASET.HMD.TARGET_KEY) == 0:
            self.target_key = self.add_serial_data([], "hand", 1, 15, False, True)
            self.target_key = self.add_serial_data(self.target_key, "knife", 1 ,6, True)
        else:
            self.target_key = cfg.DATASET.HMD.TARGET_KEY

        # divide dataset
        train_list, val_list, _ = train_val_split(data_root_dir, 2)
        if mode == 'train':
            main_data_list = train_list
        elif (mode == 'val') or (mode == 'test'):
            main_data_list = val_list
        else:
            raise ValueError("Tawake")

        # create or load dataset
        self._json_file_name = 'HMD_{}_{}.json'.format(mode, self.pred_len)
        if (self._json_file_name not in os.listdir(data_root_dir)) or save_dataset:
            print('create dataset')
            self.add_data(main_data_list)
            print('save dataset')
            with open(os.path.join(data_root_dir,'HMD_{}_{}.json'.format(mode, self.pred_len)), 'w') as f:
                json.dump(self.data_dict,f,indent=4)
        else:
            print('load dataset')
            with open(os.path.join(data_root_dir,'HMD_{}_{}.json'.format(mode, self.pred_len))) as f:
                self.data_dict = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.data_dict['index_list'])

    def __getitem__(self, i):
        # get image
        # print('i:{}'.format(i))
        idx = self.data_dict['index_list'][i]
        input_dict = {}
        input_dict['current_index_for_datadict'] = idx
        dict_data1 = self.data_dict[str(idx)]
        max_next_index = dict_data1['end_index']
        start_index = dict_data1['start_index']

        if idx - self.frame_interval >= start_index:
            past_index = idx - self.frame_interval
        else:
            past_index = start_index
        index_list = [past_index, idx]

        for i in range(self.pred_len):
            next_index = idx + (self.frame_interval * (i+1))
            if next_index > max_next_index:
                next_index = max_next_index
            index_list.append(next_index)
        input_dict['index_list'] = torch.tensor(index_list)
        
        pred_len = min(self.pred_len, max_next_index - idx)
        input_dict['pred_len'] = pred_len

        for i,index in enumerate(index_list):
            dict_index = self.data_dict[str(index)]
            
            # get rgb image
            rgb_path = dict_index['rgb_image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, idx)
        
            # get depth image
            depth_path = dict_index['depth_image_path']
            depth_image = Image.open(depth_path)
            depth_image = self.transform_depth(depth_image)
        
            # get current pose
            pos = dict_index['pose'][2:]
        
            # get current pose image
            json_data = dict_index['camera_info']
            pos_image, uv, uv_mask = self.transform_pos2image(pos, json_data, image_size)

            pos,pos_mask = self.transform_pos(pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                depth_batch = torch.unsqueeze(depth_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(pos, 0)
                pose_mask_batch = torch.unsqueeze(pos_mask, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                depth_batch = torch.cat((depth_batch, torch.unsqueeze(depth_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(pos_mask, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        input_dict['rgb'] = rgb_batch
        input_dict['depth'] = depth_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['dist'] = json_data['dist']
        input_dict['mtx'] = torch.tensor(json_data['mtx'])
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(json_data['mtx']))

        # get action
        action = dict_data1['action']
        if self.class_setting == 'normal':
            onehot_action = torch.eye(len(self.data_dict['action_list']))[self.data_dict['action_list'].index(action)]
            input_dict['action'] = onehot_action
            input_dict['action_name'] = action
        elif self.class_setting == 'summary':
            if 'cut' in action:
                input_dict['action'] = torch.tensor([1.,0.])
                input_dict['action_name'] = 'cut'
            elif 'move' in action:
                input_dict['action'] = torch.tensor([0.,1.])
                input_dict['action_name'] = 'move'
        
        return input_dict

    def add_serial_data(self,main_data, targetname, start, end, marker=True, semicolon=True):
        if start == 0:
            raise ValueError("start num >= 1")

        for i in range(start, end + 1):
            for j in ["",".1",".2"]:
                if marker:
                    main_data.append("{}:Marker{}{}".format(targetname,i,j))
                elif semicolon:
                    main_data.append("{}:{}{}".format(targetname,i,j))
                else:
                    main_data.append("{}{}{}".format(targetname,i,j))
        return main_data
    
    def add_data(self, folder_list):
        # for data preparation
        self.data_dict = {}
        self.data_dict['index_list'] = []
        self.data_dict['action_list'] = []
        self.data_dict['sequence'] = []
        self.data_dict['sequence_index2action'] = []
        image_index = 0
        
        for folder_path in folder_list:
            start_index = image_index
            data_folder_path = os.path.abspath(os.path.join(folder_path,'output'))
            json_path = os.path.abspath(os.path.join(folder_path,'calibration_data','realsensecalib_result.json'))

            csv_list = os.listdir(data_folder_path)
            if len(csv_list) == 4:
                for i in csv_list:
                    if 'calib_pos_' in i:
                        csv_file_name = i
            else:
                csv_file_name = 'calib_pos.csv'
                
            csv_path = os.path.join(data_folder_path,csv_file_name)
            
            df1 = pd.read_csv(csv_path)

            df = df1.loc[:,:'Opti_time']
            for i in self.target_key:
                df2 = df1.loc[:,i]
                df = pd.concat([df,df2],axis=1)

            self.label_list = list(df)
            csv_data_list = df.values.tolist()

            image_name_list = df.loc[:,self.label_list[0]].values.tolist()
            
            hand_data_list = df.loc[:,'hand:1':'hand:15.2'].values.tolist()
            center_check = True
            for i in range(len(image_name_list)):
                if self.skip_until_move:
                    if i == 0:
                        hand_center = self.hand_centor(hand_data_list[i])
                        continue
                    elif center_check:
                        new_hand_center = self.hand_centor(hand_data_list[i])
                        diff = np.sum(np.abs(new_hand_center - hand_center)) / 3
                        hand_center = new_hand_center

                    if diff <= 0.001:
                        continue
                    else:
                        center_check = False

                _, decimal = "{0:.5f}".format(image_name_list[i]).split('.')
                integer, _ = str(image_name_list[i]).split('.')
                integer = integer.zfill(5)
                image_name = "{}.{}.png".format(integer, decimal)
                rgb_image_path = os.path.join(data_folder_path,'RGB',image_name)
                depth_image_path = os.path.join(data_folder_path,'DEPTH',image_name)
                posture = csv_data_list[i]
                action = self.get_action(rgb_image_path)
                if action not in self.data_dict['action_list']:
                    self.data_dict['action_list'].append(action)
                    
                self.data_dict[str(image_index)] = {}
                self.data_dict[str(image_index)]['pose'] = posture
                self.data_dict[str(image_index)]['rgb_image_path'] = rgb_image_path
                self.data_dict[str(image_index)]['depth_image_path'] = depth_image_path
                self.data_dict[str(image_index)]['camera_info'] = self.load_json(json_path)
                self.data_dict[str(image_index)]['action'] = action

                image_index += 1

            for i in range(start_index, image_index - self.pred_len):
                self.data_dict[str(i)]['start_index'] = start_index
                self.data_dict[str(i)]['end_index'] = image_index - 1
                self.data_dict['index_list'].append(i)

            self.data_dict['sequence'].append([start_index, image_index - 1])
            self.data_dict['sequence_index2action'].append(action)

    def get_action(self, path):
        start_index = path.find('mocap_data/') + len('mocap_data/')
        path = path[start_index:]
        action = path[:path.find('/')]
        return action
    
    def transform_depth(self, depth_image):
        # change 1000
        if self.ToTensor != None:
            depth_image = torch.unsqueeze(torch.from_numpy(np.array(depth_image, np.int32)) / 1000.0, 0)
            # depth_image = self.ToTensor(depth_image) / 1000.0
            #depth_image = self.ToTensor(depth_image)
            #depth_image = depth_image / 1000.0
        if self.depth_trans != None:
            depth_image = self.depth_trans(depth_image)
        return depth_image

    def get_uv(self, pos_data, dist_vec, intrinsic_matrix):
        # transfer position data(based on motive coordinate) to camera coordinate
        split = self.split_posdata(pos_data)
        uv_result_list = []
        for ins in split:
            ins = np.array(ins)
            ins = ins / ins[2]
            mod_ins = self.modify_distortion(ins,dist_vec)
            uv_result = np.dot(intrinsic_matrix,mod_ins)
            uv_result_list.append(uv_result)
        return uv_result_list
    
    def split_posdata(self,pos_data):
        # to split all of position data into each position data
        assert len(pos_data) % 3 == 0, 'dimension of position data is wrong'
        iteration = int(len(pos_data) / 3)
        output_list = []
        for i in range(iteration):
            output_list.append(pos_data[3*i:3*(i+1)])

        return output_list
    
    def hand_centor(self,pos):
        pos_list = self.split_posdata(pos)
        center_pos = np.zeros(3)
        for i in pos_list:
            center_pos = center_pos + np.array(i)
        center_pos = center_pos / len(pos_list)
        return center_pos
    
    def modify_distortion(self,vec,dist_vec):
        # modify distortion
        x,y,_ = vec
        r2 = (x*x) + (y*y)
        r4 = r2*r2
        r6 = r4*r2
        dist = dist_vec[0]
        mod_x = x*(1+(dist[0]*r2)+(dist[1]*r4)+(dist[4]*r6)) + 2*dist[2]*x*y + dist[3]*(r2+(2*x*x))
        mod_y = y*(1+(dist[0]*r2)+(dist[1]*r4)+(dist[4]*r6)) + 2*dist[3]*x*y + dist[2]*(r2+(2*y*y))
        ### TODO confirm this effect ###
        return [x,y,1]

    def get_image_size(self):
        dict_data = self.data_dict['0']
        img_path = dict_data['rgb_image_path']
        img = Image.open(img_path)
        w, h = img.size
        self.size = (h , w)

class Softargmax_dataset_VP(Softargmax_dataset):

    def __init__(self,cfg,save_dataset=False,mode='train',random_len=0):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = cfg.DATASET.HMD.PATH

        self.get_future_image = True
        self.use_future_past_frame = True
        self.skip_until_move = True
        self.size = None
        self.data_dict = None
        self.label_list = None
        self.seed = 0
        
        self.frame_interval = cfg.DATASET.HMD.FRAME_INTERVAL

        if random_len == 0:
            self.random_len = cfg.DATASET.HMD.RANDOM_LEN
        else:
            self.random_len = random_len 

        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None

        self.pred_len = 1
        print('length of future is {} frame'.format(self.pred_len))
        
        if len(cfg.DATASET.HMD.TARGET_KEY) == 0:
            self.target_key = self.add_serial_data([], "hand", 1, 15, False, True)
            self.target_key = self.add_serial_data(self.target_key, "knife", 1 ,6, True)
        else:
            self.target_key = cfg.DATASET.HMD.TARGET_KEY
        
        self.class_setting = cfg.DATASET.HMD.ACTION_CLASS

        # divide dataset
        train_list, val_list, _ = train_val_split(data_root_dir, 2)
        if mode == 'train':
            main_data_list = train_list
        elif (mode == 'val') or (mode == 'test'):
            main_data_list = val_list
        else:
            raise ValueError("Tawake")
        
        # crate or load dataset
        self._json_file_name = 'HMD_{}_{}_VP.json'.format(mode, self.pred_len)
        if (self._json_file_name not in os.listdir(data_root_dir)) or save_dataset:
            print('create dataset')
            self.add_data(main_data_list)
            print('save dataset')
            with open(os.path.join(data_root_dir,'HMD_{}_{}_VP.json'.format(mode, self.pred_len)), 'w') as f:
                json.dump(self.data_dict,f,indent=4)
        else:
            print('load dataset')
            with open(os.path.join(data_root_dir,'HMD_{}_{}_VP.json'.format(mode, self.pred_len))) as f:
                self.data_dict = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, i):
        # get image
        # print('i:{}'.format(i))
        idx = self.data_dict['index_list'][i]
        input_dict = {}
        input_dict['current_index_for_datadict'] = idx
        dict_data1 = self.data_dict[str(idx)]
        end_index = dict_data1['end_index']
        start_index = dict_data1['start_index']

        random_max = self.frame_interval * self.random_len
        
        if idx - random_max >= start_index:
            past_index = idx - random.randint(1, random_max)
        elif idx == start_index:
            past_index = start_index
        else:
            past_index = idx - random.randint(1, idx - start_index)
            if past_index < start_index:
                raise ValueError('BAKATARE')
        index_list = [past_index, idx]


        max_future_range = min(end_index - idx, 2*random_max)
        
        half_range = math.floor(max_future_range/2)
        target_index = idx + random.randint(1, half_range)
        index_list.append(target_index)
        
        future_range = max_future_range - half_range
        future_index = target_index + random.randint(1,future_range)
        if future_index > end_index:
            raise ValueError('BAKATARE')
        index_list.append(future_index)

        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            dict_index = self.data_dict[str(index)]
            
            # get rgb image
            rgb_path = dict_index['rgb_image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, idx)
        
            # get depth image
            depth_path = dict_index['depth_image_path']
            depth_image = Image.open(depth_path)
            depth_image = self.transform_depth(depth_image)
        
            # get current pose
            pos = dict_index['pose'][2:]
        
            # get current pose image
            json_data = dict_index['camera_info']
            pos_image, uv, uv_mask = self.transform_pos2image(pos, json_data, image_size)

            pos,pos_mask = self.transform_pos(pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                depth_batch = torch.unsqueeze(depth_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(pos, 0)
                pose_mask_batch = torch.unsqueeze(pos_mask, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                depth_batch = torch.cat((depth_batch, torch.unsqueeze(depth_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(pos_mask, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        # get action
        action = dict_data1['action']
        if self.class_setting == 'normal':
            onehot_action = torch.eye(len(self.data_dict['action_list']))[self.data_dict['action_list'].index(action)]
            input_dict['action'] = onehot_action
            input_dict['action_name'] = action
        elif self.class_setting == 'summary':
            if 'cut' in action:
                input_dict['action'] = torch.tensor([1.,0.])
                input_dict['action_name'] = 'cut'
            elif 'move' in action:
                input_dict['action'] = torch.tensor([0.,1.])
                input_dict['action_name'] = 'move'

        input_dict['rgb'] = rgb_batch
        input_dict['depth'] = depth_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['dist'] = json_data['dist']
        input_dict['mtx'] = torch.tensor(json_data['mtx'])
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(json_data['mtx']))
        
        return input_dict

    def add_data(self, folder_list):
        # for data preparation
        self.data_dict = {}
        self.data_dict['index_list'] = []
        self.data_dict['action_list'] = []
        self.data_dict['sequence'] = []
        self.data_dict['sequence_index2action'] = []
        image_index = 0
        
        for folder_path in folder_list:
            start_index = image_index
            data_folder_path = os.path.abspath(os.path.join(folder_path,'output'))
            json_path = os.path.abspath(os.path.join(folder_path,'calibration_data','realsensecalib_result.json'))

            csv_list = os.listdir(data_folder_path)
            if len(csv_list) == 4:
                for i in csv_list:
                    if 'calib_pos_' in i:
                        csv_file_name = i
            else:
                csv_file_name = 'calib_pos.csv'
                
            csv_path = os.path.join(data_folder_path,csv_file_name)
            
            df1 = pd.read_csv(csv_path)

            df = df1.loc[:,:'Opti_time']
            for i in self.target_key:
                df2 = df1.loc[:,i]
                df = pd.concat([df,df2],axis=1)

            self.label_list = list(df)
            csv_data_list = df.values.tolist()

            image_name_list = df.loc[:,self.label_list[0]].values.tolist()
            
            hand_data_list = df.loc[:,'hand:1':'hand:15.2'].values.tolist()
            center_check = True
            for i in range(len(image_name_list)):
                if self.skip_until_move:
                    if i == 0:
                        hand_center = self.hand_centor(hand_data_list[i])
                        continue
                    elif center_check:
                        new_hand_center = self.hand_centor(hand_data_list[i])
                        diff = np.sum(np.abs(new_hand_center - hand_center)) / 3
                        hand_center = new_hand_center

                    if diff <= 0.001:
                        continue
                    else:
                        center_check = False

                _, decimal = "{0:.5f}".format(image_name_list[i]).split('.')
                integer, _ = str(image_name_list[i]).split('.')
                integer = integer.zfill(5)
                image_name = "{}.{}.png".format(integer, decimal)
                rgb_image_path = os.path.join(data_folder_path,'RGB',image_name)
                depth_image_path = os.path.join(data_folder_path,'DEPTH',image_name)
                posture = csv_data_list[i]
                action = self.get_action(rgb_image_path)
                if action not in self.data_dict['action_list']:
                    self.data_dict['action_list'].append(action)
                    
                self.data_dict[str(image_index)] = {}
                self.data_dict[str(image_index)]['pose'] = posture
                self.data_dict[str(image_index)]['rgb_image_path'] = rgb_image_path
                self.data_dict[str(image_index)]['depth_image_path'] = depth_image_path
                self.data_dict[str(image_index)]['camera_info'] = self.load_json(json_path)
                self.data_dict[str(image_index)]['action'] = action
                image_index += 1

            for i in range(start_index, image_index - self.pred_len - 1):
                self.data_dict[str(i)]['start_index'] = start_index
                self.data_dict[str(i)]['end_index'] = image_index - 1
                self.data_dict['index_list'].append(i)    

class Softargmax_dataset_test(Softargmax_dataset):

    def __init__(self,cfg,save_dataset=False,mode='test'):
        data_root_dir = cfg.DATASET.HMD.PATH

        self.get_future_image = True
        self.use_future_past_frame = True
        self.skip_until_move = True
        self.size = None
        self.data_dict = None
        self.label_list = None
        self.img_trans = None
        self.depth_trans = None
        self.seed = 0
        
        self.pred_len = 1
        self.frame_interval = cfg.DATASET.HMD.FRAME_INTERVAL
        print('length of future is {} frame'.format(1))
        
        if len(cfg.DATASET.HMD.TARGET_KEY) == 0:
            self.target_key = self.add_serial_data([], "hand", 1, 15, False, True)
            self.target_key = self.add_serial_data(self.target_key, "knife", 1 ,6, True)
        else:
            self.target_key = cfg.DATASET.HMD.TARGET_KEY
        
        self.class_setting = cfg.DATASET.HMD.ACTION_CLASS

        # divide dataset
        train_list, val_list, _ = train_val_split(data_root_dir, 2)
        if mode == 'train':
            main_data_list = train_list
        elif (mode == 'val') or (mode == 'test'):
            main_data_list = val_list
        else:
            raise ValueError("Tawake")
        
        # create and load dataset
        self._json_file_name = 'HMD_{}_{}.json'.format(mode,self.pred_len)
        if (self._json_file_name not in os.listdir(data_root_dir)) or save_dataset:
            print('create dataset')
            self.add_data(main_data_list)
            print('save dataset')
            with open(os.path.join(data_root_dir,'HMD_VP_{}_{}.json'.format(mode, self.pred_len)), 'w') as f:
                json.dump(self.data_dict,f,indent=4)
        else:
            print('load dataset')
            with open(os.path.join(data_root_dir,'HMD_VP_{}_{}.json'.format(mode,self.pred_len))) as f:
                self.data_dict = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.data_dict['sequence'])

    def __getitem__(self, i):
        # get image
        # print('i:{}'.format(i))
        start_index, end_index = self.data_dict['sequence'][i]
        input_dict = {}
        input_dict['current_index_for_datadict'] = start_index
        dict_data1 = self.data_dict[str(start_index)]
        end_index = dict_data1['end_index']


        index_list = [start_index, start_index]
        index = start_index + self.frame_interval
        while index < end_index:
            index_list.append(index)
            index += self.frame_interval
            
        if index == end_index:
            pass
        else:
            index_list.append(end_index)

        input_dict['index_list'] = torch.tensor(index_list)
        
        pred_len = min(self.pred_len, end_index - start_index)
        input_dict['pred_len'] = pred_len

        for i,index in enumerate(index_list):
            dict_index = self.data_dict[str(index)]
            
            # get rgb image
            rgb_path = dict_index['rgb_image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, start_index)
        
            # get depth image
            depth_path = dict_index['depth_image_path']
            depth_image = Image.open(depth_path)
            depth_image = self.transform_depth(depth_image)
        
            # get current pose
            pos = dict_index['pose'][2:]
        
            # get current pose image
            json_data = dict_index['camera_info']
            pos_image, uv, uv_mask = self.transform_pos2image(pos, json_data, image_size)

            pos,pos_mask = self.transform_pos(pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                depth_batch = torch.unsqueeze(depth_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(pos, 0)
                pose_mask_batch = torch.unsqueeze(pos_mask, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                depth_batch = torch.cat((depth_batch, torch.unsqueeze(depth_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(pos_mask, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        input_dict['rgb'] = rgb_batch
        input_dict['depth'] = depth_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['dist'] = json_data['dist']
        input_dict['mtx'] = torch.tensor(json_data['mtx'])
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(json_data['mtx']))

        # get action
        action = dict_data1['action']
        if self.class_setting == 'normal':
            onehot_action = torch.eye(len(self.data_dict['action_list']))[self.data_dict['action_list'].index(action)]
            input_dict['action'] = onehot_action
            input_dict['action_name'] = action
        elif self.class_setting == 'summary':
            if 'cut' in action:
                input_dict['action'] = torch.tensor([1.,0.])
                input_dict['action_name'] = 'cut'
            elif 'move' in action:
                input_dict['action'] = torch.tensor([0.,1.])
                input_dict['action_name'] = 'move'
        
        return input_dict

class RLBench_dataset(Dataset_Template):

    def __init__(self,cfg,save_dataset=False,mode='train'):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None

        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None
        
        self.size = None
        self.numpose = None # the number of key point
        
        self.pred_len = cfg.PRED_LEN
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)
        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        index = self.index_list[data_index]
        data_dict = self.data_list[index]
        
        start_index = data_dict['start_index']
        end_index = data_dict['end_index']
        input_dict = {}

        if index - 1 >= start_index:
            past_index = index - 1
        else:
            past_index = start_index
        index_list = [past_index, index]

        for i in range(self.pred_len):
            next_index = index + (i+1)
            if next_index > end_index:
                next_index = end_index
            index_list.append(next_index)
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = data_dict['image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['intrinsic_matrix']
            camera_extrinsic = pickle_data['extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        
        return input_dict

    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list
        data_list = [data_dict * n]
        data_dict = {
        'image_path': path
        'pickle_path': path
        'end_index': index of data where task will finish
        'start_index': index of date where task start
        }
        """
        # for data preparation
        self.data_list = []
        self.index_list = []
        self.sequence_index_list = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        for task_name in task_list:
            if 'all' in cfg.DATASET.RLBENCH.TASK_LIST:
                pass
            elif task_name not in cfg.DATASET.RLBENCH.TASK_LIST:
                continue

            if task_name == 'json':
                continue

            if 'RL_Becnh_dataset' in task_name:
                continue
            
            task_path = os.path.join(folder_path, task_name)
            
            sequence_list = os.listdir(task_path)
            sequence_list.sort()
            for sequence_index in sequence_list:
                start_index = index
                image_folder_path = os.path.join(task_path, sequence_index, 'image')
                pickle_folder_path = os.path.join(task_path, sequence_index, 'pickle')
                
                image_list = os.listdir(image_folder_path)
                image_list.sort()
                pickle_list = os.listdir(pickle_folder_path)
                pickle_list.sort()
                for image_name, pickle_name in zip(image_list, pickle_list):
                    data_dict = {}
                    data_dict['image_path'] = os.path.join(image_folder_path, image_name)
                    data_dict['pickle_path'] = os.path.join(pickle_folder_path, pickle_name)
                    data_dict['start_index'] = start_index
                    data_dict['end_index'] = start_index + len(image_list) - 1
                    
                    self.data_list.append(data_dict)
                    if index <= start_index + (len(image_list) - 1) - self.pred_len:
                        self.index_list.append(index)
                        
                    index += 1
            
                self.sequence_index_list.append([start_index, start_index + len(image_list) - 1])

    def transform_depth(self, depth_image):
        if self.ToTensor != None:
            depth_image = torch.unsqueeze(torch.tensor(np.array(depth_image), dtype=torch.float), 0)
        if self.depth_trans:
            depth_image = self.depth_trans(depth_image)
        return depth_image

    def transform_pos2image(self, pos, intrinsic, image_size):
        uv = self.get_uv(pos, intrinsic)
        pos_image = self.make_pos_image(image_size,uv)
        if self.ToTensor != None:
            pos_image = self.ToTensor(pos_image)
        uv = torch.tensor(uv)[:,:2]
        uv_mask = torch.where(torch.isnan(uv), 0., 1.)
        uv = torch.where(torch.isnan(uv), 0., uv)
        return pos_image, uv, uv_mask

    def get_gripper(self,pickle_data):
        gripper_pos_WorldCor = np.append(pickle_data['gripper_pose'][:3], 1)
        gripper_matrix_WorldCor = np.append(pickle_data['gripper_matrix'],[0,0,0,1]).reshape([4, 4])
        gripper_open = pickle_data['gripper_open']
        
        world2camera_matrix = pickle_data['extrinsic_matrix']
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.dot(camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.dot(camera2world_matrix, gripper_matrix_WorldCor)
        
        return torch.tensor(gripper_pose_CamCor[:3], dtype=torch.float), torch.tensor(gripper_matrix_CamCor[:3,:3], dtype=torch.float), torch.tensor(gripper_open, dtype=torch.float)
    
    def get_uv(self, pos_data, intrinsic_matrix):
        # transfer position data(based on motive coordinate) to camera coordinate
        pos_data = np.array(pos_data) # x,y,z
        pos_data = pos_data / pos_data[2] # u,v,1
        uv_result = np.dot(intrinsic_matrix, pos_data)
        return [uv_result]
        
    def get_image_size(self):
        dict_data = self.data_list[0]
        img_path = dict_data['image_path']
        img = Image.open(img_path)
        w, h = img.size
        self.size = (h , w)

    @staticmethod
    def get_task_names(task_list):
        task_name = ""
        for task in task_list:
            task_name = task_name + "_" + task
        return task_name

class RLBench_dataset_VP(RLBench_dataset):

    def __init__(self,cfg,save_dataset=False,mode='train',random_len=0):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, mode)

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None
        self.size = None
        self.numpose = None # the number of key point
        
        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None

        self.root_dir = data_root_dir
        
        self.pred_len = 1
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0

        if random_len == 0:
            self.random_len = cfg.DATASET.RLBENCH.RANDOM_LEN
        else:
            self.random_len = random_len 
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_VP_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)

        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        index = self.index_list[data_index]
        data_dict = self.data_list[index]
        
        start_index = data_dict['start_index']
        end_index = data_dict['end_index']
        input_dict = {}

        past_random_max = min(index - start_index, self.random_len)
        if past_random_max == 0:
            past_index = start_index
        else:
            diff = random.randint(1,past_random_max)
            past_index = index - diff
        index_list = [past_index, index]

        max_future_range = min(end_index - index, 2*self.random_len)
        
        half_range = math.floor(max_future_range/2)
        target_index = index + random.randint(1, half_range)
        index_list.append(target_index)
        
        future_range = max_future_range - half_range
        future_index = target_index + random.randint(1,future_range)
        index_list.append(future_index)
        
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            if (index < start_index) or (index > end_index):
                raise ValueError('hoge')
                
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = data_dict['image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['intrinsic_matrix']
            camera_extrinsic = pickle_data['extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        relative_path = os.path.relpath(rgb_path, self.root_dir)
        task_name = relative_path[:relative_path.find('/')]

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        input_dict['action_name'] = task_name
        
        return input_dict

    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list
        data_list = [data_dict * n]
        data_dict = {
        'image_path': path
        'pickle_path': path
        'end_index': index of data where task will finish
        'start_index': index of date where task start
        }
        """
        # for data preparation
        self.data_list = []
        self.index_list = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        for task_name in task_list:
            if 'all' in cfg.DATASET.RLBENCH.TASK_LIST:
                pass
            elif task_name not in cfg.DATASET.RLBENCH.TASK_LIST:
                continue

            if task_name == 'json':
                continue
            
            if 'RL_Becnh_dataset' in task_name:
                continue
            
            task_path = os.path.join(folder_path, task_name)
            
            sequence_list = os.listdir(task_path)
            sequence_list.sort()
            for sequence_index in sequence_list:
                start_index = index
                image_folder_path = os.path.join(task_path, sequence_index, 'image')
                pickle_folder_path = os.path.join(task_path, sequence_index, 'pickle')
                
                image_list = os.listdir(image_folder_path)
                image_list.sort()
                pickle_list = os.listdir(pickle_folder_path)
                pickle_list.sort()
                for image_name, pickle_name in zip(image_list, pickle_list):
                    data_dict = {}
                    data_dict['image_path'] = os.path.join(image_folder_path, image_name)
                    data_dict['pickle_path'] = os.path.join(pickle_folder_path, pickle_name)
                    data_dict['start_index'] = start_index
                    data_dict['end_index'] = start_index + len(image_list) - 1
                    
                    self.data_list.append(data_dict)
                    if index <= start_index + (len(image_list) - 1) - (self.pred_len + 1):
                        self.index_list.append(index)
                        
                    index += 1

class RLBench_dataset_test(RLBench_dataset):

    def __init__(self,cfg,save_dataset=False,mode='test'):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH, 'val')

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None
        self.size = None
        self.numpose = None # the number of key point
        self.img_trans = None
        self.depth_trans = None

        self.root_dir = data_root_dir
        self.pred_len = 1
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_Test_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)

        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list, self.sequence_index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list, self.sequence_index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.sequence_index_list)

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        start_index, end_index = self.sequence_index_list[data_index]
        index = start_index
        
        input_dict = {}

        index_list = [index, index]

        for i in range(end_index - start_index):
            next_index = index + (i+1)
            if next_index > end_index:
                next_index = end_index
            index_list.append(next_index)
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = data_dict['image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['intrinsic_matrix']
            camera_extrinsic = pickle_data['extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        relative_path = os.path.relpath(rgb_path, self.root_dir)
        task_name = relative_path[:relative_path.find('/')]

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        input_dict['action_name'] = task_name

        return input_dict

# temporally add RLBench dataset2
# remove RLBench_dataset and replaced with this
class RLBench_dataset2(RLBench_dataset):

    def __init__(self,cfg,save_dataset=False,mode='train'):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH2, mode)

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None

        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None
        
        self.size = None
        self.numpose = None # the number of key point
        
        self.pred_len = cfg.PRED_LEN
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)
        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        index = self.index_list[data_index]
        data_dict = self.data_list[index]
        
        start_index = data_dict['start_index']
        end_index = data_dict['end_index']
        input_dict = {}

        if index - 1 >= start_index:
            past_index = index - 1
        else:
            past_index = start_index
        index_list = [past_index, index]

        for i in range(self.pred_len):
            next_index = index + (i+1)
            if next_index > end_index:
                next_index = end_index
            index_list.append(next_index)
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = data_dict['image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['front_intrinsic_matrix']
            camera_extrinsic = pickle_data['front_extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        
        return input_dict
    
    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list
        data_list = [data_dict * n]
        data_dict = {
        'image_path': path
        'pickle_path': path
        'end_index': index of data where task will finish
        'start_index': index of date where task start
        }
        """
        # for data preparation
        self.data_list = []
        self.index_list = []
        self.sequence_index_list = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        for task_name in task_list:
            if 'all' in cfg.DATASET.RLBENCH.TASK_LIST:
                pass
            elif task_name not in cfg.DATASET.RLBENCH.TASK_LIST:
                continue

            if task_name == 'json':
                continue

            if 'RL_Becnh_dataset' in task_name:
                continue
            
            task_path = os.path.join(folder_path, task_name)
            
            sequence_list = os.listdir(task_path)
            sequence_list.sort()
            for sequence_index in sequence_list:
                start_index = index
                image_folder_path = os.path.join(task_path, sequence_index, 'image')
                pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
                
                image_list = os.listdir(image_folder_path)
                image_list.sort()
                pickle_list = os.listdir(pickle_folder_path)
                pickle_list.sort()
                for image_name, pickle_name in zip(image_list, pickle_list):
                    data_dict = {}
                    data_dict['image_path'] = os.path.join(image_folder_path, image_name)
                    data_dict['pickle_path'] = os.path.join(pickle_folder_path, pickle_name)
                    data_dict['start_index'] = start_index
                    data_dict['end_index'] = start_index + len(image_list) - 1
                    
                    self.data_list.append(data_dict)
                    if index <= start_index + (len(image_list) - 1) - self.pred_len:
                        self.index_list.append(index)
                        
                    index += 1
            
                self.sequence_index_list.append([start_index, start_index + len(image_list) - 1])

    def get_gripper(self, pickle_data):
        gripper_pos_WorldCor = np.append(pickle_data['gripper_pose'][:3], 1)
        gripper_matrix_WorldCor = pickle_data['gripper_matrix']
        gripper_open = pickle_data['gripper_open']
        
        world2camera_matrix = pickle_data['front_extrinsic_matrix']
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.dot(camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.dot(camera2world_matrix, gripper_matrix_WorldCor)
        
        return torch.tensor(gripper_pose_CamCor[:3], dtype=torch.float), torch.tensor(gripper_matrix_CamCor[:3,:3], dtype=torch.float), torch.tensor(gripper_open, dtype=torch.float)

class RLBench_dataset2_VP(RLBench_dataset2):

    def __init__(self,cfg,save_dataset=False,mode='train',random_len=0):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH2, mode)

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None
        self.size = None
        self.numpose = None # the number of key point
        
        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None

        self.root_dir = data_root_dir
        
        self.pred_len = 1
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0

        if random_len == 0:
            self.random_len = cfg.DATASET.RLBENCH.RANDOM_LEN
        else:
            self.random_len = random_len 
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_VP_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)

        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        index = self.index_list[data_index]
        data_dict = self.data_list[index]
        
        start_index = data_dict['start_index']
        end_index = data_dict['end_index']
        input_dict = {}

        past_random_max = min(index - start_index, self.random_len)
        if past_random_max == 0:
            past_index = start_index
        else:
            diff = random.randint(1,past_random_max)
            past_index = index - diff
        index_list = [past_index, index]

        max_future_range = min(end_index - index, 2*self.random_len)
        
        half_range = math.floor(max_future_range/2)
        target_index = index + random.randint(1, half_range)
        index_list.append(target_index)
        
        future_range = max_future_range - half_range
        future_index = target_index + random.randint(1,future_range)
        index_list.append(future_index)
        
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            if (index < start_index) or (index > end_index):
                raise ValueError('hoge')
                
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = data_dict['image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['front_intrinsic_matrix']
            camera_extrinsic = pickle_data['front_extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        relative_path = os.path.relpath(rgb_path, self.root_dir)
        task_name = relative_path[:relative_path.find('/')]

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        input_dict['action_name'] = task_name
        
        return input_dict

    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list
        data_list = [data_dict * n]
        data_dict = {
        'image_path': path
        'pickle_path': path
        'end_index': index of data where task will finish
        'start_index': index of date where task start
        }
        """
        # for data preparation
        self.data_list = []
        self.index_list = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        for task_name in task_list:
            if 'all' in cfg.DATASET.RLBENCH.TASK_LIST:
                pass
            elif task_name not in cfg.DATASET.RLBENCH.TASK_LIST:
                continue

            if task_name == 'json':
                continue
            
            if 'RL_Becnh_dataset' in task_name:
                continue
            
            task_path = os.path.join(folder_path, task_name)
            
            sequence_list = os.listdir(task_path)
            sequence_list.sort()
            for sequence_index in sequence_list:
                start_index = index
                image_folder_path = os.path.join(task_path, sequence_index, 'image')
                pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
                
                image_list = os.listdir(image_folder_path)
                image_list.sort()
                pickle_list = os.listdir(pickle_folder_path)
                pickle_list.sort()
                for image_name, pickle_name in zip(image_list, pickle_list):
                    data_dict = {}
                    data_dict['image_path'] = os.path.join(image_folder_path, image_name)
                    data_dict['pickle_path'] = os.path.join(pickle_folder_path, pickle_name)
                    data_dict['start_index'] = start_index
                    data_dict['end_index'] = start_index + len(image_list) - 1
                    
                    self.data_list.append(data_dict)
                    if index <= start_index + (len(image_list) - 1) - (self.pred_len + 1):
                        self.index_list.append(index)
                        
                    index += 1

class RLBench_dataset2_test(RLBench_dataset2):

    def __init__(self,cfg,save_dataset=False,mode='test'):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH2, 'val')

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None
        self.size = None
        self.numpose = None # the number of key point
        self.img_trans = None
        self.depth_trans = None

        self.root_dir = data_root_dir
        self.pred_len = 1
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_Test_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)

        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list, self.sequence_index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list, self.sequence_index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.sequence_index_list)

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        start_index, end_index = self.sequence_index_list[data_index]
        index = start_index
        
        input_dict = {}

        index_list = [index, index]

        for i in range(end_index - start_index):
            next_index = index + (i+1)
            if next_index > end_index:
                next_index = end_index
            index_list.append(next_index)
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = data_dict['image_path']
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data.front_intrinsic_matrix
            camera_extrinsic = pickle_data.front_extrinsic_matrix # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)

        relative_path = os.path.relpath(rgb_path, self.root_dir)
        task_name = relative_path[:relative_path.find('/')]

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        input_dict['action_name'] = task_name

        return input_dict

# temporally add RLBench dataset2
# remove RLBench_dataset and replaced with this
class RLBench_dataset3(RLBench_dataset):

    def __init__(self,cfg,save_dataset=False,mode='train'):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH3, mode)

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None
        self.size = None
        self.numpose = None # the number of key point

        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None
        
        self.use_front_depth = cfg.HOURGLASS.INPUT_DEPTH
        self.pred_trajectory = cfg.HOURGLASS.PRED_TRAJECTORY
        self.pred_len = cfg.PRED_LEN
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)
        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        index = self.index_list[data_index]
        data_dict = self.data_list[index]
        
        start_index = data_dict['start_index']
        end_index = data_dict['end_index']
        input_dict = {}

        if index - 1 >= start_index:
            past_index = index - 1
        else:
            past_index = start_index
        index_list = [past_index, index]

        for i in range(self.pred_len):
            next_index = index + (i+1)
            if next_index > end_index:
                next_index = end_index
            index_list.append(next_index)
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = os.path.join(data_dict['image_dir'], "front_rgb_{}.png".format(data_dict['filename']))
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get depth image
            if self.use_front_depth:
                depth_path = os.path.join(data_dict['image_dir'], "front_depth_{}.pickle".format(data_dict['filename']))
                with open(depth_path, 'rb') as f:
                    depth_image = pickle.load(f)
                depth_image = self.transform_depth(depth_image)

            # get trajectory image
            if self.pred_trajectory:
                trajectory_path = os.path.join(data_dict['image_dir'][:-5], "additional_info", "goal_trajectory_{}.png".format(data_dict['filename']))
                trajectory_image = Image.open(trajectory_path)
                trajectory_image = self.ToTensor(trajectory_image)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['front_intrinsic_matrix']
            camera_extrinsic = pickle_data['front_extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
                if self.use_front_depth:
                    depth_batch = torch.unsqueeze(depth_image, 0)
                if self.pred_trajectory:
                    trajectory_batch = torch.unsqueeze(trajectory_image, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)
                if self.use_front_depth:
                    depth_batch = torch.cat((depth_batch, torch.unsqueeze(depth_image, 0)), 0)
                if self.pred_trajectory:
                    trajectory_batch = torch.cat((trajectory_batch, torch.unsqueeze(trajectory_image, 0)), 0)

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        if self.use_front_depth:
            input_dict['depth'] = depth_batch
        if self.pred_trajectory:
            input_dict['trajectory'] = trajectory_batch

        return input_dict
    
    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list
        data_list = [data_dict * n]
        data_dict = {
        'image_path': path
        'pickle_path': path
        'end_index': index of data where task will finish
        'start_index': index of date where task start
        }
        """
        # for data preparation
        self.data_list = []
        self.index_list = []
        self.sequence_index_list = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        for task_name in task_list:
            if 'all' in cfg.DATASET.RLBENCH.TASK_LIST:
                pass
            elif task_name not in cfg.DATASET.RLBENCH.TASK_LIST:
                continue

            if task_name == 'json':
                continue

            if 'RL_Becnh_dataset' in task_name:
                continue
            
            task_path = os.path.join(folder_path, task_name)
            
            sequence_list = os.listdir(task_path)
            sequence_list.sort()
            for sequence_index in sequence_list:
                start_index = index
                image_folder_path = os.path.join(task_path, sequence_index, 'image')
                pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
                pickle_list = os.listdir(pickle_folder_path)
                pickle_list.sort()
                for pickle_name in pickle_list:
                    head, ext = os.path.splitext(pickle_name)
                    data_dict = {}
                    data_dict['image_dir'] = image_folder_path
                    data_dict['filename'] = os.path.join(head)
                    data_dict['pickle_path'] = os.path.join(pickle_folder_path, pickle_name)
                    data_dict['start_index'] = start_index
                    data_dict['end_index'] = start_index + len(pickle_list) - 1
                    
                    self.data_list.append(data_dict)
                    if index <= start_index + (len(pickle_list) - 1) - self.pred_len:
                        self.index_list.append(index)
                        
                    index += 1
            
                self.sequence_index_list.append([start_index, start_index + len(pickle_list) - 1])

    def get_gripper(self, pickle_data):
        gripper_pos_WorldCor = np.append(pickle_data['gripper_pose'][:3], 1)
        gripper_matrix_WorldCor = pickle_data['gripper_matrix']
        gripper_open = pickle_data['gripper_open']
        
        world2camera_matrix = pickle_data['front_extrinsic_matrix']
        camera2world_matrix = np.linalg.inv(world2camera_matrix)
        
        gripper_pose_CamCor = np.dot(camera2world_matrix, gripper_pos_WorldCor)
        gripper_matrix_CamCor = np.dot(camera2world_matrix, gripper_matrix_WorldCor)
        
        return torch.tensor(gripper_pose_CamCor[:3], dtype=torch.float), torch.tensor(gripper_matrix_CamCor[:3,:3], dtype=torch.float), torch.tensor(gripper_open, dtype=torch.float)
    
    def get_image_size(self):
        data_dict = self.data_list[0]
        img_path = os.path.join(data_dict['image_dir'], "front_rgb_{}.png".format(data_dict['filename']))
        img = Image.open(img_path)
        w, h = img.size
        self.size = (h , w)

class RLBench_dataset3_VP(RLBench_dataset3):

    def __init__(self,cfg,save_dataset=False,mode='train',random_len=0):
        """
        output: image_t,posture_t,image_t+1,posture_t+1
        
        # variable
        data_root_dir: path to root directory of data
        target_key: key of data in motive csv data. (e.g. hand)
        img_trans: transform(torch.transform) list
        seed: seed for data augmentation
        """
        data_root_dir = os.path.join(cfg.DATASET.RLBENCH.PATH3, mode)

        self.data_list = None
        self.index_list = None
        self.sequence_index_list = None
        self.size = None
        self.numpose = None # the number of key point
        
        # augmentation
        if (cfg.DATASET.RGB_AUGMENTATION) and (mode == 'train'):
            self.img_trans = imageaug_full_transform(cfg)
        else:
            self.img_trans = None
        
        if (cfg.DATASET.DEPTH_AUGMENTATION) and (mode == 'train'):
            self.depth_trans = depth_aug(cfg)
        else:
            self.depth_trans = None

        self.root_dir = data_root_dir
        
        self.use_front_depth = cfg.VIDEO_HOUR.INPUT_DEPTH
        self.pred_len = 1
        print('length of future is {} frame'.format(self.pred_len))
        
        self.seed = 0

        if random_len == 0:
            self.random_len = cfg.DATASET.RLBENCH.RANDOM_LEN
        else:
            self.random_len = random_len 
        
        task_names = self.get_task_names(cfg.DATASET.RLBENCH.TASK_LIST)
        self._json_file_name = 'RL_Becnh_dataset_VP_{}_{}{}.json'.format(mode,self.pred_len,task_names)
        json_path = os.path.join(data_root_dir, 'json', self._json_file_name)

        if not os.path.exists(json_path) or save_dataset:
            # create dataset
            print('There is no json data')
            print('create json data')
            self.add_data(data_root_dir, cfg)
            print('done')
            
            # save json data
            print('save json data')
            os.makedirs(os.path.join(data_root_dir, 'json'), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump([self.data_list, self.index_list],f,indent=4)
            print('done')
        else:
            # load json data
            print('load json data')
            with open(json_path) as f:
                [self.data_list, self.index_list] = json.load(f)

        self.get_image_size()
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, data_index):
        # get image
        # print('i:{}'.format(i))

        index = self.index_list[data_index]
        data_dict = self.data_list[index]
        
        start_index = data_dict['start_index']
        end_index = data_dict['end_index']
        input_dict = {}

        past_random_max = min(index - start_index, self.random_len)
        if past_random_max == 0:
            past_index = start_index
        else:
            diff = random.randint(1,past_random_max)
            past_index = index - diff
        index_list = [past_index, index]

        max_future_range = min(end_index - index, 2*self.random_len)
        
        half_range = math.floor(max_future_range/2)
        target_index = index + random.randint(1, half_range)
        index_list.append(target_index)
        
        future_range = max_future_range - half_range
        future_index = target_index + random.randint(1,future_range)
        index_list.append(future_index)
        
        input_dict['index_list'] = torch.tensor(index_list)
        input_dict['pred_len'] = self.pred_len

        for i,index in enumerate(index_list):
            if (index < start_index) or (index > end_index):
                raise ValueError('hoge')
                
            data_dict = self.data_list[index]
            
            # get rgb image
            rgb_path = os.path.join(data_dict['image_dir'], "front_rgb_{}.png".format(data_dict['filename']))
            rgb_image = Image.open(rgb_path)
            image_size = rgb_image.size
            rgb_image = self.transform_rgb(rgb_image, index)

            # get depth image
            if self.use_front_depth:
                depth_path = os.path.join(data_dict['image_dir'], "front_depth_{}.pickle".format(data_dict['filename']))
                with open(depth_path, 'rb') as f:
                    depth_image = pickle.load(f)
                depth_image = self.transform_depth(depth_image)

            # get pickle data
            pickle_path = data_dict['pickle_path']
            with open(pickle_path, 'rb') as f:
                pickle_data = pickle.load(f)
            
            # get camera info
            camera_intrinsic = pickle_data['front_intrinsic_matrix']
            camera_extrinsic = pickle_data['front_extrinsic_matrix'] # world2camera
            
            # get gripper info
            gripper_pos, gripper_matrix, gripper_open = self.get_gripper(pickle_data)
            
            # get uv cordinate and pose image
            pos_image, uv, uv_mask = self.transform_pos2image(gripper_pos, camera_intrinsic, image_size)
            
            # convert position data
            gripper_pos,gripper_pos_mask = self.transform_pos(gripper_pos)

            if i == 0:
                rgb_batch = torch.unsqueeze(rgb_image, 0)
                pose_image_batch = torch.unsqueeze(pos_image, 0)
                pose_batch = torch.unsqueeze(gripper_pos, 0)
                pose_mask_batch = torch.unsqueeze(gripper_pos_mask, 0)
                rotation_batch = torch.unsqueeze(gripper_matrix, 0)
                grasp_batch = torch.unsqueeze(gripper_open, 0)
                uv_batch = torch.unsqueeze(uv, 0)
                uv_mask_batch = torch.unsqueeze(uv_mask, 0)
                if self.use_front_depth:
                    depth_batch = torch.unsqueeze(depth_image, 0)
            else:
                rgb_batch = torch.cat((rgb_batch, torch.unsqueeze(rgb_image, 0)), 0)
                pose_image_batch = torch.cat((pose_image_batch, torch.unsqueeze(pos_image, 0)), 0)
                pose_batch = torch.cat((pose_batch, torch.unsqueeze(gripper_pos, 0)), 0)
                pose_mask_batch = torch.cat((pose_mask_batch, torch.unsqueeze(gripper_pos_mask, 0)), 0)
                rotation_batch = torch.cat((rotation_batch, torch.unsqueeze(gripper_matrix, 0)), 0)
                grasp_batch = torch.cat((grasp_batch, torch.unsqueeze(gripper_open, 0)), 0)
                uv_batch = torch.cat((uv_batch, torch.unsqueeze(uv, 0)), 0)
                uv_mask_batch = torch.cat((uv_mask_batch, torch.unsqueeze(uv_mask, 0)), 0)
                if self.use_front_depth:
                    depth_batch = torch.cat((depth_batch, torch.unsqueeze(depth_image, 0)), 0)

        relative_path = os.path.relpath(rgb_path, self.root_dir)
        task_name = relative_path[:relative_path.find('/')]

        input_dict['rgb'] = rgb_batch
        input_dict['pose'] = pose_image_batch
        input_dict['pose_xyz'] = pose_batch
        input_dict['pose_xyz_mask'] = pose_mask_batch
        input_dict['rotation_matrix'] = rotation_batch
        input_dict['grasp'] = grasp_batch
        input_dict['uv'] = uv_batch
        input_dict['uv_mask'] = uv_mask_batch
        input_dict['index_list'] = index_list
        input_dict['mtx'] = torch.tensor(camera_intrinsic)
        input_dict['inv_mtx'] = torch.tensor(np.linalg.inv(camera_intrinsic))
        input_dict['action_name'] = task_name
        if self.use_front_depth:
            input_dict['depth'] = depth_batch
        
        return input_dict

    def add_data(self, folder_path, cfg):
        """
        output:
        data_list: list
        data_list = [data_dict * n]
        data_dict = {
        'image_path': path
        'pickle_path': path
        'end_index': index of data where task will finish
        'start_index': index of date where task start
        }
        """
        # for data preparation
        self.data_list = []
        self.index_list = []
        index = 0
        
        task_list = os.listdir(folder_path) # get task list
        task_list.sort() # sort task
        for task_name in task_list:
            if 'all' in cfg.DATASET.RLBENCH.TASK_LIST:
                pass
            elif task_name not in cfg.DATASET.RLBENCH.TASK_LIST:
                continue

            if task_name == 'json':
                continue
            
            if 'RL_Becnh_dataset' in task_name:
                continue
            
            task_path = os.path.join(folder_path, task_name)
            
            sequence_list = os.listdir(task_path)
            sequence_list.sort()
            for sequence_index in sequence_list:
                start_index = index
                image_folder_path = os.path.join(task_path, sequence_index, 'image')
                pickle_folder_path = os.path.join(task_path, sequence_index, 'base_data')
                pickle_list = os.listdir(pickle_folder_path)
                pickle_list.sort()
                for pickle_name in pickle_list:
                    head, ext = os.path.splitext(pickle_name)
                    data_dict = {}
                    data_dict['image_dir'] = image_folder_path
                    data_dict['filename'] = os.path.join(head)
                    data_dict['pickle_path'] = os.path.join(pickle_folder_path, pickle_name)
                    data_dict['start_index'] = start_index
                    data_dict['end_index'] = start_index + len(pickle_list) - 1
                    
                    self.data_list.append(data_dict)
                    if index <= start_index + (len(pickle_list) - 1) - (self.pred_len + 1):
                        self.index_list.append(index)
                        
                    index += 1