import os
import pickle
from PIL import Image, ImageDraw
import numpy as np

### configuration ###
dataset_root = os.path.abspath("../dataset/RLBench4/val")
skip_task_lists = ['none']

### main ###
while 1:
    ans = input('Have you confirmed the configuration in the code? y or n: ')
    if ans == 'y':
        break
    elif ans == 'n':
        raise ValueError("Please confirm and change the configuration before running the code")
    else:
        print('please type y or n')

for task_name in os.listdir(dataset_root):
    if 'json' in task_name:
        continue
        
    if 'pickle' in task_name:
        continue

    if task_name in skip_task_lists:
        continue
    
    task_dir = os.path.join(dataset_root, task_name)
    for data_index in os.listdir(task_dir):
        print("task:{}  data:{}".format(task_name, data_index))
        data_dir = os.path.join(task_dir, data_index)
        base_data_dir = os.path.join(data_dir, "base_data")
        additional_dir = os.path.join(data_dir, "additional_info")
        os.makedirs(additional_dir, exist_ok=True)
        u_list, v_list = [], []
        pickle_file_list = os.listdir(base_data_dir)
        pickle_file_list.sort()
        for pickle_filename in pickle_file_list:
            with open(os.path.join(base_data_dir, pickle_filename), 'rb') as f:
                pickle_data = pickle.load(f)
                intrinsic_matrix = pickle_data['front_intrinsic_matrix']
                extrinsic_matrix = pickle_data['front_extrinsic_matrix']
                camera2world_matrix = np.linalg.inv(extrinsic_matrix)

                pose = np.append(pickle_data['gripper_pose'][:3], 1)
                pose = np.array(np.dot(camera2world_matrix, pose))
                pose = pose / pose[2]
                uv = np.dot(intrinsic_matrix, pose[:3])
                u_list.append(uv[0])
                v_list.append(uv[1])
        
        for i, pickle_filename in enumerate(pickle_file_list):
            head, ext = os.path.splitext(pickle_filename)
            img_path = os.path.join(additional_dir, 'goal_trajectory_{}.png'.format(head))
            if os.path.exists(img_path):
                continue
                
            pos_img = Image.new('L',(256,256))
            d = ImageDraw.Draw(pos_img)
            d.line([(int(u), int(v)) for u, v in zip(u_list[i:],v_list[i:])], fill=(255), width=10) 
            pos_img.save(img_path)
        