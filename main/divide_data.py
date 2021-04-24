import pickle
import sys
import os
from joblib import Parallel, delayed

task_dir = os.path.abspath(sys.argv[1])
save_base = True
save_front = False
save_left = False
save_right = False
save_wrist = False

def save_pickle(path,data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def process(pickle_file_name):
    pickle_file_path = os.path.join(pickle_dir, pickle_file_name)
    # print('open file')
    with open(pickle_file_path, 'rb') as f:
        pickle_data = pickle.load(f)

    base_data = {}
    front_camera_data = {}
    left_camera_data = {}
    right_camera_data = {}
    wrist_camera_data = {}

    if save_base:
        base_data['front_extrinsic_matrix'] = pickle_data.front_extrinsic_matrix
        base_data['front_intrinsic_matrix'] = pickle_data.front_intrinsic_matrix
        base_data['left_shoulder_extrinsic_matrix'] = pickle_data.left_shoulder_extrinsic_matrix
        base_data['left_shoulder_intrinsic_matrix'] = pickle_data.left_shoulder_intrinsic_matrix
        base_data['right_shoulder_extrinsic_matrix'] = pickle_data.right_shoulder_extrinsic_matrix
        base_data['right_shoulder_intrinsic_matrix'] = pickle_data.right_shoulder_intrinsic_matrix
        base_data['wrist_extrinsic_matrix'] = pickle_data.wrist_extrinsic_matrix
        base_data['wrist_intrinsic_matrix'] = pickle_data.wrist_intrinsic_matrix
        base_data['gripper_joint_positions'] = pickle_data.gripper_joint_positions
        base_data['gripper_matrix'] = pickle_data.gripper_matrix
        base_data['gripper_open'] = pickle_data.gripper_open
        base_data['gripper_pose'] = pickle_data.gripper_pose
        base_data['gripper_touch_forces'] = pickle_data.gripper_touch_forces
        base_data['joint_forces'] = pickle_data.joint_forces
        base_data['joint_positions'] = pickle_data.joint_positions
        base_data['joint_velocities'] = pickle_data.joint_velocities
        base_data['joints'] = pickle_data.joints
        base_data['task_low_dim_state'] = pickle_data.task_low_dim_state
        base_data_path = os.path.join(base_data_dir, pickle_file_name)
        save_pickle(base_data_path, base_data)
        # print('save_pickle base')

    if save_front:
        front_camera_data['rgb'] = pickle_data.front_rgb
        front_camera_data['mask'] = pickle_data.front_mask
        front_camera_data['depth'] = pickle_data.front_depth
        front_camera_data_path = os.path.join(front_camera_dir, pickle_file_name)
        save_pickle(front_camera_data_path, front_camera_data)
        # print('save_pickle front')

    if save_left:
        left_camera_data['rgb'] = pickle_data.left_shoulder_rgb
        left_camera_data['mask'] = pickle_data.left_shoulder_mask
        left_camera_data['depth'] = pickle_data.left_shoulder_depth
        left_camera_data_path = os.path.join(left_camera_dir, pickle_file_name)
        save_pickle(left_camera_data_path, left_camera_data)
        # print('save_pickle left')

    if save_right:
        right_camera_data['rgb'] = pickle_data.right_shoulder_rgb
        right_camera_data['mask'] = pickle_data.right_shoulder_mask
        right_camera_data['depth'] = pickle_data.right_shoulder_depth
        right_camera_data_path = os.path.join(right_camera_dir, pickle_file_name)
        save_pickle(right_camera_data_path, right_camera_data)
        # print('save_pickle right')

    if save_wrist:
        wrist_camera_data['rgb'] = pickle_data.wrist_rgb
        wrist_camera_data['mask'] = pickle_data.wrist_mask
        wrist_camera_data['depth'] = pickle_data.wrist_depth
        wrist_camera_data_path = os.path.join(wrist_camera_dir, pickle_file_name)
        save_pickle(wrist_camera_data_path, wrist_camera_data)
        # print('save_pickle wrist')

for number in os.listdir(task_dir):
    pickle_dir = os.path.join(task_dir, number, 'pickle')
    pickle_file_list = os.listdir(pickle_dir)
    pickle_file_list.sort()
    
    base_data_dir = os.path.join(task_dir, number, 'base_data')
    front_camera_dir = os.path.join(task_dir, number, 'front_camera_data')
    left_camera_dir = os.path.join(task_dir, number, 'left_camera_data')
    right_camera_dir = os.path.join(task_dir, number, 'right_camera_data')
    wrist_camera_dir = os.path.join(task_dir, number, 'wrist_camera_data')
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(front_camera_dir, exist_ok=True)
    os.makedirs(left_camera_dir, exist_ok=True)
    os.makedirs(right_camera_dir, exist_ok=True)
    os.makedirs(wrist_camera_dir, exist_ok=True)
    
    print('number: {}'.format(number))
    Parallel(n_jobs=-1, verbose=10)(delayed(process)(pickle_file_name) for pickle_file_name in pickle_file_list)