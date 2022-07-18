import rlbench
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np
import os

from PIL import Image, ImageDraw, ImageOps
import pickle
import json
import subprocess
import shutil
import math

mode_list = ["val"] # [tran, val]
num_data = [100] # [1000, 100]
start_index = 0
task_list = ["PickUpCup"]
robot = "panda" # panda, ur5
server_path = "tendon@dl20:/misc/dl001/dataset/ooba"
# server_path = "" # if you do not want to use server, please activate this.
dataset_name = "RLBench4-{}".format(robot)
dataset_base_dir = "../dataset"
dataset_path = os.path.join(dataset_base_dir, dataset_name)
os.makedirs(dataset_path, exist_ok=True)

while 1:
    ans = input('Have you confirmed the configuration in the code? y or n: ')
    if ans == 'y':
        break
    elif ans == 'n':
        raise ValueError("Please confirm and change the configuration before running the code")
    else:
        print('please type y or n')
            
def save_demo(demo, task_dir, demo_index):
    
    def save_pickle(path,data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
    image_dir = os.path.join(task_dir, demo_index, 'image')
    os.makedirs(image_dir, exist_ok=True)

    # pickle_dir = os.path.join(task_dir, demo_index, 'pickle')
    # os.makedirs(pickle_dir, exist_ok=True)

    video_dir = os.path.join(task_dir, demo_index, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    base_dir = os.path.join(task_dir, demo_index, 'base_data')
    os.makedirs(base_dir, exist_ok=True)
    
    save_pickle(os.path.join(task_dir, demo_index, 'seed.pickle'), demo[0].random_seed)
    
    demo = np.array(demo).flatten()
    for obs_index, obs in enumerate(demo):
        obs.front_rgb = (obs.front_rgb).astype(np.uint8)
        obs.front_depth = obs.front_depth.astype(np.float16)
        front_rgb = Image.fromarray(obs.front_rgb)
        
        obs.wrist_rgb = (obs.wrist_rgb).astype(np.uint8)
        obs.wrist_depth = obs.wrist_depth.astype(np.float16)
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        
        obs.left_shoulder_rgb = (obs.left_shoulder_rgb).astype(np.uint8)
        obs.left_shoulder_depth = obs.left_shoulder_depth.astype(np.float16)
        left_rgb = Image.fromarray(obs.left_shoulder_rgb)
        
        obs.right_shoulder_rgb = (obs.right_shoulder_rgb).astype(np.uint8)
        obs.right_shoulder_depth = obs.right_shoulder_depth.astype(np.float16)
        right_rgb = Image.fromarray(obs.right_shoulder_rgb)

        time_step = str(obs_index).zfill(8)
        
        front_rgb_path = os.path.join(image_dir, 'front_rgb_{}.png'.format(time_step))
        front_rgb.save(front_rgb_path)
        front_depth_path = os.path.join(image_dir, 'front_depth_{}.pickle'.format(time_step))
        save_pickle(front_depth_path, obs.front_depth)
        
        # left_rgb_path = os.path.join(image_dir, 'left_rgb_{}.png'.format(time_step))
        # left_rgb.save(left_rgb_path)
        # left_depth_path = os.path.join(image_dir, 'left_depth_{}.pickle'.format(time_step))
        # save_pickle(left_depth_path, obs.left_shoulder_depth)

        # right_rgb_path = os.path.join(image_dir, 'right_rgb_{}.png'.format(time_step))
        # right_rgb.save(right_rgb_path)
        # right_depth_path = os.path.join(image_dir, 'right_depth_{}.pickle'.format(time_step))
        # save_pickle(right_depth_path, obs.right_shoulder_depth)

        # wrist_rgb_path = os.path.join(image_dir, 'wrist_rgb_{}.png'.format(time_step))
        # wrist_rgb.save(wrist_rgb_path)
        # wrist_depth_path = os.path.join(image_dir, 'wrist_depth_{}.pickle'.format(time_step))
        # save_pickle(wrist_depth_path, obs.wrist_depth)
        
        base_data = {}
        base_data['front_extrinsic_matrix'] = obs.misc["front_camera_extrinsics"]
        base_data['front_intrinsic_matrix'] = obs.misc["front_camera_intrinsics"]
        base_data['left_shoulder_extrinsic_matrix'] = obs.misc["left_shoulder_camera_extrinsics"]
        base_data['left_shoulder_intrinsic_matrix'] = obs.misc["left_shoulder_camera_intrinsics"]
        base_data['right_shoulder_extrinsic_matrix'] = obs.misc["right_shoulder_camera_extrinsics"]
        base_data['right_shoulder_intrinsic_matrix'] = obs.misc["right_shoulder_camera_intrinsics"]
        base_data['wrist_extrinsic_matrix'] = obs.misc["wrist_camera_extrinsics"]
        base_data['wrist_intrinsic_matrix'] = obs.misc["wrist_camera_intrinsics"]
        base_data['gripper_joint_positions'] = obs.gripper_joint_positions
        base_data['gripper_matrix'] = obs.gripper_matrix
        base_data['gripper_open'] = obs.gripper_open
        base_data['gripper_pose'] = obs.gripper_pose
        base_data['gripper_touch_forces'] = obs.gripper_touch_forces
        base_data['joint_forces'] = obs.joint_forces
        base_data['joint_positions'] = obs.joint_positions
        base_data['joint_velocities'] = obs.joint_velocities
        base_data['task_low_dim_state'] = obs.task_low_dim_state
        base_data_path = os.path.join(base_dir, '{}.pickle'.format(time_step))
        save_pickle(base_data_path, base_data)
        
    save_video(demo, video_dir)
    
def save_video(demo, path):
    import cv2
    size = (256, 256)
    frame_rate = 20
    filename = os.path.join(path, 'video.mp4')
    
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(filename, fmt, frame_rate, size)
    for obs in demo:
        cv2_image = cv2.cvtColor(obs.front_rgb, cv2.COLOR_RGB2BGR)
        writer.write(cv2_image)
    
    writer.release()
    cv2.destroyAllWindows()

def move_to_server(path, server_path):
    print('copy to {} from {}'.format(server_path, path))
    subprocess.run(["scp","-r",path, server_path])
    shutil.rmtree(path)

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.gripper_touch_forces = False

# change action mode
action_mode = MoveArmThenGripper(
  arm_action_mode=EndEffectorPoseViaPlanning(),
  gripper_action_mode=Discrete()
)

# set up enviroment
env = Environment(
    action_mode, '', obs_config, False, robot_setup=robot)
env.launch()

env._scene._cam_front.set_resolution([256,256])
env._scene._cam_front.set_position(env._scene._cam_front.get_position() + np.array([0.3,0,0.3]))

env._scene._cam_over_shoulder_left.set_resolution([256,256])
env._scene._cam_over_shoulder_left.set_position(np.array([0.32500029, 1.54999971, 1.97999907]))
env._scene._cam_over_shoulder_left.set_orientation(np.array([ 2.1415925 ,  0., 0.]))

env._scene._cam_over_shoulder_right.set_resolution([256,256])
env._scene._cam_over_shoulder_right.set_position(np.array([0.32500029, -1.54999971, 1.97999907]))
env._scene._cam_over_shoulder_right.set_orientation(np.array([-2.1415925,  0., math.pi]))

for current_task in task_list:
    for mode, max_num in zip(mode_list, num_data):
        print('mode: {}'.format(mode))
        print('task_name: {}'.format(current_task))
        exec_code = 'task = {}'.format(current_task)
        exec(exec_code)

        # set up task
        task = env.get_task(task)
        task_path = os.path.join(dataset_path, mode, current_task)
        
        if server_path != "":
            subprocess.run(["scp","-r", dataset_path, server_path])

        for demo_index in range(start_index, start_index + max_num):
            print("demo index: {}".format(demo_index))
            
            success = False
            while not success:
                try:
                    demos = task.get_demos(1, live_demos=True)  # -> List[List[Observation]]
                    success = True
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    pass
                    
            save_demo(demos, task_path, str(demo_index).zfill(5))
            
            if server_path != "":
                move_to_server(os.path.join(dataset_path, mode, current_task), os.path.join(server_path, dataset_name, mode))
