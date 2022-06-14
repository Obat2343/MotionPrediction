import os
import math
import pickle
import copy
import sys
import datetime
import json

import rlbench
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np

# set config
seq_start_index = 0
seq_last_index = 5
max_try = 10
max_frame = 100
save_gt_image = True
mode = "val"
debug = False
robot = 'panda'

checkpoint_path_list = ["../output/RLBench4/PickUpCup/1stage_hourglass_trajectory_lr_1e-4_past1_skip3/model_log/checkpoint_iter100000/mp.pth"]

# get env
obs_config = ObservationConfig()
obs_config.set_all(True)

if robot == "panda":
    obs_config.gripper_touch_forces = True
else:
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

import torch
from torchvision import transforms
from eval_util import get_env, line_notification, evaluate_error, make_video, prepare_result_dict, save_result_to_csv
from hourglass_util import data_process, get_various_image

sys.path.append("../")
from pycode.model.Hourglass import stacked_hourglass_model
from pycode.config import _C as base_cfg

from pycode.misc import load_checkpoint, load_hourglass

def get_action(obs, model, current_step, action_list=None, do_demo_until=0):
    inputs = dp.make_inputs(obs)
    with torch.no_grad():
        outputs = model(inputs)
    if current_step < do_demo_until:
        print("pred:action")
        print(dp.outputs2action(outputs, obs))
        action = action_list[current_step+1]
    else:
        # print('network action')
        action = dp.outputs2action(outputs, obs)
    return action, outputs, inputs

# set seed and cuda
torch.manual_seed(base_cfg.BASIC.SEED)
cuda = torch.cuda.is_available()
device = torch.device(base_cfg.BASIC.DEVICE)

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(base_cfg.BASIC.SEED)

for checkpoint_path in checkpoint_path_list:

    # load cfg
    cfg = copy.deepcopy(base_cfg)
    yaml_file_name = [filename for filename in os.listdir(checkpoint_path[:checkpoint_path.find("model_log")]) if filename[-5:] == ".yaml"]
    if len(yaml_file_name) != 1:
        raise ValueError(f"no yaml file or too many file were found: {yaml_file_name}")

    config_path = os.path.join(checkpoint_path[:checkpoint_path.find("model_log")], yaml_file_name[0])
    cfg.merge_from_file(config_path)

    # get task name
    print(cfg.DATASET.NAME)
    current_task = checkpoint_path[checkpoint_path.find(cfg.DATASET.NAME) + len(cfg.DATASET.NAME) + 1:]
    current_task = current_task[:current_task.find("/")]

    # set up task
    print('task_name: {}'.format(current_task))
    exec_code = 'task = {}'.format(current_task)
    exec(exec_code)
    task = env.get_task(task)

    # change cfg for eval
    cfg.BASIC.NUM_GPU = 1
    cfg.BASIC.BATCH_SIZE = 1
    cfg.PRED_LEN = 1

    model = stacked_hourglass_model(cfg, output_dim=1)
    model = model.to('cuda')

    # load checkpoint 
    if cfg.MP_MODEL_NAME == 'sequence_hourglass':
        model = load_hourglass(model, checkpoint_path)
    elif cfg.MP_MODEL_NAME == 'hourglass':
        model, _, _, _, _ = load_checkpoint(model, checkpoint_path, fix_parallel=True)

    dataset_name = checkpoint_path[10:]
    dataset_name = dataset_name[:dataset_name.find("/")]
    mode = "val"
    task_dir = "../dataset/{}/{}/{}".format(dataset_name, mode, current_task)
    checkpoint_name = checkpoint_path[checkpoint_path.find(current_task) + len(current_task) + 1:-7]
    results_dir = os.path.join("../results/{}/{}/{}".format(dataset_name, mode, current_task), checkpoint_name + "_hoge")
    json_name = '{}_{}_{}_{}_skip_{}.json'.format(cfg.DATASET.NAME,mode,cfg.PRED_LEN,current_task,cfg.SKIP_LEN)
    json_path = os.path.join("../dataset",cfg.DATASET.NAME,mode,'json',json_name)
    print(json_path)

    print('load json data')
    with open(json_path) as f:
        [data_list, index_list, next_index_list, _] = json.load(f)
                
    print(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    csv_path = "../results/{}/{}/{}/result_table_{}_100.csv".format(dataset_name, mode, current_task, cfg.SKIP_LEN)
    pose_error_list_path = "../results/{}/{}/{}/error_list_pose_{}_100.csv".format(dataset_name, mode, current_task, cfg.SKIP_LEN)
    angle_error_list_path = "../results/{}/{}/{}/error_list_angle_{}_100.csv".format(dataset_name, mode, current_task, cfg.SKIP_LEN)

    total_index_list = [] # shape: Num_sequece * Num_frame
    frame_start_index = 0
    print("####################")
    for j in range(seq_last_index):
        print(f"load sequence data: {j} and skip frame")
        index_list = []
        index = frame_start_index
        frame_end_index = data_list[index]["end_index"]
        
        while index <= frame_end_index:
            index_list.append(index)
            if index == frame_end_index:
                break

            index = next_index_list[index]
        
        total_index_list.append([i - frame_start_index for i in index_list])
        frame_start_index = index + 1

    print("complete")
    dp = data_process(cfg.PAST_LEN + 1)

    ### train_setting
    seq_index_list = os.listdir(task_dir)
    seq_index_list.sort()
    if seq_last_index > len(seq_index_list):
        raise ValueError(f"invalid seq_last_index: max is {len(seq_index_list)}")

    result_dict = prepare_result_dict()
    additional_data = {}
    index = 0
    while index < seq_last_index:
        control_list = []
        image_list = []
        seed_path = os.path.join(task_dir, seq_index_list[index], 'seed.pickle')
        with open(seed_path, 'rb') as f:
            seed = pickle.load(f)
        
        # get gt action_list
        base_dir = os.path.join(task_dir, seq_index_list[index], 'base_data')
        pickle_list = os.listdir(base_dir)
        pickle_list.sort()
        gt_state_list = []
        gt_matrix_list = []
        gt_image_path_list = []
        for pickle_index, pickle_name in enumerate(pickle_list):
            # skip frame based on total_index_list
            if pickle_index in total_index_list[index]:
                pickle_path = os.path.join(base_dir, pickle_name)
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                gt_state_list.append(np.append(data["gripper_pose"], data["gripper_open"]))
                gt_matrix_list.append(data["gripper_matrix"])

                if save_gt_image:
                    head, ext = os.path.splitext(pickle_name)
                    gt_image_path_list.append(os.path.join(task_dir, seq_index_list[index], "image/front_rgb_{}.png".format(head)))
            
        result_dict["gt_state_list"].append(gt_state_list)
        result_dict["gt_matrix_list"].append(gt_matrix_list)
        result_dict["gt_image_path_list"].append(gt_image_path_list)
        
        ToTensor = transforms.ToTensor()
            
        print("num: {}".format(index))
        iteration = 0
        
        try_iter = 0
        while try_iter < max_try:
            try:
                descriptions, obs = task.reset_to_seed(seed)
                break
            except:
                try_iter += 1
                pass
            
        dp.reset()
        action_list = []
        while 1:
            # pred action
            # action: robot 6d pose and grasp state, outputs: model outputs include trajectory, input: model inputs
            action, outputs, inputs = get_action(obs, model, iteration, action_list=gt_state_list, do_demo_until=0)
            
            # record action and images
            action_list.append(action)
            image_list.append(get_various_image(inputs, outputs, obs, iteration, gt_matrix_list, gt_image_path_list, save_gt_image))
            
            # try action
            try_iter = 0
            while try_iter < max_try:
                try:
                    # apply action to the simulater
                    obs, reward, terminate = task.step(action)
                    control_list.append(1)
                    break
                except Exception as e:
                    print("error: " + str(e))
                    if try_iter == max_try - 1:
                        control_list.append(0)
                    try_iter += 1
                    pass
            
            iteration += 1
            
            # judge
            if terminate:
                result_dict['success'] += 1
                result_dict["seed_path"].append(seed_path)
                result_dict["success_list"].append(1)
                result = "success"
                control_list.append(1)
                image_list.append(get_various_image(inputs, outputs, obs, iteration, gt_matrix_list, gt_image_path_list, save_gt_image))
                break

            if iteration >= max_frame:
                result_dict["success_list"].append(0)
                result ="fail"
                break
                
            if try_iter >= max_try:
                result_dict["success_list"].append(0)
                result ="fail"
                break
            
        result_dict["out of control"].append(np.min(control_list))
        result_dict["action_list"].append(action_list)
        
        ### evaluate
        result_dict = evaluate_error(result_dict, index)
            
        ### save data
        if not debug:
            for time_step, image in enumerate(image_list):
                img_save_dir = os.path.join(results_dir, "{}_{}".format(str(index).zfill(3),result))
                os.makedirs(img_save_dir, exist_ok=True)
                front_rgb_path = os.path.join(img_save_dir, 'front_rgb_{}.png'.format(time_step))
                image.save(front_rgb_path)

            video_save_path = os.path.join(img_save_dir, "movie.mp4")
            make_video(image_list, video_save_path, image.size, control_list)
        index += 1

    dt_now = datetime.datetime.now()

    if not debug:
        save_result_to_csv(csv_path, pose_error_list_path, angle_error_list_path, checkpoint_path, dt_now, mode, seq_last_index, max_frame, result_dict)
    
    line_notification("acrab")