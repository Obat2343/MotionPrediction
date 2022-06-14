import rlbench
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *

import os
import math
import cv2
import csv
import numpy as np
from PIL import Image, ImageDraw
from fastdtw import fastdtw # https://github.com/slaypni/fastdtw
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean

def get_env(robot):
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

    return env

def line_notification(message):
    # ラインでメッセージを送信してくれる
    import requests
    url = "https://notify-api.line.me/api/notify"
    token = 'glAs1p3yWlkEt0LsRCwLZBHaPxIixNF0Q9aa6V7XACp'
    headers = {"Authorization" : "Bearer "+ token}
    message = 'プログラム名:acrab\n{}'.format(message)
    payload = {"message" :  message}
    r = requests.post(url ,headers = headers ,params=payload)

def get_pil_image(obs):
    front_rgb = (obs.front_rgb).astype(np.uint8)
    front_rgb = Image.fromarray(front_rgb)
    return front_rgb

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def make_overlay_image(base_image, over_image):
    # image2 is base image
    image1 = np.array(base_image)
    image2 = np.array(over_image)
    image2 = np.repeat(image2[:,:,np.newaxis], 3, axis=2)
    image = cv2.addWeighted(image2, 0.7, image1, 0.3, 0)
    return Image.fromarray(image)

def draw_matrix(image, pose_matrix, intrinsic_matrix, color=155):
    """
    image: PIL.Image
    pose_matrix: np.array (4X4)
        pose is position and orientation in the camera coordinate.
    intrinsic_matrix: np.array(4X4)
    """
    cordinate_vector_array = np.array([[0,0,0,1],[0,0,0.1,1],[0,0.1,0,1],[0.1,0,0,1]]).T
    cordinate_matrix = np.dot(pose_matrix, cordinate_vector_array)
    
    draw = ImageDraw.Draw(image)
    color_list = [(color,0,0), (0,color,0), (0,0,color)]
    
    base_cordinate = cordinate_matrix.T[0]
    cordinates = cordinate_matrix.T[1:]

    base_cordinate = base_cordinate[:3] / base_cordinate[2]
    base_uv = np.dot(intrinsic_matrix, base_cordinate)
    base_u, base_v = base_uv[0], base_uv[1]
    
    for i in range(len(cordinates)):
        cordinate = cordinates[i]
        cordinate = cordinate[:3] / cordinate[2]
        uv = np.dot(intrinsic_matrix, cordinate)
        u, v = uv[0], uv[1]
        
        draw.line((base_u, base_v, u, v), fill=color_list[i], width=3)
    
    return image

def prepare_result_dict():
    result_dict = {}
    result_dict['success'] = 0
    result_dict["seed_path"] = []
    result_dict["success_list"] = []
    result_dict["out of control"] = []
    result_dict["action_list"] = []
    result_dict["gt_state_list"] = []
    result_dict["gt_matrix_list"] = []
    result_dict["gt_image_path_list"] = []
    result_dict["pose_dtw_xyz"] = []
    result_dict["pose_dtw_x"] = []
    result_dict["pose_dtw_y"] = []
    result_dict["pose_dtw_z"] = []
    result_dict["angle_dtw_xyz"] = []
    result_dict["angle_dtw_x"] = []
    result_dict["angle_dtw_y"] = []
    result_dict["angle_dtw_z"] = []
    result_dict["pose_error_list_xyz"] = []
    result_dict["pose_error_list_x"] = []
    result_dict["pose_error_list_y"] = []
    result_dict["pose_error_list_z"] = []
    result_dict["angle_error_list_xyz"] = []
    result_dict["angle_error_list_x"] = []
    result_dict["angle_error_list_y"] = []
    result_dict["angle_error_list_z"] = []
    return result_dict

def evaluate_error(result_dict, index):
    pose_error_xyz, pose_error_x, pose_error_y, pose_error_z, pose_error_list_xyz, pose_error_list_x, pose_error_list_y, pose_error_list_z = calculate_dtw_pos(result_dict["action_list"][index], result_dict["gt_state_list"][index][1:])
    angle_error_xyz, angle_error_x, angle_error_y, angle_error_z, angle_error_list_xyz, angle_error_list_x, angle_error_list_y, angle_error_list_z= calculate_dtw_angle(result_dict["action_list"][index], result_dict["gt_state_list"][index][1:])
    print(f"pose error: {pose_error_xyz}")
    result_dict["pose_dtw_xyz"].append(pose_error_xyz)
    result_dict["pose_dtw_x"].append(pose_error_x)
    result_dict["pose_dtw_y"].append(pose_error_y)
    result_dict["pose_dtw_z"].append(pose_error_z)
    result_dict["angle_dtw_xyz"].append(angle_error_xyz)
    result_dict["angle_dtw_x"].append(angle_error_x)
    result_dict["angle_dtw_y"].append(angle_error_y)
    result_dict["angle_dtw_z"].append(angle_error_z)
    result_dict["pose_error_list_xyz"].append(pose_error_list_xyz)
    result_dict["pose_error_list_x"].append(pose_error_list_x)
    result_dict["pose_error_list_y"].append(pose_error_list_y)
    result_dict["pose_error_list_z"].append(pose_error_list_z)
    result_dict["angle_error_list_xyz"].append(angle_error_list_xyz)
    result_dict["angle_error_list_x"].append(angle_error_list_x)
    result_dict["angle_error_list_y"].append(angle_error_list_y)
    result_dict["angle_error_list_z"].append(angle_error_list_z)
    return result_dict

def make_video(pil_list, file_path, size, control_list, fps=20,):
    videodims = size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
    video = cv2.VideoWriter(file_path, fourcc, fps, videodims)
    #draw stuff that goes on every frame here
    for index, pil_img in enumerate(pil_list):
        imtemp = pil_img.copy()
        image_editable = ImageDraw.Draw(imtemp)
        if control_list[index]:
            control_error = "False"
            color = (237, 230, 211)
        else:
            control_error = True
            color = (255, 0, 0)
        image_editable.text((15,15), 'index:{}\nc_error: {}'.format(index,control_error), color)
        # draw frame specific stuff here.
        video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    video.release()

def save_result_to_csv(csv_path, pose_error_list_path, angle_error_list_path, checkpoint_path, dt_now, mode, seq_last_index, max_frame, result_dict):
    if not os.path.exists(csv_path):
        head_list = ["checkpoint_path", "date", "mode", "total_try", "num_succes",
                    "pose_dtw_xyz","pose_dtw_x","pose_dtw_y","pose_dtw_z",
                    "angle_dtw_xyz","angle_dtw_x","angle_dtw_y","angle_dtw_z",
                    "num of out of control", "\n"]
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(head_list)
            
    if not os.path.exists(pose_error_list_path):
        head_list = ["checkpoint_path", "date", "mode", "total_try", "axis", "\n"]
        with open(pose_error_list_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(head_list)
        
        with open(angle_error_list_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(head_list)

    ### save csv
    list_to_csv = [checkpoint_path, dt_now, mode, seq_last_index, result_dict["success"],
                np.mean(result_dict["pose_dtw_xyz"]), np.mean(result_dict["pose_dtw_x"]),
                np.mean(result_dict["pose_dtw_y"]), np.mean(result_dict["pose_dtw_z"]),
                np.mean(result_dict["angle_dtw_xyz"]), np.mean(result_dict["angle_dtw_x"]),
                np.mean(result_dict["angle_dtw_y"]), np.mean(result_dict["angle_dtw_z"]),
                seq_last_index - sum(result_dict["out of control"]),
                max_frame]

    # calculate mean and var for error plot
    def make_csv_data(error_list, min_length, checkpoint_path, dt_now, mode, seq_last_index, axis):
        error_list_np = np.array([error_ins[:min_length] for error_ins in error_list])
        error_list_to_csv_mean = [checkpoint_path, dt_now, mode, seq_last_index, axis]
        error_list_to_csv_mean.extend(np.mean(error_list_np,0).tolist())
        error_list_to_csv_var = [checkpoint_path, dt_now, mode, seq_last_index, axis]
        error_list_to_csv_var.extend(np.var(error_list_np,0).tolist())
        return error_list_to_csv_mean, error_list_to_csv_var
        
    min_length = min([len(error_list) for error_list in result_dict["pose_error_list_xyz"]])

    pose_error_list_to_csv_mean, pose_error_list_to_csv_var = make_csv_data(result_dict["pose_error_list_xyz"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "xyz")
    pose_error_list_to_csv_mean_x, pose_error_list_to_csv_var_x = make_csv_data(result_dict["pose_error_list_x"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "x")
    pose_error_list_to_csv_mean_y, pose_error_list_to_csv_var_y = make_csv_data(result_dict["pose_error_list_y"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "y")
    pose_error_list_to_csv_mean_z, pose_error_list_to_csv_var_z = make_csv_data(result_dict["pose_error_list_z"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "z")

    angle_error_list_to_csv_mean, angle_error_list_to_csv_var = make_csv_data(result_dict["angle_error_list_xyz"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "xyz")
    angle_error_list_to_csv_mean_x, angle_error_list_to_csv_var_x = make_csv_data(result_dict["angle_error_list_x"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "x")
    angle_error_list_to_csv_mean_y, angle_error_list_to_csv_var_y = make_csv_data(result_dict["angle_error_list_y"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "y")
    angle_error_list_to_csv_mean_z, angle_error_list_to_csv_var_z = make_csv_data(result_dict["angle_error_list_z"], min_length, checkpoint_path, dt_now, mode, seq_last_index, "z")

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(list_to_csv)

    with open(pose_error_list_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(pose_error_list_to_csv_mean)
        writer.writerow(pose_error_list_to_csv_var)
        writer.writerow(pose_error_list_to_csv_mean_x)
        writer.writerow(pose_error_list_to_csv_var_x)
        writer.writerow(pose_error_list_to_csv_mean_y)
        writer.writerow(pose_error_list_to_csv_var_y)
        writer.writerow(pose_error_list_to_csv_mean_z)
        writer.writerow(pose_error_list_to_csv_var_z)

    with open(angle_error_list_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(angle_error_list_to_csv_mean)
        writer.writerow(angle_error_list_to_csv_var)
        writer.writerow(angle_error_list_to_csv_mean_x)
        writer.writerow(angle_error_list_to_csv_var_x)
        writer.writerow(angle_error_list_to_csv_mean_y)
        writer.writerow(angle_error_list_to_csv_var_y)
        writer.writerow(angle_error_list_to_csv_mean_z)
        writer.writerow(angle_error_list_to_csv_var_z)

# def get_demo_and_reset():
#     demos = task.get_demos(1, live_demos=live_demos)
#     discriptions, obs = task.reset_to_demo(demos[0])
#     return demos[0], discriptions, obs


def calculate_dtw_pos(pred_action, gt_action):
    pred_xyz = np.array(pred_action)[:,:3]
    gt_xyz = np.array(gt_action)[:,:3]

    print("calculate dtw pose")
    dtw_error_xyz, path_xyz = fastdtw(pred_xyz, gt_xyz, dist=euclidean)
    error_xyz_list = error_divide_time(pred_xyz, gt_xyz, euclidean, path_xyz)
    mean_dtw_xyz = dtw_error_xyz / len(path_xyz)

    dtw_error_x, path_x = fastdtw(pred_xyz[:,0], gt_xyz[:,0], dist=euclidean)
    error_x_list = error_divide_time(pred_xyz[:,0], gt_xyz[:,0], euclidean, path_x)
    mean_dtw_x = dtw_error_x / len(path_x)

    dtw_error_y, path_y = fastdtw(pred_xyz[:,1], gt_xyz[:,1], dist=euclidean)
    error_y_list = error_divide_time(pred_xyz[:,1], gt_xyz[:,1], euclidean, path_y)
    mean_dtw_y = dtw_error_y / len(path_y)

    dtw_error_z, path_z = fastdtw(pred_xyz[:,2], gt_xyz[:,2], dist=euclidean)
    error_z_list = error_divide_time(pred_xyz[:,2], gt_xyz[:,2], euclidean, path_z)
    mean_dtw_z = dtw_error_z / len(path_z)

    return mean_dtw_xyz, mean_dtw_x, mean_dtw_y, mean_dtw_z, error_xyz_list, error_x_list, error_y_list, error_z_list

def calculate_dtw_angle(pred_action, gt_action):
    pred_quat = np.array(pred_action)[:,3:7]
    gt_quat = np.array(gt_action)[:,3:7]

    r = R.from_quat(pred_quat)
    pred_eular = r.as_euler('xyz')

    r = R.from_quat(gt_quat)
    gt_eular = r.as_euler('xyz')

    def angle_euclidean(angle1, angle2):
        diff_eular = angle1 - angle2
        diff_eular = np.where(abs(diff_eular) > np.pi, (2*np.pi) - abs(diff_eular), abs(diff_eular))
        return np.linalg.norm(diff_eular)

    print("calculate dtw angle")
    dtw_error_xyz, path_xyz = fastdtw(pred_eular, gt_eular, dist=angle_euclidean)
    error_xyz_list = error_divide_time(pred_eular, gt_eular, angle_euclidean, path_xyz)
    mean_dtw_xyz = dtw_error_xyz / len(path_xyz)

    dtw_error_x, path_x = fastdtw(pred_eular[:,0], gt_eular[:,0], dist=angle_euclidean)
    error_x_list = error_divide_time(pred_eular[:,0], gt_eular[:,0], angle_euclidean, path_x)
    mean_dtw_x = dtw_error_x / len(path_x)

    dtw_error_y, path_y = fastdtw(pred_eular[:,1], gt_eular[:,1], dist=angle_euclidean)
    error_y_list = error_divide_time(pred_eular[:,1], gt_eular[:,1], angle_euclidean, path_y)
    mean_dtw_y = dtw_error_y / len(path_y)

    dtw_error_z, path_z = fastdtw(pred_eular[:,2], gt_eular[:,2], dist=angle_euclidean)
    error_z_list = error_divide_time(pred_eular[:,2], gt_eular[:,2], angle_euclidean, path_z)
    mean_dtw_z = dtw_error_z / len(path_z)

    return mean_dtw_xyz, mean_dtw_x, mean_dtw_y, mean_dtw_z, error_xyz_list, error_x_list, error_y_list, error_z_list

def error_divide_time(pred, gt, dist, path):    
    error_list = [0] * len(gt)
    for i,j in path:
        error = dist(pred[i],gt[j])
        error_list[j] = error
    return error_list