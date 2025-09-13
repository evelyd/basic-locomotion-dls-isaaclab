import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'
scene = 'flat'  # flat, random_boxes, random_pyramids, perlin

# policy_path = "/home/iit.local/gturrisi/isaaclab_ws_home/basic-locomotion-dls-isaaclab/tested_policies/hyqreal/2025-07-23_09-19-46_8k_128_128_128_hyq/exported/policy.onnx"
# policy_path = dir_path + "/../tested_policies/" + robot + "/2025-09-07_19-13-16_go2_cuncurrent_se" + "/exported/policy.onnx"
# policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_10200_2025-09-09_15-00-11_def_rew_ang_vel_cmd.onnx"
# policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_7800_2025-09-11_01-24-20_base_est_base_lin_vel.onnx"
# policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_9900_2025-09-11_17-12-07_baseline_base_est.onnx"
# policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_9900_2025-09-11_17-14-58_emlp_ecdae_online_base_est.onnx"
# policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_5700_2025-09-12_16-35-23_fixed_sit_joint_init.onnx"
policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_9750_2025-09-09_23-02-03_dof_acc_e6.onnx"
# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 25.
    Kd_walking = 2.

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

elif(robot == "go2"):
    Kp_walking = 25.
    Kd_walking = 2.

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.
elif(robot == "b2"):
    Kp_walking = 20.
    Kd_walking = 1.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.
elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.
else:
    raise ValueError(f"Robot {robot} not supported")



RL_FREQ = 50  # Hz, frequency of the RL controller
action_scale = 0.5  # Scale of the RL actions, used to scale the actions from the RL policy

use_clip_actions = True  # If True, clip the actions to avoid too high torques
clip_actions = 3.0  # Clip the actions to avoid too high torques

use_observation_history = True  # If True, use the history of the actions to compute the RL policy
history_length = 3  # Length of the history of the actions to be used in the RL policy

use_clock_signal = False  # If True, use the clock signal in the RL policy

use_vision = False  # If True, use the vision observations in the RL policy
if(use_vision):
    resolution_heightmap = 0.2  # Resolution of the heightmap in meters
    size_x_heightmap = 0.6  # Size of the heightmap in meters
    size_y_heightmap = 0.6  # Size of the heightmap in meters

observation_space = 53  # Number of observations in the RL policy

use_imu = False
use_rma = False
use_cuncurrent_state_est = use_imu
# cuncurrent_state_est_network_path = dir_path + "/../tested_policies/" + robot + "/2025-09-07_19-13-16_go2_cuncurrent_se" + "/exported/cuncurrent_state_estimator.pth"
# cuncurrent_state_est_network_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/cuncurrent_state_estimator_baseline_final.pth"
cuncurrent_state_est_network_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/cuncurrent_state_estimator_emlp_ecdae_online_final.pth"

# stand dance specific
default_gait_freq = 2.5
obs_scale_joint_pos = 1.0
obs_scale_joint_vel = 0.05
obs_scale_lin_vel = 2.0
obs_scale_ang_vel = 0.25
init_base_height = 0.22
init_qpos = [0.0, 1.2, -2.2,
             0.0, 1.2, -2.2,
             0.0, 1.2, -2.2,
             0.0, 1.2, -2.2]  # Sit pose