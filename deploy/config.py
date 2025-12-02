import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'
scene = 'flat'  # flat, random_boxes, random_pyramids, perlin

#policy_path = "/home/iit.local/gturrisi/isaaclab_ws_home/basic-locomotion-dls-isaaclab/tested_policies/hyqreal/2025-07-23_09-19-46_8k_128_128_128_hyq/exported/policy.onnx"
# policy_path = dir_path + "/../tested_policies/" + robot + "/2025-09-07_19-13-16_go2_cuncurrent_se" + "/exported/policy.onnx"
# policy_path = dir_path + "/../tested_policies/" + robot + "/policy_7950_2025-09-16_12-24-25_rough_blind_emlp_ecdae_online.onnx"
# policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_9900_2025-09-16_13-53-36_baseline_freq1_4.onnx"
policy_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_29999_2025-11-29_11-18-06.onnx"
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
history_length = 3 #5  # Length of the history of the actions to be used in the RL policy

use_clock_signal = False #True  # If True, use the clock signal in the RL policy

use_vision = False  # If True, use the vision observations in the RL policy
if(use_vision):
    resolution_heightmap = 0.2  # Resolution of the heightmap in meters
    size_x_heightmap = 0.6  # Size of the heightmap in meters
    size_y_heightmap = 0.6  # Size of the heightmap in meters

observation_space = 53#48  # Number of observations in the RL policy

use_imu = True #False
use_rma = False
use_cuncurrent_state_est = True
# cuncurrent_state_est_network_path = dir_path + "/../tested_policies/" + robot + "/2025-09-07_19-13-16_go2_cuncurrent_se" + "/exported/cuncurrent_state_estimator.pth"
cuncurrent_state_est_network_path = "/home/evelyd/git/basic-locomotion-dls-isaaclab/tested_policies/go2/cuncurrent_state_estimator_2025-11-29_11-18-06.pth"

# stand dance specific
default_gait_freq = 0.67
obs_scale_joint_pos = 1.0
obs_scale_joint_vel = 0.05
obs_scale_lin_vel = 2.0
obs_scale_ang_vel = 0.25