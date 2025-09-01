import sys
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'aliengo'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah' 
scene = 'flat'  # flat, random_boxes, random_pyramids, perlin

#policy_path = "/home/iit.local/gturrisi/isaaclab_ws_home/basic-locomotion-dls-isaaclab/tested_policies/hyqreal/2025-07-23_09-19-46_8k_128_128_128_hyq/exported/policy.onnx"
policy_path = dir_path + "/../tested_policies/" + robot + "/8k_128_128_128_aliengo_stop_and_go_correct_offset" + "/exported/policy.onnx"

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 25.
    Kd_walking = 2.

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

elif(robot == "go2"):
    Kp_walking = 20.
    Kd_walking = 1.5

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
history_length = 5  # Length of the history of the actions to be used in the RL policy

use_clock_signal = True  # If True, use the clock signal in the RL policy

use_vision = False  # If True, use the vision observations in the RL policy
if(use_vision):
    resolution_heightmap = 0.2  # Resolution of the heightmap in meters
    size_x_heightmap = 0.6  # Size of the heightmap in meters
    size_y_heightmap = 0.6  # Size of the heightmap in meters

observation_space = 48  # Number of observations in the RL policy

use_imu = False
use_rma = False
use_cuncurrent_state_est = False