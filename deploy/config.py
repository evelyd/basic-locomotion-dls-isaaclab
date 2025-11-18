import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'
scene = 'flat'  # flat, random_boxes, random_pyramids, perlin

#policy_path = "/home/iit.local/gturrisi/isaaclab_ws_home/basic-locomotion-dls-isaaclab/tested_policies/hyqreal/2025-07-23_09-19-46_8k_128_128_128_hyq/exported/policy.onnx"
# policy_path = dir_path + "/../tested_policies/" + robot + "/8k_128_128_128_aliengo_stop_and_go_correct_offset" + "/exported/policy.onnx"
policy_path = "/home/edelia-iit.local/git/basic-locomotion-dls-isaaclab/tested_policies/go2/policy_9750_2025-09-09_23-02-03_dof_acc_e6.onnx"

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 21.5
    Kd_walking = 3.5

    Kp_stand_up_and_down = 25.
    Kd_stand_up_and_down = 2.

elif(robot == "go2"):
    Kp_walking = 21.5
    Kd_walking = 3.5

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

    Kp_stand_up_and_down = 175.
    Kd_stand_up_and_down = 20.
else:
    raise ValueError(f"Robot {robot} not supported")

# ----------------------------------------------------------------------------------------------------------------

policy_folder_path = dir_path + "/../tested_policies/" + robot + "/aliengo_symmetricactor"
#policy_folder_path = dir_path + "/../tested_policies/" + robot + "/go2_5asymm"

cuncurrent_state_est_network = policy_folder_path + "/exported/cuncurrent_state_estimator.pth"
rma_network = policy_folder_path + "/exported/rma.pth"

# Load specific training parameters
import yaml
with open(policy_folder_path + "/params/env.yaml", "r") as file:
    training_env = yaml.unsafe_load(file)

use_observation_history = True  # If True, use the history of the actions to compute the RL policy
history_length = 3 #5  # Length of the history of the actions to be used in the RL policy

use_clock_signal = False #True  # If True, use the clock signal in the RL policy

use_vision = False  # If True, use the vision observations in the RL policy
if(use_vision):
    resolution_heightmap = 0.2  # Resolution of the heightmap in meters
    size_x_heightmap = 0.6  # Size of the heightmap in meters
    size_y_heightmap = 0.6  # Size of the heightmap in meters

observation_space = 53 #48  # Number of observations in the RL policy

use_imu = False
use_rma = False
use_cuncurrent_state_est = True #False
default_gait_freq = 2.5  # Hz, frequency of the gait
#TODO needs to be updated per policy
cuncurrent_state_est_network_path = dir_path + "/../tested_policies/" + robot + "/2025-09-07_19-13-16_go2_cuncurrent_se" + "/exported/cuncurrent_state_estimator.pth"
