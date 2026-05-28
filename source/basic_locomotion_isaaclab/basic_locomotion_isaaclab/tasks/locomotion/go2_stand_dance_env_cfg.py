import torch
from isaaclab.utils import configclass
from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

@configclass
class Go2StandDanceEnvCfg(Go2FlatEnvCfg):
    # Overwrite observation space (Gravity(3) + Forward(3) + Cmds(3) + DofPos(12) + DofVel(12) + Actions(12) = 45)
    num_envs = 8192
    observation_space = 47
    action_scale = 0.5
    use_filter_actions = True
    randomize_initial_state = True

    # Disable components from the locomotion template not needed for stand dance
    use_clock_signal = False
    use_observation_history = True
    history_length = 3
    if use_observation_history:
        single_observation_space = observation_space
        observation_space *= history_length

    # Define reward curriculum
    cl_init = 0.4
    cl_step = 0.2
    metric_threshold = 10.0
    term_metric_threshold = 30.0

    # Define command curriculum
    curriculum_cl_step = 0.2

    # Stand Dance Specific Parameters
    lift_up_threshold = [0.25, 0.45]
    scale_factor_low = 0.25
    scale_factor_high = 0.35
    upright_vec = [-0.0524078, 0.0, 1.0]
    allow_contact_steps = 100

    # ------------------------------------------------------------------------
    # Reward Scales (Mapped directly from go2_standdance_config.py)
    # ------------------------------------------------------------------------
    # Set standard locomotion tracking to 0 to avoid interference
    lin_vel_reward_scale = 0.0
    yaw_rate_reward_scale = 0.0
    z_vel_reward_scale = 0.0
    ang_vel_reward_scale = 0.0
    orientation_reward_scale = 0.0
    height_reward_scale = 0.0

    # Stand Dance Rewards
    tracking_lin_vel_stand_scale = 5.0
    tracking_ang_vel_stand_scale = 5.0
    lift_up_linear_scale = 0.8
    upright_scale = 10.0
    # upright_balance_scale = 2.0
    support_polygon_scale = 10.0

    # Penalties
    termination_reward_scale = -100.0
    undesired_contact_reward_scale = -2.0
    joints_torque_reward_scale = -2.0e-5
    joints_accel_reward_scale = -2.5e-5
    rear_air_scale = -0.5
    action_rate_reward_scale = 1.0
    action_q_diff_scale = 0.5
    hip_still_scale = -0.5
    feet_clearance_cmd_linear_scale = 3.0
    feet_slip_scale = -0.4
    foot_shift_scale = -50.0
    joints_energy_reward_scale = 0.0
    collision_scale = -2.0 # in isaacgym was -2.0

    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state.pos = (0.0, 0.0, 0.38) # Standard standing height
        self.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        self.robot.init_state.joint_pos = {
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.6,
        }

@configclass
class Go2StandDanceEnvPlayCfg(Go2StandDanceEnvCfg):
    curriculum_cl_step = 0.0