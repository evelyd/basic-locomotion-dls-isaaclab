from isaaclab.envs.direct_rl_env import DirectRLEnvCfg
from basic_locomotion_dls_isaaclab.assets.aliengo_asset import ALIENGO_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ImuCfg

class LeggedRobotCfg(DirectRLEnvCfg):

    episode_length_s = 20.0
    decimation = 4
    action_space = 12
    observation_space = 235
    state_space = 0
    # -----------------------------------------------------------
    # Aliengo setup
    # -----------------------------------------------------------
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        #disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            min_position_iteration_count=4,
        ),

        # # class physx:
        # num_threads = 10
        # solver_type = 1  # 0: pgs, 1: tgs
        # num_position_iterations = 4
        # num_velocity_iterations = 0
        # contact_offset = 0.01  # [m]
        # rest_offset = 0.0   # [m]
        # bounce_threshold_velocity = 0.5 #0.5 [m/s]
        # max_depenetration_velocity = 1.0
        # max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
        # default_buffer_size_multiplier = 5
        # contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        horizontal_scale=0.1,  # [m]
        vertical_scale=0.005,  # [m]
        border_size=25,  # [m]
        curriculum=True,
        size=[8.0, 8.0],
        num_rows=10,
        num_cols=20,
        max_init_terrain_level=5,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # # class terrain:
    # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
    # horizontal_scale = 0.1 # [m]
    # vertical_scale = 0.005 # [m]
    # border_size = 25 # [m]
    # curriculum = True
    # static_friction = 1.0
    # dynamic_friction = 1.0
    # restitution = 0.
    # # rough terrain only:
    # measure_heights = True
    # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
    # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    # selected = False # select a unique terrain type and pass all arguments
    # terrain_kwargs = None # Dict of arguments for selected terrain
    # max_init_terrain_level = 5 # starting curriculum state
    # terrain_length = 8.
    # terrain_width = 8.
    # num_rows= 10 # number of terrain rows (levels)
    # num_cols = 20 # number of terrain cols (types)
    # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    # # trimesh only:
    # slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        #ray_alignment='yaw',
        #pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.4, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # an imu sensor in case we don't want any state estimator
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/base", debug_vis=True)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # -----------------------------------------------------------
    # End of Aliengo setup
    # -----------------------------------------------------------

    # class env:
    # num_envs = 4096
    # num_observations = 235
    num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
    # num_actions = 12
    num_state_history = 1
    num_stacked_obs = 1
    num_env_factors = 0
    # env_spacing = 3.  # not used with heightfields/trimeshes
    send_timeouts = True # send time out information to the algorithm
    # episode_length_s = 20 # episode length in seconds
    obs_base_vel = True
    obs_base_vela = True
    obs_height = False
    binarize_base_vela = False
    single_base_vel = False
    single_base_vela = False
    single_height = False
    priv_obs_friction = False
    priv_obs_restitution = False
    priv_obs_joint_friction = False
    priv_obs_height = False
    priv_obs_contact = False

    # class record:
    record = False
    folder = ""

    # class commands:
    curriculum = False
    max_curriculum = 1.
    num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    resampling_time = 10. # time before command are changed[s]
    heading_command = True # if true: compute ang vel command from heading error
    random_gait = True # True in low-level environment training
    discretize = False
    zero_cmd_threshold = 0.2
    separate_lin_ang = False
    # class ranges:
    lin_vel_x = [-1.0, 1.0] # min max [m/s]
    lin_vel_y = [-1.0, 1.0]   # min max [m/s]
    ang_vel_yaw = [-1, 1]    # min max [rad/s]
    heading = [-3.14, 3.14]

    # These are assigned in the asset in isaaclab
    # # class init_state:
    # pos = [0.0, 0.0, 1.] # x,y,z [m]
    # rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
    # lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
    # ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
    # default_joint_angles = { # target angles when action = 0.0
    #     "joint_a": 0.,
    #     "joint_b": 0.}
    # init_joint_angles = {
    #     "joint_a": 0.,
    #     "joint_b": 0.
    # }
    # randomize_rot = False
    # reset_from_buffer = False
    # reset_file_name = None

    # This should also be defined by the robot and the asset
    # # class control:
    # control_type = 'P' # P: position, V: velocity, T: torques
    # # PD Drive parameters:
    # stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
    # damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
    # # action scale: target angle = actionScale * action + defaultAngle
    # # action_scale = 0.5
    # action_mode = "bias"
    # hip_reduction_scale = 1.0
    # # decimation: Number of control action updates @ sim DT per policy DT
    # # decimation = 4
    # kp_factor_range = [0.8, 1.3]
    # kd_factor_range = [0.5, 1.5]
    # ratio_delay = 0.0
    # decimation_range = [4, 8]
    # torque_scale = 1.0

    # again, gets defined in the asset
    # # class asset:
    # file = ""
    # name = "legged_robot"  # actor name
    foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
    penalize_contacts_on = []
    terminate_after_contacts_on = []
    allow_initial_contacts_on = []
    # disable_gravity = False
    # collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
    # fix_base_link = False # fixe the base of the robot
    # default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
    # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    # replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
    # flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    density = 0.001
    angular_damping = 0.
    linear_damping = 0.
    max_angular_velocity = 1000.
    max_linear_velocity = 1000.
    armature = 0.
    thickness = 0.01

    # class domain_rand:
    randomize_friction = True
    friction_range = [0.5, 1.25]
    randomize_restitution = False
    restitution_range = [0.0, 0.4]
    randomize_base_mass = False
    added_mass_range = [-1., 1.]
    randomize_com_displacement = False
    com_displacement_range = [[-0.15, -0.15, -0.15], [0.15, 0.15, 0.15]]
    randomize_joint_props = False
    joint_friction_range = [0.0, 0.2]
    joint_damping_range = [0.0, 0.02]
    randomize_foot_mass = True
    randomize_hip_mass = True
    randomize_thigh_mass = True
    randomize_calf_mass = True
    push_robots = True
    push_interval_s = 15
    max_push_vel_xy = 1.
    lag_timesteps = 0
    swing_lag_timesteps = [0, 0]
    stance_lag_timesteps = [0, 0]
    use_dynamic_kp_scale = False
    # class rewards:
    # class scales:
    termination = -0.0
    tracking_lin_vel = 1.0
    tracking_ang_vel = 0.5
    lin_vel_z = -2.0
    ang_vel_xy = -0.05
    orientation = -0.
    torques = -0.00001
    dof_vel = -0.
    dof_acc = -2.5e-7
    base_height = -0.
    feet_air_time =  1.0
    collision = -1.
    feet_stumble = -0.0
    action_rate = -0.01
    stand_still = -0.

    only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 1.
    max_contact_force = 100. # forces above this value are penalized
    curriculum = False

    # class normalization:
    # class obs_scales:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    height_measurements = 5.0
    clip_observations = 100.
    clip_actions = 100.

    # class noise:
    add_noise = True
    noise_level = 1.0 # scales other values
    # class noise_scales:
    dof_pos = 0.01
    dof_vel = 1.5
    lin_vel = 0.1
    ang_vel = 0.2
    ang_range = [1.0, 1.0]
    gravity = 0.05
    height_measurements = 0.1

    # viewer camera:
    # class viewer:
    ref_env = 0
    pos = [10, 0, 6]  # [m]
    lookat = [11., 5, 3.]  # [m]

    # in Isaaclab, defined in the sim
    # # class sim:
    # dt =  0.005
    # substeps = 1
    # gravity = [0., 0. ,-9.81]  # [m/s^2]
    # up_axis = 1  # 0 is y, 1 is z

# in Isaaclab this is not defined here, it's defined in the pporunnercfg
# class LeggedRobotCfgPPO(BaseConfig):
#     seed = 13
#     runner_class_name = 'OnPolicyRunner'
#     use_wandb = False
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [512, 256, 128]
#         critic_hidden_dims = [512, 256, 128]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         # rnn_type = 'lstm'
#         # rnn_hidden_size = 512
#         # rnn_num_layers = 1

#     class algorithm:
#         # training params
#         value_loss_coef = 1.0
#         use_clipped_value_loss = True
#         clip_param = 0.2
#         entropy_coef = 0.01
#         num_learning_epochs = 5
#         num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
#         learning_rate = 1.e-3 #5.e-4
#         schedule = 'adaptive' # could be adaptive, fixed
#         gamma = 0.99
#         lam = 0.95
#         desired_kl = 0.005
#         max_grad_norm = 1.

#     class runner:
#         policy_class_name = 'ActorCritic'
#         algorithm_class_name = 'PPO'
#         num_steps_per_env = 24 # per iteration
#         max_iterations = 1500 # number of policy updates

#         # logging
#         save_interval = 50 # check for potential saves every this many iterations
#         experiment_name = 'test'
#         run_name = ''
#         # load and resume
#         resume = False
#         load_optimizer = True
#         load_run = -1 # -1 = last run
#         checkpoint = -1 # -1 = last saved model
#         resume_path = None # updated from load_run and chkpt