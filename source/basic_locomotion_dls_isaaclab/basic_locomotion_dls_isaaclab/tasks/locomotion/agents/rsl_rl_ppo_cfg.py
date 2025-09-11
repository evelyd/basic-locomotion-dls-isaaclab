# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from pathlib import Path
from dataclasses import MISSING

@configclass
class DiscriminatorCfg:
    """Configuration for the discriminator network."""

    class_name: str = "Discriminator"
    """The discriminator class name. Default is Discriminator."""

    hidden_dims: list[int] = MISSING
    """The hidden dimensions of the discriminator network."""

    reward_scale: float = MISSING
    """The reward coefficient."""


@configclass
class MorphologycalSymmetriesCfg:
    """Configuration for using morphosymm-rl."""

    class_name: str = "MorphologycalSymmetries"
    """The class name."""

    obs_space_names_actor =  None
    """The observation space names for the actor network."""

    obs_space_names_critic = None
    """The observation space names for the critic network."""

    action_space_names = None
    """The action space names."""

    joints_order = None
    """The order of the joints in the robot."""

    robot_name = None
    """The name of the robot to use inside Morphosymm."""


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 50
    experiment_name = "flat_direct"
    empirical_normalization = False

    wandb_project = "aliengo-rl"

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="fixed", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    #AMP Related Stuff
    amp_data_path = "./../../../amp_dataset/"
    dataset_names = ["flat"]
    dataset_weights = [1.0, 1.0]
    slow_down_factor = 1.0
    discriminator = DiscriminatorCfg(
        hidden_dims=[128, 128],
        reward_scale=0.1,
    )

    # Symmetry Related Stuff
    history_length = 5
    obs_space_names_actor = [
            "base_lin_vel:base",
            "base_ang_vel:base",
            "gravity:base",
            "ctrl_commands",
            "default_qpos_js_error",
            "qvel_js",
            "actions",
            "clock_data",
        ]*int(history_length)
    obs_space_names_critic = obs_space_names_actor
    """obs_space_names_critic += ["position_gains",
            "velocity_gains",
            "friction_static",
            "friction_dynamic",
            "armature"
        ]"""

    morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        action_space_names = ["actions"],
        joints_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ],
        robot_name = "a1",
    )


@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 300 #50
    experiment_name = "rough_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        #actor_hidden_dims=[512, 256, 128],
        #critic_hidden_dims=[512, 256, 128],
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    #AMP Related Stuff
    amp_data_path = "./../../../amp_dataset/"
    dataset_names = ["flat", "boxes", "stairs"]
    dataset_weights = [1.0, 1.0, 1.0, 1.0]
    slow_down_factor = 1.0
    discriminator = DiscriminatorCfg(
        hidden_dims=[1024, 512],
        reward_scale=1.0,
    )

    # Symmetry Related Stuff
    history_length = 5
    obs_space_names_actor = [
            "base_lin_vel:base",
            "base_ang_vel:base",
            "gravity:base",
            "ctrl_commands",
            "default_qpos_js_error",
            "qvel_js",
            "actions",
            "clock_data",
        ]*int(history_length)
    obs_space_names_critic = obs_space_names_actor
    """obs_space_names_critic += ["position_gains",
            "velocity_gains",
            "friction_static",
            "friction_dynamic",
            "armature"
        ]"""
    #obs_space_names_actor += "heightmap:rows4xcols4"

    morphologycal_symmetries_cfg = MorphologycalSymmetriesCfg(
        obs_space_names_actor = obs_space_names_actor,
        obs_space_names_critic = obs_space_names_critic,
        action_space_names = ["actions"],
        joints_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ],
        robot_name = "a1",
    )

@configclass
class FlatPPOEMLPRunnerCfg(FlatPPORunnerCfg):
    """Runner configuration for locomotion with EMLP."""

    experiment_name = "locomotion-flat-emlp"
    wandb_project = experiment_name

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSymm",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

@configclass
class KoopmanCfg:
    model: dict[str, list[float]] = MISSING
    robot: dict[str, list[float]] = MISSING

@configclass
class FlatPPOCDAEOnlineRunnerCfg(FlatPPORunnerCfg):
    """Runner configuration for ErgoCub with online C-DAE. Only defined for non-ideal velocity task."""

    experiment_name = "locomotion-flat-cdae-online"
    wandb_project = experiment_name

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPODAEOnline", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="fixed", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    koopman = KoopmanCfg(
        model={'name': 'cdae',
               'equivariant': False,
               'activation': 'ELU',
               'num_layers': 5,
               'num_hidden_units': 128,
               'batch_norm': False,
               'obs_pred_w': 1.0,
               'orth_w': 0.0,
               'corr_w': 0.0,
               'bias': True,
               'constant_function': True,
               'num_mini_batches': 8,
               'mini_batch_size': 256,
               'beta_initial': 0.4,
               'beta_annealing_steps': 4000},
        robot={'name': 'aliengo',
                'lr': 1e-3,
                'max_epochs': 200,
                'obs_state_ratio': 3,
                'state_obs': ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'velocity_commands_xy', 'velocity_commands_z', 'joint_pos', 'joint_vel', 'prev_action', 'clock_data'],
                'action_obs': ['action'],
                'state_dim': 48 + 4,
                'action_dim': 12,
                'pred_horizon': 5,
                'frames_per_state': 1},
    )

@configclass
class FlatPPOEMLPECDAEOnlineRunnerCfg(FlatPPOCDAEOnlineRunnerCfg):
    """Runner configuration for ErgoCub with EC-DAE."""

    experiment_name = "ergocub-amp-emlp-ecdae-online"
    wandb_project = experiment_name

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSymm",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    koopman = KoopmanCfg(
        model={'name': 'ecdae',
               'equivariant': False,
               'activation': 'ELU',
               'num_layers': 5,
               'num_hidden_units': 128,
               'batch_norm': False,
               'obs_pred_w': 1.0,
               'orth_w': 0.0,
               'corr_w': 0.0,
               'bias': True,
               'constant_function': True,
               'num_mini_batches': 8,
               'mini_batch_size': 256,
               'beta_initial': 0.4,
               'beta_annealing_steps': 4000,
               'equivariant': True,
               'group_avg_trick': True,
               'state_dependent_obs_dyn': False},
        robot={'name': 'aliengo',
                'lr': 1e-3,
                'max_epochs': 200,
                'obs_state_ratio': 3,
                'state_obs': ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'velocity_commands_xy', 'velocity_commands_z', 'joint_pos', 'joint_vel', 'prev_action', 'clock_data'],
                'action_obs': ['action'],
                'state_dim': 48 + 4,
                'action_dim': 12,
                'pred_horizon': 5,
                'frames_per_state': 1},
    )

@configclass
class LeggedRobotCfgPPO(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500

    save_interval = 50 # check for potential saves every this many iterations
    experiment_name = 'test'
    run_name = ''
    # load and resume TODO not sure about the below
    resume = False
    load_optimizer = True
    load_run = -1 # -1 = last run
    checkpoint = -1 # -1 = last saved model
    resume_path = None # updated from load_run and chkpt

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

@configclass
class CommonCfgPPO(LeggedRobotCfgPPO):
    """Common configuration for the legged robot tasks."""
    use_wandb = True

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

@configclass
class StandDanceCfgPPO(CommonCfgPPO):
    """Configuration for the stand dance task using PPO."""
    experiment_name = "stand_dance_go2"
    wandb_project = experiment_name
    max_iterations = 30000
    save_interval = 300

@configclass
class StandDanceEMLPCfgPPO(StandDanceCfgPPO):
    """Configuration for the stand dance task using PPO."""
    experiment_name = "stand_dance_go2_emlp"
    wandb_project = experiment_name
    max_iterations = 30000
    save_interval = 300
    robot_name = 'go2'

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSymm", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

@configclass
class StandDanceCDAEOnlineCfgPPO(StandDanceCfgPPO):
    """Configuration for the stand dance task using PPO."""
    experiment_name = "stand_dance_go2_cdae_online"
    wandb_project = experiment_name
    max_iterations = 30000
    save_interval = 300

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPODAEOnline", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

    koopman = KoopmanCfg(
        model={'name': 'cdae',
               'equivariant': False,
               'activation': 'ELU',
               'num_layers': 5,
               'num_hidden_units': 128,
               'batch_norm': False,
               'obs_pred_w': 1.0,
               'orth_w': 0.0,
               'corr_w': 0.0,
               'bias': True,
               'constant_function': True,
               'num_mini_batches': 8,
               'mini_batch_size': 256,
               'beta_initial': 0.4,
               'beta_annealing_steps': 4000},
        robot={'name': 'go2',
                'lr': 1e-3,
                'max_epochs': 200,
                'obs_state_ratio': 3, #TODO change the obs, state dim, etc.
                'state_obs': ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'forward_vec', 'velocity_commands_xy', 'velocity_commands_z', 'joint_pos', 'joint_vel', 'prev_action', 'clock_input'],
                'action_obs': ['action'],
                'state_dim': 53,
                'action_dim': 12,
                'pred_horizon': 5,
                'frames_per_state': 1},
    )

@configclass
class StandDanceEMLPECDAEOnlineCfgPPO(StandDanceCfgPPO):
    """Configuration for the stand dance task using PPO."""
    experiment_name = "stand_dance_go2_emlp_ecdae_online"
    wandb_project = experiment_name
    max_iterations = 30000
    save_interval = 300

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticSymm", #ActorCritic, ActorCriticRecurrent, ActorCriticSymm, ActorCriticMoE
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPODAEOnline", #PPO, PPOSymmDataAugmented #AMP_PPO
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

    koopman = KoopmanCfg(
        model={'name': 'ecdae',
               'equivariant': False,
               'activation': 'ELU',
               'num_layers': 5,
               'num_hidden_units': 128,
               'batch_norm': False,
               'obs_pred_w': 1.0,
               'orth_w': 0.0,
               'corr_w': 0.0,
               'bias': True,
               'constant_function': True,
               'num_mini_batches': 8,
               'mini_batch_size': 256,
               'beta_initial': 0.4,
               'beta_annealing_steps': 4000,
               'equivariant': True,
               'group_avg_trick': True,
               'state_dependent_obs_dyn': False},
        robot={'name': 'go2',
                'lr': 1e-3,
                'max_epochs': 200,
                'obs_state_ratio': 3,
                'state_obs': ['base_lin_vel', 'base_ang_vel', 'projected_gravity', 'forward_vec', 'velocity_commands_xy', 'velocity_commands_z', 'joint_pos', 'joint_vel', 'prev_action', 'clock_input'],
                'action_obs': ['action'],
                'state_dim': 53,
                'action_dim': 12,
                'pred_horizon': 5,
                'frames_per_state': 1},
    )