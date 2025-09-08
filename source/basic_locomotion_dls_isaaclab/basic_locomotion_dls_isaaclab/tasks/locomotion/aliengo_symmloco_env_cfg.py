import numpy as np
import math

from omegaconf import MISSING

import isaaclab.envs.mdp as mdp
import basic_locomotion_dls_isaaclab.tasks.locomotion.mdp as custom_mdp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs.direct_rl_env import DirectRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils import configclass, dataclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import assets as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import basic_locomotion_dls_isaaclab.tasks.custom_events as custom_events
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

# Assume these are imported from their respective files or defined globally
# For this full file, they are placeholders. In your actual setup,
# ensure these are correctly imported or defined.
# Example: from .aliengo_robot_cfg import ALIENGO_CFG, ALIENGO_HIP_ACTUATOR_CFG, ALIENGO_THIGH_ACTUATOR_CFG, ALIENGO_CALF_ACTUATOR_CFG

from basic_locomotion_dls_isaaclab.assets.aliengo_asset import ALIENGO_CFG


# --- Global Flags ---
USE_VEL_CMD = False #True
OBS_T = False
INIT_POSE = "sit"

# ---------------------------------------------------------------------------- #
#                                Reward Settings                               #
# ---------------------------------------------------------------------------- #

@configclass
class AliengoActionCfg:
    """
    Action configuration for a typical quadruped robot in Isaac Lab.
    """
    controller = JointPositionActionCfg(
        asset_name="robot", # Identifies the asset (robot) this action controller applies to
        joint_names=[".*"],
        scale=0.5,
    )

@configclass
class AliengoCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0.3, 0.3), heading=(-0.5 * math.pi, 0.5 * math.pi)
        ),
    )

@configclass
class AliengoRewardCfg:
    """Reward terms for AliengoStandDance environment."""

    # --- Tracking Rewards ---
    tracking_lin_vel: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_tracking_lin_vel(tracking_sigma=0.05),
        func=custom_mdp.tracking_lin_vel,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "tracking_sigma": 0.05, "vel_cmd": USE_VEL_CMD, "scale_factor_low": 0.25, "scale_factor_high": 0.35, "command_name": "base_velocity"},
        weight=0.8, # From CyberCommonCfg.rewards.scales
    )
    tracking_ang_vel: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_tracking_ang_vel(tracking_ang_sigma=0.2),
        func=custom_mdp.tracking_ang_vel,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "tracking_sigma": 0.05, "ang_rew_mode": "heading", "vel_cmd": USE_VEL_CMD, "scale_factor_low": 0.25, "scale_factor_high": 0.35, "command_name": "base_velocity"},
        weight=0.5, # From CyberCommonCfg.rewards.scales
    )

    # --- Height / Orientation Rewards ---
    upright: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_upright(),
        func=custom_mdp.upright,
        weight=100., # From CyberCommonCfg.rewards.scales
    )
    base_height_reward: RewardTermCfg = RewardTermCfg( # This was previously orientation_penalty
        # func=lambda env: env._reward_base_height(desired_base_height=0.35),
        func=custom_mdp.base_height,
        params={"desired_base_height": 0.35},
        weight=1.0,  # Keeping original weight for base_height from AliengoFlatEnvCfg
    )

    # --- Joint / Action Penalties ---
    torque_limits: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_torque_limits(),
        func=custom_mdp.torque_limits,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_torque_limit": 0.5},
        weight=-0.01, # From CyberCommonCfg.rewards.scales
    )
    dof_vel: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_dof_vel(),
        func=custom_mdp.dof_vel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        weight=-1e-4, # From CyberCommonCfg.rewards.scales
    )
    dof_acc: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_dof_acc(),
        func=custom_mdp.dof_acc,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        weight=-2.5e-7, # From CyberCommonCfg.rewards.scales
    )
    dof_pos_limits: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_dof_pos_limits(),
        func=custom_mdp.dof_pos_limits,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        weight=-10, # From CyberCommonCfg.rewards.scales
    )
    action_rate: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_action_rate(),
        func=custom_mdp.action_rate,
        weight=-0.03, # From CyberCommonCfg.rewards.scales
    )
    action_q_diff: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_action_q_diff(),
        func=custom_mdp.action_q_diff,
        params={"allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50},
        weight=-0.5 * 2 if INIT_POSE == "sit" else 0., # Conditional weight
    )

    # --- Feet / Contact Rewards ---
    feet_slip: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_feet_slip(),
        func=custom_mdp.feet_slip,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]), "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"])},
        weight=-0.04 * 10, # From CyberCommonCfg.rewards.scales
    )
    feet_clearance_cmd_linear: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_feet_clearance_cmd_linear(),
        func=custom_mdp.feet_clearance_cmd_linear,
        params={"foot_target": 0.05, "allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50},
        weight=-300, # From CyberCommonCfg.rewards.scales
    )
    collision: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_collision(), # This likely corresponds to undesired_contact_penalty
        func=custom_mdp.collision,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "FR_foot", "FL_foot", "RL_calf", "RR_calf", "RL_thigh", "RR_thigh"]), "allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50},
        weight=-2.0, # From CyberCommonCfg.rewards.scales
    )
    foot_twist: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_foot_twist(),
        func=custom_mdp.foot_twist,
        weight=-0, # From CyberCommonCfg.rewards.scales
    )
    foot_shift: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_foot_shift(),
        func=custom_mdp.foot_shift,
        params={"allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50},
        weight=-50, # From CyberCommonCfg.rewards.scales
    )
    # Assuming 'front_contact_force' and 'hip_still' are custom terms from your original environment
    front_contact_force: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_front_contact_force(),
        func=custom_mdp.front_contact_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FR_foot", "FL_foot"])},
        weight=0.0 # Placeholder weight, set as needed
    )
    hip_still: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_hip_still(),
        func=custom_mdp.hip_still,
        params={"allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50},
        weight=0.0 # Placeholder weight, set as needed
    )

    # --- Air Time / Lift Up Rewards ---
    rear_air: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_rear_air(),
        func=custom_mdp.rear_air,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_foot", "RR_foot"])},
        weight=-0.5, # From CyberCommonCfg.rewards.scales
    )
    lift_up_linear: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_lift_up_linear(),
        func=custom_mdp.lift_up_linear,
        params={"lift_up_threshold_range": [0.15, 0.42]},
        weight=0.8, # From CyberCommonCfg.rewards.scales
    )
    lift_up: RewardTermCfg = RewardTermCfg( # This was inferred from the compute_rewards snippet
        # func=lambda env: env._reward_lift_up(),
        func=custom_mdp.lift_up,
        params={"liftup_target": 0.42},
        weight=0.0 # Placeholder weight, set as needed
    )
    stand_air: RewardTermCfg = RewardTermCfg(
        # func=lambda env: env._reward_stand_air(),
        func=custom_mdp.stand_air,
        params={"allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50},
        weight=-50 * 0, # From CyberCommonCfg.rewards.scales, effectively 0
    )

# ---------------------------------------------------------------------------- #
#                              Observation Settings                            #
# ---------------------------------------------------------------------------- #

@configclass
class AliengoObservationCfg:
    """Observation terms for AliengoStandDance environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=mdp.imu_projected_gravity,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        projected_forward_vec_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_forward_vec,
            func=custom_mdp.get_projected_forward_vec_from_imu,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        commands_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.commands[:, :3] * env.commands_scale,
            func=custom_mdp.vel_commands,
            params={"command_name": "base_velocity"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        joint_pos_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: (env.robot.data.dof_pos - env.robot.data.dof_default_pos) * 1.0,
            func=mdp.joint_pos_rel,
            # noise={"scale": 0.002, "type": "uniform"}
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        joint_vel_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.robot.data.dof_vel * 0.0, # Normalization from CyberDog config
            func=mdp.joint_vel_rel,
            # noise={"scale": 0.002, "type": "uniform"}
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        actions_obs: ObservationTermCfg = ObservationTermCfg(
            func=mdp.last_action,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        clock_inputs_obs: ObservationTermCfg = ObservationTermCfg(
            func=custom_mdp.rear_clock_inputs,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ValueCfg(ObsGroup):
        projected_gravity_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=mdp.imu_projected_gravity,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        projected_forward_vec_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_forward_vec,
            func=custom_mdp.get_projected_forward_vec_from_imu,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        commands_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.commands[:, :3] * env.commands_scale,
            func=custom_mdp.vel_commands,
            params={"command_name": "base_velocity"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        joint_pos_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: (env.robot.data.dof_pos - env.robot.data.dof_default_pos) * 1.0,
            func=mdp.joint_pos_rel,
            # noise={"scale": 0.002, "type": "uniform"}
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        joint_vel_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.robot.data.dof_vel * 0.0, # Normalization from CyberDog config
            func=mdp.joint_vel_rel,
            # noise={"scale": 0.002, "type": "uniform"}
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        actions_obs: ObservationTermCfg = ObservationTermCfg(
            func=mdp.last_action,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        clock_inputs_obs: ObservationTermCfg = ObservationTermCfg(
            func=custom_mdp.rear_clock_inputs,
            noise=Unoise(n_min=-0.002, n_max=0.002),
            history_length=3,
        )
        base_lin_vel_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        base_ang_vel_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        friction_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=custom_mdp.friction,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        restitution_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=custom_mdp.restitution,
            # noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        joint_friction_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=custom_mdp.joint_friction,
            # noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        com_disp_obs: ObservationTermCfg = ObservationTermCfg(
            # func=lambda env: env.projected_gravity,
            func=custom_mdp.com_displacement,
            # noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        mass_offset_obs: ObservationTermCfg = ObservationTermCfg(
            func=custom_mdp.mass_offset,
            # noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        link_in_contact_obs: ObservationTermCfg = ObservationTermCfg(
            func=custom_mdp.link_in_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "FR_foot", "FL_foot", "RL_calf", "RR_calf", "RL_thigh", "RR_thigh"])},
            # noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: ValueCfg = ValueCfg()


# ---------------------------------------------------------------------------- #
#                               Termination Settings                           #
# ---------------------------------------------------------------------------- #

@configclass
class AliengoTerminationCfg:
    """Termination terms for AliengoStandDance environment."""

    # robot.terminate_bodies = ["base", "trunk", "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"]

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    illegal_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]), "threshold": 1.0},
    )
    contact_with_mercy = TerminationTermCfg(
        func=custom_mdp.contact_with_mercy,
        params={"term_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", "FR_thigh", "FL_thigh", "FR_calf", "FL_calf", "FR_foot", "FL_foot", "RL_thigh", "RR_thigh"]), "allow_init_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot", "RL_calf", "RR_calf"]), "allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50}
    )
    joint_position_protect = TerminationTermCfg(
        func=custom_mdp.joint_position_protect,
    )
    stand_air = TerminationTermCfg(
        func=custom_mdp.stand_air,
        params={"allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50}
    )
    abrupt_change = TerminationTermCfg(
        func=custom_mdp.abrupt_change,
        params={"allow_contact_steps": 0 if INIT_POSE == "upright" else 30 if INIT_POSE == "sit" else 50, "max_dof_change": 0.3}
    )
    # base_pitch_invalid = TerminationTermCfg(
    #     func=custom_mdp.is_base_pitch_invalid,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "threshold": 1.5},
    # )

# ---------------------------------------------------------------------------- #
#                              Randomization Settings                          #
# ---------------------------------------------------------------------------- #

@configclass
class AliengoEventCfg:
    """Configuration for randomization."""

    physics_material = EventTermCfg(
        func=custom_mdp.randomize_rigid_body_material_save,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    add_com_disp = EventTermCfg(
        func=custom_mdp.randomize_rigid_body_com_save,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.0, 0.0), "z": (-0.01, 0.01)},
            # "operation": "add",
        },
    )

    add_base_mass = EventTermCfg(
        func=custom_mdp.randomize_rigid_body_mass_save,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # scale_ = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
    #             "operation": "scale"},
    # )

    #TODO isn't this a duplicate? -> no, because it's applied to the other links, not base
    scale_all_link_masses = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip", ".*_thigh", ".*_calf"]), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )


    base_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )


    """add_all_joint_default_pos = EventTermCfg(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "pos_distribution_params": (-0.05, 0.05),
                "operation": "add"},
    )"""

    """joint_parameters = EventTermCfg(
    func=mdp.randomize_joint_parameters,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "armature_distribution_params": (0.0, 0.2),
        "operation": "add",
        "distribution": "uniform",
    },
    )"""

    scale_all_joint_friction_model = EventTermCfg(
        func=custom_mdp.randomize_joint_parameters_save,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "friction_distribution_params": (0.2, 2.0),
                "armature_distribution_params": (0.0, 1.0),
                "operation": "scale"},
    )


    # scale_all_joint_armature_model = EventTermCfg(
    #     func=custom_events.randomize_joint_friction_model,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #             "armature_distribution_params": (0.0, 1.0),
    #             "operation": "scale"},
    # )



    actuator_gains = EventTermCfg(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (-5.0, 5.0),
        "damping_distribution_params": (-1.0, 1.0),
        "operation": "add",
        "distribution": "uniform",
    },
    )

    # interval
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
    interval_range_s=(15.0, 15.0), # interval 15 in isaacgym
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.0, 0.0), # -1, 1 in isaacgym, z not used
                                   "roll": (-0.0, 0.0), "pitch": (-0.0, 0.0), "yaw": (-0.0, 0.0)}}, # not used in isaacgym
    )

    # zero command velocity
    """zero_command_velocity = EventTermCfg(
        func=custom_events.zero_command_velocity,
        mode="interval",
        interval_range_s=(19.0, 19.0),
    )"""

    """# reset command velocity
    resample_command_velocity = EventTermCfg(
        func=custom_events.resample_command_velocity,
        mode="interval",
        interval_range_s=(11.0, 11.0),
    )"""

    # TODO i didn't have this in mine
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # TODO i didn't have this one either
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

SIT_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.19),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            ".*_thigh_joint": 1.3,
            ".*_calf_joint": -2.5,
        },
        joint_vel={".*": 0.0},
)

UPRIGHT_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.55),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            ".*_thigh_joint": 2.3,
            ".*_calf_joint": -2.0,
        },
        joint_vel={".*": 0.0},
)

@configclass
class AliengoStandDanceSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        # max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # These direct attributes are now part of the base ManagerBasedEnvCfg.EnvCfg
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/base", debug_vis=True)


# ---------------------------------------------------------------------------- #
#                               Main Environment Configuration                 #
# ---------------------------------------------------------------------------- #

@configclass
class AliengoStandDanceEnvCfg(ManagerBasedRLEnvCfg): # <--- Inherit from ManagerBasedEnvCfg
    """Configuration for AliengoStandDance environment in Isaac Lab."""

    # Curriculum settings
    # curriculum: bool = (INIT_POSE == "sit") # Adapt as needed for Aliengo
    cl_init: float = 0.6
    cl_step: float = 0.2
    allow_contact_steps: int = 50

    # Specialized settings
    default_gait_freq = 2.5

    # Simulation settings (as before, but ensure full path for PhysxCfg if you uncomment it)
    # sim: SimulationCfg = SimulationCfg(
    #     dt=1 / 200,
    #     # render_interval=decimation,
    #     #disable_contact_processing=True,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     #physx=PhysxCfg(
    #     #    gpu_max_rigid_contact_count=2**20,
    #     #    gpu_max_rigid_patch_count=2**24,
    #     #),
    # )

    # Asset configuration (as before)
    # robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Rigid body groups for terminations/rewards (as before)


    # Terrain configuration (as before)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", # Use TerrainFlatCfg() for a flat plane with more options
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Assign separate configuration classes
    scene: AliengoStandDanceSceneCfg = AliengoStandDanceSceneCfg()
    rewards: AliengoRewardCfg = AliengoRewardCfg()
    observations: AliengoObservationCfg = AliengoObservationCfg()
    terminations: AliengoTerminationCfg = AliengoTerminationCfg()
    events: AliengoEventCfg = AliengoEventCfg() # Changed from randomization to events
    actions: AliengoActionCfg = AliengoActionCfg()
    commands: AliengoCommandsCfg = AliengoCommandsCfg()

    def __post_init__(self):
        """Post initialization."""

        # Call parent's post-init for ManagerBasedEnvCfg
        super().__post_init__()

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.scene.num_envs = 8192
        self.scene.env_spacing = 3.0
        self.scene.replicate_physics = True

        if INIT_POSE == "sit":
            self.scene.robot.replace(init_state=SIT_INIT_STATE)

@configclass
class AliengoStandDanceDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for AliengoStandDance environment in Isaac Lab."""

    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12

    # Desired clip actions
    desired_clip_actions = 3.0
    use_filter_actions = True

    observation_space = 3 # projected gravity
    observation_space += 3 # projected_forward_vec
    observation_space += 3 # velocity commands
    observation_space += 12 # joint pos
    observation_space += 12 # joint vel
    observation_space += 12 # past action
    observation_space += 2 # clock inputs

    # observation history
    use_observation_history = True
    history_length = 3
    if(use_observation_history):
        single_observation_space = observation_space # Placeholder. Later we may add map, but only from the latest obs
        observation_space *= history_length

    obs_scale_joint_pos = 1.0
    obs_scale_joint_vel = 0.0
    obs_scale_lin_vel = 2.0
    obs_scale_ang_vel = 0.25
    add_noise = True
    noise_level = 1.0
    noise_scale_joint_pos = 0.01
    noise_scale_joint_vel = 1.5
    noise_scale_lin_vel = 0.1
    noise_scale_ang_vel = 0.2
    noise_scale_ang_range = [1.0, 1.0]
    noise_scale_gravity = 0.05

    default_gait_freq = 2.5
    kappa_gait_probs = 0.07
    measure_heights = False

    terminate_after_contacts_on = ["base", ".*hip", ".*thigh", "FR_calf", "FL_calf", "FR_foot", "FL_foot"]
    penalize_contacts_on = ["base", ".*hip", ".*thigh", ".*calf", "FL_foot", "FR_foot"] # stand
    allow_initial_contacts_on = [".*foot", "RL_calf", "RR_calf"]
    max_dof_change = 0.3
    command_clip_ang_vel = 0.25 * np.pi

    use_privileged_obs = True
    state_space = 0
    if(use_privileged_obs):
        state_space = observation_space
        state_space += 3 + 3 + 1 + 1 + 12 + 1 + 1 + len(penalize_contacts_on)

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Asset configuration
    robot: ArticulationCfg = ALIENGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # Terrain configuration
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        # attach_yaw_only=True,
        ray_alignment='yaw',
        #pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.4, 1.0]),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[0.6, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # an imu sensor in case we don't want any state estimator
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/base", debug_vis=True)

    command_curriculum = USE_VEL_CMD
    command_cl_init = 0.6
    command_cl_step = 0.2
    command_max_curriculum = 1.
    command_resampling_time = 10.

    @configclass
    class MyCommandsCfg:
        ranges: dict[str, list[float]] = MISSING

    commands: MyCommandsCfg = MyCommandsCfg(
        ranges={
            # "lin_vel_x": [-0.3, 0.3],
            # "lin_vel_y": [-0.0, 0.0],
            # "ang_vel_z": [-0.3, 0.3],
            "lin_vel_x": [-0.0, 0.0],
            "lin_vel_y": [-0.0, 0.0],
            "ang_vel_z": [-0.0, 0.0],
            # "heading": [-0.5 * np.pi, 0.5 * np.pi],
            "heading": [0.0, 0.0],
        }
    )

    init_state_randomize_rot = (INIT_POSE == "upright")

    reward_curriculum = (INIT_POSE == "sit")
    reward_cl_init = 0.6
    reward_cl_step = 0.2
    reward_allow_contact_steps = 0 if (INIT_POSE == "upright") else 30 if (INIT_POSE == "sit") else 50
    reward_kappa_gait_probs = 0.07

    reward_base_height_target = 0.54
    reward_soft_dof_vel_limit = 1.
    reward_max_contact_force = 100.
    control_action_scale = 0.5 # TODO in isaacgym also used for other stuff, here only used in reward
    reward_upright_vec = [0.0, 0.0, 1.0]

    # Stand dance specific
    reward_tracking_liftup_sigma = 0.03
    reward_liftup_target = reward_base_height_target
    reward_lift_up_threshold = [0.23, reward_base_height_target]
    reward_tracking_sigma = 0.05
    reward_scale_factor_low = 0.25
    reward_scale_factor_high = 0.35
    reward_ang_rew_mode = "heading"
    reward_tracking_ang_sigma = 0.2
    reward_foot_target = 0.05

    @configclass
    class MyRewardsCfg:
        scales: dict[str, list[float]] = MISSING

    feet_slip_wt = -0.04 * 10
    action_q_diff_wt = -0.5 * 2 if INIT_POSE == "sit" else 0.
    rewards: MyRewardsCfg = MyRewardsCfg(
        scales={
            "feet_slip": feet_slip_wt,
            "feet_clearance_cmd_linear": -300,
            "collision": -2.0e2,
            "torque_limits": -0.01,
            "tracking_lin_vel": 0.8,
            "tracking_ang_vel": 0.5,
            "rear_air": -0.5,
            "action_rate": -0.03,
            "action_q_diff": action_q_diff_wt,
            "stand_air": -50 * 0,
            "dof_vel": -1e-4,
            "dof_acc": -2.5e-7,
            "dof_pos_limits": -10,
            "upright": 1.2e2,
            "lift_up_linear": 0.8e2,
            "time_upright": 500, # new
            "foot_twist": -0,
            "foot_shift": -50,
            # "termination": -50.0,
            "lin_vel_z": -2.0e5,
            "ang_vel_xy": -0.05e5,
            "orientation": -0.,
            "torques": -0.00001,
            "base_height": -0.,
            "feet_air_time":  1.0,
            "stumble": -0.0,
            # "front_feet_air_time": 500,
        }
    ) #TODO make terminations less strict with curriculum?

    # events
    events: AliengoEventCfg = AliengoEventCfg()
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=3.0, replicate_physics=True)

    #TODO use the action and obs noise models same as in aliengo env cfg?

    def __post_init__(self):
        """Post initialization."""

        # Call parent's post-init for ManagerBasedEnvCfg
        super().__post_init__()

        if INIT_POSE == "sit":
            self.robot.replace(init_state=SIT_INIT_STATE)