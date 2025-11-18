# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.assets import Articulation, DeformableObject, RigidObject
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        pos_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos



def randomize_joint_friction_model(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    first_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    second_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the friction parameters used in joint friction model.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # -- friction
    if friction_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                friction = actuator.friction_static.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_static[env_ids[:, None], actuator_joint_ids] = friction

                friction = actuator.friction_dynamic.to(asset.device).clone()
                friction = _randomize_prop_by_op(
                    friction, friction_distribution_params, env_ids, torch.arange(friction.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.friction_dynamic[env_ids[:, None], actuator_joint_ids] = friction

    if armature_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                armature = actuator.armature.to(asset.device).clone()
                armature = _randomize_prop_by_op(
                    armature, armature_distribution_params, env_ids, torch.arange(armature.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.armature[env_ids[:, None], actuator_joint_ids] = armature


def randomize_joint_delay_model(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    first_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    second_order_delay_filter_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):

    """
    Randomize the delay used in joint hydraulic model.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if first_order_delay_filter_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                first_order_delay_filter = actuator.first_order_delay_filter.to(asset.device).clone()
                first_order_delay_filter = _randomize_prop_by_op(
                    first_order_delay_filter, first_order_delay_filter_distribution_params, env_ids, torch.arange(first_order_delay_filter.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.first_order_delay_filter[env_ids[:, None], actuator_joint_ids] = first_order_delay_filter

    if second_order_delay_filter_distribution_params is not None:
        for actuator in asset.actuators.values():
            actuator_joint_ids = [joint_id in joint_ids for joint_id in actuator.joint_indices]
            if sum(actuator_joint_ids) > 0:
                second_order_delay_filter = actuator.second_order_delay_filter.to(asset.device).clone()
                second_order_delay_filter = _randomize_prop_by_op(
                    second_order_delay_filter, second_order_delay_filter_distribution_params, env_ids, torch.arange(second_order_delay_filter.shape[1]), operation=operation, distribution=distribution
                )[env_ids][:, actuator_joint_ids]
                actuator.second_order_delay_filter[env_ids[:, None], actuator_joint_ids] = second_order_delay_filter


def zero_command_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):

    env._commands[env_ids, 0] = 0.0
    env._commands[env_ids, 1] = 0.0
    env._commands[env_ids, 2] = 0.0


def resample_command_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):

    # Sample new commands
    env._commands[env_ids] = torch.zeros_like(env._commands[env_ids]).uniform_(-1.0, 1.0)
    env._commands[env_ids, 0] *= 0.5
    env._commands[env_ids, 1] *= 0.25
    env._commands[env_ids, 2] *= 0.3

def reset_multiple_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    state1: dict = None,
    state2: dict = None,
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # get halfway point
    half_point = len(env_ids) // 2

    # Get the default root state for the environments to be reset
    default_root_states = asset.data.default_root_state[env_ids].clone()

    # --- Create State 1 Tensors ---
    # The number of environments for this group
    num_group1 = half_point
    # Expand the root pose and velocity to match the number of environments
    # group1_root_state = torch.tensor(state1["root_state"], device=asset.device).repeat(num_group1, 1)
    group1_root_state = torch.tensor(state1.pos + state1.rot + (0,)*6, device=asset.device).repeat(num_group1, 1)

    # --- Create State 2 Tensors ---
    # The number of environments for this group
    num_group2 = len(env_ids) - half_point
    # Expand the root pose and velocity to match the number of environments
    # group2_root_state = torch.tensor(state2["root_state"], device=asset.device).repeat(num_group2, 1)
    group2_root_state = torch.tensor(state2.pos + state2.rot + (0,)*6, device=asset.device).repeat(num_group2, 1)

    # --- Combine and Finalize the Tensors ---
    combined_root_states = torch.cat((group1_root_state, group2_root_state), dim=0)

    # Preserve original x and y positions
    combined_root_states[:, 0:2] = default_root_states[:, 0:2]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = combined_root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(combined_root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = combined_root_states[:, 7:13] + rand_samples

    print(f"writing base positions {positions} and orientations {orientations} and velocities {velocities}")

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_multiple_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    state1: dict = None,
    state2: dict = None,
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    default_joint_pos = asset.data.default_joint_pos[env_ids].to(asset.device).clone()

    # get halfway point
    half_point = len(env_ids) // 2

    # --- Create State 1 Tensors ---
    # The number of environments for this group
    num_group1 = half_point
    # Extract the joint positions from the dictionary and convert to a tensor
    # group1_joint_pos = torch.tensor(list(state1["joint_pos"].values()), device=asset.device).repeat(num_group1, 1)
    group1_joint_pos = torch.tensor(list(state1.joint_pos), device=asset.device).repeat(num_group1, 1)
    # Joint velocity is likely zero
    group1_joint_vel = torch.zeros_like(group1_joint_pos, device=asset.device)

    # --- Create State 2 Tensors ---
    # The number of environments for this group
    num_group2 = len(env_ids) - half_point
    # Extract the joint positions from the dictionary and convert to a tensor
    # group2_joint_pos = torch.tensor(list(state2["joint_pos"].values()), device=asset.device).repeat(num_group2, 1)
    group2_joint_pos = torch.tensor(list(state2.joint_pos), device=asset.device).repeat(num_group2, 1)
    # Joint velocity is likely zero
    group2_joint_vel = torch.zeros_like(group2_joint_pos, device=asset.device)

    combined_joint_pos = torch.cat((group1_joint_pos, group2_joint_pos), dim=0)
    combined_joint_vel = torch.cat((group1_joint_vel, group2_joint_vel), dim=0)

    # scale these values randomly
    combined_joint_pos *= math_utils.sample_uniform(*position_range, combined_joint_pos.shape, combined_joint_pos.device)
    combined_joint_vel *= math_utils.sample_uniform(*velocity_range, combined_joint_vel.shape, combined_joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, asset_cfg.joint_ids]
    joint_pos = combined_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids, asset_cfg.joint_ids]
    joint_vel = combined_joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=asset_cfg.joint_ids,
    )