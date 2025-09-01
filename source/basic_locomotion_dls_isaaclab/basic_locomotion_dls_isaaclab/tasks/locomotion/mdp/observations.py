# Copyright (c) 2022-2024, The LAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg

from isaaclab.sensors import Imu

from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def get_projected_forward_vec_from_imu(env: ManagerBasedEnv, imu_asset_cfg: SceneEntityCfg = SceneEntityCfg("imu"), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """Computes the projected forward vector from the IMU orientation."""
        # extract the used quantities (to enable type-hinting)
        imu_asset: Imu = env.scene[imu_asset_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        # Get the IMU orientation (w, x, y, z)
        imu_quat = imu_asset.data.quat_w

        # Apply the yaw component of the IMU orientation to the unit forward vector
        projected_forward = math_utils.quat_apply_inverse(imu_quat, asset.data.FORWARD_VEC_B)
        return projected_forward

def vel_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Gets the velocity commands, omitting the heading command."""
    commands = env.command_manager.get_command(command_name)
    return commands[:, :3]

def rear_clock_inputs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gets the rear foot clock inputs."""
    return env._clock_inputs[:, -2:]

def friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gets the friction coefficients."""
    if hasattr(env, "joint_friction_coeffs") and env.joint_friction_coeffs is not None:
        return env.friction_coeffs
    else:
        return torch.zeros(
            (env.num_envs, 1), dtype=torch.float, device=env.device
        )

def restitution(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gets the restitution."""
    if hasattr(env, "restitution") and env.restitution is not None:
        return env.restitution
    else:
        return torch.zeros(
            (env.num_envs, 1), dtype=torch.float, device=env.device
        )

def joint_friction(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gets the joint friction coefficients."""

    # Check if the attribute is already populated (after env's __init__ and randomization)
    if hasattr(env, "joint_friction_coeffs") and env.joint_friction_coeffs is not None:
        return env.joint_friction_coeffs
    else:
        return torch.zeros(
            (env.num_envs, 12), dtype=torch.float, device=env.device
        )



def com_displacement(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gets the center of mass displacement."""

    # Check if the attribute is already populated (after env's __init__ and randomization)
    if hasattr(env, "com_displacement") and env.com_displacement is not None:
        return env.com_displacement
    else:

        # Return a zero tensor with the correct shape. This is for shape inference.
        # It uses env.device and env.num_envs which ARE available at this point.
        return torch.zeros((env.num_envs, 3), dtype=torch.float, device=env.device)


def mass_offset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gets the base mass offset."""
    if hasattr(env, "mass_offset") and env.mass_offset is not None:
        return env.mass_offset
    else:
        return torch.zeros((env.num_envs, 1), dtype=torch.float, device=env.device)

def link_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Gets the undesired links in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    link_in_contact = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 0.1, dim=1).float()
    return link_in_contact