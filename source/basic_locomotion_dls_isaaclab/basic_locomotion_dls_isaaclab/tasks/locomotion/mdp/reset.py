# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np
from isaaclab.utils.math import matrix_from_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length

"""
Joint terminations.
"""
def contact_with_mercy(env: ManagerBasedRLEnv, term_sensor_cfg: SceneEntityCfg, allow_init_sensor_cfg: SceneEntityCfg, allow_contact_steps: int) -> torch.Tensor:
    term_contact_sensor:ContactSensor = env.scene.sensors[term_sensor_cfg.name]
    allow_init_contact_sensor:ContactSensor = env.scene.sensors[allow_init_sensor_cfg.name]
    mercy_contacts = torch.logical_and(
            torch.any(torch.norm(term_contact_sensor.data.net_forces_w_history[:, :, term_sensor_cfg.body_ids, :], dim=-1) > 1., dim=(1, 2)), #term contacts
            torch.logical_not(torch.logical_and(
                torch.any(torch.norm(allow_init_contact_sensor.data.net_forces_w_history[:, :, allow_init_sensor_cfg.body_ids, :], dim=-1) > 1., dim=(1, 2)), # allow init contact inds
                env.episode_length_buf <= allow_contact_steps
            ))
        )
    return mercy_contacts

def joint_position_protect(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)

    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > limits[..., 1] - 5 / 180 * np.pi, dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < limits[..., 0] + 5 / 180 * np.pi, dim=1)
    return torch.logical_and(
            env.episode_length_buf > 3, torch.any(torch.logical_or(out_of_lower_limits, out_of_upper_limits), dim=-1))
    # return torch.logical_or(out_of_upper_limits, out_of_lower_limits)

def stand_air(env: ManagerBasedRLEnv, allow_contact_steps: int) -> torch.Tensor:
    return torch.logical_and(
            torch.logical_and(env.episode_length_buf > 3, env.episode_length_buf <= allow_contact_steps),
            torch.any((env.foot_positions[:, -2:, 2]) > 0.06, dim=-1) # - self._get_heights_at_points(self.foot_positions[:, -2:, :2]) assume flat terrain
    )

def abrupt_change(env: ManagerBasedRLEnv, allow_contact_steps: int, max_dof_change: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.logical_and(
            torch.logical_and(env.episode_length_buf > 3, env.episode_length_buf <= allow_contact_steps),
            torch.any(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - env.last_dof_pos) > max_dof_change, dim=-1)
        )

def is_base_pitch_invalid(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
    Terminates the episode if the robot's pitch orientation exceeds a threshold.

    The function calculates the pitch angle from the base's orientation and
    returns a boolean tensor indicating if the angle is outside the valid range.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        asset_cfg (SceneEntityCfg): The configuration for the robot asset.
        threshold (float): The maximum allowed pitch angle in radians.

    Returns:
        torch.Tensor: A boolean tensor of shape (num_envs,) indicating
                      if the pitch is outside the valid range.
    """
    # Get the base orientation (quaternion) in the world frame
    base_quat = env.scene[asset_cfg.name].data.root_quat_w[:, :]

    # Convert quaternion to rotation matrix
    base_rot_mat = matrix_from_quat(base_quat)

    # Extract the pitch angle from the rotation matrix.
    # The pitch can be calculated from the element at (2, 0) of the rotation matrix.
    pitch = torch.atan2(-base_rot_mat[:, 2, 0], base_rot_mat[:, 2, 2])

    # Check if the absolute pitch angle exceeds the threshold
    return torch.abs(pitch) > threshold