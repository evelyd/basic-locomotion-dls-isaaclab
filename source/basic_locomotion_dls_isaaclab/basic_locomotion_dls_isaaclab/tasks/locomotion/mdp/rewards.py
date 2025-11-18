import torch
import numpy as np
from typing import TYPE_CHECKING, List
import isaaclab.utils.math as math_utils
import basic_locomotion_dls_isaaclab.tasks.locomotion.utils as custom_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, ObservationTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.command_manager import CommandTerm

def lin_vel_z(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize z axis base linear velocity."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    reward = torch.square(env.base_lin_vel[:, 2])
    base_in_collision = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 0.1, dim=-1).float()
    reward = reward * (1 - base_in_collision.float())
    return reward

def ang_vel_xy(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize xy axes base angular velocity."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    reward = torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)
    base_in_collision = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 0.1, dim=-1).float()
    reward = reward * (1 - base_in_collision.float())
    return reward

def orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    reward = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)
    return reward

def base_height(env: ManagerBasedRLEnv, desired_base_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base height away from target."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = torch.mean(asset.data.root_pos_w[:, 2].unsqueeze(1), dim=1) #  - env.measured_heights assume flat terrain
    return torch.square(base_height - desired_base_height)

def torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torques."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.torques), dim=1)
    return reward

def dof_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize dof velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return reward

def dof_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize dof accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
    return reward

def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize changes in actions."""
    # last_actions needs to be an attribute of env
    reward = torch.sum(torch.square(env.action_manager.prev_action - env.action_manager.action), dim=1)
    return reward

def action_overshoot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action overshoot."""
    # overshoot_buf needs to be an attribute of env
    reward = torch.sum(env.overshoot_buf, dim=-1)
    return reward

def action_q_diff(env: ManagerBasedRLEnv, allow_contact_steps: int) -> torch.Tensor:
    """Penalize q_diff based on contact steps."""
    # q_diff_buf and episode_length_buf needs to be an attribute of env
    condition = env.episode_length_buf <= allow_contact_steps
    reward = torch.sum(torch.square(env.q_diff_buf), dim=-1) * condition.float()
    return reward

def collision(env: ManagerBasedRLEnv, allow_contact_steps: int, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize collisions after initial contact steps."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    reward = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 0.1, dim=(1, 2)).float()
    cond = env.episode_length_buf > allow_contact_steps
    reward = reward * cond.float()
    return reward

def termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminal reward / penalty."""
    return env.reset_buf * ~env.time_out_buf

def dof_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize dof positions too close to the limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]).clip(max=0.)
    out_of_limits += (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

def dof_vel_limits(env: ManagerBasedRLEnv, soft_dof_vel_limit: float) -> torch.Tensor:
    """Penalize dof velocities too close to the limit."""
    return torch.sum((torch.abs(env.robot.data.joint_vel) - env.dof_vel_limits * soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

def torque_limits(env: ManagerBasedRLEnv, soft_torque_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize torques too close to the limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum((torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - asset.data.computed_torque[:, asset_cfg.joint_ids] * soft_torque_limit).clip(min=0.), dim=1)

def tracking_lin_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, tracking_sigma: float, vel_cmd: bool, scale_factor_low: float, scale_factor_high: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Tracking of linear velocity commands (xy axes)."""

    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    commands_lin_vel = env.command_manager.get_command(command_name)[:, :2]
    if not vel_cmd:
        commands_lin_vel = torch.zeros_like(commands_lin_vel) # Adjust if command is 4D

    actual_lin_vel = custom_utils.quat_apply_yaw_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)
    lin_vel_error = torch.sum(torch.square(commands_lin_vel - actual_lin_vel[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / tracking_sigma)

    base_in_collision = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 0.1, dim=(1, 2)).float()
    reward = reward * (1 - base_in_collision.squeeze().float())
    forward = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B) # Uses _forward_vec from env
    upright_vec_rotated = custom_utils.quat_apply_yaw(asset.data.root_quat_w, env.upright_vec) # Uses upright_vec from env
    is_stand = (torch.sum(forward * upright_vec_rotated, dim=-1) / torch.norm(upright_vec_rotated, dim=-1)) > 0.9

    scaling_factor = (torch.clip(
        asset.data.root_pos_w[:, 2], # - torch.mean(env._get_heights_at_points(env.foot_positions[:, -2:, :2]), dim=1), # assuming flat terrain
        min=scale_factor_low, max=scale_factor_high
    ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
    reward = reward * is_stand.float() * scaling_factor
    return reward

def tracking_ang_vel(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, tracking_sigma: float, ang_rew_mode: str, vel_cmd: bool, scale_factor_low: float, scale_factor_high: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Tracking of angular velocity commands (yaw)."""

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    commands_ang_vel = env.command_manager.get_command(command_name)[:, 2]
    if not vel_cmd:
        commands_ang_vel = torch.zeros_like(commands_ang_vel)

    heading = env._get_cur_heading() # Assuming _get_cur_heading is an env method

    if ang_rew_mode == "heading":
        heading_error = torch.square(math_utils.wrap_to_pi(commands_ang_vel - heading) / np.pi)
        reward = torch.exp(-heading_error / tracking_sigma)
    elif ang_rew_mode == "heading_with_pen":
        heading_error = torch.square(math_utils.wrap_to_pi(commands_ang_vel - heading) / np.pi)
        reward = torch.exp(-heading_error / tracking_sigma)
        # last_heading needs to be an env attribute
        est_ang_vel = math_utils.wrap_to_pi(heading - env.last_heading) / env.dt_sim_env # Use env.dt_sim_env
        penalty = (torch.abs(est_ang_vel) - 1.0).clamp(min=0)
        reward = reward - 0.1 * penalty
    else: # Default mode from original code (likely angular velocity tracking)
        est_ang_vel = math_utils.wrap_to_pi(heading - env.last_heading) / env.dt_sim_env
        ang_vel_error = torch.abs(env.command_manager.get_command(command_name)[:, 2] - est_ang_vel)
        reward = torch.exp(-ang_vel_error / tracking_sigma)
    base_in_collision = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 0.1, dim=(1, 2)).float()
    reward = reward * (1 - base_in_collision.squeeze().float())
    forward = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    upright_vec_rotated = custom_utils.quat_apply_yaw(asset.data.root_quat_w, env.upright_vec)
    is_stand = (torch.sum(forward * upright_vec_rotated, dim=-1) / torch.norm(upright_vec_rotated, dim=-1)) > 0.9

    scaling_factor = (torch.clip(
        asset.data.root_pos_w[:, 2], # - torch.mean(env._get_heights_at_points(env.foot_positions[:, -2:, :2]), dim=1), # assuming flat terrain
        min=scale_factor_low, max=scale_factor_high
    ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
    reward = reward * is_stand.float() * scaling_factor
    return reward

def feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Reward long steps."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # last_contacts and feet_air_time need to be attributes of env (updated in _pre_physics_step) #TODO this needs to be foot indices byt the last one is the base
    contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[1:], 2] > 1.
    contact_filt = torch.logical_or(contact, env.last_contacts) # env.last_contacts updated in env's _pre_physics_step
    # Note: env.feet_air_time and env.last_contacts need to be updated in env._pre_physics_step
    # The increment and reset logic should live in the environment, not the reward function.
    # The reward function should just *read* the current feet_air_time.
    # So, the following lines of original code need to be handled in StandDanceEnv's _pre_physics_step:
    # self.feet_air_time += self.dt
    # self.last_contacts = contact
    # self.feet_air_time *= ~contact_filt

    first_contact = (env.feet_air_time > 0.) * contact_filt
    rew_airTime = torch.sum((env.feet_air_time - 0.5) * first_contact, dim=1)
    rew_airTime *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    base_in_collision = torch.any(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[0], :], dim=-1) > 0.1, dim=(1, 2)).float()
    rew_airTime = rew_airTime * (1 - base_in_collision.float())
    return rew_airTime

def stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get horizontal forces
    horizontal_forces = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :2], dim=-1)

    vertical_forces = torch.abs(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2])

    # rear feet_indices need to be accessible
    return torch.any(horizontal_forces > 5.0 * vertical_forces, dim=(1, 2)).float()

def stand_still(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize motion at zero commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    # default_dof_pos needs to be an attribute of env
    return torch.sum(torch.abs(asset.data.joint_pos - env.default_dof_pos), dim=1) * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1)

def feet_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_contact_force: float) -> torch.Tensor:
    """Penalize high contact forces."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # feet_indices need to be accessible
    return torch.sum((torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) - max_contact_force).clip(min=0.), dim=1)

def lift_up(env: ManagerBasedRLEnv, liftup_target: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for lifting the base."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    reward = torch.exp(root_height - liftup_target) - 1
    return reward

def lift_up_linear(env: ManagerBasedRLEnv, lift_up_threshold_range: list, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear reward for lifting the base based on a range."""
    asset: RigidObject = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    # Assuming _get_heights_at_points and foot_positions are env methods/attributes
    # root_height -= torch.mean(env._get_heights_at_points(env.foot_positions[:, -2:, :2]), dim=1) # assume flat terrain
    reward = (root_height - lift_up_threshold_range[0]) / (lift_up_threshold_range[1] - lift_up_threshold_range[0])
    reward = torch.clamp(reward, 0., 1.)
    return reward

def rear_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for rear feet/calf being in air."""
    contact_sensor: SceneEntityCfg = env.scene.sensors[sensor_cfg.name]
    # rear feet_indices and calf_indices need to be accessible
    contact_feet_rear = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[:2], 2], dim=-1) < 1. # Z-component of force
    contact_calf_rear = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[2:], 2], dim=-1) < 1.
    unhealthy_condition = torch.logical_and(~contact_calf_rear, contact_feet_rear)
    reward = torch.all(contact_feet_rear, dim=1).float() + unhealthy_condition.sum(dim=-1).float()
    return reward

def stand_air(env: ManagerBasedRLEnv, allow_contact_steps: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for standing in air early in episode."""

    asset: RigidObject = env.scene[asset_cfg.name]

    # _forward_vec, upright_vec, _get_heights_at_points, foot_positions need to be accessible from env
    forward_rotated = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    upright_vec_rotated = custom_utils.quat_apply_yaw(asset.data.root_quat_w, env.upright_vec)
    is_stand = (torch.sum(forward_rotated * upright_vec_rotated, dim=-1) / torch.norm(upright_vec_rotated, dim=-1)) > 0.9

    stand_air_condition = torch.logical_and(
        torch.logical_and(
            env.episode_length_buf < allow_contact_steps,
            is_stand # Replaced original check as it seems to be what 'is_stand' calculates
        ), torch.any((env.foot_positions[:, 2, -2:]) > 0.03, dim=1) # - env._get_heights_at_points(env.foot_positions[:, -2:, :2]) assume flat terrain
    )
    return stand_air_condition.float()

def foot_twist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize foot twisting (xy linear vel and yaw angular vel)."""
    # foot_velocities, foot_velocities_ang, foot_positions, _get_heights_at_points need to be accessible from env
    vxy = torch.norm(env.foot_velocities[:, :, :2], dim=-1)
    vang = torch.norm(env.foot_velocities_ang, dim=-1) # Assuming this is 3D angular velocity
    condition = (env.foot_positions[:, 2, :]) < 0.025 #  - env._get_heights_at_points(env.foot_positions[:, :, :2]) assume flat terrain
    reward = torch.mean((vxy + 0.1 * vang) * condition.float(), dim=1)
    return reward

def feet_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet slipping."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: SceneEntityCfg = env.scene.sensors[sensor_cfg.name]
    # feet_indices, last_contacts, foot_velocities, foot_velocities_ang need to be accessible from env
    # The update of last_contacts for filtering should happen in env._pre_physics_step
    contact = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1) > 1.
    # Check if any contacts over the history
    contact_filt = torch.any(contact, dim=1)
    # NOTE: The original also has: self.last_contacts = contact, this logic should be in _pre_physics_step of env
    # xy lin vel
    foot_velocities = torch.square(torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=2).view(env.num_envs, -1))
    # yaw ang vel
    foot_ang_velocities = torch.square(torch.norm(asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :] / np.pi, dim=2).view(env.num_envs, -1))
    rew_slip = torch.sum(contact_filt * (foot_velocities + foot_ang_velocities), dim=1)
    return rew_slip

def foot_shift(env: ManagerBasedRLEnv, allow_contact_steps: int) -> torch.Tensor:
    """Penalize foot shift from initial position during early steps."""
    # init_feet_positions, foot_positions, _get_heights_at_points needs to be accessible from env
    desired_foot_positions = torch.clone(env.init_feet_positions[:, 2:, :]) # Assuming rear feet
    desired_foot_positions[:, :, 2] = 0.02
    # desired_foot_positions[:, 2, :] += env._get_heights_at_points(env.foot_positions[:, :2, -2:]) assume flat terrain
    rear_foot_shift = torch.norm(env.foot_positions[:, 2:] - desired_foot_positions, dim=-1).mean(dim=1)

    init_foot_positions = torch.clone(env.init_feet_positions[:, :, :2]) # Assuming front feet
    front_foot_shift = torch.norm(torch.stack([
        (init_foot_positions[:, :2, 0] - env.foot_positions[:, :2, 0]).clamp(min=0),
        torch.abs(init_foot_positions[:, :2, 1] - env.foot_positions[:, :2, 1])
    ], dim=-1), dim=-1).mean(dim=1)

    condition = env.episode_length_buf < allow_contact_steps
    reward = (front_foot_shift + rear_foot_shift) * condition.float()
    return reward

def front_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces on front feet."""
    contact_sensor: SceneEntityCfg = env.scene.sensors[sensor_cfg.name]
    # termination_contact_indices needs to be accessible from env
    # Example assumes termination_contact_indices[5:7] refers to front feet
    # You need to ensure these indices map correctly to your feet.
    # It's better to use env.feet_indices or specific named indices if possible.
    # For now, mimicking original code:
    force = torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1).mean(dim=1)
    reward = force
    return reward

def hip_still(env: ManagerBasedRLEnv, allow_contact_steps: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize hip movement during early steps."""
    asset: Articulation = env.scene[asset_cfg.name]
    # default_dof_pos should be an env attribute initialized in _setup_scene
    movement = torch.abs(asset.data.joint_pos.view(env.num_envs, 4, 3)[:, :, 0] - 0.).mean(dim=1) # Original has 0., adjust if needed
    condition = env.episode_length_buf < allow_contact_steps
    reward = movement * condition.float()
    return reward

def upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for upright orientation."""
    # _forward_vec and upright_vec need to be accessible from env
    asset: Articulation = env.scene[asset_cfg.name]
    forward_rotated = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    upright_vec_rotated = custom_utils.quat_apply_yaw(asset.data.root_quat_w, env.upright_vec)
    cosine_dist = torch.sum(forward_rotated * upright_vec_rotated, dim=-1) / torch.norm(upright_vec_rotated, dim=-1)
    reward = torch.square(0.5 * cosine_dist + 0.5)
    return reward

def feet_clearance_cmd_linear(env: ManagerBasedRLEnv, foot_target: float, allow_contact_steps: int) -> torch.Tensor:
        # phases, foot_indices, desired_contact_states need to be accessible
        # This assumes your CyberEnv provides these.

        # phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0) # foot_indices needs to be properly handled
        # For now, a placeholder for phases and desired_contact_states
        phases = torch.ones(env.num_envs, 2, device=env.device) # Placeholder
        desired_contact_states = torch.zeros(env.num_envs, 4, device=env.device) # Placeholder

        foot_height = (env.foot_positions[:, 2, -2:]).view(env.num_envs, -1)
        terrain_at_foot_height = 0 # env._get_heights_at_points(env.foot_positions[:, -2:, :2]) assume flat terrain
        target_height = foot_target * phases + terrain_at_foot_height + 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - desired_contact_states[:, -2:])
        condition = env.episode_length_buf > allow_contact_steps
        rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        return torch.sum(rew_foot_clearance, dim=1)