from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
import basic_locomotion_dls_isaaclab.tasks.locomotion.utils as custom_utils

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg as EventTerm

from .symmloco_common_env import SymmlocoCommonEnv
from .aliengo_symmloco_env_cfg import AliengoStandDanceDirectEnvCfg

class AliengoStandDanceEnv(SymmlocoCommonEnv):
    cfg: AliengoStandDanceDirectEnvCfg

    def __init__(self, cfg: AliengoStandDanceDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.last_heading = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.init_feet_positions = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device)
        self.last_joint_pos = torch.zeros((self.num_envs, 12))
        # self.term_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.mercy_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.position_protect_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stand_air_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.abrupt_change_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _compute_common_obs(self):
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.projected_gravity_b,
                    #TODO not using imu base info!!
                    math_utils.quat_apply_inverse(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B),
                    self._commands[:, :3] * self.command_scale,
                    (self._robot.data.joint_pos - self._robot.data.default_joint_pos) * self.cfg.obs_scale_joint_pos,
                    self._robot.data.joint_vel * self.cfg.obs_scale_joint_vel,
                    self._actions,
                    self._clock_inputs[:, -2:],
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # add perceptive inputs if not blind
        if self.cfg.measure_heights:
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)

        return obs

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.single_observation_space, dtype=torch.float, device=self.device)
        self.add_noise = self.cfg.add_noise
        noise_level = self.cfg.noise_level
        start_index = 0
        noise_vec[start_index:start_index + 3] = self.cfg.noise_scale_gravity * noise_level
        noise_vec[start_index + 3: start_index + 6] = self.cfg.noise_scale_gravity * noise_level
        start_index += 6
        noise_vec[start_index: start_index + 3] = 0.
        noise_vec[start_index + 3:start_index + 15] = self.cfg.noise_scale_joint_pos * noise_level * self.cfg.obs_scale_joint_pos
        noise_vec[start_index + 15:start_index + 27] = self.cfg.noise_scale_joint_vel * noise_level * self.cfg.obs_scale_joint_vel
        noise_vec[start_index + 27:start_index + 39] = 0. # previous actions
        noise_vec[start_index + 39: start_index + 41] = 0. # clock input
        start_index = start_index + 41
        assert start_index == self.cfg.single_observation_space
        return noise_vec

    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # only explicitly allow foot contact in these mercy steps
        self.mercy_contact_buf = torch.logical_and(
            torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._termination_ids, :], dim=-1) > 1., dim=1),
            torch.logical_not(torch.logical_and(
                torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._allow_initial_contact_ids, :], dim=-1) > 1., dim=1),
                self.episode_length_buf <= self.cfg.reward_allow_contact_steps
            ))
        )
        self.position_protect_buf = torch.logical_and(
            self.episode_length_buf > 3, torch.any(torch.logical_or(
            self._robot.data.joint_pos < self._robot.data.joint_pos_limits[:, :, 0] + 5 / 180 * np.pi,
            self._robot.data.joint_pos > self._robot.data.joint_pos_limits[:, :, 1] - 5 / 180 * np.pi
            ), dim=-1)
        )
        foot_positions = self._robot.data.body_pos_w[:, self._feet_ids_robot, :]
        self.stand_air_buf = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= self.cfg.reward_allow_contact_steps),
            torch.any((foot_positions[:, -2:, 2] - self._get_heights_at_points(foot_positions[:, -2:, :2])) > self.init_feet_positions[:, -2:, 2] + 0.06, dim=-1)
        )

        self.abrupt_change_buf = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= self.cfg.reward_allow_contact_steps),
            torch.any(torch.abs(self._robot.data.joint_pos - self.last_joint_pos) > self.cfg.max_dof_change, dim=-1)
        )

        all_conditions = torch.stack([
            self.mercy_contact_buf,
            self.position_protect_buf,
            self.stand_air_buf,
            self.abrupt_change_buf])
        resets = torch.any(all_conditions, dim=0)

        return resets

    def _update_command_curriculum(self, env_ids: torch.Tensor, extras: dict):
        """Updates the command ranges based on episode performance."""

        # Access the average reward sums directly from the log dictionary
        avg_tracking_lin_vel = extras["log"].get("Episode_Reward/tracking_lin_vel", 0.0)
        avg_tracking_ang_vel = extras["log"].get("Episode_Reward/tracking_ang_vel", 0.0)

        # The values are already the average over the environments that reset,
        # so you don't need to compute the mean again.
        normalized_lin_vel_reward = avg_tracking_lin_vel / self.cfg.rewards.scales["tracking_lin_vel"]
        normalized_ang_vel_reward = avg_tracking_ang_vel / self.cfg.rewards.scales["tracking_ang_vel"]

        if normalized_lin_vel_reward > 0.8:
            # self.command_ranges["lin_vel_x"][0] = 0. # no backward vel
            self._command_ranges["lin_vel_x"][0] = torch.clip(self._command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.command_max_curriculum, 0.)
            self._command_ranges["lin_vel_x"][1] = torch.clip(self._command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.command_max_curriculum)
            # no side vel
            self._command_ranges["lin_vel_y"][0] = torch.clip(self._command_ranges["lin_vel_y"][0] - 0.2, -self.cfg.command_max_curriculum, 0.)
            self._command_ranges["lin_vel_y"][1] = torch.clip(self._command_ranges["lin_vel_y"][1] + 0.2, 0., self.cfg.command_max_curriculum)

        if normalized_ang_vel_reward > 0.8:
            self._command_ranges["ang_vel_yaw"][0] = torch.clip(self._command_ranges["ang_vel_yaw"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self._command_ranges["ang_vel_yaw"][1] = torch.clip(self._command_ranges["ang_vel_yaw"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        conti_velx_cmd = self._commands[env_ids, 0:1]
        self._commands[env_ids, 0:1] = torch.sign(conti_velx_cmd) * torch.round(torch.abs(conti_velx_cmd) / 0.1) * 0.1

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        self.init_feet_positions[env_ids] = self._robot.data.body_pos_w[env_ids][:, self._feet_ids_robot, :]

        # add the custom terminations to the log
        extras = dict()
        # extras["Episode_Termination/termination_contacts"] = torch.count_nonzero(self.term_contact_buf[env_ids]).item()
        extras["Episode_Termination/mercy_contacts"] = torch.count_nonzero(self.mercy_contact_buf[env_ids]).item()
        extras["Episode_Termination/position_protect_termination"] = torch.count_nonzero(self.position_protect_buf[env_ids]).item()
        extras["Episode_Termination/stand_air_termination"] = torch.count_nonzero(self.stand_air_buf[env_ids]).item()
        extras["Episode_Termination/abrupt_change_termination"] = torch.count_nonzero(self.abrupt_change_buf[env_ids]).item()

        self.extras["log"].update(extras)

        heading = self._get_cur_heading()
        self.last_heading[env_ids] = heading[env_ids]
        self.last_joint_pos = self._robot.data.joint_pos.clone()

    def _recompute_ang_vel(self):
        heading = self._get_cur_heading()
        self._commands[:, 2] = torch.clip(
            0.5*math_utils.wrap_to_pi(self._commands[:, 3] - heading), -self.cfg.command_clip_ang_vel, self.cfg.command_clip_ang_vel
        ) * (0.5 * np.pi / self.cfg.command_clip_ang_vel)

    def _get_heights_at_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        Finds the terrain height at specified points using the height scanner data.

        Args:
            points (torch.Tensor): A tensor of shape (num_envs, num_points, 2) in the world frame.

        Returns:
            torch.Tensor: A tensor of shape (num_envs, num_points) with the terrain heights.
        """
        # Get the height scanner ray hits in world coordinates
        ray_hits_w = self._height_scanner.data.ray_hits_w

        # Repeat the points tensor to compare with all ray hits
        # Shape: (num_envs, num_points, num_rays, 2)
        points_expanded = points.unsqueeze(2).expand(-1, -1, ray_hits_w.shape[1], -1)

        # Calculate the horizontal distance between each point and all ray hits
        # Shape: (num_envs, num_points, num_rays)
        xy_distance = torch.norm(points_expanded - ray_hits_w.unsqueeze(1)[..., :2], dim=-1)

        # Find the index of the closest ray hit for each point
        # Shape: (num_envs, num_points)
        closest_ray_indices = torch.argmin(xy_distance, dim=-1)

        # Use the indices to gather the terrain heights (Z-coordinate of ray hits)
        # Shape: (num_envs, num_points)
        # The `torch.gather` function needs the input tensor to have the same
        # number of dimensions as the index tensor.
        ray_hits_z = self._height_scanner.data.ray_hits_w[..., 2]
        # Expand the ray hits z-coordinates to match the dimensions of the index tensor
        ray_hits_z_expanded = ray_hits_z.unsqueeze(1).expand(-1, points.shape[1], -1)

        heights = torch.gather(ray_hits_z_expanded, 2, closest_ray_indices.unsqueeze(2)).squeeze(2)

        return heights

    #------------ reward functions----------------
    def _reward_lift_up(self):
        root_height = self._robot.data.root_pos_w[:, 2]
        root_height -= torch.mean(self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2]), dim = 1)
        delta_height = root_height - self.cfg.reward_liftup_target
        error = torch.square(delta_height)
        reward = torch.exp(- error / self.cfg.reward_tracking_liftup_sigma) #use tracking sigma
        return reward

    def _reward_lift_up_linear(self):
        root_height = self._robot.data.root_pos_w[:, 2]
        root_height -= torch.mean(self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2]), dim = 1)
        reward = (root_height - self.cfg.reward_lift_up_threshold[0]) / (self.cfg.reward_lift_up_threshold[1] - self.cfg.reward_lift_up_threshold[0])
        reward = torch.clamp(reward, 0., 1.)
        return reward

    def _reward_tracking_lin_vel(self):
        if not self.cfg.command_curriculum:
            self._commands[:, :2] = 0.
        actual_lin_vel = custom_utils.quat_apply_yaw_inverse(self._robot.data.root_quat_w, self._robot.data.root_lin_vel_w)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - actual_lin_vel[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.reward_tracking_sigma)
        forward = math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        reward_upright_vec = torch.tensor(self.cfg.reward_upright_vec, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        upright_vec = custom_utils.quat_apply_yaw(self._robot.data.root_quat_w, reward_upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9

        scale_factor_low = self.cfg.reward_scale_factor_low
        scale_factor_high = self.cfg.reward_scale_factor_high
        scaling_factor = (torch.clip(
            self._robot.data.root_pos_w[:, 2] - torch.mean(self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2]), dim = 1), min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        reward = reward * is_stand.float() * scaling_factor
        return reward

    def _reward_tracking_ang_vel(self):
        if not self.cfg.command_curriculum:
            self._commands[:, 3] = 0.
        heading = self._get_cur_heading()
        if self.cfg.reward_ang_rew_mode == "heading":
            # old
            heading_error = torch.square(math_utils.wrap_to_pi(self._commands[:, 3] - heading) / np.pi)
            reward = torch.exp(-heading_error / self.cfg.reward_tracking_sigma)
        elif self.cfg.reward_ang_rew_mode == "heading_with_pen":
            heading_error = torch.square(math_utils.wrap_to_pi(self._commands[:, 3] - heading) / np.pi)
            reward = torch.exp(-heading_error / self.cfg.reward_tracking_sigma)
            est_ang_vel = math_utils.wrap_to_pi(heading - self.last_heading) / 0.02
            penalty = (torch.abs(est_ang_vel) - 1.0).clamp(min=0)
            reward = reward - 0.1 * penalty
        else:
            # new, trying
            est_ang_vel = math_utils.wrap_to_pi(heading - self.last_heading) / 0.02
            # ang_vel_error = torch.abs(self._commands[:, 2] - est_ang_vel) / torch.abs(self._commands[:, 2]).clamp(min=1e-6)
            ang_vel_error = torch.abs(self._commands[:, 2] - est_ang_vel)
            reward = torch.exp(-ang_vel_error/self.cfg.reward_tracking_ang_sigma)
        forward = math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        reward_upright_vec = torch.tensor(self.cfg.reward_upright_vec, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        upright_vec = custom_utils.quat_apply_yaw(self._robot.data.root_quat_w, reward_upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9
        scale_factor_low = self.cfg.reward_scale_factor_low
        scale_factor_high = self.cfg.reward_scale_factor_high
        scaling_factor = (torch.clip(
            self._robot.data.root_pos_w[:, 2] - torch.mean(self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2]), dim=1), min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        reward = reward * is_stand.float() * scaling_factor
        return reward

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, -2:] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], 2])
        terrain_at_foot_height = self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2])
        target_height = self.cfg.reward_foot_target * phases + terrain_at_foot_height + 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self._desired_contact_states[:, -2:])
        condition = self.episode_length_buf > self.cfg.reward_allow_contact_steps
        rew_foot_clearance = rew_foot_clearance * condition.unsqueeze(dim=-1).float()
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_rear_air(self):
        contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids[-2:], 2] > 1.
        calf_contact = self._contact_sensor.data.net_forces_w[:, self._calf_ids[-2:], 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        reward = torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()
        return reward

    def _reward_stand_air(self):
        stand_air_condition = torch.logical_and(
            torch.logical_and(
                self.episode_length_buf < self.cfg.reward_allow_contact_steps,
                math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)[..., 2] < 0.9
            ), torch.any((self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], 2] - self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2])) > self.init_feet_positions[:, -2:, 2] + 0.03, dim=1)
        )
        return stand_air_condition.float()

    def _reward_foot_twist(self):
        vxy = torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2], dim=-1)
        vang = torch.norm(self._robot.data.body_ang_vel_w[:, self._feet_ids_robot, 2:], dim=-1)
        condition = (self._robot.data.body_pos_w[:, self._feet_ids_robot, 2] - self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot, :2])) < 0.025
        reward = torch.mean((vxy + 0.1 * vang) * condition.float(), dim=1)
        return reward

    def _reward_feet_slip(self):
        condition = (self._robot.data.body_pos_w[:, self._feet_ids_robot, 2] - self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot, :2])) < 0.03
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2], dim=2))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self._robot.data.body_ang_vel_w[:, self._feet_ids_robot, 2:] / np.pi, dim=2))
        rew_slip = torch.sum(condition.float() * (foot_velocities + foot_ang_velocities), dim=1)
        return rew_slip

    def _reward_foot_shift(self):
        desired_foot_positions = torch.clone(self.init_feet_positions[:, 2:])
        desired_foot_positions[:, :, 2] = 0.02
        desired_foot_positions[:, :, 2] += self._get_heights_at_points(self._robot.data.body_pos_w[:, self._feet_ids_robot[-2:], :2])
        rear_foot_shift = torch.norm(self._robot.data.body_pos_w[:, self._feet_ids_robot[2:]] - desired_foot_positions, dim=-1).mean(dim=1)
        init_foot_positions = torch.clone(self.init_feet_positions[:, :2])
        front_foot_shift = torch.norm( torch.stack([
            (init_foot_positions[:, :, 0] - self._robot.data.body_pos_w[:, self._feet_ids_robot[:2], 0]).clamp(min=0),
            torch.abs(init_foot_positions[:, :, 1] - self._robot.data.body_pos_w[:, self._feet_ids_robot[:2], 1])
        ], dim=-1), dim=-1).mean(dim=1)
        condition = self.episode_length_buf < self.cfg.reward_allow_contact_steps
        reward = (front_foot_shift + rear_foot_shift) * condition.float()
        return reward

    def _reward_front_contact_force(self):
        force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._termination_ids[5: 7]], dim=-1).mean(dim=1)
        reward = force
        return reward

    def _reward_hip_still(self):
        movement = torch.abs(self._robot.data.joint_pos[:, :, 0] - 0.).mean(dim=1)
        condition = self.episode_length_buf < self.cfg.reward_allow_contact_steps
        reward = movement * condition.float()
        return reward