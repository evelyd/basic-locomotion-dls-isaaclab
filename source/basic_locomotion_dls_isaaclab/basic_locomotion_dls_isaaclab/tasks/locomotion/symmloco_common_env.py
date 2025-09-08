# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
import basic_locomotion_dls_isaaclab.tasks.locomotion.utils as custom_utils

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


from .aliengo_env_cfg import AliengoFlatEnvCfg, AliengoRoughBlindEnvCfg, AliengoRoughVisionEnvCfg
from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg
from .hyqreal_env_cfg import HyQRealFlatEnvCfg, HyQRealRoughVisionEnvCfg, HyQRealRoughBlindEnvCfg
from .b2_env_cfg import B2FlatEnvCfg, B2RoughVisionEnvCfg, B2RoughBlindEnvCfg



class SymmlocoCommonEnv(DirectRLEnv):
    cfg: AliengoFlatEnvCfg | AliengoRoughBlindEnvCfg | AliengoRoughVisionEnvCfg | Go2FlatEnvCfg | Go2RoughVisionEnvCfg | Go2RoughBlindEnvCfg | HyQRealFlatEnvCfg | HyQRealRoughVisionEnvCfg | HyQRealRoughBlindEnvCfg

    def __init__(self, cfg: AliengoFlatEnvCfg | AliengoRoughBlindEnvCfg | AliengoRoughVisionEnvCfg | Go2FlatEnvCfg | Go2RoughVisionEnvCfg | Go2RoughBlindEnvCfg | HyQRealFlatEnvCfg | HyQRealRoughVisionEnvCfg | HyQRealRoughBlindEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity, yaw angular velocity, and heading commands
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)

        # Swing peak
        self._swing_peak = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)

        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.rewards.scales.keys()
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
        self._hip_ids, _ = self._contact_sensor.find_bodies(".*hip")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._undesired_contact_body_ids = self._base_id + self._hip_ids + self._thigh_ids
        self._penalized_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalize_contacts_on)
        self._termination_ids, _ = self._contact_sensor.find_bodies(self.cfg.terminate_after_contacts_on)
        self._allow_initial_contact_ids, _ = self._contact_sensor.find_bodies(self.cfg.allow_initial_contacts_on)

        self._feet_ids_robot, _ = self._robot.find_bodies(".*foot")
        self._hip_ids_robot, _ = self._robot.find_bodies(".*hip")

        self._feet_air_time = torch.zeros(self.num_envs, len(self._feet_ids), dtype=torch.float, device=self.device, requires_grad=False)

        # Setup for command curriculum
        self._command_ranges = self.cfg.commands.ranges

        self._reward_scales = self.cfg.rewards.scales
        # Setup for reward curriculum
        if self.cfg.reward_curriculum:
            self._reward_scales_final = self.cfg.rewards.scales
            for key in self._reward_scales:
                if self._reward_scales_final[key] < 0:
                    self._reward_scales[key] *= self.cfg.reward_cl_init

        # Initialize vectors
        self._clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self._desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self._gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._noise_scale_vec = self._get_noise_scale_vec(self.cfg)

        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)
        self._clipped_episode_sums = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        self.command_scale = torch.tensor([self.cfg.obs_scale_lin_vel, self.cfg.obs_scale_lin_vel, self.cfg.obs_scale_ang_vel], device=self.device, requires_grad=False,)

        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

        # Sample new commands
        resample_ids = (self.episode_length_buf % int(self.cfg.command_resampling_time / self.step_dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_ids)

        # Update contact targets
        self._step_contact_targets()

        # Recompute the angular velocity command based on the current heading
        self._recompute_ang_vel()

        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.5
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:

        # Resample commands
        # resample_time = self.episode_length_buf == self.max_episode_length - 200
        # commands_resample = torch.zeros_like(self._commands).uniform_(-1.0, 1.0)
        # # commands_resample[:, 0] *= 0.5 * self._velocity_gait_multiplier
        # commands_resample[:, 0] *= 0.3
        # commands_resample[:, 1] *= 0.0
        # commands_resample[:, 2] *= 0.3
        # commands_resample[:, 3] *= 0.5 * np.pi
        # self._commands[:, :3] = self._commands[:, :3] * ~resample_time.unsqueeze(1).expand(-1, 3) + commands_resample * resample_time.unsqueeze(1).expand(-1, 3)

        # # Stop
        # rest_time = self.episode_length_buf >= self.max_episode_length - 50
        # self._commands[:, :3] *= ~rest_time.unsqueeze(1).expand(-1, 3)


        """# Stop and Go
        rest_time = (self.episode_length_buf >= self.max_episode_length - 150) & (self.episode_length_buf <= self.max_episode_length - 100)
        self._commands[:, :3] *= ~rest_time.unsqueeze(1).expand(-1, 3)

        restart_time = self.episode_length_buf == self.max_episode_length - 99
        commands_restart = torch.zeros_like(self._commands).uniform_(-1.0, 1.0)
        commands_restart[:, 0] *= 0.5 * self._velocity_gait_multiplier
        commands_restart[:, 1] *= 0.25
        commands_restart[:, 2] *= 0.3
        self._commands[:, :3] = self._commands[:, :3] * ~restart_time.unsqueeze(1).expand(-1, 3) + commands_restart * restart_time.unsqueeze(1).expand(-1, 3)"""

        obs = self._compute_common_obs()

        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self._noise_scale_vec

        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)

        # Final observations dictionary
        observations = {"policy": obs}

        # Priv obs don't receive noise
        if self.cfg.use_privileged_obs:
            observations["critic"] = self._compute_privileged_obs(obs)

        sum_nan = torch.isnan(obs).sum() + (torch.isnan(observations["critic"]).sum() if observations["critic"] is not None else 0)
        sum_inf = torch.isinf(obs).sum() + (torch.isinf(observations["critic"]).sum() if observations["critic"] is not None else 0)

        if(sum_nan > 0):
            print("Nan in observation computation")
            breakpoint()

        if(sum_inf > 0):
            print("Inf in observation computation")
            breakpoint()

        return observations

    def _compute_common_obs(self):
        obs = torch.cat(
            [
                # values are scaled in noise_vec function
                tensor
                for tensor in (
                    self._robot.data.projected_gravity_b,
                    self._commands[:, :3] * self.cfg.command_scale,
                    (self._robot.data.joint_pos - self._robot.data.default_joint_pos),
                    self._robot.data.joint_vel,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )

        # if isinstance(self.cfg, AliengoRoughVisionEnvCfg) or isinstance(self.cfg, Go2RoughVisionEnvCfg):
        # add perceptive inputs if not blind
        if self.cfg.measure_heights:
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            obs = torch.cat((obs, height_data), dim=-1)

        return obs

    def _compute_privileged_obs(self, observations):
        obs = observations.clone()

        extra_obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_w * self.cfg.obs_scale_lin_vel,
                    self._robot.data.root_ang_vel_w * self.cfg.obs_scale_ang_vel,
                    self.friction_coeffs,
                    self.restitution,
                    self.joint_friction_coeffs,
                    self.com_displacement[:, self._base_id],
                    self.mass_offset,
                    (torch.norm(self._contact_sensor.data.net_forces_w[:, self._penalized_contact_body_ids, :], dim=-1) > 0.1).float(),
                )
                if tensor is not None
            ],
            dim=-1,
        )

        privileged_obs = torch.cat([obs, extra_obs], dim=-1)
        return privileged_obs

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.cfg.single_observation_space)
        self.add_noise = self.cfg.add_noise
        noise_level = self.cfg.noise_level
        noise_vec[:3] = self.cfg.noise_scale_lin_vel * noise_level * self.cfg.obs_scale_lin_vel
        noise_vec[3:6] = self.cfg.noise_scale_ang_vel * noise_level * self.cfg.obs_scale_ang_vel
        noise_vec[6:9] = self.cfg.noise_scale_gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = self.cfg.noise_scale_joint_pos * noise_level * self.cfg.obs_scale_joint_pos
        noise_vec[24:36] = self.cfg.noise_scale_joint_vel * noise_level * self.cfg.obs_scale_joint_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.measure_heights:
            noise_vec[48:235] = self.cfg.noise_scale_height_measurements * noise_level * self.cfg.obs_scale_height_measurements
        return noise_vec

    def _recompute_ang_vel(self):
        heading = self._get_cur_heading()
        self._commands[:, 2] = torch.clip(0.5*math_utils.wrap_to_pi(self._commands[:, 3] - heading), -1., 1.)

    def _get_cur_heading(self):
        heading_vec = math_utils.quat_apply_yaw(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        return heading

    def _get_rewards(self) -> torch.Tensor:

        rewards = {}

        for reward_name, weight in self._reward_scales.items():
            # Construct the name of the reward function
            reward_func_name = f"_reward_{reward_name}"
            reward_func = getattr(self, reward_func_name, None)

            # Check if the function exists
            if reward_func:
                # Call the function and apply the weight and step duration
                rewards[reward_name] = reward_func() * weight * self.step_dt
            else:
                # Handle cases where the reward function is not defined
                print(f"Warning: Reward function '{reward_func_name}' not found.")

        # Sum the rewards
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Apply clipping
        clipped_reward = torch.clip(reward, min=0.0)

        # Add termination reward after clipping
        if "termination" in self._reward_scales:
            termination_rew = self._reward_termination() * self._reward_scales["termination"] * self.step_dt
            clipped_reward += termination_rew

        # Accumulate the clipped reward to the episode sum buffer
        self._clipped_episode_sums += clipped_reward

        # You can also add other logging here
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.time_out_buf = time_out

        resets = self._check_termination()

        return resets, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            # Curriculum based on the distance the robot walked
            distance = torch.norm(self._robot.data.root_state_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1)
            # robots that walked far enough progress to harder terrains
            move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2
            # robots that walked less than half of their required distance go to simpler terrains
            move_down = distance < torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
            move_down *= ~move_up
            # update terrain levels
            self._terrain.update_env_origins(env_ids, move_up, move_down)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0

        # Update reward curriculum
        if self.cfg.reward_curriculum and (self.common_step_counter % 200 == 0):
            self._update_reward_curriculum(env_ids)

        # Reset episode rew buf after using it
        self._clipped_episode_sums[env_ids] = 0.

        # Sample new commands
        self._resample_commands(env_ids)

        # Update contact targets
        self._step_contact_targets()

        # Recompute angular velocity command based on new heading
        self._recompute_ang_vel()

        # Reset swing peak
        self._swing_peak[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

        # Reset robot state
        joint_pos, joint_vel, default_root_state = self._reset_robot_states(env_ids)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        # extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())

        self.extras["log"].update(extras)

        # Update command curriculum after extras is populated
        if self.cfg.command_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids, self.extras)

    def _update_command_curriculum(self, extras: dict):
        """Updates the command ranges based on episode performance."""

        # Access the average reward sums directly from the log dictionary
        avg_tracking_lin_vel = extras["log"].get("Episode_Reward/tracking_lin_vel", 0.0)

        # Normalize the reward sum
        normalized_lin_vel_reward = avg_tracking_lin_vel / self._reward_scales["tracking_lin_vel"]

        # Apply the curriculum logic
        if normalized_lin_vel_reward > 0.8:
            self._command_ranges["lin_vel_x"][0] = torch.clip(
                self._command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.command_max_curriculum,
                0.0,
            )
            self._command_ranges["lin_vel_x"][1] = torch.clip(
                self._command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.command_max_curriculum,
            )

    def _update_reward_curriculum(self, env_ids):
        metric = torch.mean(self._clipped_episode_sums[env_ids])
        print("reward metric", metric)
        if metric > 0.2:
            for key in self._reward_scales:
                if self._reward_scales_final[key] < 0:
                    scale = max(self._reward_scales[key] + self.cfg.reward_cl_step * self._reward_scales_final[key], self._reward_scales_final[key])
                    self._reward_scales[key] = scale

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        self._commands[env_ids, 0] = self._sample_fn("continuous", [self._command_ranges["lin_vel_x"][0], self._command_ranges["lin_vel_x"][1]], (len(env_ids),))
        self._commands[env_ids, 1] = self._sample_fn("continuous", [self._command_ranges["lin_vel_y"][0], self._command_ranges["lin_vel_y"][1]], (len(env_ids),))
        # relatively sample
        cur_heading = self._get_cur_heading()[env_ids]
        self._commands[env_ids, 3] = cur_heading + torch_rand_float(
            self.cfg.commands.ranges["heading"][0], self.cfg.commands.ranges["heading"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

    def _sample_fn(self, mode: str, support: list, shape, low: float = None, high: float = None):
        """
        mode: continuous or discrete
        support: [low, high] if `mode` is `continuous`, [limit low, limit, hight, num bins] if `mode` is `discrete`
        shape:
        low: additional range for discrete distribution
        high: additional range for discrete distribution
        """
        if mode == "continuous":
            return torch.rand(shape, dtype=torch.float, device=self.device) * (support[1] - support[0]) + support[0]
        elif mode == "discrete":
            candidates = np.linspace(support[0], support[1], support[2])
            if low is None:
                low = support[0]
            if high is None:
                high = support[1]
            candidates = candidates[np.logical_and(candidates >= low, candidates <= high)]
            assert len(candidates) > 0
            return torch.from_numpy(np.random.choice(candidates, size=shape), dtype=torch.float).to(self.device)

    def _step_contact_targets(self):
        # order of feet: ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']

        frequencies = self.cfg.default_gait_freq
        phases = 0.5
        offsets = 0
        bounds = 0

        durations = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._gait_indices = torch.remainder(self._gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [self._gait_indices + phases + offsets + bounds,
                        self._gait_indices + offsets,
                        self._gait_indices + bounds,
                        self._gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        self._clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self._clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self._clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self._clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # von mises distribution
        if hasattr(self.cfg, "kappa_gait_probs"):
            #print("kappa aaaaaaaaaaaaa")
            kappa = self.cfg.reward_kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
            smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                        smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                                1 - smoothing_cdf_start(
                                            torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

            self._desired_contact_states[:, 0] = smoothing_multiplier_FL
            self._desired_contact_states[:, 1] = smoothing_multiplier_FR
            self._desired_contact_states[:, 2] = smoothing_multiplier_RL
            self._desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _check_termination(self):
        """ Check if environments need to be reset
        """
        resets = torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._termination_ids, :], dim=-1) > 1., dim=1)
        return resets

    def _reset_robot_states(self, env_ids):
        joint_pos, joint_vel = self._reset_dofs_rand(env_ids)
        default_root_state = self._reset_root_states_rand(env_ids)

        return joint_pos, joint_vel, default_root_state

    def _reset_dofs_rand(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if not self.cfg.init_state_randomize_rot:
            joint_pos = self._robot.data.default_joint_pos[env_ids] + torch_rand_float(-0.1, 0.1, self._robot.data.default_joint_pos[env_ids].shape, device=self.device)
        else:
            joint_pos = self._robot.data.joint_pos_limits[..., 0] + (
                self._robot.data.joint_pos_limits[..., 1] - self._robot.data.joint_pos_limits[..., 0]) * torch_rand_float(
                    0.0, 1.0, self._robot.data.default_joint_pos[env_ids].shape, device=self.device
                )
        joint_vel = torch_rand_float(-0.1, 0.1, self._robot.data.default_joint_pos[env_ids].shape, device=self.device)

        return joint_pos, joint_vel

    def _reset_root_states_rand(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            default_root_state[:, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        if self.cfg.init_state_randomize_rot:
            rand_rpy = torch_rand_float(-np.pi, np.pi, (len(env_ids), 3), device=self.device)
            default_root_state[:, 3:7] = math_utils.quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        # base velocities
        default_root_state[:, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        return default_root_state

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        reward = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        forward = math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        reward_upright_vec = torch.tensor(self.cfg.reward_upright_vec, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        upright_vec = custom_utils.quat_apply_yaw(self._robot.data.root_quat_w, reward_upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9

        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        # base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        base_in_collision = torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._base_id, :], dim=-1) > 0.1, dim=-1).float()
        reward = reward * (1 - base_in_collision.float()) * is_stand.float()
        return reward

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        reward = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        forward = math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        reward_upright_vec = torch.tensor(self.cfg.reward_upright_vec, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        upright_vec = custom_utils.quat_apply_yaw(self._robot.data.root_quat_w, reward_upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9

        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._base_id, :], dim=-1) > 0.1, dim=-1).float()
        reward = reward * (1 - base_in_collision.float()) * is_stand.float()
        return reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # "flat" corresponding to the terrain the robot is on
        reward = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        # base_height = torch.mean(self._robot.data.root_pos_w[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        base_height = torch.mean(self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2], dim=1)
        return torch.square(base_height - self.cfg.reward_base_height_target)

    def _reward_torques(self):
        # Penalize torques
        reward = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        return reward

    def _reward_dof_vel(self):
        # Penalize dof velocities
        reward = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        return reward

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        reward = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        return reward

    def _reward_action_rate(self):
        # Penalize changes in actions
        reward = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        return reward

    def _reward_action_overshoot(self):
        # reward = torch.sum(self.overshoot_buf, dim=-1)
        # return reward
        """
        Penalizes the agent for commanding joint positions outside of the hard limits.
        """
        actions = self._actions

        # Get the joint position limits from the robot's data.
        joint_pos_limits = self._robot.data.joint_pos_limits

        # Check for overshoot above the upper limit
        upper_overshoot = (actions - joint_pos_limits[:, :, 1]).clip(min=0)

        # Check for overshoot below the lower limit
        lower_overshoot = (joint_pos_limits[:, :, 0] - actions).clip(min=0)

        # The total penalty is the sum of overshoots from both ends
        penalty = upper_overshoot + lower_overshoot

        # Return the sum of penalties across all joints for each environment
        return -torch.sum(penalty, dim=-1)

    # def _reward_action_q_diff(self):
    #     q_diff_buf = torch.abs(self._robot.data.default_joint_pos + self.cfg.control_action_scale * self._actions - self._robot.data.joint_pos)
    #     reward = torch.sum(q_diff_buf, dim=-1)
    #     return reward

    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1.*(torch.norm(self._contact_sensor.data.net_forces_w[:, self._penalized_contact_body_ids, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(
            self._robot.data.joint_pos - (self._robot.data.soft_joint_pos_limits[:, :, 0])
        ).clip(max=0.0)
        out_of_limits += (
            self._robot.data.joint_pos - (self._robot.data.soft_joint_pos_limits[:, :, 1])
        ).clip(min=0.0)

        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        # compute out of limits constraints

        # return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.reward_soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
        # compute out of limits constraints
        out_of_limits = (
            torch.abs(self._robot.data.joint_vel)
            - self._robot.data.joint_vel_limits * self.cfg.reward_soft_dof_vel_limit
        )
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        # return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.reward_soft_torque_limit).clip(min=0.), dim=1)
        # compute out of limits constraints
        # TODO: We need to fix this to support implicit joints.
        out_of_limits = torch.abs(
            self._robot.data.applied_torque - self._robot.data.computed_torque
        )
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        reward = torch.exp(-lin_vel_error/self.cfg.reward_tracking_sigma)
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._base_id, :], dim=-1) > 0.1, dim=-1).float()
        reward = reward * (1 - base_in_collision.float())
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_w[:, 2])
        reward = torch.exp(-ang_vel_error/self.cfg.reward_tracking_sigma)
        # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        base_in_collision = torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._base_id, :], dim=-1) > 0.1, dim=-1).float()
        reward = reward * (1 - base_in_collision.float())
        return reward

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2] > 1.
        # contact_filt = torch.logical_or(contact, self.last_contacts)
        # self.last_contacts = contact
        # first_contact = (self._feet_air_time > 0.) * contact_filt
        # self._feet_air_time += self.dt
        # rew_airTime = torch.sum((self._feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # # is_in_collision = torch.any(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1, dim=1)
        # base_in_collision = torch.norm(self.contact_forces[:, self.base_contact_indice, :], dim=-1) > 0.1
        # rew_airTime = rew_airTime * (1 - base_in_collision.float())
        # self._feet_air_time *= ~contact_filt
        # return rew_airTime
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        return feet_air_time

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids, :2], dim=2) >
             5 *torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2]), dim=1)

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1) * (torch.norm(self._commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids, :], dim=-1) -  self.cfg.reward_max_contact_force).clip(min=0.), dim=1)

    # from common env
    def _reward_upright(self):
        forward = math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        reward_upright_vec = torch.tensor(self.cfg.reward_upright_vec, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        upright_vec = math_utils.quat_apply_yaw(self._robot.data.root_quat_w, reward_upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
        # dot product with [0, 0, 1]
        # cosine_dist = forward[:, 2]
        reward = torch.square(0.5 * cosine_dist + 0.5)
        return reward

    def _reward_lift_up(self):
        root_height = self._robot.data.root_pos_w[:, 2]
        # four leg stand is ~0.28
        # sit height is ~0.385
        reward = torch.exp(root_height - self.cfg.reward_lift_up_threshold) - 1
        return reward

    def _reward_collision(self):
        reward = torch.sum(1.*(torch.norm(self._contact_sensor.data.net_forces_w[:, self._penalized_contact_body_ids, :], dim=-1) > 0.1), dim=1) #was foot ids first
        cond = self.episode_length_buf > self.cfg.reward_allow_contact_steps
        reward = reward * cond.float()
        return reward

    def _reward_action_q_diff(self):
        condition = self.episode_length_buf <= self.cfg.reward_allow_contact_steps
        q_diff_buf = torch.abs(self._robot.data.default_joint_pos + self.cfg.control_action_scale * self._actions - self._robot.data.joint_pos)
        # reward = torch.sum(q_diff_buf, dim=-1)
        reward = torch.sum(torch.square(q_diff_buf), dim=-1) * condition.float()
        return reward

    def _reward_feet_slip(self):
        contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2], dim=2))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self._robot.data.body_ang_vel_w[:, self._feet_ids_robot, 2:] / np.pi, dim=2))
        rew_slip = torch.sum(contact_filt * (foot_velocities + foot_ang_velocities), dim=1)
        return rew_slip

    def _reward_front_feet_air_time(self):
        # Assuming front feet indices are 0 and 1
        is_front_contact = self._contact_sensor.data.net_forces_w[:, self._feet_ids[:2], 2] > 1.0
        # Sum up the time steps where the front feet are in the air
        return torch.sum(~is_front_contact, dim=-1).float()

    def _reward_time_upright(self):
        """
        Reward for staying in an upright position, using the same definition
        as the main _reward_upright function.

        The reward is a linear function of time, active only when the robot is upright.

        Returns:
            torch.Tensor: A tensor of rewards, one for each environment.
        """
        # Use the same logic as the main upright reward function
        forward = math_utils.quat_apply(self._robot.data.root_quat_w, self._robot.data.FORWARD_VEC_B)
        reward_upright_vec = torch.tensor(self.cfg.reward_upright_vec, device=self.device).unsqueeze(0).expand(self.num_envs, -1)
        upright_vec = math_utils.quat_apply_yaw(self._robot.data.root_quat_w, reward_upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)

        # Check if the robot is "upright" based on a threshold
        is_upright = cosine_dist > 0.9  # You can adjust this threshold

        # Return a linear reward that is active only when the robot is upright.
        reward = is_upright.float() * 1.0  # Reward of 1.0 for every step the robot is upright

        return reward

def normalize_range(x: torch.Tensor, limit):
    if isinstance(limit[0], list):
        low = torch.tensor(np.array(limit[0])).to(x.device)
        high = torch.tensor(np.array(limit[1])).to(x.device)
    else:
        low = limit[0]
        high = limit[1]
    mean = (low + high) / 2
    scale = (high - low) / 2
    if isinstance(scale, torch.Tensor):
        scale = torch.clamp(scale, min=1e-5)
    else:
        scale = max(scale, 1e-5)

    return (x - mean) / scale

def torch_rand_float(min_val, max_val, shape, device='cpu'):
    """
    Generates a tensor of random floats in the range [min_val, max_val].
    """
    return min_val + (max_val - min_val) * torch.rand(shape, device=device)