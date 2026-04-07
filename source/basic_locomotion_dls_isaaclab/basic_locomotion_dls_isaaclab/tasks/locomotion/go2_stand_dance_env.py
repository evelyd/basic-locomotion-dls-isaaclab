from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .go2_stand_dance_env_cfg import Go2StandDanceEnvCfg

class Go2StandDanceEnv(DirectRLEnv):
    cfg: Go2StandDanceEnvCfg

    def __init__(self, cfg: Go2StandDanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)
        # self._commands = torch.zeros(self.num_envs, 3, device=self.device) # vx, vy, yaw_vel
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)
        self._sit_baseline = torch.zeros(self.num_envs, 12, device=self.device)

        self._clipped_episode_sums = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self._command_ranges = {
            "lin_vel_x": torch.tensor([0.0, 0.0], device=self.device),
            "lin_vel_y": torch.tensor([0.0, 0.0], device=self.device),
            "ang_vel_z": torch.tensor([0.0, 0.0], device=self.device),
            "heading": torch.tensor([-0.5 * torch.pi, 0.5 * torch.pi], device=self.device),
        }
        self.command_max_curriculum = 1.0
        self.command_clip_ang_vel = 0.25 * torch.pi

        # Get body indices for Front/Rear feet mapping
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._front_feet_ids, _ = self._contact_sensor.find_bodies("F.*foot")
        self._rear_feet_ids, _ = self._contact_sensor.find_bodies("R.*foot")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._rear_calf_ids, _ = self._contact_sensor.find_bodies("R.*calf")
        self._feet_ids_robot, _ = self._robot.find_bodies(".*foot")
        self._rear_feet_ids_robot, _ = self._robot.find_bodies("R.*foot")
        self._undesired_contact_body_ids = self._base_id + self._contact_sensor.find_bodies(".*hip")[0]
        self._hip_joint_ids = self._robot.find_joints(".*hip_joint")[0]

        self._penalized_contact_ids, _ = self._contact_sensor.find_bodies(
            ["base", ".*hip", ".*thigh", ".*calf.*", "F.*foot"]
        )

        # Find termination contact bodies
        self._term_contact_ids, _ = self._contact_sensor.find_bodies(["base", ".*hip", ".*thigh", ".*calf.*"])

        # Find grace period contact bodies
        self._allow_init_contact_ids, _ = self._contact_sensor.find_bodies([".*foot", "R.*calf.*"])

        # Upright vector pre-allocation
        self._upright_vec_w = torch.tensor(self.cfg.upright_vec, device=self.device).repeat(self.num_envs, 1)
        self._forward_vec_w = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Buffers to track the initial footprint for the foot_shift penalty
        self._init_front_feet_pos_w = torch.zeros(self.num_envs, len(self._front_feet_ids), 3, device=self.device)
        self._init_rear_feet_pos_w = torch.zeros(self.num_envs, len(self._rear_feet_ids_robot), 3, device=self.device)

        # Initialize episode sums for WandB logging
        self.episode_sums = {
            "lift_up_linear": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "tracking_lin_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "tracking_ang_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "upright": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "upright_balance": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "support_polygon": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "rear_air": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_q_diff": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "action_rate": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "joints_torque": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "hip_still": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # "base_pitch": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "feet_clearance_cmd_linear": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "feet_slip": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "foot_shift": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "collision": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        }

        # Initialize reward curriculum level (defaults to 1.0 if not using curriculum)
        self.reward_cl = getattr(self.cfg, "cl_init", 1.0) # Matches cl_init from config

        # Clock buffers for walking rhythm
        self._rear_foot_indices = torch.zeros(self.num_envs, 2, device=self.device)
        self._rear_clock_inputs = torch.zeros(self.num_envs, 2, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):

        self._recompute_ang_vel()
        self._previous_actions = self._actions.clone()
        self._previous_joint_pos = self._robot.data.joint_pos.clone()
        self._actions = torch.clamp(actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:

        # If the episode just reset (step 0), capture the exact world-frame coordinates of the feet
        reset_env_ids = self.episode_length_buf == 0
        if reset_env_ids.any():
            self._init_front_feet_pos_w[reset_env_ids] = self._robot.data.body_pos_w[reset_env_ids][:, self._front_feet_ids, :].clone()
            self._init_rear_feet_pos_w[reset_env_ids] = self._robot.data.body_pos_w[reset_env_ids][:, self._rear_feet_ids_robot, :].clone()

        # Sample commands (Simplified for template)
        # resample_mask = self.episode_length_buf % 200 == 0
        # self._commands[resample_mask] = torch.zeros_like(self._commands[resample_mask]).uniform_(-0.3, 0.3)
        # self._commands[resample_mask, 1] *= 0.0 # Restrict lateral movement
        # Remove the old command sampling logic and use this:
        resample_mask = (self.episode_length_buf % int(10.0 / self.step_dt) == 0) # e.g. every 10 seconds
        resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # Vectors
        base_quat = self._robot.data.root_quat_w
        projected_gravity = self._robot.data.projected_gravity_b
        projected_forward = math_utils.quat_apply_inverse(base_quat, self._forward_vec_w)

        # Scale commands (lin_vel scale = 2.0, ang_vel scale = 0.25)
        scaled_commands = self._commands.clone()
        scaled_commands[:, :2] *= 2.0
        scaled_commands[:, 2] *= 0.25

        # Scale DOF pos (scale = 1.0) and DOF vel (scale = 0.0)
        scaled_dof_pos = (self._robot.data.joint_pos - self._robot.data.default_joint_pos) * 1.0
        scaled_dof_vel = self._robot.data.joint_vel * 0.0

        # -----------------------------------------------------
        # NEW: Clock / Rhythm Generator
        # -----------------------------------------------------
        gait_freq = 2.5
        gait_indices = torch.remainder(self.episode_length_buf * self.step_dt * gait_freq, 1.0)

        # Rear Left (RL) and Rear Right (RR) offset by 0.5 for a trot
        foot_indices_RL = torch.remainder(gait_indices + 0.0, 1.0)
        foot_indices_RR = torch.remainder(gait_indices + 0.5, 1.0)
        self._rear_foot_indices = torch.stack([foot_indices_RL, foot_indices_RR], dim=1)

        # Calculate sine waves for the neural network
        self._rear_clock_inputs = torch.sin(2 * torch.pi * self._rear_foot_indices)

        obs = torch.cat([
            projected_gravity,
            projected_forward,
            scaled_commands[:,:3], # don't use heading as an obs
            scaled_dof_pos,
            scaled_dof_vel,
            self._actions,
            self._rear_clock_inputs, # Add the rear foot clock signals to the observation
        ], dim=-1)

        if self.cfg.use_observation_history:
            self._observation_history = torch.cat((self._observation_history[:, 1:, :], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:

        # Generate the rhythmic clock (2.5 Hz Trot)
        gait_freq = 2.5
        # Calculate where we are in the phase (0.0 to 1.0)
        self.gait_indices = torch.remainder(self.episode_length_buf * self.step_dt * gait_freq, 1.0)

        # Rear Left (RL) and Rear Right (RR) are offset by 0.5 for a trot
        self.foot_indices_RL = torch.remainder(self.gait_indices + 0.0, 1.0)
        self.foot_indices_RR = torch.remainder(self.gait_indices + 0.5, 1.0)
        self.rear_foot_indices = torch.stack([self.foot_indices_RL, self.foot_indices_RR], dim=1)

        # Sine waves for the neural network observation
        clock_RL = torch.sin(2 * torch.pi * self.foot_indices_RL)
        clock_RR = torch.sin(2 * torch.pi * self.foot_indices_RR)
        self.rear_clock_inputs = torch.stack([clock_RL, clock_RR], dim=1)

        # liftup linear rew
        delta_height = self._robot.data.root_pos_w[:, 2] - 0.016  # offset for foot radius

        lift_up_reward = (delta_height - self.cfg.lift_up_threshold[0]) / (self.cfg.lift_up_threshold[1] - self.cfg.lift_up_threshold[0])
        lift_up_reward = torch.clamp(lift_up_reward, 0.0, 1.0)

        base_quat = self._robot.data.root_quat_w

        # Transform the robot's local forward vector into the world frame
        forward_w = math_utils.quat_apply(base_quat, self._forward_vec_w)
        # Rotate the upright target by the robot's current yaw
        upright_target_w = math_utils.quat_apply_yaw(base_quat, self._upright_vec_w)

        dot_product = torch.sum(forward_w * upright_target_w, dim=-1)
        is_stand = (dot_product / torch.norm(upright_target_w, dim=-1)) > 0.9

        scaling_factor = (torch.clip(delta_height, min=self.cfg.scale_factor_low, max=self.cfg.scale_factor_high) - self.cfg.scale_factor_low) / (self.cfg.scale_factor_high - self.cfg.scale_factor_low)

        # Extract yaw from the base quaternion to build the heading frame
        _, _, yaw = math_utils.euler_xyz_from_quat(base_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Get world-frame linear velocities
        vx_w = self._robot.data.root_lin_vel_w[:, 0]
        vy_w = self._robot.data.root_lin_vel_w[:, 1]

        # Rotate world velocities into the flat heading frame (Equivalent to quat_apply_yaw_inverse)
        vx_heading = vx_w * cos_yaw + vy_w * sin_yaw
        vy_heading = -vx_w * sin_yaw + vy_w * cos_yaw
        heading_lin_vel_xy = torch.stack([vx_heading, vy_heading], dim=1)

        # Track linear velocity in the heading frame (Flat to the ground!)
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - heading_lin_vel_xy), dim=1)
        track_lin_vel = torch.exp(-lin_vel_error / 0.05) * is_stand.float() * scaling_factor

        # Track yaw rate in the WORLD frame (World Z is always upright, avoiding barrel rolls!)
        # This is mathematically equivalent to your (heading - last_heading) / dt estimation
        # ang_vel_error = torch.abs(self._commands[:, 2] - self._robot.data.root_ang_vel_w[:, 2])
        heading = self._get_cur_heading()
        est_ang_vel = math_utils.wrap_to_pi(heading - self.last_heading) / self.step_dt
        ang_vel_error = torch.abs(self._commands[:, 2] - est_ang_vel)
        track_ang_vel = torch.exp(-ang_vel_error / 0.2) * is_stand.float() * scaling_factor

        # upright rew
        cosine_dist = dot_product / torch.norm(upright_target_w, dim=-1)
        upright_reward = torch.square(0.5 * cosine_dist + 0.5)

        # rear air pen
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Check contacts (Force > 1.0 means touching)
        rear_foot_touching = torch.max(torch.norm(net_contact_forces[:, :, self._rear_feet_ids], dim=-1), dim=1)[0] > 1.0
        rear_calf_touching = torch.max(torch.norm(net_contact_forces[:, :, self._rear_calf_ids], dim=-1), dim=1)[0] > 1.0

        # Penalize if BOTH rear feet are in the air simultaneously
        both_feet_in_air = torch.all(~rear_foot_touching, dim=1)

        # Penalize if the calf is touching BUT the foot is not (Knee-walking!)
        unhealthy_condition = rear_calf_touching & ~rear_foot_touching

        # Sum the penalties
        rear_air_reward = both_feet_in_air.float() + unhealthy_condition.sum(dim=-1).float()

        # Create a float mask that is 1.0 during sit-to-stand, and 0.0 afterwards
        mercy_mask_float = (self.episode_length_buf <= self.cfg.allow_contact_steps).float()

        # action q diff rew
        # Bulletproof explicit motor target calculation
        motor_targets = (self._actions * self.cfg.action_scale) + self._robot.data.default_joint_pos
        q_diff_raw = torch.sum(torch.square(motor_targets - self._robot.data.joint_pos), dim=-1)
        # q_diff = q_diff_raw * mercy_mask_float
        q_diff = q_diff_raw
        # Relax the tracking sigma to match Isaac Gym (0.25 instead of 0.05)
        track_lin_vel = torch.exp(-lin_vel_error / 0.25) * is_stand.float() * scaling_factor

        # action rate pen
        action_rate_reward = torch.sum(torch.square(self._previous_actions - self._actions), dim=1)

        # joint torque pen
        torques_reward = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # hip still pen
        # Hips are typically the 0th, 3rd, 6th, and 9th joints in the 12-DOF array
        hip_movement = torch.abs(self._robot.data.joint_pos[:, self._hip_joint_ids]).mean(dim=1)
        hip_still_reward = hip_movement * mercy_mask_float

        # foot clearance pen
        # Creates a triangle wave (0 -> 1 -> 0) based on the clock from _get_observations
        phases = 1 - torch.abs(1.0 - torch.clip((self._rear_foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)

        rear_foot_heights = self._robot.data.body_pos_w[:, self._rear_feet_ids_robot, 2]
        terrain_at_foot_height = 0.0

        # Target an arc of 0.05m height
        target_height = 0.05 * phases + terrain_at_foot_height + 0.02

        # MINIMAL CHANGE: Replace active_mask_float with command velocity check
        # is_commanded_to_move = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()

        clearance_error = torch.square(target_height - rear_foot_heights)

        # # Multiply by the command mask so it only marches when told to walk!
        # rew_foot_clearance = torch.sum(clearance_error, dim=1) * is_commanded_to_move
        rew_foot_clearance = torch.sum(clearance_error, dim=1)

        # feet slip pen
        # 1. Condition: Foot Z-height < 0.03m (Assuming flat terrain at Z=0.0 for now)
        foot_pos = self._robot.data.body_pos_w[:, self._feet_ids_robot, :]
        slip_condition = (foot_pos[:, :, 2] - 0.0) < 0.03

        # 2. XY Linear Velocity squared
        foot_lin_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, 0:2]
        foot_velocities = torch.square(torch.norm(foot_lin_vel, dim=-1))

        # 3. Yaw Angular Velocity squared (divided by pi)
        foot_ang_vel_yaw = self._robot.data.body_ang_vel_w[:, self._feet_ids_robot, 2:3]
        foot_ang_velocities = torch.square(torch.norm(foot_ang_vel_yaw / torch.pi, dim=-1))

        rew_feet_slip = torch.sum(slip_condition.float() * (foot_velocities + foot_ang_velocities), dim=1)

        # foot shift pen
        rear_foot_pos = self._robot.data.body_pos_w[:, self._rear_feet_ids_robot, :]
        desired_rear_foot_pos = self._init_rear_feet_pos_w.clone()
        desired_rear_foot_pos[:, :, 2] = 0.02 # Fixed 2cm height target from your code

        rear_foot_shift = torch.norm(rear_foot_pos - desired_rear_foot_pos, dim=-1).mean(dim=1)

        # --- FRONT FEET SHIFT ---
        front_foot_pos = self._robot.data.body_pos_w[:, self._front_feet_ids, :]
        init_front_foot_pos = self._init_front_feet_pos_w

        # X distance: clamped to penalize backward sliding, but allow forward sliding!
        dx = (init_front_foot_pos[:, :, 0] - front_foot_pos[:, :, 0]).clamp(min=0)
        # Y distance: absolute deviation
        dy = torch.abs(init_front_foot_pos[:, :, 1] - front_foot_pos[:, :, 1])

        front_foot_shift = torch.norm(torch.stack([dx, dy], dim=-1), dim=-1).mean(dim=1)

        # --- APPLY THE MASK ---
        # Only active during the initial sit-to-stand phase
        shift_condition = (self.episode_length_buf <= self.cfg.allow_contact_steps).float()
        rew_foot_shift = (front_foot_shift + rear_foot_shift) * shift_condition

        # 1. Get current instantaneous contact forces on all penalized bodies
        penalized_contact_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._penalized_contact_ids], dim=-1)

        # 2. Count how many prohibited bodies are touching the ground (Force > 0.1 N)
        collision_count = torch.sum((penalized_contact_forces > 0.1).float(), dim=1)

        # 3. Apply the Mask (Only active AFTER the sit-to-stand mercy steps)
        collision_cond = (self.episode_length_buf > self.cfg.allow_contact_steps).float()

        # 4. Final collision penalty count
        rew_collision = collision_count * collision_cond

        # implement upright balance reward
        # 1. Measure Pitch Angular Velocity (rocking forward/backward)
        pitch_vel = self._robot.data.root_ang_vel_b[:, 1]
        
        # 2. Penalize high pitch velocity (Damper)
        # Using the tracking_ang_y_sigma = 0.1 from your old config
        balance_reward = torch.exp(-torch.square(pitch_vel) / 0.1) 
        
        # 3. Only apply this damper when the robot is actually standing
        balance_reward = balance_reward * is_stand.float()
        
        # 1. Find the XY center point between the two rear feet
        rear_feet_xy = torch.mean(self._robot.data.body_pos_w[:, self._rear_feet_ids_robot, :2], dim=1)
        
        # 2. Get the Base CoM position in the XY plane
        base_xy = self._robot.data.root_pos_w[:, :2]
        
        # 3. Calculate the squared distance between CoM and the support center
        com_shift_error = torch.sum(torch.square(base_xy - rear_feet_xy), dim=-1)
        
        # 4. Reward keeping the CoM directly over the feet (only when trying to stand!)
        # (Using a tight 0.02 sigma so it requires precision)
        support_polygon_reward = torch.exp(-com_shift_error / 0.02) * is_stand.float()

        rewards = {
            "lift_up_linear": lift_up_reward * self.cfg.lift_up_linear_scale * self.step_dt,
            "tracking_lin_vel": track_lin_vel * self.cfg.tracking_lin_vel_stand_scale * self.step_dt,
            "tracking_ang_vel": track_ang_vel * self.cfg.tracking_ang_vel_stand_scale * self.step_dt,
            "upright": upright_reward * self.cfg.upright_scale * self.step_dt,
            # 4. Add to your rewards dictionary (Weight: ~2.0)
            # "upright_balance": balance_reward * self.cfg.upright_balance_scale * self.step_dt,
            "support_polygon": support_polygon_reward * self.cfg.support_polygon_scale * self.step_dt,

            # Apply curriculum multiplier
            "rear_air": rear_air_reward.float() * self.cfg.rear_air_scale * self.reward_cl * self.step_dt,
            "action_q_diff": q_diff * self.cfg.action_q_diff_scale * self.reward_cl * self.step_dt,

            # Regularization
            "action_rate": action_rate_reward * self.cfg.action_rate_reward_scale * self.step_dt,
            "joints_torque": torques_reward * self.cfg.joints_torque_reward_scale * self.step_dt,
            "hip_still": hip_still_reward * self.cfg.hip_still_scale * self.step_dt,

            # Clock stepping
            "feet_clearance_cmd_linear": rew_foot_clearance * self.cfg.feet_clearance_cmd_linear_scale * self.step_dt,
            "feet_slip": rew_feet_slip * self.cfg.feet_slip_scale * self.step_dt,
            "foot_shift": rew_foot_shift * self.cfg.foot_shift_scale * self.step_dt,
            "collision": rew_collision * self.cfg.undesired_contact_reward_scale * self.step_dt,
        }

        # Add to episodic sums for logging
        for key, value in rewards.items():
            self.episode_sums[key] += value

        # 1. Sum the step rewards
        total_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for key, reward_value in rewards.items():
            total_reward += reward_value
            self.episode_sums[key] += reward_value  # Added ONLY ONCE!

        # 2. Log the pitch (but don't put it in the rewards dictionary!)
        # _, pitch, _ = math_utils.euler_xyz_from_quat(base_quat)
        # self.episode_sums["base_pitch"] += pitch    # Added ONLY ONCE!

        # 3. Clamp the total step reward (Floor of 0.0)
        total_reward = torch.clamp(total_reward, min=0.0)

        # 4. Add the clamped reward to the curriculum buffer
        self._clipped_episode_sums += total_reward

        # Return clamped or unclamped depending on what your RL algorithm expects
        # (Usually, RL algorithms want the raw, unclamped reward returned here)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # 1. EXACT ISAAC GYM CONTACT IMMUNITY LOGIC
        any_term_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._term_contact_ids], dim=-1), dim=1)[0] > 1.0, dim=1)
        any_allow_init_contact = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._allow_init_contact_ids], dim=-1), dim=1)[0] > 1.0, dim=1)

        mercy_steps = self.episode_length_buf <= self.cfg.allow_contact_steps

        # Die if a term contact is hit, UNLESS an allowed initial contact is also touching during mercy steps!
        contact_died = any_term_contact & ~mercy_steps

        # 2. POSITION PROTECT (Limits + 5 degrees)
        grace_period = self.episode_length_buf > 30
        joint_pos = self._robot.data.joint_pos

        # EXACT ISAAC GYM FIX: Use true joint_pos_limits, NOT soft_joint_pos_limits!
        hard_limits = self._robot.data.joint_pos_limits

        margin = 5.0 * 3.14159 / 180.0
        position_protect = grace_period & torch.any(
            (joint_pos < hard_limits[:, :, 0] + margin) | (joint_pos > hard_limits[:, :, 1] - margin), dim=-1
        )

        # 3. STAND AIR CONDITION (Rear feet > 6cm during mercy steps)
        rear_foot_heights = self._robot.data.body_pos_w[:, self._rear_feet_ids_robot, 2]
        # # (Assuming flat terrain at Z=0 for height calculation)
        # stand_air = grace_period & mercy_steps & torch.any((rear_foot_heights > 0.06), dim=-1)
        # Only kill for stand_air AFTER the launch phase is over
        physics_settle_steps = self.episode_length_buf > 3
        
        rear_foot_heights = self._robot.data.body_pos_w[:, self._rear_feet_ids_robot, 2]
        init_rear_foot_heights = self._init_rear_feet_pos_w[:, :, 2]
        
        # Only penalize jumping DURING the mercy steps, and allow a 15cm bounce margin for PhysX 5
        stand_air = physics_settle_steps & mercy_steps & torch.any(
            (rear_foot_heights > init_rear_foot_heights + 0.06), dim=-1
        )
        # TODO do i need termination rewards?

        # 4. ABRUPT CHANGE (Joints move > 0.3 rad)
        abrupt_change = physics_settle_steps & mercy_steps & torch.any(
                    torch.abs(self._robot.data.joint_pos - self._previous_joint_pos) > 0.3, dim=-1
                )
        
        # COMBINE EXACTLY LIKE ISAAC GYM (reset_buf |= ...)
        died = contact_died | position_protect | stand_air | abrupt_change

        self.extras["time_outs"] = time_out
        if "episode" not in self.extras:
            self.extras["episode"] = {}

        self.extras["episode"]["died_base_contact"] = torch.mean(contact_died.float())
        self.extras["episode"]["died_position_protect"] = torch.mean(position_protect.float())
        self.extras["episode"]["died_stand_air"] = torch.mean(stand_air.float())
        self.extras["episode"]["died_abrupt_change"] = torch.mean(abrupt_change.float())

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ====================================================================
        # 1. EVALUATE CURRICULUM FIRST!
        # ====================================================================
        metric = torch.mean(self._clipped_episode_sums[env_ids])

        print(f"metric: {metric}")

        if metric > 0.2:
            cl_step = getattr(self.cfg, "cl_step", 0.2)
            self.reward_cl = min(1.0, self.reward_cl + cl_step)

        self._clipped_episode_sums[env_ids] = 0.0

        if "episode" not in self.extras:
            self.extras["episode"] = {}
        self.extras["episode"]["reward_cl"] = torch.tensor(self.reward_cl, device=self.device)

        # ====================================================================
        # 2. STANDARD RESET (No Hacks!)
        # ====================================================================
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.use_observation_history:
            self._observation_history[env_ids] = 0.0

        # EXACT ISAAC GYM SPAWN: Fetch the Stand states...
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]

        # ...but OVERRIDE the base height to be low to the ground!
        default_root_state[:, 2] = 0.221

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # OVERRIDE the joints to the Sit Pose!
        hip_idx = self._robot.find_joints(".*hip_joint")[0]
        thigh_idx = self._robot.find_joints(".*thigh_joint")[0]
        calf_idx = self._robot.find_joints(".*calf_joint")[0]

        joint_pos[:, hip_idx] = 0.0
        joint_pos[:, thigh_idx] = 1.2
        joint_pos[:, calf_idx] = -2.15

        self._sit_baseline[env_ids] = joint_pos.clone()

        # Initialize previous joint pos to the SIT pose so abrupt_change doesn't trigger!
        self._previous_joint_pos[env_ids] = joint_pos.clone()
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Log and reset episode sums
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length
            self.episode_sums[key][env_ids] = 0.0

        # Resample commands
        self._resample_commands(env_ids)
        self.last_heading[env_ids] = self._get_cur_heading()[env_ids]

        # Update command curriculum
        if self.common_step_counter % 200 == 0:
            self._update_command_curriculum()

    def _get_cur_heading(self):
        # Calculate current yaw heading from the base quaternion
        heading_vec = math_utils.quat_apply_yaw(self._robot.data.root_quat_w, self._forward_vec_w)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        return heading

    def _recompute_ang_vel(self):
        # Dynamically recalculate the yaw velocity command based on current heading error
        heading = self._get_cur_heading()
        self._commands[:, 2] = torch.clip(
            0.5 * math_utils.wrap_to_pi(self._commands[:, 3] - heading),
            -self.command_clip_ang_vel,
            self.command_clip_ang_vel
        ) * (0.5 * torch.pi / self.command_clip_ang_vel)

    def _resample_commands(self, env_ids):
        # Sample continuous velocities from current curriculum bounds
        self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (self._command_ranges["lin_vel_x"][1] - self._command_ranges["lin_vel_x"][0]) + self._command_ranges["lin_vel_x"][0]
        self._commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (self._command_ranges["lin_vel_y"][1] - self._command_ranges["lin_vel_y"][0]) + self._command_ranges["lin_vel_y"][0]

        # Sample heading relative to current heading
        cur_heading = self._get_cur_heading()[env_ids]
        heading_noise = torch.rand(len(env_ids), device=self.device) * (self._command_ranges["heading"][1] - self._command_ranges["heading"][0]) + self._command_ranges["heading"][0]
        self._commands[env_ids, 3] = cur_heading + heading_noise

        # Discretize forward velocity to 0.1 intervals (from your specific stand_dance_direct_env.py)
        conti_velx_cmd = self._commands[env_ids, 0:1]
        self._commands[env_ids, 0:1] = torch.sign(conti_velx_cmd) * torch.round(torch.abs(conti_velx_cmd) / 0.1) * 0.1

    def _update_command_curriculum(self):
        # This should be called periodically (e.g., when the max_episode_length is reached)
        avg_tracking_lin_vel = torch.mean(self.episode_sums["tracking_lin_vel"]) / self.max_episode_length
        avg_tracking_ang_vel = torch.mean(self.episode_sums["tracking_ang_vel"]) / self.max_episode_length

        # Normalize by the reward scales you set in your config
        normalized_lin_vel_reward = avg_tracking_lin_vel / self.cfg.tracking_lin_vel_stand_scale
        normalized_ang_vel_reward = avg_tracking_ang_vel / self.cfg.tracking_ang_vel_stand_scale

        if normalized_lin_vel_reward > 0.8:
            self._command_ranges["lin_vel_x"][0] = torch.clip(self._command_ranges["lin_vel_x"][0] - 0.2, -self.command_max_curriculum, 0.)
            self._command_ranges["lin_vel_x"][1] = torch.clip(self._command_ranges["lin_vel_x"][1] + 0.2, 0., self.command_max_curriculum)
            self._command_ranges["lin_vel_y"][0] = torch.clip(self._command_ranges["lin_vel_y"][0] - 0.2, -self.command_max_curriculum, 0.)
            self._command_ranges["lin_vel_y"][1] = torch.clip(self._command_ranges["lin_vel_y"][1] + 0.2, 0., self.command_max_curriculum)

        if normalized_ang_vel_reward > 0.8:
            self._command_ranges["ang_vel_z"][0] = torch.clip(self._command_ranges["ang_vel_z"][0] - 0.2, -self.command_max_curriculum, 0.)
            self._command_ranges["ang_vel_z"][1] = torch.clip(self._command_ranges["ang_vel_z"][1] + 0.2, 0., self.command_max_curriculum)