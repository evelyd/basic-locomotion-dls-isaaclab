import torch
import isaaclab.utils.math as math_utils
# from .aliengo_env_cfg import AliengoFlatEnvCfg, AliengoRoughBlindEnvCfg, AliengoRoughVisionEnvCfg
from .aliengo_symmloco_env_cfg import AliengoStandDanceEnvCfg
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs import VecEnvStepReturn

class StandDanceEnv(ManagerBasedRLEnv):
    """Reinforcement learning environment for quadruped standing and dancing."""

    cfg: AliengoStandDanceEnvCfg

    _cached_clock_inputs: torch.Tensor = None

    # def __init__(self, cfg: AliengoStandDanceEnvCfg, sim_params, physics_engine, sim_device, headless):
    #     super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    def __init__(self, cfg: AliengoStandDanceEnvCfg, env_idx: int | None = None, **kwargs):
        # Init values that have to exist before super call
        # self._forward_vec = torch.tensor([1, 0, 0])

        # render_mode is handled by the framework based on cfg.sim
        super().__init__(cfg, env_idx=env_idx, **kwargs)

        # Initialize environment-specific tensors
        self.last_heading = torch.zeros(self.num_envs, device=self.device)
        self.init_feet_positions = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # Additional initializations from CyberEnv parent
        # self._forward_vec = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.upright_vec = torch.tensor([0.2, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.gravity_vec = torch.tensor([0, 0, -1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # Initialize gait-related tensors
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.foot_positions = torch.zeros(self.num_envs, 4, 3, device=self.device) # Initialize

        self.robot: Articulation = self.scene["robot"]
        self.q_diff_buf = torch.zeros(self.num_envs, self.robot.num_joints, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = self.robot.data.joint_pos
        self.last_contacts = torch.zeros(self.num_envs, self.robot.num_bodies, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.robot.data.joint_pos)

        self.friction_coeffs = torch.zeros(
            (cfg.scene.num_envs, 1), dtype=torch.float, device=self.device
        )

        self.restitution = torch.zeros(
            (cfg.scene.num_envs, 1), dtype=torch.float, device=self.device
        )

        self.joint_friction_coeffs = torch.zeros(
            (cfg.scene.num_envs, 12), dtype=torch.float, device=self.device
        )

        # Assuming the observation expects a flattened vector: num_bodies * 3 (for x, y, z)
        self.com_displacement = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device
        )

        # self.foot_asset_cfg = SceneEntityCfg("robot", body_names=[".*_foot"])
        # input(f"foot indices updated in step: {self.foot_asset_cfg}")
        self.feet_indices, feet_names = self.robot.find_bodies(".*_foot")
        self.upright_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    @property
    def _clock_inputs(self) -> torch.Tensor:
        """
        Lazily initializes the clock inputs the first time it's accessed.
        This ensures self.device and self.num_envs are available.
        """
        if self._cached_clock_inputs is None:
            self._cached_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        return self._cached_clock_inputs

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """

        # call pre physics updates
        self._update_gait_phase_variables()
        self.foot_positions = self.robot.data.body_pos_w[:, self.feet_indices, :] # Update
        self.last_contacts = self.scene.sensors["contact_forces"].data.net_forces_w_history > 1.

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # call post physics updates
        self.last_heading[:] = self._get_cur_heading()
        self.init_feet_positions[self.episode_length_buf == 1] = self.foot_positions[self.episode_length_buf == 1]
        self.last_dof_pos[:] = self.robot.data.joint_pos[:]
        self.q_diff_buf = torch.abs(self.default_dof_pos.to(self.device) + 0.25 * self.action_manager.action.to(self.device) - self.robot.data.joint_pos.to(self.device))

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: torch.Tensor):

        # input(f"reset being called")

        super()._reset_idx(env_ids)
        heading = self._get_cur_heading()
        self.last_heading[env_ids] = heading[env_ids]
        self.init_feet_positions[env_ids] = self.foot_positions[env_ids] # Initialize on reset
        self.last_dof_pos[env_ids] = self.robot.data.joint_pos[env_ids] # Reset last dof pos

        # Reset gait indices for the specified environments
        self.gait_indices[env_ids] = torch.zeros(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False)
        self._clock_inputs[env_ids] = torch.zeros(len(env_ids), 4, dtype=torch.float, device=self.device, requires_grad=False)

    def _update_gait_phase_variables(self):
        """
        Computes the phase variable per foot, as per the original Isaac Gym code.
        """
        # Assuming these are fixed values as per your original code's 'else' ranch
        # If they become dynamic from commands, you'd access them like self.commands.some_gait_param (if defined in CommandsCfg)
        # Or from specific entries in self.commands.data if you have a custom CommandTerm
        frequencies = torch.full((self.num_envs,), self.cfg.default_gait_freq, dtype=torch.float, device=self.device)
        phases = torch.full((self.num_envs,), 0.5, dtype=torch.float, device=self.device)
        offsets = torch.full((self.num_envs,), 0.0, dtype=torch.float, device=self.device)
        bounds = torch.full((self.num_envs,), 0.0, dtype=torch.float, device=self.device)

        durations = 0.5 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0) # self.dt is env.dt_sim_env from ManagerBasedEnv

        # Note: Your original code has a mix-up in which indices refer to which foot.
        # Ensure the mapping below (idx 0 to 3) correctly corresponds to your robot's foot order.
        # Assuming standard quadruped order: FR, FL, RR, RL or similar.
        # This mapping assumes:
        # foot_indices[0] -> FL
        # foot_indices[1] -> FR
        # foot_indices[2] -> RL
        # foot_indices[3] -> RR
        # Adjust as per your robot's limb conventions.
        foot_phases_temp = [
        self.gait_indices + offsets, # Example: FL
        self.gait_indices + phases + offsets + bounds, # Example: FR
        self.gait_indices + phases, # Example: RL
        self.gait_indices + bounds # Example: RR
        ]

        # Concatenate and normalize to [0, 1)
        self.foot_indices = torch.remainder(torch.cat([fp.unsqueeze(1) for fp in foot_phases_temp], dim=1), 1.0)

        # Apply the transformation for swing/stance phases
        # Iterate over the columns of self.foot_indices (each column is a foot's phase)
        for i in range(self.foot_indices.shape[1]): # Iterate 4 times for 4 feet
            idxs = self.foot_indices[:, i].clone() # Work on a copy of the column
            stance_idxs_mask = torch.remainder(idxs, 1) < durations
            swing_idxs_mask = torch.remainder(idxs, 1) >= durations # Changed to >= from > for full coverage

            # Apply transformation for stance phase
            self.foot_indices[stance_idxs_mask, i] = torch.remainder(idxs[stance_idxs_mask], 1) * (0.5 / durations[stance_idxs_mask])

            # Apply transformation for swing phase
            self.foot_indices[swing_idxs_mask, i] = 0.5 + (torch.remainder(idxs[swing_idxs_mask], 1) - durations[swing_idxs_mask]) * (
            0.5 / (1.0 - durations[swing_idxs_mask])) # Use 1.0 for float division

        # Compute clock inputs based on the transformed foot phases
        # self.foot_indices[:, 0] corresponds to foot_phases_temp[0], etc.
        self._clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[:, 0])
        self._clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[:, 1])
        self._clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[:, 2])
        self._clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[:, 3])

    def _recompute_ang_vel(self):
        heading = self._get_cur_heading()
        self.commands[:, 2] = torch.clip(
            0.5 * math_utils.wrap_to_pi(self.commands[:, 3] - heading),
            -self.cfg.commands_cfg.desired_vel_cfg.clip_ang_vel, self.cfg.commands_cfg.desired_vel_cfg.clip_ang_vel
        ) * (0.5 * np.pi / self.cfg.commands_cfg.desired_vel_cfg.clip_ang_vel)

    def _get_cur_heading(self):
        """ Get current heading of the robot. """
        return torch.atan2(self.robot.data.root_quat_w[:, 2], self.robot.data.root_quat_w[:, 0]) * 2.0

    def _get_heights_at_points(self, points):
        """ Get terrain heights at given (x,y) points. """
        return torch.zeros_like(points[:, :, 0]) # Assuming flat terrain for now