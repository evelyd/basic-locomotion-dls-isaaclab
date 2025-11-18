# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import importlib
import numpy as np

from dha.nn.DynamicsAutoEncoder import DAE
from dha.nn.EquivDynamicsAutoencoder import EquivDAE
from dha.nn.ControlledDynamicsAutoEncoder import ControlledDAE
from dha.nn.ControlledEquivDynamicsAutoencoder import ControlledEquivDAE

import escnn
from escnn.nn import FieldType
from morpho_symm.utils.rep_theory_utils import group_rep_from_gens
import math

from escnn.group import Group, Representation
from morpho_symm.utils.algebra_utils import permutation_matrix

from typing import Tuple, Union

class RunningMeanStd:
    """
    Calculates the running mean and standard deviation of a data stream.
    Based on the parallel algorithm for calculating variance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Args:
        epsilon (float): Small constant to initialize the count for numerical stability.
        shape (Tuple[int, ...]): Shape of the data (e.g., observation shape).
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: Tuple[int, ...] = (),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        """
        Updates the running statistics using a new batch of data.

        Args:
            arr (torch.Tensor): Batch of data (batch_size, *shape).
        """
        batch = arr.to(self.device, dtype=torch.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            batch.shape[0], dtype=torch.float32, device=self.device
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        """
        Updates statistics using precomputed batch mean, variance, and count.

        Args:
            batch_mean (torch.Tensor): Mean of the batch.
            batch_var (torch.Tensor): Variance of the batch.
            batch_count (torch.Tensor): Number of samples in the batch.
        """
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)


class Normalizer(RunningMeanStd):
    """
    A normalizer that uses running statistics to normalize inputs, with optional clipping.

    Args:
        input_dim (Tuple[int, ...]): Shape of the input observations.
        epsilon (float): Small constant added to variance to avoid division by zero.
        clip_obs (float): Maximum absolute value to clip the normalized observations.
    """

    def __init__(
        self,
        input_dim: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        shape = (input_dim,) if isinstance(input_dim, int) else tuple(input_dim)
        super().__init__(epsilon=epsilon, shape=shape, device=device)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """
        Normalizes input using running mean and std, and clips the result.

        Args:
            input (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized and clipped tensor.
        """
        x = input.to(self.device, dtype=torch.float32)
        std = (self.var + self.epsilon).sqrt()
        y = (x - self.mean) / std
        return torch.clamp(y, -self.clip_obs, self.clip_obs)

    @torch.no_grad()
    def update_normalizer(self, rollouts, expert_loader) -> None:
        """
        Updates running statistics using samples from both policy and expert trajectories.

        Args:
            rollouts: Object with method `feed_forward_generator_amp(...)`.
            expert_loader: Dataloader or similar object providing expert batches.
        """
        policy_generator = rollouts.feed_forward_generator_amp(
            None, mini_batch_size=expert_loader.batch_size
        )
        expert_generator = expert_loader.dataset.feed_forward_generator_amp(
            expert_loader.batch_size
        )

        for expert_batch, policy_batch in zip(expert_generator, policy_generator):
            batch = torch.cat((*expert_batch, *policy_batch), dim=0)
            self.update(batch)

def fill_replay_buffer(algorithm_instance, env_instance, obs_normalizer, critic_obs_normalizer, num_initial_steps=None):
    """
    Initializes the replay buffers within the algorithm by performing dummy rollouts,
    mimicking the regular training loop's data collection process.
    This version correctly handles rsl_rl's RolloutStorage by performing full rollouts
    and clearing storage periodically.

    Args:
        algorithm_instance: PPO or similar
        env_instance: env
        obs_normalizer: Normalizer for observations
        critic_obs_normalizer: Normalizer for critic observations
        num_initial_steps: The total number of environment steps to take for pre-filling
    """

    # Determine num_initial_rollouts based on num_initial_steps
    if num_initial_steps is None:
        if hasattr(algorithm_instance, 'replay_buffer') and hasattr(algorithm_instance.replay_buffer.states, 'shape'):
            required_steps_for_koopman = algorithm_instance.replay_buffer.states.shape[0]
        else:
            required_steps_for_koopman = 10000 # Fallback if buffer info not available
            print("Warning: Could not determine Koopman buffer size. Defaulting to 10000 steps.")

        steps_per_full_rollout = env_instance.num_envs * algorithm_instance.storage.num_transitions_per_env
        if steps_per_full_rollout == 0: # Avoid division by zero if not configured
            steps_per_full_rollout = 1 # Dummy value, will lead to few rollouts
            print("Warning: steps_per_full_rollout is zero, likely due to num_envs or num_steps_per_env. Check config.")


        num_initial_rollouts = max(1, (required_steps_for_koopman + steps_per_full_rollout - 1) // steps_per_full_rollout)
        print(f"No num_initial_steps provided. Will perform {num_initial_rollouts} full rollouts to fill Koopman buffer.")
    else:
        # If num_initial_steps is provided, convert it to rollouts
        steps_per_full_rollout = env_instance.num_envs * algorithm_instance.num_steps_per_env
        if steps_per_full_rollout == 0:
            steps_per_full_rollout = 1
            print("Warning: steps_per_full_rollout is zero, likely due to num_envs or num_steps_per_env. Check config.")

        num_initial_rollouts = max(1, (num_initial_steps + steps_per_full_rollout - 1) // steps_per_full_rollout)
        print(f"num_initial_steps ({num_initial_steps}) will result in {num_initial_rollouts} full rollouts.")


    print(f"Initializing replay buffers by performing {num_initial_rollouts} full rollouts...")

    # Set algorithm's policy to evaluation mode during initialization
    if hasattr(algorithm_instance.policy, 'eval'):
        algorithm_instance.policy.eval()

    # Reset environment to get initial observations for the very first rollout
    obs, extras = env_instance.get_observations()
    critic_obs = extras["observations"].get("critic", obs)
    obs, critic_obs = obs.to(algorithm_instance.device), critic_obs.to(algorithm_instance.device)

    # Ensure koopman_transition is initialized and clear if needed
    if not hasattr(algorithm_instance, 'koopman_transition'):
        raise AttributeError("Algorithm instance missing 'koopman_transition' attribute.")
    algorithm_instance.koopman_transition.clear()

    if hasattr(algorithm_instance, 'storage'):
        algorithm_instance.storage.observations[0].copy_(obs)
        if hasattr(algorithm_instance.storage, 'critic_observations') and extras["observations"]["critic"] is not None:
             algorithm_instance.storage.critic_observations[0].copy_(critic_obs)
    else:
        print("Warning: Algorithm instance does not have 'storage' attribute. Ensure your PPO handles initial observation internally.")

    for rollout_idx in range(num_initial_rollouts):
        for i in range(algorithm_instance.storage.num_transitions_per_env):
            actions = algorithm_instance.act(obs, critic_obs)
            algorithm_instance.act_koopman(obs, actions)
            obs, rewards, dones, infos = env_instance.step(actions)
            obs = obs_normalizer(obs)
            if "critic" in infos["observations"]:
                critic_obs = critic_obs_normalizer(
                    infos["observations"]["critic"]
                )
            else:
                critic_obs = obs
            obs, critic_obs, rewards, dones = (
                obs.to(algorithm_instance.device),
                critic_obs.to(algorithm_instance.device),
                rewards.to(algorithm_instance.device),
                dones.to(algorithm_instance.device),
            )
            algorithm_instance.process_env_step(rewards, dones, infos)

            algorithm_instance.process_koopman_step(obs)

        algorithm_instance.compute_returns(critic_obs.clone().detach())
        algorithm_instance.storage.clear()

    print("Replay buffer initialization complete.")

    # Switch back to train mode
    if hasattr(algorithm_instance.policy, 'train'):
        algorithm_instance.policy.train()


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

def initialize_dae_model(cfg, G, task: str, dt: int, device: torch.device, is_ideal: bool = False) -> torch.nn.Module:
    """
    Initializes a Koopman model based on the provided configuration (from train_cfg.koopman_model).
    Can also load pre-trained weights if cfg.load_path is specified.

    Args:
        cfg: The Koopman model configuration object from train_cfg.
        state_dim (int): The dimension of the observation space (from environment).
        action_dim (int): The dimension of the action space (from environment).
        dt (float): The environment's delta time (from environment).
        device (torch.device): The torch device (e.g., 'cuda:0', 'cpu').

    Returns:
        torch.nn.Module: An initialized Koopman model (new or loaded).
    """

    # Create the state representations
    gspace = escnn.gspaces.no_base_space(G)
    # Extract the representations from G.representations.items()
    rep_Rd = G.representations['R3']
    rep_TqQ_js = G.representations['TqQ_js']
    rep_xy = group_rep_from_gens(G, rep_H={h: rep_Rd(h)[:2, :2].reshape((2, 2)) for h in G.elements if h != G.identity})
    rep_xy.name = "base_xy"
    rep_euler_xyz = G.representations['euler_xyz']
    rep_euler_z = group_rep_from_gens(G, rep_H={h: rep_euler_xyz(h)[2, 2].reshape((1, 1)) for h in G.elements if h != G.identity})
    rep_euler_z.name = "euler_z"
    rep_kin_three_two = get_kinematic_three_rep_two(G)
    rep_kin_three = get_kinematic_three_rep(G)

    # Create dict to define which obs match which representations
    obs_rep_dict = {
        'base_lin_vel': rep_Rd,
        'base_ang_vel': rep_euler_xyz,
        'projected_gravity': rep_Rd,
        'forward_vec': rep_Rd,
        'velocity_commands_xy': rep_xy,
        'velocity_commands_z': rep_euler_z,
        'joint_pos': rep_TqQ_js,
        'joint_vel': rep_TqQ_js,
        'prev_action': rep_TqQ_js,
        'clock_data': rep_kin_three,
        'clock_input': rep_kin_three_two,
        'action': rep_TqQ_js,
    }

    state_reps = []
    action_reps = []
    for state_obs in cfg["robot"]["state_obs"]:
        if state_obs in obs_rep_dict:
            state_reps.append(obs_rep_dict[state_obs])
        else:
            raise ValueError(f"Observation '{state_obs}' not found in the defined representations.")
    for action_obs in cfg["robot"]["action_obs"]:
        if action_obs in obs_rep_dict:
            action_reps.append(obs_rep_dict[action_obs])
        else:
            raise ValueError(f"Action '{action_obs}' not found in the defined representations.")

    state_type = FieldType(gspace, representations=state_reps)
    action_type = FieldType(gspace, representations=action_reps)

    state_dim = cfg["robot"]["state_dim"]
    action_dim = cfg["robot"]["action_dim"]

    # Ensure that with duplicate reps the size matches the expected dimensions
    state_type.size = state_dim
    action_type.size = action_dim

    obs_state_dim = math.ceil(cfg["robot"]["obs_state_ratio"] * state_dim)
    num_hidden_neurons = cfg["model"]["num_hidden_units"]
    if obs_state_dim > num_hidden_neurons:
        num_hidden_neurons = 2 ** math.ceil(math.log2(obs_state_dim))

    activation = cfg["model"]["activation"]

    if not cfg["model"]["equivariant"]:
        activation = class_from_name("torch.nn", activation)

    obs_fn_params = {'num_layers': cfg["model"]["num_layers"], 'num_hidden_units': cfg["model"]["num_hidden_units"], 'activation': activation, 'bias': cfg["model"]["bias"], 'batch_norm': cfg["model"]["batch_norm"]}

    initial_rng_state = torch.get_rng_state()

    if "edae" in task.lower():
        model = EquivDAE(
            state_rep=state_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            group_avg_trick=cfg["model"]["group_avg_trick"],
            state_dependent_obs_dyn=cfg["model"]["state_dependent_obs_dyn"],
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    elif "ecdae" in task.lower():
        model = ControlledEquivDAE(
            state_rep=state_type.representation,
            action_rep=action_type.representation,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            group_avg_trick=cfg["model"]["group_avg_trick"],
            state_dependent_obs_dyn=cfg["model"]["state_dependent_obs_dyn"],
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    elif "cdae" in task.lower():
        model = ControlledDAE(
            state_dim=state_dim,
            action_dim=action_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    elif "dae" in task.lower():
        model = DAE(
            state_dim=state_dim,
            obs_state_dim=obs_state_dim,
            dt=dt,
            obs_pred_w=cfg["model"]["obs_pred_w"],
            orth_w=cfg["model"]["orth_w"],
            obs_fn_params=obs_fn_params,
            enforce_constant_fn=cfg["model"]["constant_function"],
        )
    else:
        raise ValueError(f"Trying to create DAE model with unsupported task: {task}")

    torch.set_rng_state(initial_rng_state)

    # Put the model on the specified device
    model.to(device)

    return model

def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def get_kinematic_three_rep(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(4, dtype=int)}
    gens = [permutation_matrix([1, 0, 3, 2]), permutation_matrix([2, 3, 0, 1]), permutation_matrix([0, 1, 2, 3])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three

def get_kinematic_three_rep_two(G: Group):
    #  [0   1    2   3]
    #  [RF, LF, RH, LH]
    rep_kin_three = {G.identity: np.eye(2, dtype=int)}
    gens = [permutation_matrix([1, 0])]
    for h, rep_h in zip(G.generators, gens):
        rep_kin_three[h] = rep_h

    rep_kin_three = group_rep_from_gens(G, rep_kin_three)
    rep_kin_three.name = "kin_three"
    return rep_kin_three