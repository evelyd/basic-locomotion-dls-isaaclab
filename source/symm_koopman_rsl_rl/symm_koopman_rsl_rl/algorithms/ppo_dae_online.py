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
# and/or other materials provided from the distribution.
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

import os
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from escnn.nn import GeometricTensor

import escnn
# Assuming ppo.py is in the same directory or accessible via PYTHONPATH
from rsl_rl.algorithms import PPO
from symm_koopman_rsl_rl.storage import ReplayBuffer, RunningStdScaler, PrioritizedReplayBuffer

from symm_koopman_rsl_rl.utils import initialize_dae_model

class PPODAEOnline(PPO):
    def __init__(self,
                 policy,
                 task,
                 koopman_cfg,
                 dt,
                 G: escnn.group.groups.cyclicgroup.CyclicGroup,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 replay_buffer_size=100000,
                 normalize_advantage_per_mini_batch=False,
                 # RND parameters
                 rnd_cfg: dict | None = None,
                 # Symmetry parameters
                 symmetry_cfg: dict | None = None,
                 # Distributed training parameters
                 multi_gpu_cfg: dict | None = None,
                 ):

        # Initialize the base PPO class first
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.koopman_transition = RolloutStorage.Transition()
        self.replay_buffer = PrioritizedReplayBuffer(koopman_cfg["robot"]["state_dim"], koopman_cfg["robot"]["action_dim"], koopman_cfg["model"]["beta_initial"], koopman_cfg["model"]["beta_annealing_steps"], replay_buffer_size) #default device will be cpu
        self.obs_action_normalizer = RunningStdScaler(koopman_cfg["robot"]["state_dim"], koopman_cfg["robot"]["action_dim"], device=self.device)

        self.state_dim = koopman_cfg["robot"]["state_dim"]
        self.action_dim = koopman_cfg["robot"]["action_dim"]

        self.task = task
        # Initialize DAE model
        self.dae_model = initialize_dae_model(
        cfg=koopman_cfg,
        G=G,
        task=self.task,
        dt=dt,
        device=self.device
        )

        # Set the DAE optimizer
        self.dae_optimizer = torch.optim.Adam(self.dae_model.parameters(), lr=koopman_cfg["robot"]["lr"])

    def act_koopman(self, koopman_obs, koopman_actions):
        # takes obs, not critic_obs
        if "stand-dance" in self.task.lower():
            raise NotImplementedError("DAE not implemented for Stand-Dance task yet.")
        else:
            state = koopman_obs[:, -self.state_dim:]
        self.koopman_transition.observations = state # take only the most recent obs
        self.koopman_transition.actions = koopman_actions

    def process_koopman_step(self, koopman_obs):
        device = self.replay_buffer.device
        if "stand-dance" in self.task.lower():
            raise NotImplementedError("DAE not implemented for Stand-Dance task yet.")
        else:
            next_state = koopman_obs[:, -self.state_dim:]
        self.replay_buffer.insert(self.koopman_transition.observations.to(device), self.koopman_transition.actions.to(device), next_state.to(device)) # take only the most recent obs
        self.koopman_transition.clear()

    def get_critic_input(self, critic_obs):
        """Processes critic_obs through DAE to get augmented input for the critic."""
        dae_input = critic_obs[:, 2*self.state_dim:3*self.state_dim] # most recent state, works with and without priv obs
        dae_input = dae_input.to(dtype=next(self.dae_model.parameters()).dtype)
        dae_input = dae_input.to(device=next(self.dae_model.parameters()).device)

        # dae_input_normed = safe_standardize(dae_input, self.state_mean, self.state_std)
        dae_input_normed = self.obs_action_normalizer.normalize_states(dae_input)

        # Wrap as GeometricTensor for E-DAE/EC-DAE
        if "edae" in self.task or "ecdae" in self.task:
            dae_input_normed = GeometricTensor(dae_input_normed, self.dae_model.obs_fn.in_type)
            latent = self.dae_model.obs_fn(dae_input_normed).tensor.detach()
        else:
            latent = self.dae_model.obs_fn(dae_input_normed).detach()
        return torch.cat([critic_obs, latent], dim=-1)

    def act(self, obs, critic_obs):
        # Override to use DAE-augmented critic input
        critic_input = self.get_critic_input(critic_obs)
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_input).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        if self.actor_critic.action_mean is not None:
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def compute_returns(self, last_critic_obs):
        critic_input = self.get_critic_input(last_critic_obs)
        last_values= self.actor_critic.evaluate(critic_input).detach() # This line changes
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        # Override to use DAE-augmented critic input during value loss calculation
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

                # Augment critic_obs_batch with DAE latent
                critic_input_batch = self.get_critic_input(critic_obs_batch) # Changed here
                value_batch = self.actor_critic.evaluate(critic_input_batch, masks=masks_batch, hidden_states=hid_states_batch[1]) # Changed here

                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL adaptation (same logic as PPO)
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss (same logic as PPO)
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss (same logic as PPO)
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step (same logic as PPO)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss