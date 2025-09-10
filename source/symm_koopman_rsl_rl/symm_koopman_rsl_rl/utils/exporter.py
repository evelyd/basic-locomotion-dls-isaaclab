# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Code taken from https://github.com/isaac-sim/IsaacLab/blob/5716d5600a1a0e45345bc01342a70bd81fac7889/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

import copy
import os
import torch
from amp_rsl_rl.networks import ActorMoE, ActorMoESymm, ExportedActorMoESymm, SimpleEMLP, ActorEMLP, ExportedActorEMLP


def export_policy_as_onnx(
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, normalizer, verbose)
    policy_exporter.to("cpu")
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        temp_actor_copy = copy.deepcopy(actor_critic.actor)
        temp_actor_copy.cpu()
        temp_actor_copy.eval()

        # CORRECTED: Use the actor's own export() function if it exists
        if hasattr(temp_actor_copy, 'export'):
            self.actor = temp_actor_copy.export()
        else:
            self.actor = temp_actor_copy

        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            # ... (recurrent code) ...
            pass
        # copy normalizer
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x):
        # Now, the self.actor is a standard torch.nn.Sequential module
        # No need for GeometricTensor handling here
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            # ... (recurrent case code) ...
            pass
        else:
            # The input for ONNX is now a standard torch.Tensor
            # Get the input size from the exported sequential model's first layer
            # This is more robust as it works for all Sequential models
            obs_size = self.actor[0].in_features
            obs = torch.zeros(1, obs_size)

            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )