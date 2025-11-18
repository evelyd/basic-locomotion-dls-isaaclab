# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .locomotion_env import LocomotionEnv
# from .stand_dance_direct_env import CyberStandDanceEnv

# Aliengo environments
from .locomotion_env import AliengoFlatEnvCfg, AliengoRoughVisionEnvCfg, AliengoRoughBlindEnvCfg
from .stand_dance_env import StandDanceEnv
from .stand_dance_direct_env import AliengoStandDanceEnv
from .aliengo_symmloco_env_cfg import AliengoStandDanceEnvCfg, AliengoStandDanceDirectEnvCfg
from .go2_symmloco_env_cfg import Go2StandDanceDirectEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

gym.register(
    id="Stand-Dance-Aliengo-Flat",
    entry_point=StandDanceEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoStandDanceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandDanceCfgPPO",
    },
)

gym.register(
    id="Stand-Dance-Aliengo-Flat-Direct",
    entry_point=AliengoStandDanceEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoStandDanceDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandDanceCfgPPO",
    },
)

gym.register(
    id="Stand-Dance-Go2-Flat-Direct",
    entry_point=AliengoStandDanceEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2StandDanceDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandDanceCfgPPO",
    },
)

gym.register(
    id="Stand-Dance-Go2-EMLP-Flat-Direct",
    entry_point=AliengoStandDanceEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2StandDanceDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandDanceEMLPCfgPPO",
    },
)

gym.register(
    id="Stand-Dance-Go2-CDAE-Online-Flat-Direct",
    entry_point=AliengoStandDanceEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2StandDanceDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandDanceCDAEOnlineCfgPPO",
    },
)

gym.register(
    id="Stand-Dance-Go2-EMLP-ECDAE-Online-Flat-Direct",
    entry_point=AliengoStandDanceEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2StandDanceDirectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:StandDanceEMLPECDAEOnlineCfgPPO",
    },
)

gym.register(
    id="Locomotion-Aliengo-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Flat-EMLP",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPOEMLPRunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Flat-CDAE-Online",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPOCDAEOnlineRunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Flat-EMLP-ECDAE-Online",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPOEMLPECDAEOnlineRunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Aliengo-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AliengoRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

# Go2 environments
from .locomotion_env import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

gym.register(
    id="Locomotion-Go2-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Go2-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-Go2-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)


# B2 environments
from .locomotion_env import B2FlatEnvCfg, B2RoughVisionEnvCfg, B2RoughBlindEnvCfg

gym.register(
    id="Locomotion-B2-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": B2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-B2-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": B2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-B2-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": B2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

# HyQReal environments
from .locomotion_env import HyQRealFlatEnvCfg, HyQRealRoughVisionEnvCfg, HyQRealRoughBlindEnvCfg

gym.register(
    id="Locomotion-HyQReal-Flat",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HyQRealFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-HyQReal-Rough-Blind",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HyQRealRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="Locomotion-HyQReal-Rough-Vision",
    entry_point=LocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HyQRealRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)