# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from basic_locomotion_dls_isaaclab.actuators import IdentifiedActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DCMotorCfg

from basic_locomotion_dls_isaaclab.assets import ISAAC_ASSET_DIR


# HYQREAL robot configuration from mujoco
stiffness_mujoco = 200.0
damping_mujoco = 20.0
friction_static_mujoco = 0.2
friction_dynamic_mujoco = 0.6
armature_mujoco = 0.01

HYQREAL_HIP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_hip_joint"],
    effort_limit=173.0,
    velocity_limit=21.0,
    saturation_effort=173.0,
    stiffness=stiffness_mujoco,
    damping=damping_mujoco,
    armature=armature_mujoco,
    friction_static=friction_static_mujoco,
    activation_vel=0.1,
    friction_dynamic=friction_dynamic_mujoco,
)

HYQREAL_THIGH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_thigh_joint"],
    effort_limit=208.0,
    velocity_limit=21.0,
    saturation_effort=208.0,
    stiffness=stiffness_mujoco,
    damping=damping_mujoco,
    armature=armature_mujoco,
    friction_static=friction_static_mujoco,
    activation_vel=0.1,
    friction_dynamic=friction_dynamic_mujoco,
)

HYQREAL_CALF_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".*_calf_joint"],
    effort_limit=249.0,
    velocity_limit=21.0,
    saturation_effort=249.0,
    stiffness=stiffness_mujoco,
    damping=damping_mujoco,
    armature=armature_mujoco,
    friction_static=friction_static_mujoco,
    activation_vel=0.1,
    friction_dynamic=friction_dynamic_mujoco,
)



HYQREAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/hyqreal_asset/from_xml/hyqreal2.usd",
        #usd_path= "/home/iit.local/gturrisi/isaaclab_ws_home/hyqreal.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            ".*_thigh_joint": 0.9,
            ".*_calf_joint": -1.7,
        },
        joint_vel={".*": 0.0},
    ),

    #actuators={
    #    "base_legs": DCMotorCfg(
    #        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    #        effort_limit=44.4,
    #        saturation_effort=44.4,
    #        velocity_limit=21.0,
    #        stiffness=25.0,
    #        damping=2,
    #        friction=0.0,
    #    ),
    #},
    
    actuators={"hip": HYQREAL_HIP_ACTUATOR_CFG, "thigh": HYQREAL_THIGH_ACTUATOR_CFG,
               "calf": HYQREAL_CALF_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
