# Description: This script is used to simulate the full model of the robot in mujoco

# Authors:
# Giulio Turrisi

import time
import numpy as np
from tqdm import tqdm
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+"/../")
sys.path.append(dir_path+"/../scripts/rsl_rl")

import matplotlib.pyplot as plt

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

from gym_quadruped.sensors.heightmap import HeightMap
from gym_quadruped.utils.mujoco.visual import render_sphere

# Locomotion Policy imports
from locomotion_policy_wrapper import LocomotionPolicyWrapper

import config

def update_plots(axs, history):
    # Convert lists of observations into a single numpy array
    times = np.array(history['time'])
    base_lin_vel = np.vstack(history['base_lin_vel'])
    base_ang_vel = np.vstack(history['base_ang_vel'])
    base_proj_gravity = np.vstack(history['base_proj_gravity'])
    forward_vec = np.vstack(history['forward_vec'])
    commands = np.vstack(history['commands'])
    joint_pos = np.vstack(history['joint_pos'])
    desired_joint_pos = np.vstack(history['desired_joint_pos']) # not squeezed
    joint_vel = np.vstack(history['joint_vel'])
    previous_action = np.vstack(history['previous_action'])
    clock_inputs = np.vstack(history['clock_inputs'])

    # Re-plot each subplot with the flattened data
    if times.size > 0:
        # Base Linear Velocity
        axs[0].cla()
        axs[0].set_title("Base Linear Velocity (3)")
        axs[0].plot(times, base_lin_vel)
        axs[0].legend(['x', 'y', 'z'])

        # Base Angular Velocity
        axs[1].cla()
        axs[1].set_title("Base Angular Velocity (3)")
        axs[1].plot(times, base_ang_vel)
        axs[1].legend(['x', 'y', 'z'])

        # Base Projected Gravity
        axs[2].cla()
        axs[2].set_title("Base Projected Gravity (3)")
        axs[2].plot(times, base_proj_gravity)
        axs[2].legend(['x', 'y', 'z'])

        # Forward Vector
        axs[3].cla()
        axs[3].set_title("Forward Vector (3)")
        axs[3].plot(times, forward_vec)
        axs[3].legend(['x', 'y', 'z'])

        # Commands
        axs[4].cla()
        axs[4].set_title("Commands (3)")
        axs[4].plot(times, commands)
        axs[4].legend(['x', 'y', 'z'])

        # Joint Positions (12)
        axs[5].cla()
        axs[5].set_title("Joint Positions (12)")
        axs[5].plot(times, joint_pos, linestyle='-')
        axs[5].plot(times, desired_joint_pos, linestyle='--', alpha=0.6)

        # Joint Velocities (12)
        axs[6].cla()
        axs[6].set_title("Joint Velocities (12)")
        axs[6].plot(times, joint_vel)

        # Previous Action (12)
        axs[7].cla()
        axs[7].set_title("Previous Action (12)")
        axs[7].plot(times, previous_action)

        # Clock Inputs
        axs[8].cla()
        axs[8].set_title("Clock Inputs (2)")
        axs[8].plot(times, clock_inputs)

    # Set x-axis label on the bottom plot only
    axs[8].set_xlabel("Time (s)")

    plt.draw()
    plt.pause(1e-6)

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    robot_name = config.robot
    scene_name = config.scene
    simulation_dt = 0.002


    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
    )

    # Set sit pose initially if specified
    if hasattr(config, "init_qpos") and hasattr(config, "init_base_height"):
        init_qpos = np.zeros(env.mjData.qpos.shape)
        init_qpos[0:7] = env.mjData.qpos[0:7]
        init_qpos[7:] = config.init_qpos
        init_qpos[2] = config.init_base_height
        init_qvel = np.zeros(env.mjData.qvel.shape)
        env.reset(random=False, qpos=init_qpos, qvel=init_qvel)
    else:
        env.reset(random=False)
    env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType

    # Initialization of variables used in the main control loop --------------------------------
    locomotion_policy = LocomotionPolicyWrapper(env=env)

    if(locomotion_policy.use_vision):
        resolution_heightmap = config.resolution_heightmap
        num_rows_heightmap = round(config.size_x_heightmap/resolution_heightmap) + 1
        num_cols_heightmap = round(config.size_y_heightmap/resolution_heightmap) + 1
        heightmap = HeightMap(num_rows=num_rows_heightmap, num_cols=num_cols_heightmap, dist_x=resolution_heightmap, dist_y=resolution_heightmap, mj_model=env.mjModel, mj_data=env.mjData)


    # --------------------------------------------------------------
    RENDER_FREQ = 30  # Hz
    last_render_time = time.time()

    input(f"starting new episode, press enter to continue...")

    # Create lists to store observation data over time
    history = {
        'time': [],
        'base_lin_vel': [],
        'base_ang_vel': [],
        'base_proj_gravity': [],
        'forward_vec': [],
        'commands': [],
        'joint_pos': [],
        'joint_vel': [],
        'previous_action': [],
        'clock_inputs': [],
    }
    # Desired joint positions will be stored separately for plotting
    history['desired_joint_pos'] = []

    # # Set up subplots for real-time plotting
    # plt.style.use('ggplot')
    # fig, axs = plt.subplots(10, 1, figsize=(12, 18), sharex=True)
    # fig.tight_layout(pad=3.0)
    # plt.ion() # Turn on interactive mode for live plotting

    while True:
        step_start = time.time()

        # Get the current state of the robot -----------------------------------------------------
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        base_lin_vel = env.base_lin_vel(frame='base')
        base_ang_vel = env.base_ang_vel(frame='base')
        base_ori_euler_xyz = env.base_ori_euler_xyz
        heading_orientation_SO3 = env.heading_orientation_SO3
        base_quat_wxyz = qpos[3:7]
        base_pos = env.base_pos

        if(config.use_imu or config.use_cuncurrent_state_est):
            imu_linear_acceleration = env.mjData.sensordata[0:3]
            imu_angular_velocity = env.mjData.sensordata[3:6]
            imu_orientation = env.mjData.sensordata[9:13]
            imu_orientation = base_quat_wxyz
        else:
            imu_linear_acceleration = np.zeros(3)
            imu_angular_velocity = np.zeros(3)
            imu_orientation = np.zeros(4)

        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]

        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()

        if(locomotion_policy.use_vision):
            heightmap.update_height_map(env.mjData.qpos[0:3], yaw=env.base_ori_euler_xyz[2])

        # RL controller --------------------------------------------------------------
        if env.step_num % round(1 / (locomotion_policy.RL_FREQ * simulation_dt)) == 0:

            desired_joint_pos, obs = locomotion_policy.compute_control(
                        base_pos=base_pos,
                        base_ori_euler_xyz=base_ori_euler_xyz,
                        base_quat_wxyz=base_quat_wxyz,
                        base_lin_vel=base_lin_vel,
                        base_ang_vel=base_ang_vel,
                        heading_orientation_SO3=heading_orientation_SO3,
                        joints_pos=joints_pos,
                        joints_vel=joints_vel,
                        ref_base_lin_vel=ref_base_lin_vel,
                        ref_base_ang_vel=ref_base_ang_vel,
                        imu_linear_acceleration=imu_linear_acceleration,
                        imu_angular_velocity=imu_angular_velocity,
                        imu_orientation=imu_orientation,
                        heightmap_data=heightmap.data if locomotion_policy.use_vision else None)

            # Store the current time
            history['time'].append(env.step_num * simulation_dt)

            # Extract and store each component with the correct dimensions
            start_idx = 0
            history['base_lin_vel'].append(obs[:, start_idx : start_idx+3])
            start_idx += 3
            history['base_ang_vel'].append(obs[:, start_idx : start_idx+3])
            start_idx += 3
            history['base_proj_gravity'].append(obs[:, start_idx : start_idx+3])
            start_idx += 3
            history['forward_vec'].append(obs[:, start_idx : start_idx+3])
            start_idx += 3
            history['commands'].append(obs[:, start_idx : start_idx+3])
            start_idx += 3
            history['joint_pos'].append(obs[:, start_idx : start_idx+12])
            start_idx += 12
            history['joint_vel'].append(obs[:, start_idx : start_idx+12])
            start_idx += 12
            history['previous_action'].append(obs[:, start_idx : start_idx+12])
            start_idx += 12
            history['clock_inputs'].append(obs[:, start_idx : start_idx+2])
            # Note: The remaining part of the obs array is for vision, which can be skipped if not needed for plotting.

            # Store desired joint positions
            desired_pos_flat = np.concatenate([
                desired_joint_pos.FL,
                desired_joint_pos.FR,
                desired_joint_pos.RL,
                desired_joint_pos.RR
            ])
            history['desired_joint_pos'].append(desired_pos_flat)

            # Update the plots
            # update_plots(axs, history)

        # PD controller --------------------------------------------------------------
        else:
            desired_joint_pos = locomotion_policy.desired_joint_pos


        Kp = locomotion_policy.Kp_walking
        Kd = locomotion_policy.Kd_walking

        error_joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        error_joints_pos.FL = desired_joint_pos.FL - joints_pos.FL
        error_joints_pos.FR = desired_joint_pos.FR - joints_pos.FR
        error_joints_pos.RL = desired_joint_pos.RL - joints_pos.RL
        error_joints_pos.RR = desired_joint_pos.RR - joints_pos.RR

        tau = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        tau.FL = Kp * (error_joints_pos.FL) - Kd * joints_vel.FL
        tau.FR = Kp * (error_joints_pos.FR) - Kd * joints_vel.FR
        tau.RL = Kp * (error_joints_pos.RL) - Kd * joints_vel.RL
        tau.RR = Kp * (error_joints_pos.RR) - Kd * joints_vel.RR


        # Set control and mujoco step ----------------------------------------------------------------------
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau.FL.reshape((3,))
        action[env.legs_tau_idx.FR] = tau.FR.reshape((3,))
        action[env.legs_tau_idx.RL] = tau.RL.reshape((3,))
        action[env.legs_tau_idx.RR] = tau.RR.reshape((3,))
        state, reward, is_terminated, is_truncated, info = env.step(action=action)


        # Sleep to match real-time ---------------------------------------------------------
        loop_elapsed_time = time.time() - step_start

        if(loop_elapsed_time < simulation_dt):
            time.sleep(simulation_dt - (loop_elapsed_time))

        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - last_render_time > 1.0 / RENDER_FREQ or env.step_num == 1:
            env.render()
            last_render_time = time.time()

            if(locomotion_policy.use_vision):
                if heightmap.data is not None:
                    for i in range(heightmap.data.shape[0]):
                        for j in range(heightmap.data.data.shape[1]):
                            heightmap.geom_ids[i, j] = render_sphere(
                                viewer=env.viewer,
                                position=([heightmap.data[i][j][0][0], heightmap.data[i][j][0][1], heightmap.data[i][j][0][2]]),
                                diameter=0.02,
                                color=[0, 1, 0, 0.5],
                                geom_id=heightmap.geom_ids[i, j],
                            )




    env.close()

