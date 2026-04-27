import os
import argparse
import numpy as np
import torch
import sys

if '--headless' in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import imageio
from collections import deque

from matplotlib import pyplot as plt

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def rotate_vec_by_inverse_quat(vec, q):
    R = quaternion_to_rotation_matrix(q)
    return R.T @ vec

def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

def get_heading(q):
    forward_vec = np.array([1., 0., 0.])
    xyzw_q = np.array([q[1], q[2], q[3], q[0]])
    xyzw_q[:2] = 0.0
    xyzw_q = xyzw_q / np.linalg.norm(xyzw_q)
    xyz = xyzw_q[:3]
    t = np.cross(xyz, forward_vec) * 2.0
    heading_vec = forward_vec + xyzw_q[3] * t + np.cross(xyz, t)
    return np.arctan2(heading_vec[1], heading_vec[0])

# --- Configuration for IsaacLab ---
class Go2SimConfig:
    dt = 0.005
    decimation = 4
    env_dt = dt * decimation
    num_dofs = 12
    num_history = 3

    init_pos = [0.0, 0.0, 0.221]

    init_quat = [1.0, 0.0, 0.0, 0.0]

    init_joints = {
        'FL_hip_joint': 0.0,   'FR_hip_joint': 0.0,   'RL_hip_joint': 0.0,   'RR_hip_joint': 0.0,
        'FL_thigh_joint': 1.2, 'FR_thigh_joint': 1.2, 'RL_thigh_joint': 1.2, 'RR_thigh_joint': 1.2,
        'FL_calf_joint': -2.15,'FR_calf_joint': -2.15,'RL_calf_joint': -2.15,'RR_calf_joint': -2.15
    }

    default_joints = {
        'FL_hip_joint': 0.0,   'FR_hip_joint': 0.0,   'RL_hip_joint': 0.0,   'RR_hip_joint': 0.0,
        'FL_thigh_joint': 0.9, 'FR_thigh_joint': 0.9, 'RL_thigh_joint': 0.9, 'RR_thigh_joint': 0.9,
        'FL_calf_joint': -1.6, 'FR_calf_joint': -1.6, 'RL_calf_joint': -1.6, 'RR_calf_joint': -1.6
    }

    # FIX 2: This order MUST exactly match the MorphologycalSymmetriesCfg in rsl_rl_ppo_cfg.py
    network_joint_order = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
    ]

    # FIX 3: Torque Limits mapped to the network_joint_order (4 Hips, 4 Thighs, 4 Calves)
    torque_limits = np.array([23.7]*4 + [23.7]*4 + [45.43]*4)

    stiffness = 50.0
    damping = 1.5
    action_scale = 0.5
    use_filter_actions = True
    action_filter_alpha = 0.8

    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.0
    gait_freq = 2.5

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to policy.pt')
    parser.add_argument('--xml_path', type=str, required=True, help='Path to converted MuJoCo XML')
    parser.add_argument('--video_name', type=str, default='go2_dance_isaaclab.mp4')
    parser.add_argument('--num_steps', type=int, default=2000)
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()

class GaitGenerator:
    def __init__(self, freq=2.5, dt=0.02):
        self.dt = dt
        self.freq = freq
        self.episode_length_buf = 0

    def step(self):
        gait_indices = (self.episode_length_buf * self.dt * self.freq) % 1.0
        foot_indices_RL = (gait_indices + 0.0) % 1.0
        foot_indices_RR = (gait_indices + 0.5) % 1.0
        self.episode_length_buf += 1
        return np.sin(2 * np.pi * np.array([foot_indices_RL, foot_indices_RR]))

class Go2EnvShim:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.history = deque(maxlen=cfg.num_history)
        self.previous_actions = np.zeros(cfg.num_dofs)

        self.qpos_indices = []
        self.dof_indices = []
        self.actuator_indices = [] # New: Explicitly track the actuator mapping

        for name in cfg.network_joint_order:
            try:
                # 1. Map qpos and dof indices
                j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if j_id == -1:
                    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.replace("_joint", ""))
                    if j_id == -1: raise ValueError(f"Joint {name} not found")

                self.qpos_indices.append(model.jnt_qposadr[j_id])
                self.dof_indices.append(model.jnt_dofadr[j_id])

                # 2. Map Actuator indices explicitly
                a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if a_id == -1:
                    a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor")
                if a_id == -1:
                    raise ValueError(f"Actuator for {name} not found")

                self.actuator_indices.append(a_id)

            except Exception as e:
                print(f"Error mapping {name}: {e}")
                sys.exit(1)

        self.qpos_indices = np.array(self.qpos_indices, dtype=int)
        self.dof_indices = np.array(self.dof_indices, dtype=int)
        self.actuator_indices = np.array(self.actuator_indices, dtype=int)

        self.default_dof_pos = np.array([cfg.default_joints[name] for name in cfg.network_joint_order])
        self.gait_gen = GaitGenerator(freq=cfg.gait_freq, dt=cfg.env_dt)

    def update_command_heading(self, data, command_base, target_heading):
        current_quat = data.qpos[3:7]
        current_heading = get_heading(current_quat)
        heading_error = wrap_to_pi(target_heading - current_heading)

        clip_ang_vel = 0.25 * np.pi
        scale = 0.5 * np.pi / clip_ang_vel
        yaw_vel = np.clip(0.5 * heading_error, -clip_ang_vel, clip_ang_vel) * scale

        new_cmd = command_base.copy()
        new_cmd[2] = yaw_vel
        return new_cmd

    def get_obs(self, data, command, last_actions):
        base_quat = data.qpos[3:7]
        proj_gravity = rotate_vec_by_inverse_quat(np.array([0., 0., -1.]), base_quat)
        proj_forward = rotate_vec_by_inverse_quat(np.array([1., 0., 0.]), base_quat)

        obs_cmd = command.copy()[:3]
        obs_cmd[0] *= self.cfg.lin_vel_scale
        obs_cmd[1] *= self.cfg.lin_vel_scale
        obs_cmd[2] *= self.cfg.ang_vel_scale

        dof_pos = data.qpos[self.qpos_indices]
        dof_vel = data.qvel[self.dof_indices]

        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.cfg.dof_pos_scale
        dof_vel_scaled = dof_vel * self.cfg.dof_vel_scale

        obs_clock = self.gait_gen.step()

        current_obs = np.concatenate([
            proj_gravity, proj_forward, obs_cmd, dof_pos_scaled, dof_vel_scaled, last_actions, obs_clock
        ])

        if len(self.history) == 0:
            for _ in range(self.cfg.num_history):
                self.history.append(current_obs.copy())
        self.history.append(current_obs)

        return torch.tensor(np.concatenate(list(self.history)), dtype=torch.float32)

def play_mujoco():
    args = get_args()
    cfg = Go2SimConfig()

    print(f"Loading model: {args.xml_path}")
    if not os.path.exists(args.xml_path):
        print(f"Error: XML file not found at {args.xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Set the armature
    for name in cfg.network_joint_order:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1:
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.replace("_joint", ""))

        dof_id = model.jnt_dofadr[j_id]

        # Values pulled exactly from your IsaacLab go2_asset.py
        if "hip" in name:
            model.dof_armature[dof_id] = 0.0152
        elif "thigh" in name:
            model.dof_armature[dof_id] = 0.0228
        elif "calf" in name:
            model.dof_armature[dof_id] = 0.0434

    # Set the friction
    for i in range(model.ngeom):
        # Set the friction values
        model.geom_friction[i] = [1.5, 0.05, 0.005]

        # UPGRADE THE SOLVER: Force MuJoCo to actually calculate twist/roll friction
        # 3 = sliding only (default)
        # 4 = sliding + torsional
        # 6 = sliding + torsional + rolling
        model.geom_condim[i] = 6

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = cfg.init_pos
    data.qpos[3:7] = cfg.init_quat

    sim_env = Go2EnvShim(cfg, model)
    init_vec = np.array([cfg.init_joints[name] for name in cfg.network_joint_order])
    data.qpos[sim_env.qpos_indices] = init_vec

    print(f"Loading policy from {args.checkpoint}...")
    try:
        loaded_dict = torch.load(args.checkpoint, map_location='cpu')
        weights = loaded_dict['model_state_dict'] if 'model_state_dict' in loaded_dict else loaded_dict

        class SimpleActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = torch.nn.Sequential(
                    torch.nn.Linear(141, 512), torch.nn.ELU(),
                    torch.nn.Linear(512, 256), torch.nn.ELU(),
                    torch.nn.Linear(256, 128), torch.nn.ELU(),
                    torch.nn.Linear(128, 12)
                )
            def forward(self, x): return self.actor(x)

        policy = SimpleActor()
        with torch.no_grad():
            policy.actor[0].weight.copy_(weights['actor.0.weight'])
            policy.actor[0].bias.copy_(weights['actor.0.bias'])
            policy.actor[2].weight.copy_(weights['actor.2.weight'])
            policy.actor[2].bias.copy_(weights['actor.2.bias'])
            policy.actor[4].weight.copy_(weights['actor.4.weight'])
            policy.actor[4].bias.copy_(weights['actor.4.bias'])
            policy.actor[6].weight.copy_(weights['actor.6.weight'])
            policy.actor[6].bias.copy_(weights['actor.6.bias'])
    except Exception as e:
        print(f"Policy Load Error: {e}")
        return

    frames = []
    target_heading = 0.0 #0.5 * np.pi
    command = np.array([0.0, 0.0, 0.0])
    last_actions = np.zeros(12)

    print("Starting simulation...")

    id_isaac = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "isaac_view")
    id_track = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    cam_name = "isaac_view" if id_isaac != -1 else ("track" if id_track != -1 else None)

    mujoco.mj_forward(model, data) # Make sure kinematics are updated before render
    renderer.update_scene(data, camera=cam_name)
    frames.append(renderer.render())

    collected_obs = []

    for i in range(args.num_steps):

        if i % 500 == 0 and i > 0:
            target_heading += 0.5 * np.pi
            print(f"--- New Target Heading: {target_heading:.2f} radians ---")

        dynamic_command = sim_env.update_command_heading(data, command, target_heading)
        obs = sim_env.get_obs(data, dynamic_command, last_actions)
        collected_obs.append(obs)

        with torch.no_grad():
            actions = policy(obs).numpy()
            # actions = np.clip(actions, -100.0, 100.0)
            actions = np.clip(actions, -3.0, 3.0)

        if cfg.use_filter_actions:
            temp = cfg.action_filter_alpha * actions + (1 - cfg.action_filter_alpha) * sim_env.previous_actions
            processed_actions = cfg.action_scale * temp + sim_env.default_dof_pos
        else:
            processed_actions = cfg.action_scale * actions + sim_env.default_dof_pos

        sim_env.previous_actions = actions.copy()
        last_actions = actions.copy()

        for _ in range(cfg.decimation):
            q_pos = data.qpos[sim_env.qpos_indices]
            q_vel = data.qvel[sim_env.dof_indices]

            torques = cfg.stiffness * (processed_actions - q_pos) - cfg.damping * q_vel
            torques = np.clip(torques, -cfg.torque_limits, cfg.torque_limits)

            # FIX 4: Apply torques to explicit actuator indices instead of blindly assigning to [:]
            data.ctrl[sim_env.actuator_indices] = torques

            mujoco.mj_step(model, data)

        if i % 2 == 0:
            id_isaac = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "isaac_view")
            id_track = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            cam_name = "isaac_view" if id_isaac != -1 else ("track" if id_track != -1 else None)

            renderer.update_scene(data, camera=cam_name)
            frames.append(renderer.render())

        if i % 50 == 0:
            cur_heading = get_heading(data.qpos[3:7])
            print(f"Step {i} | Base Height: {data.qpos[2]:.3f} | Heading: {cur_heading:.2f}")

    print(f"Saving video to {args.video_name}...")
    vid_path = f"{args.video_name}"
    imageio.mimsave(vid_path, frames, fps=30)

    # Save collected observations
    collected_obs = np.stack(collected_obs)
    npy_path = f"collected_obs.npy"
    print(f"[INFO] Saving collected observations to: {npy_path}")
    np.save(npy_path, collected_obs)

if __name__ == "__main__":
    play_mujoco()