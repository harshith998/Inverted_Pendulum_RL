"""
Variable inverted pendulum Gymnasium environment.

Each episode, PendulumConfig is sampled (n_links, lengths, masses, masses, cart_mass).

Observation space (Dict)
------------------------
All arrays are padded to max_links + 1 nodes (cart + max_links joints) and
 2 * max_links edges (bidirectional for each link)

NODES:
  [is_cart, is_joint, is_end, sin_theta, cos_theta, angular velocity, x, xvecloicty]
  Cart node  : [1, 0, 0, 0, 0, 0, x_cart, xdot_cart]
  Joint node : [0, 1, 0, sin(θ), cos(θ), θ̇, 0, 0]
  End node   : [0, 0, 1, sin(θ), cos(θ), θ̇, 0, 0]

EDGES:
[length_m, mass_kg]

Action space
------------
  Box([-max_force], [max_force], dtype=float32) — scalar force on the cart.
"""

from __future__ import annotations

import numpy as np
import mujoco
import gymnasium
from gymnasium import spaces

from env.mujoco_builder import PendulumConfig, build_mjcf
from env.rewards import compute_reward
from graph.graph_builder import build_graph, NODE_FEAT_DIM, EDGE_FEAT_DIM


class VariablePendulumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        n_links_range: tuple[int, int] = (2, 4),
        cart_mass_range: tuple[float, float] = (0.5, 3.0),
        link_length_range: tuple[float, float] = (0.3, 1.2),
        link_mass_range: tuple[float, float] = (0.1, 2.0),
        rail_limit: float = 2.5,
        max_force: float = 20.0,
        timestep: float = 0.001,
        frame_skip: int = 4,
        max_episode_steps: int = 1000,
        termination_angle: float = np.pi / 4,
        angle_noise: float = 0.05,
        vel_noise: float = 0.01,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.n_links_range = n_links_range
        self.cart_mass_range = cart_mass_range
        self.link_length_range = link_length_range
        self.link_mass_range = link_mass_range
        self.rail_limit = rail_limit
        self.max_force = max_force
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.termination_angle = termination_angle
        self.angle_noise = angle_noise
        self.vel_noise = vel_noise
        self.render_mode = render_mode

        self.max_links = n_links_range[1]
        self._max_nodes = self.max_links + 1   # cart + max_links joints
        self._max_edges = 2 * self.max_links   # bidirectional

        # MuJoCo model/data — built fresh at each reset when config changes.
        self._mj_model: mujoco.MjModel | None = None
        self._mj_data: mujoco.MjData | None = None
        self._config: PendulumConfig | None = None
        self._step_count: int = 0

        self.action_space = spaces.Box(
            low=np.array([-max_force], dtype=np.float32),
            high=np.array([max_force], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._max_nodes, NODE_FEAT_DIM), dtype=np.float32,
            ),
            "edge_index": spaces.Box(
                low=0, high=self._max_nodes - 1,
                shape=(2, self._max_edges), dtype=np.int64,
            ),
            "edge_features": spaces.Box(
                low=0.0, high=np.inf,
                shape=(self._max_edges, EDGE_FEAT_DIM), dtype=np.float32,
            ),
            "n_nodes": spaces.Box(low=2, high=self._max_nodes, shape=(1,), dtype=np.int64),
            "n_edges": spaces.Box(low=2, high=self._max_edges, shape=(1,), dtype=np.int64),
        })

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0

        config = self._sample_config()
        self._load_model(config)
        self._set_initial_state()

        obs = self._get_obs()
        info = {"config": self._config}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_force, self.max_force)
        self._mj_data.ctrl[0] = float(action[0])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self._mj_model, self._mj_data)

        self._step_count += 1
        obs = self._get_obs()

        joint_angles = self._get_joint_angles()
        cart_pos = float(self._mj_data.qpos[0])
        reward, reward_info = compute_reward(joint_angles, cart_pos, float(action[0]))

        terminated = self._is_terminated(joint_angles, cart_pos)
        truncated = self._step_count >= self.max_episode_steps

        info = {"reward_components": reward_info}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Rendering support can be added here using mujoco.Renderer.
        raise NotImplementedError("Rendering not yet implemented.")

    def close(self):
        self._mj_model = None
        self._mj_data = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_config(self) -> PendulumConfig:
        rng = self.np_random
        n_links = int(rng.integers(self.n_links_range[0], self.n_links_range[1] + 1))
        lengths = rng.uniform(*self.link_length_range, size=n_links).tolist()
        masses = rng.uniform(*self.link_mass_range, size=n_links).tolist()
        cart_mass = float(rng.uniform(*self.cart_mass_range))
        return PendulumConfig(n_links=n_links, lengths=lengths, masses=masses, cart_mass=cart_mass)

    def _load_model(self, config: PendulumConfig):
        xml = build_mjcf(config, rail_limit=self.rail_limit, max_force=self.max_force, timestep=self.timestep)
        self._mj_model = mujoco.MjModel.from_xml_string(xml)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._config = config

    def _set_initial_state(self):
        """Start near upright with small random perturbations."""
        n = self._config.n_links
        # qpos layout: [x_cart, theta_0, theta_1, ..., theta_{n-1}]
        # qvel layout: [xdot_cart, thetadot_0, ..., thetadot_{n-1}]
        self._mj_data.qpos[0] = 0.0  # cart at centre
        self._mj_data.qpos[1:n + 1] = self.np_random.uniform(
            -self.angle_noise, self.angle_noise, size=n
        )
        self._mj_data.qvel[:] = self.np_random.uniform(
            -self.vel_noise, self.vel_noise, size=n + 1
        )
        mujoco.mj_forward(self._mj_model, self._mj_data)

    def _get_joint_angles(self) -> np.ndarray:
        """Return the n_links joint angles as a numpy array."""
        return np.array(self._mj_data.qpos[1:self._config.n_links + 1], dtype=np.float32)

    def _get_obs(self) -> dict:
        """
        Build the padded graph observation from current MuJoCo state.
        Real entries fill [0 : n_nodes] / [0 : n_edges]; rest are zero.
        """
        config = self._config
        data = self._mj_data

        cart_pos = float(data.qpos[0])
        cart_vel = float(data.qvel[0])
        joint_angles = np.array(data.qpos[1:config.n_links + 1], dtype=float)
        joint_vels = np.array(data.qvel[1:config.n_links + 1], dtype=float)

        graph = build_graph(config, cart_pos, cart_vel, joint_angles, joint_vels)

        n_nodes = graph.node_features.shape[0]
        n_edges = graph.edge_features.shape[0]

        # Pad to max size so the observation shape is fixed.
        node_features_pad = np.zeros((self._max_nodes, NODE_FEAT_DIM), dtype=np.float32)
        edge_index_pad = np.zeros((2, self._max_edges), dtype=np.int64)
        edge_features_pad = np.zeros((self._max_edges, EDGE_FEAT_DIM), dtype=np.float32)

        node_features_pad[:n_nodes] = graph.node_features
        edge_index_pad[:, :n_edges] = graph.edge_index
        edge_features_pad[:n_edges] = graph.edge_features

        return {
            "node_features": node_features_pad,
            "edge_index": edge_index_pad,
            "edge_features": edge_features_pad,
            "n_nodes": np.array([n_nodes], dtype=np.int64),
            "n_edges": np.array([n_edges], dtype=np.int64),
        }

    def _is_terminated(self, joint_angles: np.ndarray, cart_pos: float) -> bool:
        angle_fail = np.any(np.abs(joint_angles) > self.termination_angle)
        rail_fail = np.abs(cart_pos) >= self.rail_limit
        return bool(angle_fail or rail_fail)
