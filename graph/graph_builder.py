"""
Nodes  : cart (index 0) + one node per joint (indices 1..n_links)
         Total n_nodes = n_links + 1

Edges  : one bidirectional pair per rod → 2 * n_links directed edges
         Rod i connects node i (parent) to node i+1 (child) and back.

Node feature vector:
  Index  Field         Cart node   Joint node  End node
  -----  -----------   ---------   ----------  --------
  0      is_cart       1           0           0
  1      is_joint      0           1           0
  2      is_end        0           0           1
  3      sin(theta)    0           sin(θᵢ)     sin(θₙ)
  4      cos(theta)    0*          cos(θᵢ)     cos(θₙ)
  5      theta vel.    0           θ̇ᵢ          θ̇ₙ
  6      x_cart        x           0           0
  7      xvel.         ẋ           0           0

  *cos(theta) is padded to 0 for the cart

  The angle is the angle of the lower edge/rod for a node, to the verticle.

Edge feature vector  (EDGE_FEAT_DIM = 2)
-----------------------------------------
  Index  Field
  0      length  (metres)
  1      mass    (kg)

Both the forward and backward edge for a given rod share the same features.
The GNN learns directionality from the edge_index structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.mujoco_builder import PendulumConfig

NODE_FEAT_DIM: int = 8
EDGE_FEAT_DIM: int = 2


@dataclass
class PendulumGraph:
    """Container for the graph representation of one timestep."""
    node_features: np.ndarray   # shape (n_nodes, NODE_FEAT_DIM),  float32
    edge_index: np.ndarray      # shape (2, n_edges),               int64
    edge_features: np.ndarray   # shape (n_edges, EDGE_FEAT_DIM),   float32

    @property
    def n_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_features.shape[0]


# Normalization constants — all features land in roughly [-1, 1]
_RAIL_LIMIT   = 2.5    # cart position range
_CART_VEL_MAX = 5.0    # m/s reasonable max
_ANG_VEL_MAX  = 10.0   # rad/s reasonable max
_LEN_MIN, _LEN_MAX = 0.3, 1.2
_MASS_MIN, _MASS_MAX = 0.1, 2.0


def build_graph(
    config: PendulumConfig,
    cart_pos: float,
    cart_vel: float,
    joint_angles: np.ndarray,
    joint_vels: np.ndarray,
) -> PendulumGraph:

    n = config.n_links
    n_nodes = n + 1
    n_edges = 2 * n

    node_features = np.zeros((n_nodes, NODE_FEAT_DIM), dtype=np.float32)
    edge_index    = np.zeros((2, n_edges), dtype=np.int64)
    edge_features = np.zeros((n_edges, EDGE_FEAT_DIM), dtype=np.float32)

    # cart node — normalize position and velocity to [-1, 1]
    node_features[0, 0] = 1.0
    node_features[0, 6] = cart_pos / _RAIL_LIMIT
    node_features[0, 7] = cart_vel / _CART_VEL_MAX

    # joint/endpoint nodes — sin/cos already [-1,1]; normalize angular velocity
    for i in range(n):
        node_idx = i + 1
        is_end = (i == n - 1)
        node_features[node_idx, 1] = float(not is_end)
        node_features[node_idx, 2] = float(is_end)
        node_features[node_idx, 3] = float(np.sin(joint_angles[i]))
        node_features[node_idx, 4] = float(np.cos(joint_angles[i]))
        node_features[node_idx, 5] = float(joint_vels[i]) / _ANG_VEL_MAX

    # edges — normalize length and mass to [0, 1]
    for i in range(n):
        fwd = 2 * i
        bwd = 2 * i + 1

        edge_index[0, fwd] = i;     edge_index[1, fwd] = i + 1
        edge_index[0, bwd] = i + 1; edge_index[1, bwd] = i

        norm_len  = (config.lengths[i] - _LEN_MIN)  / (_LEN_MAX  - _LEN_MIN)
        norm_mass = (config.masses[i]  - _MASS_MIN) / (_MASS_MAX - _MASS_MIN)

        edge_features[fwd] = [norm_len, norm_mass]
        edge_features[bwd] = [norm_len, norm_mass]

    return PendulumGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
    )