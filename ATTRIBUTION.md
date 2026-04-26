# Attribution

## Overview

This project implements a graph neural network-based reinforcement learning system for controlling a variable inverted pendulum. The codebase is organized into environment simulation, graph encoding, model architectures, training loops, and evaluation infrastructure.

---

## Contributions by Component

### Written by Harshith

- **Environment (`env/pendulum_env.py`, `env/mujoco_builder.py`, `env/rewards.py`)**
  Designed and implemented the full custom Gymnasium environment from scratch. This includes the programmatic MuJoCo XML builder that dynamically generates valid MJCF models for arbitrary link counts, the domain randomization logic (per-episode sampling of link lengths, masses, cart mass), termination conditions, and the composite reward function with upright bonus, alive bonus, force penalty, and rail penalty.

- **Graph encoding (`graph/graph_builder.py`, `graph/graph_utils.py`)**
  Designed the full graph observation representation: node feature encoding (8D: type flags, sin/cos angles, angular velocity, cart state), bidirectional edge construction with normalized physical parameters, padding and masking scheme for variable-topology graphs, and feature normalization to [-1, 1].

- **Training infrastructure (`training/train_dqn.py`, `training/train_ppo.py`, `training/ablation_ppo.py`)**
  Wrote the full DQN training loop (replay buffer, double-DQN target computation, epsilon-greedy schedule, auxiliary loss, checkpointing) and the PPO training loop (parallel environment collection, GAE advantage estimation, clipped surrogate objective, LR annealing, best-model saving). Also built the hyperparameter ablation framework.

- **Evaluation infrastructure (`eval/eval_dqn.py`, `eval/eval_ppo.py`)**
  Designed and implemented the OOD evaluation suite: 1D parameter sweeps (100 points × 200 episodes), 2D heatmap grid, result caching system, and all plotting code.

- **GNN-Transformer model (`models/gnn_transformer_ppo.py`)**
  Designed and implemented the graph attention transformer architecture: multi-head scaled dot-product attention restricted to graph edges, edge-feature attention bias, numerically stable scatter softmax (log-sum-exp trick via `scatter_reduce_`), residual connections, LayerNorm, and position-wise feed-forward sublayers.

- **Configuration and project structure (`configs/default.yaml`, `requirements.txt`)**
  Wrote the centralized YAML configuration and project layout.

---

### Written Rohan (notebookes, codebase integrated by Harshith)

- **MLP baselines (`models/mlp_dqn.py`, `models/mlp_ppo.py`)**
  Rohan wrote MLP policy notebooks for an earlier flat-observation version of this project. I adapted these into the modular class-based architecture, integrated them with the graph observation format (flattening padded node and edge features), added dropout regularization, and wired them into the shared base classes.

- **Base GNN structure (informed `models/gnn_dqn.py` encoder)**
  Rohan wrote initial GNN notebook experiments with basic message passing. I ported this into the production encoder, added LayerNorm, residual connections, masked mean pooling, the two-layer Q-head, and the auxiliary rod-parameter prediction head.



### Written with Claude (Claude Sonnet, Anthropic)

- **GNN-MPNN model (`models/gnn_mpnn_ppo.py`, `models/gnn_dqn.py`)**
  I designed the MPNN architecture and wrote the initial message-passing structure. Claude assisted with debugging numerically stable aggregation, the residual connection fix (identifying embedding drift as the cause of Q-value divergence), and refining the masked scatter operations. The final architecture decisions (LayerNorm placement, disabling dropout inside message-passing layers, residual formulation) were made collaboratively through iterative training runs.

- **Iterative debugging and hyperparameter decisions**
  Claude assisted with diagnosing training instability (Q-value divergence, PPO collapse, NaN in attention softmax) and suggesting fixes. I evaluated each fix against training curves and accepted or rejected them based on observed results.

---

### Written by Claude

- **Physics validation tests (`tests/test_physics.py`)**
  Claude wrote the MuJoCo-vs-scipy trajectory validation test, energy conservation check, and graph structure integrity tests. I specified what to test (trajectory accuracy tolerances, energy drift bounds) and Claude implemented the numerical integration comparison.

---

## AI Tool Usage Summary

Claude Code (claude-sonnet-4-6) was used throughout this project as a coding assistant. Specifically:

- **What Claude generated**: physics test code, portions of the MPNN message-passing internals, the numerically stable scatter softmax implementation in the transformer, the parallel environment training loop structure, and the OOD evaluation caching system.
- **What we modified**: every generated component was reviewed, often substantially reworked (e.g., the transformer attention was rewritten after Claude's first version produced NaN gradients), and integrated into the broader system I designed.
- **What we debugged**: training instability across all three PPO architectures required iterative diagnosis using training curves — identifying residual connections as missing, dropout as harmful inside message-passing, and LR as too high. These conclusions came from reading training curves and reasoning about gradient flow, not from Claude's suggestions alone.
- **What we designed end-to-end**: the graph observation format, the custom MuJoCo environment with domain randomization, the reward function, the OOD evaluation methodology, and the overall research framing (GNN generalization across variable pendulum configurations).
