# GNN-RL Inverted Pendulum

## What it Does / Goal

This project trains reinforcement learning agents to balance a variable inverted pendulum — a cart-pole system where the number of links, their lengths, and their masses all change between episodes. Rather than encoding the physical state as a flat vector (which would break when the topology changes), the system represents the pendulum as a graph: cart and joints are nodes, rods are edges carrying physical parameters (length, mass). A Graph Neural Network (GNN) encoder reads this graph and produces a fixed-size embedding regardless of how many links are present, enabling a single trained model to generalize zero-shot to pendulum configurations it has never seen — including out-of-distribution lengths and masses. Two GNN architectures are compared (message-passing MPNN and graph-attention transformer) against flat MLP baselines, under both DQN (discrete, off-policy) and PPO (continuous, on-policy) training regimes.

---

## Research Context

Inverted pendulum control is one of the oldest benchmarks in reinforcement learning. Barto, Sutton & Anderson (1983) first demonstrated a neural network learning to balance a cart-pole from scratch. Since then, deep RL methods — DQN (Mnih et al., 2015) and PPO (Schulman et al., 2017) — have been widely applied to pendulum variants, consistently achieving near-perfect balance. However, **all prior pendulum RL work assumes fixed, known physical parameters**: a single link length and mass set at the start, unchanged across all episodes. A controller trained on one configuration cannot transfer to another without retraining.

Separately, Wang et al. (2018) introduced **NerveNet**, which applied Graph Neural Networks to multi-body locomotion control (ant, cheetah, humanoid robots), showing that GNNs can generalize across different robot *morphologies* (structural changes). NerveNet's setting is fundamentally different from ours: it targets complex locomotion across entirely different robot bodies, not the controlled OOD analysis of physical parameters within a single system type. The locomotion complexity also makes isolating what the model has actually learned difficult.

**This project sits at the intersection:** we apply graph-structured encoding to the classical pendulum setting, but with *variable physical parameters* (link lengths and masses sampled fresh each episode). A single model must balance pendulums it has never seen. This setup is novel in two ways: (1) no prior pendulum RL work trains a single model across variable configurations, and (2) the simplicity of the pendulum — relative to NerveNet's locomotion robots — enables rigorous, interpretable OOD analysis: we can sweep a single physical parameter and measure exactly how far outside training distribution the model remains effective.

**References**
- Barto, Sutton & Anderson (1983). *Neuronlike adaptive elements that can solve difficult learning control problems.* IEEE SMC.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Wang et al. (2018). *NerveNet: Learning Structured Policy with Graph Neural Networks.* ICLR.

---

## Quick Start
more detailed start in setup.md

```bash
# 1. Clone and set up environment
git clone <repo-url>
cd Inverted_Pendulum_RL
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train
python3.12 training/train_dqn.py --policy gnn          # GNN DQN
python3.12 training/train_dqn.py --policy mlp          # MLP DQN baseline
python3.12 training/train_ppo.py --policy gnn_mpnn     # GNN MPNN PPO
python3.12 training/train_ppo.py --policy gnn_transformer  # GNN Transformer PPO
python3.12 training/train_ppo.py --policy mlp          # MLP PPO baseline

# 3. Evaluate (OOD generalization sweeps)
python3.12 eval/eval_dqn.py --policy gnn
python3.12 eval/eval_ppo.py --policy gnn_mpnn

# 4. Hyperparameter ablation
python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep lr
python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep all

# 5. Physics validation
python3.12 tests/test_physics.py
```

All hyperparameters are controlled from `configs/default.yaml`. Checkpoints are saved to `checkpoints/`, training curves to `checkpoints/*.png`, and eval plots to `eval/plots/`.

---

## Video Links

| Video | Link |
|---|---|
| Demo (non-technical) | https://youtu.be/ukRrmbrJezs |
| Technical Walkthrough | https://youtu.be/omyPsI9fVvs |

---

## Evaluation

All models trained for 2,000,000 environment steps. Evaluation uses 100 evenly-spaced parameter values × 200 episodes per point, extending ±100% beyond the training distribution on each axis.

### Training Curves

| Model | Curve |
|---|---|
| GNN DQN | `checkpoints/gnn_dqn_training_curve.png` |
| MLP DQN | `checkpoints/mlp_dqn_training_curve.png` |
| GNN MPNN PPO | `checkpoints/gnn_mpnn_ppo_training_curve.png` |
| GNN Transformer PPO | `checkpoints/gnn_transformer_ppo_training_curve.png` |
| MLP PPO | `checkpoints/mlp_ppo_training_curve.png` |

### OOD Generalization — Link Length Sweep

| Model | Plot |
|---|---|
| GNN DQN | `eval/plots/gnn_dqn_link_length_sweep.png` |
| MLP DQN | `eval/plots/mlp_dqn_link_length_sweep.png` |
| GNN MPNN PPO | `eval/plots/gnn_mpnn_ppo_link_length_sweep.png` |

### OOD Generalization — Link Mass Sweep

| Model | Plot |
|---|---|
| GNN DQN | `eval/plots/gnn_dqn_link_mass_sweep.png` |
| MLP DQN | `eval/plots/mlp_dqn_link_mass_sweep.png` |
| GNN MPNN PPO | `eval/plots/gnn_mpnn_ppo_link_mass_sweep.png` |

### OOD Heatmaps (Length × Mass)

| Model | Plot |
|---|---|
| GNN DQN | `eval/plots/gnn_dqn_ood_heatmap.png` |
| MLP DQN | `eval/plots/mlp_dqn_ood_heatmap.png` |
| GNN MPNN PPO | `eval/plots/gnn_mpnn_ppo_ood_heatmap.png` |

### Key Findings

- **GNN DQN outperforms MLP DQN on OOD generalization** — the graph encoder maintains stable reward across parameter ranges the MLP cannot handle
- **DQN outperforms PPO** on this task due to the replay buffer's variance reduction; PPO reaches comparable peak rewards but with higher volatility
- **GNN MPNN and GNN Transformer are comparable in-distribution**; transformer shows stronger attention-weighted generalization on extreme OOD mass values
- **MLP baselines degrade sharply OOD** — fixed input layout cannot adapt to unseen physical configurations

---

## Individual Contributions

**Harshith**
- Custom MuJoCo environment with domain randomization (`env/`)
- Graph observation encoding (`graph/`)
- Full training infrastructure for DQN and PPO (`training/`)
- GNN-Transformer architecture (`models/gnn_transformer_ppo.py`)
- OOD evaluation suite (`eval/`)
- Project configuration and structure

**Rohan**
- MLP policy implementations (notebooks, adapted into codebase by Harshith)
- Base GNN message-passing experiments (notebooks, ported and extended by Harshith)

See [ATTRIBUTION.md](ATTRIBUTION.md) for full detail including AI tool usage.
