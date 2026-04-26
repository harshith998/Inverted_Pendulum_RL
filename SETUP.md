# Setup Instructions

## Requirements

- Python 3.12
- macOS or Linux (MuJoCo works on both; Windows is untested)
- ~2 GB disk space for dependencies and checkpoints
- GPU optional — training runs on CPU but is significantly faster with CUDA

---

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd Inverted_Pendulum_RL
```

### 2. Create a Python 3.12 virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `mujoco`, `gymnasium`, `torch`, `numpy`, `scipy`, `pyyaml`, `matplotlib`.

### 4. Verify MuJoCo works

```bash
python3.12 tests/test_physics.py
```

Expected output: trajectory accuracy, energy conservation, and graph structure checks all pass. If MuJoCo is not found, install it manually:

```bash
pip install mujoco
```

MuJoCo 2.3+ bundles its own binaries — no separate license or system install required.

### 5. (Optional) Verify visual rendering

```bash
python3.12 tests/test_visual.py
```

Opens a MuJoCo viewer window showing the pendulum for 3 seconds. Requires a display; skip on headless servers.

---

## Running Training

All hyperparameters are in `configs/default.yaml`. Modify there before running.

```bash
# DQN variants
python3.12 training/train_dqn.py --policy gnn        # GNN-MPNN DQN
python3.12 training/train_dqn.py --policy mlp        # MLP baseline DQN

# PPO variants
python3.12 training/train_ppo.py --policy gnn_mpnn        # GNN MPNN PPO
python3.12 training/train_ppo.py --policy gnn_transformer # GNN Transformer PPO
python3.12 training/train_ppo.py --policy mlp             # MLP baseline PPO
```

Checkpoints saved to `checkpoints/{policy}_dqn_best.pt` / `checkpoints/{policy}_ppo_best.pt`.
Training curves saved to `checkpoints/{policy}_*_training_curve.png`.

---

## Running Evaluation

Requires a trained checkpoint to exist in `checkpoints/`.

```bash
# DQN OOD evaluation
python3.12 eval/eval_dqn.py --policy gnn
python3.12 eval/eval_dqn.py --policy mlp

# PPO OOD evaluation
python3.12 eval/eval_ppo.py --policy gnn_mpnn
python3.12 eval/eval_ppo.py --policy gnn_transformer
python3.12 eval/eval_ppo.py --policy mlp

# Run only specific tests (1=length sweep, 2=mass sweep, 3=heatmap)
python3.12 eval/eval_dqn.py --policy gnn --tests 1 2
python3.12 eval/eval_ppo.py --policy gnn_mpnn --tests 3

# Custom checkpoint path
python3.12 eval/eval_dqn.py --policy gnn --checkpoint checkpoints/my_model.pt
```

Results cached to `eval/cache/`, plots saved to `eval/plots/`.

---

## Running Hyperparameter Ablation

```bash
# Single sweep (3 runs × 1M steps each)
python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep lr
python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep n_envs

# All sweeps (18 runs — leave overnight)
python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep all

# Shorter runs for quick signal
python3.12 training/ablation_ppo.py --policy gnn_mpnn --sweep lr --steps 500000
```

Available sweeps: `lr`, `n_envs`, `rollout_steps`, `gae_lambda`, `entropy_coef`, `n_epochs`.
Results cached to `checkpoints/ablation/` and reused on re-runs.

---

## Project Structure

```
Inverted_Pendulum_RL/
├── configs/
│   └── default.yaml          # All hyperparameters
├── env/
│   ├── pendulum_env.py       # Custom Gymnasium environment
│   ├── mujoco_builder.py     # Programmatic MJCF XML generation
│   └── rewards.py            # Composite reward function
├── graph/
│   ├── graph_builder.py      # Graph observation construction
│   └── graph_utils.py        # Validation utilities
├── models/
│   ├── base_dqn.py           # Abstract DQN base
│   ├── base_ppo.py           # Abstract PPO actor-critic base
│   ├── gnn_dqn.py            # GNN-MPNN DQN
│   ├── mlp_dqn.py            # MLP DQN baseline
│   ├── gnn_mpnn_ppo.py       # GNN-MPNN PPO
│   ├── gnn_transformer_ppo.py# GNN-Transformer PPO
│   └── mlp_ppo.py            # MLP PPO baseline
├── training/
│   ├── train_dqn.py          # DQN training loop
│   ├── train_ppo.py          # PPO training loop (parallel envs)
│   └── ablation_ppo.py       # Hyperparameter ablation framework
├── eval/
│   ├── eval_dqn.py           # OOD evaluation for DQN
│   └── eval_ppo.py           # OOD evaluation for PPO
├── tests/
│   ├── test_physics.py       # MuJoCo vs scipy validation
│   └── test_visual.py        # Visual rendering test
├── checkpoints/              # Saved model weights and training curves
├── eval/plots/               # OOD evaluation plots
├── eval/cache/               # Cached evaluation results (.npz)
├── README.md
├── SETUP.md
├── ATTRIBUTION.md
├── ITERATIONS.md
└── requirements.txt
```
