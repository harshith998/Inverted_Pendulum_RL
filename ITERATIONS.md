# Model Iteration Log

Documents each major iteration of the training pipeline: what was observed, what changed, and what the result was. Training curves referenced below are in `checkpoints/`, eval plots in `eval/plots/`.

---

## Iteration 1 — DQN Baselines: GNN vs MLP Initial Comparison

**Observation:**
Both initial GNN DQN and MLP DQN had zero win rate (no episode survived to max steps). MLP DQN showed higher mean reward and more consistent convergence — but OOD evaluation revealed it generalized horribly. GNN DQN generalized significantly better, especially on link mass: reward stayed high across a wide OOD mass range while MLP reward collapsed.

The link length OOD result was confounded: longer links are physically easier to balance (lower center of mass sensitivity), so both models appeared to handle length OOD — this isn't a generalization win, it's a physics artifact. Mass OOD is the cleaner test.

The core tradeoff: MLP is more accurate in-distribution but brittle OOD. GNN uses message passing to build a structured representation of the physical system — understanding which nodes and edges matter — rather than just concatenating features, giving it meaningful OOD generalization.

**See:** `eval/plots/gnn_dqn_link_mass_sweep.png`, `eval/plots/mlp_dqn_link_mass_sweep.png`, `eval/plots/gnn_dqn_ood_heatmap.png`, `eval/plots/mlp_dqn_ood_heatmap.png`

---

## Iteration 2 — GNN DQN: Divergence and Residual Fix

**Observation:**
MLP DQN converged to a stable reward plateau and stayed there — more training steps wouldn't help, it had reached its capacity limit. GNN DQN, however, rose initially then dipped and failed to recover. Inspecting the architecture revealed the problem: at each message-passing step, the node embedding was being fully replaced (`h = h_new`) rather than updated residually.

**Root cause:**
Without a residual connection, the encoder had no gradient path back to the original embedding. As embeddings drifted in scale over training, Q-values became unstable and the network collapsed.

**Changes made:**
- Residual connection added: `h = h + f(h)` (i.e. `return self.norm(h + h_new)`)
- Learning rate reduced: `3e-4 → 5e-5` for slower, more stable updates
- Training steps increased so the slower LR has room to converge
- Extended epsilon decay for longer exploration before policy commits
- Added `LayerNorm` after each message-passing layer to prevent embedding scale drift
- L2 weight decay activated in optimizer

**Result:**
GNN DQN trained stably. Episodes were successful, training grew and converged to a clear plateau rather than diverging. Win rate increased from 0%.

**See:** `checkpoints/gnn_dqn_training_curve.png`

---

## Iteration 3 — GNN DQN OOD Analysis and Motivation for Transformer

**Observation:**
After the residual fix, OOD evaluation showed GNN DQN generalizing well — reward stayed high across OOD mass and length ranges, with only slight dropoffs at the far ends. The 2D heatmap confirmed it generalized across both variables simultaneously, both in-distribution and OOD.

MLP OOD plots looked visually smooth, but this is misleading: the MLP's in-distribution and OOD rewards were both near-zero — consistent failure is not generalization. The GNN's OOD performance had meaningful signal with actual reward.

**Motivation for transformer:**
Standard message passing (MPNN) treats all neighbouring nodes equally — each message contributes the same weight to the aggregation. A graph attention transformer computes attention weights dynamically: given the current state, it decides which neighbours matter more or less. For a variable pendulum where a heavy distant link might dominate the dynamics in some states but not others, learned per-state attention could be more expressive than fixed mean aggregation. This motivated developing a GNN Transformer PPO variant alongside GNN MPNN PPO.

**See:** `eval/plots/gnn_dqn_link_length_sweep.png`, `eval/plots/gnn_dqn_link_mass_sweep.png`, `eval/plots/gnn_dqn_ood_heatmap.png`

---

## Iteration 4 — GNN Transformer PPO: NaN Crash

**Observation:**
GNN Transformer PPO crashed immediately with `NaN` values in the action distribution. Training was completely non-functional from step 1.

**Root cause:**
The scatter softmax in graph attention computed `exp(logit)` directly. As logits grew during training, this overflowed to `inf`. The subsequent `inf / inf` normalization produced `NaN` which propagated through the entire network.

**Fix:**
Replaced with numerically stable scatter softmax using the log-sum-exp trick:
1. Compute per-destination-node max logit via `scatter_reduce_(..., reduce="amax")`
2. Subtract max before exponentiation: `exp(logit - max_per_node)`
3. Normalize with clamped denominator to avoid division by zero

Padding edges are masked to `-inf` before computing the max so they don't affect the softmax.

**Result:**
No more NaN. Training proceeded past initialization.

---

## Iteration 5 — GNN Transformer PPO: Instability and Hyperparameter Fix

**Observation:**
After the NaN fix, the transformer PPO overshot — reward climbed fast then crashed and couldn't recover. The training curve showed large swings and no stable convergence.

**Root cause:**
- Learning rate too high — gradient updates too large, policy overshot stable region
- Too many epochs per update (`n_epochs=10`) — reusing the same on-policy rollout 10 times caused the policy to drift far from what collected the data, making PPO's clipping ineffective
- Entropy too low — the policy's action distribution (Gaussian std) collapsed too early, making it hard to recover from a bad state once the policy fell into one

**Changes made:**
- `lr: 3e-4 → 1e-4`
- `anneal_lr: true` — linearly decay learning rate to 0 over training so updates become conservative as the policy matures
- `n_epochs: 10 → 5`
- `entropy_coef: 0.01 → 0.02` — higher entropy keeps the Gaussian std wider, preserving the ability to explore and recover
- `total_steps: 1M → 2M` — slower LR requires more steps to converge

**Result:**
Stable convergence without catastrophic collapse. See `checkpoints/gnn_transformer_ppo_training_curve.png`.

---

## Iteration 6 — GNN MPNN PPO: Missing Residual (Same Bug, PPO Context)

**Observation:**
GNN MPNN PPO (ported from the working GNN DQN) showed the same divergence pattern as pre-fix GNN DQN, but much faster — collapsing at ~250k steps vs ~1.3M for DQN.

**Root cause:**
The MPNN residual fix from Iteration 2 had not been carried over to the PPO model. The PPO encoder still had `return self.norm(h_new)`. PPO's on-policy gradient updates are more aggressive than DQN's buffered updates, so the embedding drift caused collapse much sooner.

**Fix:**
`return self.norm(h_new)` → `return self.norm(h + h_new)` in `gnn_mpnn_ppo.py`.

**Result:**
Stable MPNN PPO training reaching comparable rewards to GNN DQN. See `checkpoints/gnn_mpnn_ppo_training_curve.png`.

---

## Iteration 7 — Parallel Environments for PPO Variance Reduction

**Observation:**
Even after stabilization, PPO training curves were more volatile than DQN — large spikes up and down, making it hard to assess convergence. This was fundamental to the algorithm: each update used only the last 2048 steps from a single environment. If that rollout happened to catch several failing episodes, the update directly degraded the policy.

**Root cause:**
On-policy learning has no replay buffer — it can't smooth over bad episodes the way DQN can. Single-environment rollouts produce temporally correlated data with no diversity.

**Change:**
- `n_envs: 1 → 4` parallel independent environments
- `rollout_steps: 2048 → 4096` per env
- `mini_batch_size: 64 → 256`
- Total data per update: `2048 → 16384` transitions (8× more diversity per update)

**Result:**
Reduced spike frequency. Each update now draws from 4 independently-randomized pendulum configurations with different link counts, masses, and lengths, breaking temporal correlation.

---

## Summary Table

| Iteration | Problem | Fix | Impact |
|---|---|---|---|
| 1 | MLP fails OOD despite higher in-dist reward | GNN established as better approach | Confirmed graph structure advantage |
| 2 | GNN DQN diverges (no residual) | Residual `h = h + f(h)` + lower LR + LayerNorm | Stable training, nonzero win rate |
| 3 | OOD analysis + architecture motivation | — (analysis iteration) | Motivated transformer variant |
| 4 | Transformer PPO NaN crash | Stable scatter softmax (log-sum-exp) | Training unblocked |
| 5 | Transformer PPO overshoots, can't recover | Lower LR + annealing + fewer epochs + higher entropy | Stable PPO convergence |
| 6 | MPNN PPO fast collapse (residual missing) | Residual added to PPO encoder | Stable MPNN PPO |
| 7 | PPO high variance (single env) | 4 parallel envs + larger rollout | Reduced volatility |
