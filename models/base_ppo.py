"""Abstract base for all PPO actor-critic policies.

Subclasses implement encode(obs) -> embedding tensor.
This base handles the shared actor/critic heads, action sampling,
log-prob computation (with tanh correction), and value estimation.

Action parameterisation
-----------------------
Raw mean from actor MLP → tanh → scale by max_force.
Log-std is a single learned scalar (state-independent).
Log-prob is corrected for the tanh change-of-variables:
    log π(a|s) = log N(raw|mean,std) - Σ log(1 - tanh²(raw))
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


LOG_STD_MIN = -5.0
LOG_STD_MAX  = 2.0


class BasePPOPolicy(ABC, nn.Module):

    def __init__(self, hidden: int, max_force: float):
        super().__init__()
        self.max_force = max_force

        # Shared actor trunk (backbone → actor hidden)
        self.actor_trunk = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # Shared critic trunk (backbone → critic hidden)
        self.critic_trunk = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        self.mean_head  = nn.Linear(hidden // 2, 1)   # raw mean before tanh
        self.log_std    = nn.Parameter(torch.zeros(1)) # learned scalar
        self.value_head = nn.Linear(hidden // 2, 1)   # V(s)

    # ------------------------------------------------------------------
    # Subclasses must implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, obs: dict) -> torch.Tensor:
        """Return graph/flat embedding of shape (B, hidden)."""

    # ------------------------------------------------------------------
    # Actor / critic interface
    # ------------------------------------------------------------------

    def get_value(self, obs: dict) -> torch.Tensor:
        """Return V(s) of shape (B, 1)."""
        emb = self.encode(obs)
        return self.value_head(self.critic_trunk(emb))

    def get_action_and_value(self, obs: dict, action: torch.Tensor | None = None):
        """
        Sample (or evaluate) an action.

        Returns
        -------
        action      : (B, 1)  tanh-squashed, scaled to [-max_force, max_force]
        log_prob    : (B,)    tanh-corrected log probability
        entropy     : (B,)    differential entropy of the Gaussian (pre-squash)
        value       : (B, 1)  V(s)
        """
        emb       = self.encode(obs)
        actor_h   = self.actor_trunk(emb)
        critic_h  = self.critic_trunk(emb)

        raw_mean  = self.mean_head(actor_h)           # (B, 1)  unbounded

        log_std   = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std       = log_std.exp().expand_as(raw_mean)

        dist      = torch.distributions.Normal(raw_mean, std)

        if action is None:
            raw_action = dist.rsample()               # reparameterised sample
        else:
            # Invert tanh to recover the raw pre-squash action for evaluation
            # Clamp to avoid log(0) at ±1
            a_norm     = (action / self.max_force).clamp(-1 + 1e-6, 1 - 1e-6)
            raw_action = torch.atanh(a_norm)

        # Tanh squash + scale
        squashed = torch.tanh(raw_action) * self.max_force   # (B, 1)

        # Log-prob with tanh correction (change-of-variables)
        log_prob = dist.log_prob(raw_action)                 # (B, 1)
        log_prob = log_prob - torch.log(
            self.max_force * (1.0 - torch.tanh(raw_action).pow(2)) + 1e-6
        )
        log_prob = log_prob.squeeze(-1)                      # (B,)

        entropy  = dist.entropy().squeeze(-1)                # (B,)
        value    = self.value_head(critic_h)                 # (B, 1)

        return squashed, log_prob, entropy, value

    # ------------------------------------------------------------------
    # Convenience: single observation (no batch dim) for env rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs: dict, device: torch.device) -> tuple[float, float, float]:
        """
        Called once per env step during rollout collection.

        Returns
        -------
        action_val  : float  force in [-max_force, max_force]
        log_prob    : float
        value       : float  V(s)
        """
        obs_t = {
            k: torch.tensor(v, dtype=torch.float32 if v.dtype != np.int64 else torch.int64)
               .unsqueeze(0).to(device)
            for k, v in obs.items()
        }
        action, log_prob, _, value = self.get_action_and_value(obs_t)
        return (float(action.squeeze()),
                float(log_prob.squeeze()),
                float(value.squeeze()))
