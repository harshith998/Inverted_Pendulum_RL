"""Param-residual CGAT with baked-in mass-conditioned force scaling.

This variant keeps the schedule inside the policy checkpoint and applies it at
the force level after tanh, matching the successful schedule-style ablations
more closely than raw-mean scaling. It remains a fixed learned policy at eval
time: no LQR solve, no gains, and no controller recomputation.
"""

import numpy as np
import torch

from .cgat_mass_schedule import CGATMassSchedulePPOPolicy
from models.base_ppo import LOG_STD_MIN, LOG_STD_MAX


class CGATMassForceSchedulePPOPolicy(CGATMassSchedulePPOPolicy):
    """Param-residual CGAT plus trainable mass-conditioned force scale."""

    VARIANT = "mass_force_schedule"

    def _scaled_force(self, raw_action: torch.Tensor, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        _, params = self._physical_state_and_params(obs)
        scale = self._mass_schedule_scale(params)
        action = torch.tanh(raw_action) * self.max_force * scale
        return action.clamp(-self.max_force, self.max_force), scale

    def get_action_and_value(self, obs: dict, action: torch.Tensor | None = None):
        emb = self.encode(obs)
        actor_h = self.actor_trunk(emb)
        critic_h = self.critic_trunk(emb)
        raw_mean = super().actor_mean(obs, emb, actor_h)

        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(raw_mean)
        dist = torch.distributions.Normal(raw_mean, std)

        _, params = self._physical_state_and_params(obs)
        scale = self._mass_schedule_scale(params)

        if action is None:
            raw_action = dist.rsample()
        else:
            a_norm = (action / (self.max_force * scale)).clamp(-1 + 1e-6, 1 - 1e-6)
            raw_action = torch.atanh(a_norm)

        squashed = torch.tanh(raw_action) * self.max_force * scale
        squashed = squashed.clamp(-self.max_force, self.max_force)

        log_prob = dist.log_prob(raw_action)
        jacobian = self.max_force * scale * (1.0 - torch.tanh(raw_action).pow(2))
        log_prob = log_prob - torch.log(jacobian + 1e-6)
        log_prob = log_prob.squeeze(-1)

        entropy = dist.entropy().squeeze(-1)
        value = self.value_head(critic_h)
        return squashed, log_prob, entropy, value

    @torch.no_grad()
    def get_deterministic_action(self, obs: dict, device: torch.device) -> float:
        obs_t = {
            k: torch.tensor(v, dtype=torch.float32 if v.dtype != np.int64 else torch.int64)
               .unsqueeze(0).to(device)
            for k, v in obs.items()
        }
        emb = self.encode(obs_t)
        actor_h = self.actor_trunk(emb)
        raw_mean = super().actor_mean(obs_t, emb, actor_h)
        action, _ = self._scaled_force(raw_mean, obs_t)
        return float(action.squeeze())
