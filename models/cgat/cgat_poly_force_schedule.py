"""Param-residual CGAT with a compact length/mass polynomial force schedule.

This keeps the adaptation inside the neural policy checkpoint. The extra
parameters are optimized from environment returns only; there are no LQR labels,
stored LQR gains, Riccati solves, or controller recomputation at eval time.
"""

import numpy as np
import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy
from models.base_ppo import LOG_STD_MIN, LOG_STD_MAX


class CGATPolyForceSchedulePPOPolicy(CGATParamResidualPPOPolicy):
    """Param-residual CGAT plus polynomial physical-parameter force scale."""

    VARIANT = "poly_force_schedule"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        # Features: 1, length_n, mass_n, length_n*mass_n, length_n^2, mass_n^2.
        self.poly_scale_coeffs = nn.Parameter(torch.tensor([
            0.18232156, 0.0, 0.30, 0.0, 0.0, 0.0,
        ], dtype=torch.float32))
        self.poly_log_clip = nn.Parameter(torch.tensor(0.72, dtype=torch.float32))

    def _poly_features(self, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links]
        masses = params[:, self.max_links:2 * self.max_links]
        length = lengths.mean(dim=1, keepdim=True)
        mass = masses.mean(dim=1, keepdim=True)
        length_n = (length - 1.075) / 1.025
        mass_n = (mass - 1.975) / 1.925
        return torch.cat([
            torch.ones_like(length_n),
            length_n,
            mass_n,
            length_n * mass_n,
            length_n.square(),
            mass_n.square(),
        ], dim=1)

    def _force_scale(self, obs: dict) -> torch.Tensor:
        _, params = self._physical_state_and_params(obs)
        features = self._poly_features(params)
        log_scale = (features * self.poly_scale_coeffs.view(1, -1)).sum(dim=1, keepdim=True)
        clip = self.poly_log_clip.abs().clamp(0.05, 1.20)
        return torch.exp(log_scale.clamp(-clip, clip))

    def _scaled_force(self, raw_action: torch.Tensor, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._force_scale(obs)
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
        scale = self._force_scale(obs)

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
        entropy = dist.entropy().squeeze(-1)
        value = self.value_head(critic_h)
        return squashed, log_prob.squeeze(-1), entropy, value

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
