"""Param-residual CGAT with an RBF length/mass force schedule.

The schedule is a compact set of trainable log-scale weights over physical
length/mass centers. It is optimized from environment returns only and remains
inside a fixed neural policy checkpoint at eval time.
"""

import numpy as np
import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy
from models.base_ppo import LOG_STD_MIN, LOG_STD_MAX


class CGATRBFForceSchedulePPOPolicy(CGATParamResidualPPOPolicy):
    """Param-residual CGAT plus RBF physical-parameter force scale."""

    VARIANT = "rbf_force_schedule"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.n_length_centers = 4
        self.n_mass_centers = 4
        self.register_buffer("length_centers", torch.linspace(0.05, 2.10, self.n_length_centers))
        self.register_buffer("mass_centers", torch.linspace(0.05, 3.90, self.n_mass_centers))
        self.log_scale_table = nn.Parameter(torch.full(
            (self.n_length_centers, self.n_mass_centers),
            0.18232156,
            dtype=torch.float32,
        ))
        # Start close to the strongest mass-force schedule: larger scales in
        # high-mass columns, but allow optimization to reshape by length.
        with torch.no_grad():
            for j in range(self.n_mass_centers):
                if self.mass_centers[j] >= 1.35:
                    self.log_scale_table[:, j].fill_(0.61518564)
        self.log_scale_clip = nn.Parameter(torch.tensor(0.75, dtype=torch.float32))
        self.length_bandwidth = nn.Parameter(torch.tensor(0.52, dtype=torch.float32))
        self.mass_bandwidth = nn.Parameter(torch.tensor(0.95, dtype=torch.float32))

    def _rbf_weights(self, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links].mean(dim=1, keepdim=True)
        masses = params[:, self.max_links:2 * self.max_links].mean(dim=1, keepdim=True)
        length_bw = self.length_bandwidth.abs().clamp(0.18, 1.20)
        mass_bw = self.mass_bandwidth.abs().clamp(0.35, 2.50)
        length_w = torch.exp(-0.5 * ((lengths - self.length_centers.view(1, -1)) / length_bw).square())
        mass_w = torch.exp(-0.5 * ((masses - self.mass_centers.view(1, -1)) / mass_bw).square())
        weights = length_w[:, :, None] * mass_w[:, None, :]
        return weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    def _force_scale(self, obs: dict) -> torch.Tensor:
        _, params = self._physical_state_and_params(obs)
        weights = self._rbf_weights(params)
        clip = self.log_scale_clip.abs().clamp(0.05, 1.20)
        table = self.log_scale_table.clamp(-clip, clip)
        log_scale = (weights * table.view(1, self.n_length_centers, self.n_mass_centers)).sum(
            dim=(1, 2), keepdim=True
        )
        return torch.exp(log_scale.view(-1, 1))

    def _scaled_force(self, raw_action: torch.Tensor, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self._force_scale(obs)
        action = torch.tanh(raw_action) * self.max_force * scale
        return action.clamp(-self.max_force, self.max_force), scale

    def get_action_and_value(self, obs: dict, action: torch.Tensor | None = None):
        emb = self.encode(obs)
        actor_h = self.actor_trunk(emb)
        critic_h = self.critic_trunk(emb)
        raw_mean = self.actor_mean(obs, emb, actor_h)

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
        raw_mean = self.actor_mean(obs_t, emb, actor_h)
        action, _ = self._scaled_force(raw_mean, obs_t)
        return float(action.squeeze())
