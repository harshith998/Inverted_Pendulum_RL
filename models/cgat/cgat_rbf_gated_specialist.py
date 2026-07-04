"""RBF force-schedule CGAT with gated learned specialist residuals.

This variant keeps the strong learned backbone intact and adds small
zero-initialized raw-action residual experts for difficult physical regimes.
The gates are deterministic functions of observed physical parameters, but the
corrections themselves are learned only from environment returns.
"""

import torch
import torch.nn as nn

from .cgat_rbf_force_schedule import CGATRBFForceSchedulePPOPolicy


class CGATRBFGatedSpecialistPPOPolicy(CGATRBFForceSchedulePPOPolicy):
    """RBF schedule plus regime-gated specialist action residuals."""

    VARIANT = "rbf_gated_specialist"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        feature_dim = self.state_dim + self.nonlinear_feature_dim + 8
        self.specialist_residual = nn.Sequential(
            nn.Linear(feature_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )
        nn.init.zeros_(self.specialist_residual[-1].weight)
        nn.init.zeros_(self.specialist_residual[-1].bias)
        self.specialist_limit = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))

    def _specialist_features(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links].clamp(min=0.03)
        masses = params[:, self.max_links:2 * self.max_links].clamp(min=0.02)
        cart_mass = params[:, 2 * self.max_links:2 * self.max_links + 1].clamp(min=0.05)
        length_mean = lengths.mean(dim=1, keepdim=True)
        mass_mean = masses.mean(dim=1, keepdim=True)
        length_std = lengths.std(dim=1, keepdim=True, unbiased=False)
        mass_std = masses.std(dim=1, keepdim=True, unbiased=False)
        total_link_mass = masses.sum(dim=1, keepdim=True)
        param_summary = torch.cat([
            length_mean,
            mass_mean,
            length_std,
            mass_std,
            total_link_mass,
            cart_mass,
            mass_mean / length_mean,
            total_link_mass / (cart_mass + total_link_mass),
        ], dim=1)
        return torch.cat([state, self._nonlinear_features(state), param_summary], dim=1)

    def _specialist_gates(self, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links]
        masses = params[:, self.max_links:2 * self.max_links]
        length_mean = lengths.mean(dim=1, keepdim=True)
        mass_mean = masses.mean(dim=1, keepdim=True)

        low_mass_long = torch.sigmoid((0.70 - mass_mean) / 0.12) * torch.sigmoid(
            (length_mean - 0.95) / 0.18
        )
        high_mass = torch.sigmoid((mass_mean - 2.35) / 0.25)
        long_midmass = torch.sigmoid((length_mean - 1.25) / 0.20) * torch.sigmoid(
            (mass_mean - 0.85) / 0.20
        ) * torch.sigmoid((2.40 - mass_mean) / 0.25)
        return torch.cat([low_mass_long, high_mass, long_midmass], dim=1)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        state, params = self._physical_state_and_params(obs)
        features = self._specialist_features(state, params)
        raw_residuals = torch.tanh(self.specialist_residual(features))
        gates = self._specialist_gates(params)
        limit = self.specialist_limit.abs().clamp(0.02, 0.80)
        residual = (gates * raw_residuals).sum(dim=1, keepdim=True) * limit
        return mean + residual
