"""Param-residual CGAT with a bounded learned raw-action residual expert.

This variant is pure learned control. It does not call an LQR solver, store
classical gains, or recompute a controller from physical parameters. The added
expert is a neural residual over physical state and parameter features, gated
smoothly toward the OOD mass/length regimes where the current learned policy
loses high-survival cells.
"""

import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy


class CGATForceResidualPPOPolicy(CGATParamResidualPPOPolicy):
    """Param-residual CGAT plus bounded direct raw-action residual."""

    VARIANT = "force_residual"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        state_dim = 2 + 2 * max_links
        nonlinear_dim = 2 + 6 * max_links
        enhanced_param_dim = 1 + 7 * max_links
        summary_dim = 8
        feature_dim = state_dim + nonlinear_dim + enhanced_param_dim + summary_dim

        self.raw_residual_limit = 1.75
        self.force_residual_net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.force_residual_net[-1].weight)
        nn.init.zeros_(self.force_residual_net[-1].bias)

    def _param_summary(self, params: torch.Tensor) -> torch.Tensor:
        lengths = params[:, :self.max_links].clamp(min=0.03)
        masses = params[:, self.max_links:2 * self.max_links].clamp(min=0.02)
        cart_mass = params[:, 2 * self.max_links:2 * self.max_links + 1].clamp(min=0.05)

        total_link_mass = masses.sum(dim=1, keepdim=True)
        max_mass = masses.max(dim=1, keepdim=True).values
        mean_mass = masses.mean(dim=1, keepdim=True)
        min_length = lengths.min(dim=1, keepdim=True).values
        max_length = lengths.max(dim=1, keepdim=True).values
        mean_length = lengths.mean(dim=1, keepdim=True)
        load_ratio = total_link_mass / cart_mass
        inertia_proxy = (masses * lengths * lengths).sum(dim=1, keepdim=True)

        return torch.cat([
            total_link_mass,
            max_mass,
            mean_mass,
            min_length,
            max_length,
            mean_length,
            load_ratio,
            inertia_proxy,
        ], dim=1)

    def _ood_gate(self, params: torch.Tensor) -> torch.Tensor:
        summary = self._param_summary(params)
        max_mass = summary[:, 1:2]
        min_length = summary[:, 3:4]
        max_length = summary[:, 4:5]
        load_ratio = summary[:, 6:7]

        heavy_gate = torch.sigmoid(2.8 * (max_mass - 1.35))
        short_gate = torch.sigmoid(8.0 * (0.35 - min_length))
        long_gate = torch.sigmoid(3.0 * (max_length - 1.15))
        load_gate = torch.sigmoid(1.3 * (load_ratio - 1.6))
        return torch.clamp(0.25 + 0.75 * torch.maximum(
            torch.maximum(heavy_gate, short_gate),
            torch.maximum(long_gate, load_gate),
        ), 0.25, 1.0)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        state, params = self._physical_state_and_params(obs)
        features = torch.cat([
            state,
            self._nonlinear_features(state),
            self._enhanced_params(params),
            self._param_summary(params),
        ], dim=1)
        residual = self.raw_residual_limit * torch.tanh(self.force_residual_net(features))
        return mean + self._ood_gate(params) * residual
