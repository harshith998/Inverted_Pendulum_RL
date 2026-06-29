"""Param-residual CGAT with a baked-in learned mass-conditioned raw scale.

The schedule is part of the neural policy checkpoint and is evaluated through
the normal CGAT path. It is not an LQR controller and does not recompute gains;
it only rescales the learned actor's raw mean based on observed physical
parameters. The defaults encode the best env-discovered scale region so the
variant can be tested as a compact policy-search hypothesis.
"""

import torch
import torch.nn as nn

from .cgat_param_residual import CGATParamResidualPPOPolicy


class CGATMassSchedulePPOPolicy(CGATParamResidualPPOPolicy):
    """Param-residual CGAT plus trainable mass-conditioned raw-mean scale."""

    VARIANT = "mass_schedule"

    def __init__(self, hidden: int = 128, n_icga_layers: int = 2,
                 n_heads: int = 2, max_links: int = 3, max_force: float = 20.0):
        super().__init__(
            hidden=hidden,
            n_icga_layers=n_icga_layers,
            n_heads=n_heads,
            max_links=max_links,
            max_force=max_force,
        )
        self.log_low_scale = nn.Parameter(torch.tensor(0.18232156))   # log(1.20)
        self.log_high_scale = nn.Parameter(torch.tensor(0.64185387))  # log(1.90)
        self.mass_switch = nn.Parameter(torch.tensor(1.20))
        self.mass_sharpness = nn.Parameter(torch.tensor(8.0))

    def _mass_schedule_scale(self, params: torch.Tensor) -> torch.Tensor:
        masses = params[:, self.max_links:2 * self.max_links]
        mean_mass = masses.mean(dim=1, keepdim=True)
        max_mass = masses.max(dim=1, keepdim=True).values
        mass_signal = 0.65 * mean_mass + 0.35 * max_mass
        switch = self.mass_switch.clamp(0.2, 3.5)
        gate = (mass_signal >= switch).float()
        log_low = self.log_low_scale.clamp(-0.4, 0.8)
        log_high = self.log_high_scale.clamp(-0.2, 1.0)
        return torch.exp((1.0 - gate) * log_low + gate * log_high)

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        mean = super().actor_mean(obs, emb, actor_h)
        _, params = self._physical_state_and_params(obs)
        return mean * self._mass_schedule_scale(params)
