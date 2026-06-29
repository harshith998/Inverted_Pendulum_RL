"""Action-scale CGAT with clipped encoder edge features.

OOD length/mass values appear as normalized edge features outside the training
range. This variant clips only the features seen by the transformer encoder's
edge-bias path, while the learned feedback/action-scale heads still receive the
true normalized physical parameters. It is still a pure learned policy: no LQR
solve, no oracle action, and no controller recomputation.
"""

import torch

from .cgat_action_scale import CGATActionScalePPOPolicy


class CGATClippedActionScalePPOPolicy(CGATActionScalePPOPolicy):
    """Action-scale CGAT with train-range-clipped encoder edge features."""

    VARIANT = "clipped_action_scale"

    def _encoder_obs(self, obs: dict) -> dict:
        out = dict(obs)
        edge = obs["edge_features"].float().clone()
        edge[:, :, 0:2] = edge[:, :, 0:2].clamp(0.0, 1.0)
        out["edge_features"] = edge
        return out

    def encode(self, obs: dict) -> torch.Tensor:
        return super().encode(self._encoder_obs(obs))
