"""CGAT gain-feedback with bounded angular-velocity inputs.

Short-link failures show high-frequency force sign flips driven by tiny angular
velocity changes. This variant applies a smooth tanh bound to joint velocity
features before the encoder and feedback head see the observation. It remains a
plain neural policy: no LQR gain, oracle action, or controller recomputation.
"""

import torch

from .cgat_gain_feedback import CGATGainFeedbackPPOPolicy


class CGATVelocityBoundedPPOPolicy(CGATGainFeedbackPPOPolicy):
    """Gain-feedback CGAT with tanh-bounded joint velocity features."""

    VARIANT = "velocity_bounded"

    def _bounded_obs(self, obs: dict) -> dict:
        node = obs["node_features"].float().clone()
        # Node feature 5 is angular velocity normalized by 10 rad/s. The tanh
        # keeps sign and small-signal sensitivity but limits OOD short-link
        # velocity spikes that cause bang-bang sign chatter.
        vel = node[:, 1:self.max_links + 1, 5]
        node[:, 1:self.max_links + 1, 5] = 0.04 * torch.tanh(vel / 0.04)
        out = dict(obs)
        out["node_features"] = node
        return out

    def encode(self, obs: dict) -> torch.Tensor:
        return super().encode(self._bounded_obs(obs))

    def actor_mean(self, obs: dict, emb: torch.Tensor, actor_h: torch.Tensor) -> torch.Tensor:
        return super().actor_mean(self._bounded_obs(obs), emb, actor_h)
