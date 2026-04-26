"""Random baseline policy — samples uniformly from the action space.

No training required. Used as a floor baseline in OOD evaluation to confirm
that learned policies are doing better than chance.

Compatible with both DQN and PPO eval interfaces:
  DQN : get_action(obs, epsilon, device) -> int   (action bin index)
  PPO : get_action(obs, device)          -> (float, float, float)  (force, log_prob, value)
"""

import numpy as np


class RandomDQNPolicy:
    """Random baseline matching the DQN eval interface."""

    def __init__(self, n_action_bins: int):
        self.n_action_bins = n_action_bins

    def get_action(self, obs: dict, epsilon: float, device) -> int:
        return np.random.randint(self.n_action_bins)

    def eval(self):
        return self

    def to(self, device):
        return self


class RandomPPOPolicy:
    """Random baseline matching the PPO eval interface."""

    def __init__(self, max_force: float):
        self.max_force = max_force

    def get_action(self, obs: dict, device) -> tuple[float, float, float]:
        force = float(np.random.uniform(-self.max_force, self.max_force))
        return force, 0.0, 0.0

    def eval(self):
        return self

    def to(self, device):
        return self
