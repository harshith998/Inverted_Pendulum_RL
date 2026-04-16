"""Abstract interface all DQN policies must implement."""

from abc import ABC, abstractmethod
import torch
import numpy as np


class BaseDQNPolicy(ABC, torch.nn.Module):

    def __init__(self, n_action_bins: int):
        super().__init__()
        self.n_action_bins = n_action_bins

    @abstractmethod
    def get_q_values(self, obs: dict) -> torch.Tensor:
        """Return Q-values of shape (batch, n_action_bins)."""

    def get_action(self, obs: dict, epsilon: float, device: torch.device) -> int:
        # if random les than epsilon  then explore
        if np.random.random() < epsilon:
            return np.random.randint(self.n_action_bins)
        #otherwise exploit, if random more than epsilon
        with torch.no_grad():
            obs_t = {k: torch.tensor(v, dtype=torch.float32 if v.dtype != np.int64 else torch.int64)
                     .unsqueeze(0).to(device)
                     for k, v in obs.items()}
            q = self.get_q_values(obs_t)
        return int(q.argmax(dim=1).item())
