"""A small GRU model learning from six historical AEC measurements."""
import torch
from torch import nn


class RULGRU(nn.Module):
    def __init__(self, num_features: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.rnn = nn.GRU(num_features, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        return self.head(hidden[-1]).squeeze(-1)
