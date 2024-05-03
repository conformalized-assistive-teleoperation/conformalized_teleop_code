from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn as nn

class Ensemble_Estimator(torch.nn.Module):
    def __init__(
        self,
        # frozen_decoder: torch.nn.Module,
        action_dim: int = 7,
        state_dim: int = 7,
        latent_dim: int = 7,
        hidden_dim: int = 30,

    ):
        super(Ensemble_Estimator, self).__init__()

        # Save Hyperparameters
        # self.frozen_decoder = frozen_decoder
        self.state_dim, self.latent_dim = state_dim, latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.q_lo = 0.1
        self.q_hi = 0.9

        # Build Model
        self.build_model()

    def build_model(self) -> None:
        # Encoder --> Takes (State, Action) --> Encodes to `z` latent space
        self.enc = nn.Sequential(
            nn.Linear(self.state_dim + self.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim * 6),
            nn.Tanh(),
            nn.Linear(self.latent_dim * 6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim * 2),
        )


    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ Default forward pass --> encode (s, a) --> z; decode (s, z) --> a. """
        x = torch.cat([s, z], 1)
        new_z = self.enc(x)

        # Return Predicted Action via Decoder
        return new_z


