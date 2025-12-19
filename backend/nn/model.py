from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmazonsNet(nn.Module):
    """
    Minimal network:
      - board trunk: (B,4,10,10) -> (B,64,10,10)
      - value head: scalar in [-1,1]
      - policy head: given board feat + action planes (3) -> scalar logit
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )

        # value head
        self.v_conv = nn.Conv2d(channels, 16, 1)
        self.v_fc1 = nn.Linear(16 * 10 * 10, 64)
        self.v_fc2 = nn.Linear(64, 1)

        # policy head: concat feat (channels) + action(3) => channels+3
        self.p_conv = nn.Conv2d(channels + 3, 16, 1)
        self.p_fc1 = nn.Linear(16 * 10 * 10, 64)
        self.p_fc2 = nn.Linear(64, 1)

    def forward_features(self, x_board: torch.Tensor) -> torch.Tensor:
        return self.trunk(x_board)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.v_conv(feat))
        z = z.flatten(1)
        z = F.relu(self.v_fc1(z))
        z = torch.tanh(self.v_fc2(z))
        return z.squeeze(-1)  # (B,)

    def policy_logit(self, feat: torch.Tensor, x_action: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,10,10); x_action: (B,3,10,10)
        z = torch.cat([feat, x_action], dim=1)
        z = F.relu(self.p_conv(z))
        z = z.flatten(1)
        z = F.relu(self.p_fc1(z))
        z = self.p_fc2(z)
        return z.squeeze(-1)  # (B,)