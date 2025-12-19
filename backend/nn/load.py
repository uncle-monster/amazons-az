from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from backend.nn.model import AmazonsNet


def load_model(checkpoint_path: str = "checkpoints/model.pt", device: str = "cpu") -> AmazonsNet:
    path = Path(checkpoint_path)
    net = AmazonsNet(channels=64).to(device)
    if path.exists():
        state = torch.load(path, map_location=device)
        net.load_state_dict(state)
    net.eval()
    return net