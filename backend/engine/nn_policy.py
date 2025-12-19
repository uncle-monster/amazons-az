from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from backend.engine.board import Board
from backend.engine.rules import Action
from backend.nn.encode import encode_action, encode_state
from backend.nn.model import AmazonsNet


@torch.no_grad()
def nn_value_and_logits(
    net: AmazonsNet,
    board: Board,
    player: int,
    actions: List[Action],
    device: str = "cpu",
) -> Tuple[float, np.ndarray]:
    """
    Returns:
      value: float in [-1,1] for player-to-move perspective
      logits: np.ndarray shape (len(actions),)
    """
    net.eval()

    x_board = torch.from_numpy(encode_state(board, player)).unsqueeze(0).to(device)  # (1,4,10,10)
    feat = net.forward_features(x_board)
    v = net.value(feat).item()

    if len(actions) == 0:
        return v, np.zeros((0,), dtype=np.float32)

    x_actions = np.stack([encode_action(a) for a in actions], axis=0)  # (A,3,10,10)
    x_actions_t = torch.from_numpy(x_actions).to(device)

    feat_rep = feat.repeat(x_actions_t.shape[0], 1, 1, 1)  # (A,C,10,10)
    logits = net.policy_logit(feat_rep, x_actions_t).detach().cpu().numpy().astype(np.float32)

    return v, logits