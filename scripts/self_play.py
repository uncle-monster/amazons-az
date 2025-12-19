from __future__ import annotations
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from backend.engine.board import Board, P1, P2
from backend.engine.mcts_puct import mcts_puct_policy
from backend.engine.rules import apply_action, is_terminal
from backend.nn.encode import encode_action, encode_state
from backend.nn.load import load_model


def other(player: int) -> int:
    return P2 if player == P1 else P1


def play_one_game(
    net,
    device: str = "cpu",
    simulations: int = 80,
    max_actions_expand: int = 300,
    temperature_moves: int = 20,
    dirichlet_alpha: float = 0.10,
    dirichlet_epsilon: float = 0.30,
    seed: int = 0,
) -> List[Dict]:
    board = Board.initial()
    player = P1
    samples: List[Dict] = []
    ply = 0

    while True:
        if is_terminal(board, player):
            winner = other(player)
            for s in samples:
                s_player = s["player"]
                s["z"] = 1.0 if s_player == winner else -1.0
            return samples

        temperature = 1.0 if ply < temperature_moves else 1e-9

        actions, pi, chosen, v_root, v_mcts_root = mcts_puct_policy(
            board=board,
            player_to_move=player,
            net=net,
            device=device,
            simulations=simulations,
            c_puct=1.5,
            max_actions_expand=max_actions_expand,
            seed=seed + ply * 1337,
            temperature=temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            dirichlet_seed=seed + ply * 99991,
        )

        x_state = encode_state(board, player)
        x_actions = np.stack([encode_action(a) for a in actions], axis=0)
        pi_arr = np.asarray(pi, dtype=np.float32)

        samples.append(
            {
                "state": x_state,
                "actions": x_actions,
                "pi": pi_arr,
                "player": int(player),
                "v_root": float(v_root),
                "v_mcts_root": float(v_mcts_root),
                "z": None,
            }
        )

        board = apply_action(board, player, chosen)
        player = other(player)
        ply += 1


def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # NEW: configurable via env
    games = int(os.environ.get("AZ_SELFPLAY_GAMES", "20"))
    simulations = int(os.environ.get("AZ_SELFPLAY_SIMS", "80"))
    max_actions_expand = int(os.environ.get("AZ_SELFPLAY_MAXA", "300"))
    temperature_moves = int(os.environ.get("AZ_TEMP_MOVES", "20"))
    dirichlet_alpha = float(os.environ.get("AZ_DIR_ALPHA", "0.10"))
    dirichlet_epsilon = float(os.environ.get("AZ_DIR_EPS", "0.30"))
    device = os.environ.get("AZ_DEVICE", "cpu")

    net_path = os.environ.get("AZ_MODEL_PATH", "checkpoints/model.pt")
    net = load_model(net_path, device=device)

    # per-run seed (default: time-based; override with AZ_SEED for reproducibility)
    env_seed = os.environ.get("AZ_SEED")
    run_seed = int(env_seed) if env_seed is not None else int(time.time_ns() % 2**31)

    print(
        "self-play config:",
        f"run_seed={run_seed}",
        f"games={games}",
        f"sims={simulations}",
        f"maxA={max_actions_expand}",
        f"temp_moves={temperature_moves}",
        f"dir_alpha={dirichlet_alpha}",
        f"dir_eps={dirichlet_epsilon}",
        f"device={device}",
        f"net={net_path}",
        sep="\n - ",
    )

    all_samples: List[Dict] = []
    t0 = time.time()
    for g in range(games):
        samples = play_one_game(
            net=net,
            device=device,
            simulations=simulations,
            max_actions_expand=max_actions_expand,
            temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            seed=run_seed + g * 1000003,  # spread games apart in RNG space
        )
        all_samples.extend(samples)
        print(f"game {g}: samples={len(samples)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"selfplay_{ts}_g{games}_puct_sims{simulations}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"saved: {out_path}  total_samples={len(all_samples)}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()