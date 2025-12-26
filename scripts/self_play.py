"""Self-play script with multiprocessing support.

Behaves like the original single-process script by default.
Set AZ_SELFPLAY_WORKERS > 1 to enable parallel generation.
"""
from __future__ import annotations
import os
# 必须在 import torch 之前设置！
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import time
import time
import pickle
import random
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# ===========================================================================
# Original Imports (Preserved)
# ===========================================================================
from backend.engine.board import Board, P1, P2
from backend.engine.mcts_puct import mcts_puct_policy
from backend.engine.rules import apply_action, is_terminal
from backend.nn.encode import encode_action, encode_state
from backend.nn.load import load_model

# ===========================================================================
# Game Logic & Helpers (Preserved from original)
# ===========================================================================

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
    """Runs a single self-play game."""
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

        # Ensure seed changes per ply for MCTS noise if needed, 
        # though usually MCTS relies on torch/numpy random state.
        # We pass derived seeds just in case the backend uses them explicitly.
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


# ===========================================================================
# Configuration Helpers
# ===========================================================================

def _get_config() -> Dict[str, Any]:
    """Reads environment variables to configure the run."""
    return {
        "games": int(os.environ.get("AZ_SELFPLAY_GAMES", "20")),
        "simulations": int(os.environ.get("AZ_SELFPLAY_SIMS", "80")),
        "max_actions_expand": int(os.environ.get("AZ_SELFPLAY_MAXA", "300")),
        "temperature_moves": int(os.environ.get("AZ_TEMP_MOVES", "20")),
        "dirichlet_alpha": float(os.environ.get("AZ_DIR_ALPHA", "0.10")),
        "dirichlet_epsilon": float(os.environ.get("AZ_DIR_EPS", "0.30")),
        "device": os.environ.get("AZ_DEVICE", "cpu"),
        "net_path": os.environ.get("AZ_MODEL_PATH", "checkpoints/model.pt"),
        "out_dir": os.environ.get("AZ_SELFPLAY_OUT", "data"),
    }

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ===========================================================================
# Single Process Logic (Original Main)
# ===========================================================================

def _run_single_process(cfg: Dict[str, Any], seed_base: int):
    """The original main loop logic, running in the current process."""
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...", cfg["net_path"])
    net = load_model(cfg["net_path"], device=cfg["device"])

    print(f"Starting single-process self-play. Games={cfg['games']}, Device={cfg['device']}")

    all_samples: List[Dict] = []
    t0 = time.time()
    
    for g in range(cfg["games"]):
        # Spread games apart in RNG space
        game_seed = seed_base + g * 1000003
        _seed_everything(game_seed)
        
        samples = play_one_game(
            net=net,
            device=cfg["device"],
            simulations=cfg["simulations"],
            max_actions_expand=cfg["max_actions_expand"],
            temperature_moves=cfg["temperature_moves"],
            dirichlet_alpha=cfg["dirichlet_alpha"],
            dirichlet_epsilon=cfg["dirichlet_epsilon"],
            seed=game_seed,
        )
        all_samples.extend(samples)
        print(f"game {g+1}/{cfg['games']}: samples={len(samples)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"selfplay_{ts}_g{cfg['games']}_puct_sims{cfg['simulations']}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(all_samples, f)

    print(f"Saved: {out_path}  total_samples={len(all_samples)}  elapsed={time.time()-t0:.1f}s")
    return str(out_path)

# ===========================================================================
# Parallel Process Logic
# ===========================================================================

@dataclass
class _WorkerArgs:
    worker_id: int
    num_games: int
    base_seed: int
    # Config is re-read inside worker to ensure clean state
    
def _worker_main(args: _WorkerArgs) -> str:
    """Entry point for a background worker process."""
    
    # 1. Setup Environment & Config
    cfg = _get_config()
    worker_seed = args.base_seed + (args.worker_id * 1_000_000)
    _seed_everything(worker_seed)
    
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Model
    net = load_model(cfg["net_path"], device=cfg["device"])
    
    all_samples = []
    
    # 3. Run Games
    for g in range(args.num_games):
        game_seed = worker_seed + g * 1003
        _seed_everything(game_seed)

        # 记录开始时间
        t_start = time.time()
        
        # 执行游戏 (只调用一次！)
        samples = play_one_game(
            net=net,
            device=cfg["device"],
            simulations=cfg["simulations"],
            max_actions_expand=cfg["max_actions_expand"],
            temperature_moves=cfg["temperature_moves"],
            dirichlet_alpha=cfg["dirichlet_alpha"],
            dirichlet_epsilon=cfg["dirichlet_epsilon"],
            seed=game_seed,
        )
        all_samples.extend(samples)
        
        # 打印进度
        duration = time.time() - t_start
        print(f"[Worker {args.worker_id}] Finished game {g+1}/{args.num_games} "
              f"({len(samples)} moves) in {duration:.1f}s", flush=True)
    
    # 4. Save Output
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"selfplay_{ts}_w{args.worker_id}_g{args.num_games}.pkl"
    out_path = out_dir / filename
    
    with open(out_path, "wb") as f:
        pickle.dump(all_samples, f)
        
    return str(out_path)

def _split_games(total: int, workers: int) -> List[int]:
    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]

# ===========================================================================
# Main Entry Point
# ===========================================================================

def main():
    # 1. Read Base Config
    cfg = _get_config()
    
    # 2. Determine Mode
    num_workers = int(os.environ.get("AZ_SELFPLAY_WORKERS", "1"))
    env_seed = os.environ.get("AZ_SEED")
    base_seed = int(env_seed) if env_seed is not None else int(time.time_ns() % 2**31)

    print("self-play config:",
          f"workers={num_workers}",
          f"games={cfg['games']}",
          f"sims={cfg['simulations']}",
          f"device={cfg['device']}",
          f"net={cfg['net_path']}",
          sep="\n - ")

    if num_workers <= 1:
        # --- Single Process Mode ---
        _run_single_process(cfg, base_seed)
    else:
        # --- Parallel Mode ---
        if cfg['games'] <= 0:
            raise ValueError("AZ_SELFPLAY_GAMES must be > 0")

        # Split work
        chunks = _split_games(cfg['games'], num_workers)
        worker_args = [
            _WorkerArgs(worker_id=i, num_games=chunks[i], base_seed=base_seed)
            for i in range(num_workers) if chunks[i] > 0
        ]

        print(f"Spawning {len(worker_args)} workers to play {cfg['games']} games total...")
        t0 = time.time()

        # Use 'spawn' context for Windows/CUDA compatibility
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_main, worker_args)
        
        print(f"All workers finished. Elapsed={time.time()-t0:.1f}s")
        print("Outputs:", results)

if __name__ == "__main__":
    main()
