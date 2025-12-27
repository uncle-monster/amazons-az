from __future__ import annotations

import os
# =================================================================
# 【关键修改】评估阶段是单进程，必须开启多核加速！
# 不要设为 "1"，否则会慢死。建议设为 4 到 8 之间。
# =================================================================
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"

import shutil
from pathlib import Path

from backend.engine.board import Board, P1, P2
from backend.engine.mcts_puct import mcts_puct_search
from backend.engine.rules import apply_action, is_terminal
from backend.nn.load import load_model


def other(player: int) -> int:
    return P2 if player == P1 else P1


def play_game(net_p1, net_p2, sims: int, max_actions: int, seed: int) -> int:
    """
    Returns winner: P1 or P2
    """
    board = Board.initial()
    player = P1
    ply = 0

    while True:
        if is_terminal(board, player):
            return other(player)

        net = net_p1 if player == P1 else net_p2
        a = mcts_puct_search(
            board=board,
            player_to_move=player,
            net=net,
            device="cpu",
            simulations=sims,
            c_puct=1.5,
            max_actions_expand=max_actions,
            seed=seed + ply * 1337,
        )
        board = apply_action(board, player, a)
        player = other(player)
        ply += 1


def main() -> int:
    sims = int(os.environ.get("AZ_EVAL_SIMS", "80"))
    max_actions = int(os.environ.get("AZ_EVAL_MAXA", "300"))
    games = int(os.environ.get("AZ_EVAL_GAMES", "20"))
    
    # 设定胜率阈值，默认为 0.60
    win_threshold = float(os.environ.get("AZ_GATE", "0.60"))

    ckpt_dir = Path("checkpoints")
    best_path = ckpt_dir / "model_best.pt"
    cand_path = ckpt_dir / "model_candidate.pt"

    # NEW: keep a "current model" alias in sync for convenience/compat
    alias_path = ckpt_dir / "model.pt"

    if not cand_path.exists():
        print("candidate checkpoint not found:", cand_path)
        return 2

    if best_path.exists():
        best = load_model(str(best_path), device="cpu")
        print("loaded best:", best_path)
    else:
        best = None
        print("no best checkpoint; accept candidate by default")
        # accept first model
        cand_path.replace(best_path)
        # NEW: also sync alias
        shutil.copyfile(best_path, alias_path)
        print("promoted candidate -> best")
        print("synced alias ->", alias_path)
        return 0

    cand = load_model(str(cand_path), device="cpu")
    print("loaded candidate:", cand_path)

    def run_match(n_games: int, start_i: int) -> tuple[int, int]:
        """Returns (cand_wins, games_played)."""
        cand_wins = 0
        half = n_games // 2
        for j in range(n_games):
            i = start_i + j
            seed = 10000 + i

            if j < half:
                # candidate plays P1
                winner = play_game(cand, best, sims=sims, max_actions=max_actions, seed=seed)
                cand_is = P1
            else:
                # candidate plays P2
                winner = play_game(best, cand, sims=sims, max_actions=max_actions, seed=seed)
                cand_is = P2

            if winner == cand_is:
                cand_wins += 1

            print(f"game {i}: winner={'cand' if winner == cand_is else 'best'}")

        return cand_wins, n_games

    # --- 运行对局 (不再分阶段) ---
    cand_wins, played = run_match(games, start_i=0)
    winrate = cand_wins / played
    
    print(
        f"final: candidate wins: {cand_wins}/{played}  "
        f"winrate={winrate:.3f}  threshold={win_threshold:.2f}"
    )

    # 只要胜率 >= 0.60 就接受
    if winrate >= win_threshold:
        # promote
        cand_path.replace(best_path)
        print("ACCEPT: promoted candidate -> best")

        # NEW: keep checkpoints/model.pt in sync with best
        shutil.copyfile(best_path, alias_path)
        print("synced alias ->", alias_path)

        return 0
    else:
        print("REJECT: keep best, discard candidate")
        cand_path.unlink(missing_ok=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())