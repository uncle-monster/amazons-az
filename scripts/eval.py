from __future__ import annotations

import os
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
    extend_games = int(os.environ.get("AZ_EVAL_EXTEND_GAMES", "40"))
    gray_low = float(os.environ.get("AZ_EVAL_GRAY_LOW", "0.47"))
    gray_high = float(os.environ.get("AZ_EVAL_GRAY_HIGH", "0.57"))

    win_threshold = float(os.environ.get("AZ_GATE", "0.55"))

    if extend_games <= games:
        extend_games = games  # disable extend if misconfigured

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

    # --- stage 1 ---
    cand_wins_1, played_1 = run_match(games, start_i=0)
    winrate_1 = cand_wins_1 / played_1
    print(
        f"stage1: candidate wins: {cand_wins_1}/{played_1}  "
        f"winrate={winrate_1:.3f}  threshold={win_threshold:.2f}"
    )

    # --- stage 2 (optional extend) ---
    cand_wins_total = cand_wins_1
    played_total = played_1

    in_gray = (gray_low <= winrate_1 <= gray_high) and (extend_games > games)
    if in_gray:
        more = extend_games - games
        print(f"gray-zone hit ({gray_low:.2f}..{gray_high:.2f}); extending eval by {more} games (total={extend_games})")
        cand_wins_2, played_2 = run_match(more, start_i=games)
        cand_wins_total += cand_wins_2
        played_total += played_2

    winrate = cand_wins_total / played_total
    print(
        f"final: candidate wins: {cand_wins_total}/{played_total}  "
        f"winrate={winrate:.3f}  threshold={win_threshold:.2f}"
    )

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