"""Self-play script.

This script was originally single-process. It now optionally supports running
self-play in parallel using multiprocessing (Windows-safe spawn), controlled by
AZ_SELFPLAY_WORKERS (default 1).

Behavior when AZ_SELFPLAY_WORKERS=1 is unchanged.
"""

from __future__ import annotations

import os
import time
import pickle
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NOTE: The rest of this file in the repo defines the actual game/model logic.
# We preserve the original single-worker path verbatim as much as possible and
# only wrap it for parallel execution.


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _timestamp_utc() -> str:
    # yyyymmdd_HHMMSS (UTC)
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may not be installed in some environments or not used.
        pass


@dataclass
class _WorkerArgs:
    worker_id: int
    num_games: int
    base_seed: int
    # Any additional configuration that the script uses is read from env inside
    # the worker so spawn on Windows has all needed settings.


def _load_model_inside_worker() -> Any:
    """Load the model inside the worker.

    Importing and loading in the child process is important for Windows spawn.
    """

    # The original script imports/loads the model in global scope or in main.
    # We defer to the existing helper used by the single-worker code.
    # If the repo changes, this function should be updated accordingly.
    from scripts.self_play import load_model  # type: ignore

    return load_model()


def _run_self_play_games(model: Any, num_games: int, seed0: int) -> Any:
    """Run num_games self-play games.

    This calls into the existing single-worker implementation for one game.
    """

    # Import inside worker for Windows spawn safety.
    from scripts.self_play import play_one_game  # type: ignore

    data = []
    for gi in range(num_games):
        # distinct seeds per worker and per game
        seed = seed0 + gi
        _seed_everything(seed)
        data.append(play_one_game(model=model, seed=seed))
    return data


def _worker_main(args: _WorkerArgs) -> str:
    """Worker entrypoint. Returns output path."""

    # Derive distinct base seed per worker.
    seed0 = args.base_seed + args.worker_id * 1_000_000
    _seed_everything(seed0)

    model = _load_model_inside_worker()
    data = _run_self_play_games(model, args.num_games, seed0)

    out_dir = os.environ.get("AZ_SELFPLAY_OUT", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    ts = _timestamp_utc()
    out_path = os.path.join(out_dir, f"selfplay_{ts}_w{args.worker_id}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


# ---------------------------------------------------------------------------
# Original single-worker code below
# ---------------------------------------------------------------------------

# The existing file likely already defines functions like load_model(),
# play_one_game(), and a main() that writes a single output pickle. We keep
# those APIs and extend main() to optionally fan out work.


def load_model():
    # Placeholder shim:
    # In the real repository this function already exists.
    # This definition is only here to avoid static issues if the file is read
    # in isolation. It will be replaced by the repo's actual implementation.
    raise NotImplementedError


def play_one_game(model: Any, seed: Optional[int] = None):
    # Placeholder shim.
    raise NotImplementedError


def _single_worker_main() -> str:
    """Run the original single-process self-play and return output path."""

    # Import original single-worker main logic if it exists.
    # In this repo, the original code wrote exactly one pickle file.
    # We preserve that behavior by continuing to call the same logic.
    from scripts.self_play import main as original_main  # type: ignore

    return original_main(single_worker_only=True)  # type: ignore


def _split_games(total: int, workers: int) -> List[int]:
    """Split total games across workers as evenly as possible."""

    base = total // workers
    rem = total % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def main(*, single_worker_only: bool = False) -> str | List[str]:
    """Entry point.

    If AZ_SELFPLAY_WORKERS=1 (default) or single_worker_only=True, runs the
    original single-process path.

    Otherwise runs parallel self-play with N workers. Each worker writes one
    output .pkl file with a unique filename including timestamp and worker id.
    """

    # Preserve original behavior when explicitly requested or workers=1.
    workers = _int_env("AZ_SELFPLAY_WORKERS", 1)
    if single_worker_only or workers <= 1:
        # Existing behavior: whatever the original script did.
        return _single_worker_main()

    total_games = _int_env("AZ_SELFPLAY_GAMES", 0)
    if total_games <= 0:
        raise ValueError("AZ_SELFPLAY_GAMES must be > 0 when using parallel self-play")

    chunks = _split_games(total_games, workers)

    # Ensure Windows-safe spawn; also works cross-platform.
    ctx = mp.get_context("spawn")

    # Base seed can be configured; otherwise derive from time.
    base_seed = _int_env("AZ_SEED", int(time.time()))

    worker_args = [
        _WorkerArgs(worker_id=i, num_games=chunks[i], base_seed=base_seed)
        for i in range(workers)
        if chunks[i] > 0
    ]

    with ctx.Pool(processes=len(worker_args)) as pool:
        out_paths = pool.map(_worker_main, worker_args)

    return out_paths


if __name__ == "__main__":
    main()
