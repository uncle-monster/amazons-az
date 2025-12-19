from __future__ import annotations

import os
import subprocess
import sys


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd)


def env_snapshot() -> dict[str, str]:
    """
    Print the env knobs that materially affect training stability / speed.
    Missing keys are shown as "<default>" so it's obvious what is and isn't set.
    """
    keys = [
        # loop control
        "AZ_ITERS",
        "AZ_SEED",

        # self-play
        "AZ_SELFPLAY_GAMES",
        "AZ_SELFPLAY_SIMS",
        "AZ_SELFPLAY_MAXA",
        "AZ_TEMP_MOVES",
        "AZ_DIR_ALPHA",
        "AZ_DIR_EPS",
        "AZ_DEVICE",
        "AZ_MODEL_PATH",

        # train
        "AZ_LR",
        "AZ_V_MIX",
        "AZ_REPLAY_FILES",
        "AZ_REPLAY_MAX_SAMPLES",

        # eval / gate
        "AZ_GATE",
        "AZ_EVAL_GAMES",
        "AZ_EVAL_EXTEND_GAMES",
        "AZ_EVAL_GRAY_LOW",
        "AZ_EVAL_GRAY_HIGH",
        "AZ_EVAL_SIMS",
        "AZ_EVAL_MAXA",
    ]
    return {k: os.environ.get(k, "<default>") for k in keys}


def main() -> int:
    iters = int(os.environ.get("AZ_ITERS", "5"))

    print("AZ config snapshot:")
    snap = env_snapshot()
    for k in sorted(snap.keys()):
        print(f" - {k}={snap[k]}")

    for k in range(iters):
        print(f"\n========== ITER {k} ==========")

        rc = run([sys.executable, "-m", "scripts.self_play"])
        if rc != 0:
            print("self_play failed")
            return rc

        rc = run([sys.executable, "-m", "scripts.train"])
        if rc != 0:
            print("train failed")
            return rc

        rc = run([sys.executable, "-m", "scripts.eval"])
        # eval returns 0 accept, 1 reject, 2 error
        if rc == 2:
            print("eval error")
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())