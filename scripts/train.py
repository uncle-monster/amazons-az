from __future__ import annotations

import glob
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend.nn.model import AmazonsNet


def list_selfplay_pkls(data_dir: str = "data") -> List[str]:
    files = sorted(glob.glob(str(Path(data_dir) / "selfplay_*.pkl")))
    return files


def load_replay(
    data_dir: str = "data",
    last_n_files: int = 10,
    max_samples: int = 50000,
    seed: int = 0,
) -> List[Dict]:
    files = list_selfplay_pkls(data_dir)
    if not files:
        raise FileNotFoundError("No selfplay_*.pkl found in data/. Run self_play first.")

    use_files = files[-last_n_files:] if last_n_files > 0 else files
    print("loading replay files:")
    for f in use_files:
        print(" -", f)

    data: List[Dict] = []
    for f in use_files:
        with open(f, "rb") as fp:
            d = pickle.load(fp)
        data.extend(d)

    # NEW: if using mixed value targets, require v_mcts_root to be present
    v_mix = float(os.environ.get("AZ_V_MIX", "0.0"))
    if v_mix > 0.0:
        before = len(data)
        data = [s for s in data if "v_mcts_root" in s]
        after = len(data)
        print(f"filtered for v_mcts_root (AZ_V_MIX>0): {before} -> {after}")

        if after == 0:
            raise RuntimeError(
                "AZ_V_MIX>0 but no samples contain v_mcts_root. "
                "Generate new self-play data with v_mcts_root first, or set AZ_V_MIX=0."
            )

    # downsample to max_samples for speed/memory
    rng = random.Random(seed)
    if max_samples is not None and len(data) > max_samples:
        data = rng.sample(data, k=max_samples)

    print(f"replay samples: {len(data)} (from {len(use_files)} files)")
    return data


def save_checkpoint(net: AmazonsNet, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    print("saved checkpoint:", out_path)


def main():
    device = "cpu"
    v_mix = float(os.environ.get("AZ_V_MIX", "0.0"))  # 0.0=´¿z£¬0.3=»ìºÏ

    # Replay buffer settings
    last_n_files = int(os.environ.get("AZ_REPLAY_FILES", "5"))
    max_samples = int(os.environ.get("AZ_REPLAY_MAX_SAMPLES", "20000"))
    seed = int(os.environ.get("AZ_SEED", "0"))

    # Training settings
    epochs = int(os.environ.get("AZ_EPOCHS", "1"))
    lr = float(os.environ.get("AZ_LR", "1e-3"))
    weight_decay = float(os.environ.get("AZ_WD", "1e-4"))
    value_weight = float(os.environ.get("AZ_VALUE_W", "0.25"))
    grad_clip = float(os.environ.get("AZ_GRAD_CLIP", "1.0"))

    # Checkpoint naming
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load replay
    data = load_replay("data", last_n_files=last_n_files, max_samples=max_samples, seed=seed)

    # Init model
    net = AmazonsNet(channels=64).to(device)

    # Load current "best" if exists
    best_path = ckpt_dir / "model_best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        net.load_state_dict(state)
        print("loaded best checkpoint:", best_path)
    else:
        print("no model_best.pt found; training from scratch")

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        random.shuffle(data)

        net.train()
        total_loss = 0.0
        total_p = 0.0
        total_v = 0.0

        for i, s in enumerate(data):
            x_state = torch.from_numpy(s["state"]).unsqueeze(0).to(device)  # (1,4,10,10)
            x_actions = torch.from_numpy(s["actions"]).to(device)          # (A,3,10,10)
            target_pi = torch.from_numpy(s["pi"]).to(device)               # (A,)
            z = float(s["z"])
            if v_mix > 0.0 and "v_mcts_root" in s:
                v0 = float(s["v_mcts_root"])
                z = (1.0 - v_mix) * z + v_mix * v0
            target_z = torch.tensor([z], device=device)

            feat = net.forward_features(x_state)
            pred_v = net.value(feat)                                       # (1,)

            feat_rep = feat.repeat(x_actions.shape[0], 1, 1, 1)            # (A,C,10,10)
            logits = net.policy_logit(feat_rep, x_actions)                 # (A,)

            log_probs = F.log_softmax(logits, dim=0)                       # (A,)
            loss_p = -(target_pi * log_probs).sum()
            loss_v = F.mse_loss(pred_v, target_z)

            loss = loss_p + value_weight * loss_v

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()

            total_loss += float(loss.item())
            total_p += float(loss_p.item())
            total_v += float(loss_v.item())

            if (i + 1) % 200 == 0:
                print(
                    f"step {i+1}/{len(data)} "
                    f"loss={total_loss/(i+1):.4f} p={total_p/(i+1):.4f} v={total_v/(i+1):.4f}"
                )

        print(f"epoch {epoch} done. avg_loss={total_loss/len(data):.4f}")

    # Save as candidate
    cand_path = ckpt_dir / "model_candidate.pt"
    save_checkpoint(net, str(cand_path))


if __name__ == "__main__":
    main()