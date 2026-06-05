"""GenTac trajectory dataset.

Each sample = a (history_frames + window_frames)-length slice of normalized 2D coords
for 23 entities (team0[0:11], team1[0:11], ball), plus a per-(frame, entity) validity mask.

Coords are normalized to [-1, 1] (paper §6.3.1: "All 2D pitch coordinates are linearly
normalized to [-1, 1]."). Missing entities are zero-filled and masked=False.

Slot assignment:
  - Per match, pick the 11 most-present players per team (by frame count).
  - Sort the chosen 11 by player id (numerically), assign slot indices 0..10.
  - This gives a stable per-match identity for the entity embedding to learn.
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import GenTacConfig


def _player_num(pid: str) -> int:
    m = re.search(r"(\d+)", pid)
    return int(m.group(1)) if m else 0


def _build_slot_map(frames_dict: dict, team_key: str, n_slots: int) -> List[str]:
    """Pick the n_slots most-present players in `team_key`, return ordered slot list."""
    counts: dict[str, int] = {}
    for f in frames_dict.values():
        for pid in f.get(team_key, {}):
            counts[pid] = counts.get(pid, 0) + 1
    chosen = sorted(counts, key=lambda p: counts[p], reverse=True)[:n_slots]
    return sorted(chosen, key=_player_num)


def _load_match_arrays(json_path: Path, cfg: GenTacConfig):
    """Convert one match JSON into (positions, mask, period, frame_ids) numpy arrays.

    positions: (n_frames, n_entities, 2)  float32, normalized to [-1, 1]
    mask:      (n_frames, n_entities)     bool
    period:    (n_frames,)                int8
    frame_ids: (n_frames,)                int32 (the original frame numbers)
    """
    data = json.loads(json_path.read_text())
    frames_dict = data["frames"]
    N = cfg.n_players_per_team
    n_ent = 2 * N + 1

    team0_slots = _build_slot_map(frames_dict, "team0", N)
    team1_slots = _build_slot_map(frames_dict, "team1", N)

    sorted_keys = sorted(frames_dict.keys(), key=int)
    F = len(sorted_keys)
    positions = np.zeros((F, n_ent, 2), dtype=np.float32)
    mask = np.zeros((F, n_ent), dtype=bool)
    period = np.zeros(F, dtype=np.int8)
    frame_ids = np.zeros(F, dtype=np.int32)

    sx = cfg.pitch_x / 2.0
    sy = cfg.pitch_y / 2.0

    for i, k in enumerate(sorted_keys):
        f = frames_dict[k]
        period[i] = f.get("period", 0)
        frame_ids[i] = int(k)
        for slot, pid in enumerate(team0_slots):
            xy = f.get("team0", {}).get(pid)
            if xy is not None:
                positions[i, slot, 0] = xy[0] / sx
                positions[i, slot, 1] = xy[1] / sy
                mask[i, slot] = True
        for slot, pid in enumerate(team1_slots):
            xy = f.get("team1", {}).get(pid)
            if xy is not None:
                positions[i, N + slot, 0] = xy[0] / sx
                positions[i, N + slot, 1] = xy[1] / sy
                mask[i, N + slot] = True
        ball = f.get("ball")
        if ball is not None:
            positions[i, 2 * N, 0] = ball[0] / sx
            positions[i, 2 * N, 1] = ball[1] / sy
            mask[i, 2 * N] = True

    return {
        "positions": positions,
        "mask": mask,
        "period": period,
        "frame_ids": frame_ids,
        "team0_slots": team0_slots,
        "team1_slots": team1_slots,
        "game_id": data["metadata"]["game_id"],
    }


class TrajectoryDataset(Dataset):
    """Yields (history + future)-length windows for diffusion training.

    Each sample dict:
      'history':  (H, n_entities, 2)   float32, normalized
      'future':   (w, n_entities, 2)   float32, normalized (the noise target)
      'mask':     (H + w, n_entities)  bool
    """

    def __init__(self, json_paths: List[Path], cfg: GenTacConfig, stride: int | None = None):
        self.cfg = cfg
        self.matches = [_load_match_arrays(p, cfg) for p in json_paths]
        self.H = cfg.history_frames
        self.w = cfg.window_frames
        self.stride = stride or cfg.train_stride_frames
        self.windows = self._index_windows()

    def _index_windows(self) -> List[Tuple[int, int]]:
        windows = []
        L = self.H + self.w
        for mi, m in enumerate(self.matches):
            F = len(m["positions"])
            for start in range(0, F - L + 1, self.stride):
                # skip windows that cross a period boundary
                if m["period"][start] != m["period"][start + L - 1]:
                    continue
                windows.append((mi, start))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        mi, start = self.windows[idx]
        m = self.matches[mi]
        pos = m["positions"][start : start + self.H + self.w]   # (H+w, 23, 2)
        mask = m["mask"][start : start + self.H + self.w]       # (H+w, 23)
        return {
            "history": torch.from_numpy(pos[: self.H].copy()),
            "future":  torch.from_numpy(pos[self.H :].copy()),
            "mask":    torch.from_numpy(mask.copy()),
        }


if __name__ == "__main__":
    # smoke test
    root = Path(__file__).resolve().parents[2]
    cfg = GenTacConfig(smoke=True)
    paths = sorted((root / "data" / "processed").glob("Sample_Game_*.json"))
    paths = [p for p in paths if "_clip" not in p.name]
    print(f"matches found: {[p.name for p in paths]}")
    ds = TrajectoryDataset(paths, cfg)
    print(f"windows: {len(ds)}")
    s = ds[0]
    print(f"history {s['history'].shape}  future {s['future'].shape}  mask {s['mask'].shape}")
    print(f"history coord range: [{s['history'].min():.3f}, {s['history'].max():.3f}]")
    print(f"mask True frac: {s['mask'].float().mean():.3f}")
