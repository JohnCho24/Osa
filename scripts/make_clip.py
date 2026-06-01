"""Regenerate renderer clips that are consistent with the model's roster.

The renderer's default view loads `data/processed/<game>_clip.json`. Those clips
must use the SAME 11-slot-per-team roster the model/server use, otherwise the
demo lets a coach click a player the model doesn't represent and Generate 400s
with "unknown player".

The model collapses a whole match into a fixed roster: the 11 most-present
players per team (see src/model/dataset._build_slot_map). A hand-sliced clip that
keeps whoever was physically on the pitch can therefore include a substituted
starter (e.g. Player1) who is NOT in that match-wide roster.

This script slices a window in which all 22 slot players AND the ball are present
for every frame, emits ONLY slot players (with their real IDs, in meters), and so
guarantees: rendered roster == server roster == model roster.

Usage:
    python scripts/make_clip.py                 # all Sample_Game_*.json
    python scripts/make_clip.py Sample_Game_1   # one match
    python scripts/make_clip.py --frames 750    # window length (default 750 = 30s)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.config import GenTacConfig
from src.model.dataset import _build_slot_map

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def _find_window(frames: dict, t0: list[str], t1: list[str], n: int) -> int:
    """First/best start key for an n-frame window with all 22 slots + ball present.

    Scans contiguous integer frame ids. Picks the window maximizing ball-present
    frames among those where every slot player is present in every frame.
    """
    keys = sorted(int(k) for k in frames)
    keyset = set(keys)

    def full22(j: int) -> bool:
        f = frames.get(str(j))
        if not f:
            return False
        a = f.get("team0") or {}
        b = f.get("team1") or {}
        return all(p in a for p in t0) and all(p in b for p in t1)

    best_start, best_ball = None, -1
    i = 0
    while i < len(keys):
        start = keys[i]
        # contiguous integer run starting here, all full22
        end = start
        while (end + 1) in keyset and full22(end + 1) and full22(start):
            end += 1
        run_len = end - start + 1 if full22(start) else 0
        if run_len >= n:
            # slide n-window across [start, end-n+1], score by ball coverage
            s = start
            while s + n - 1 <= end:
                ball = sum(1 for j in range(s, s + n) if frames[str(j)].get("ball"))
                if ball > best_ball:
                    best_ball, best_start = ball, s
                s += max(1, n // 3)
            i = keys.index(end) + 1 if end in keyset else i + 1
        else:
            i += 1
    if best_start is None:
        raise SystemExit("no window with all 22 slot players present found")
    return best_start


def make_clip(game: str, n: int) -> None:
    src = PROC / f"{game}.json"
    if not src.exists():
        raise SystemExit(f"{src} not found")
    cfg = GenTacConfig(smoke=True)
    N = cfg.n_players_per_team
    full = json.loads(src.read_text())
    frames = full["frames"]
    t0 = _build_slot_map(frames, "team0", N)
    t1 = _build_slot_map(frames, "team1", N)

    start = _find_window(frames, t0, t1, n)
    out_frames: dict[str, dict] = {}
    ball_missing = 0
    for j in range(start, start + n):
        f = frames[str(j)]
        ball = f.get("ball")
        if not ball:
            ball_missing += 1
        out_frames[str(j)] = {
            "period": f.get("period", 1),
            "ball": ball if ball else None,
            "team0": {p: f["team0"][p] for p in t0 if p in f.get("team0", {})},
            "team1": {p: f["team1"][p] for p in t1 if p in f.get("team1", {})},
        }

    meta = dict(full["metadata"])
    meta["team0"] = {"name": meta.get("team0", {}).get("name", "Home"), "players": t0}
    meta["team1"] = {"name": meta.get("team1", {}).get("name", "Away"), "players": t1}
    meta["n_frames"] = n
    meta["clip_start"] = start
    meta["clip_end"] = start + n

    out = {"metadata": meta, "frames": out_frames}
    dst = PROC / f"{game}_clip.json"
    dst.write_text(json.dumps(out))
    print(
        f"{game}: wrote {dst.name}  frames {start}..{start + n - 1}  "
        f"team0={t0}  team1={t1}  ball_missing={ball_missing}/{n}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", help="game id(s), e.g. Sample_Game_1")
    ap.add_argument("--frames", type=int, default=750, help="clip length in frames")
    args = ap.parse_args()
    games = args.games or [p.stem for p in sorted(PROC.glob("Sample_Game_*.json")) if "_clip" not in p.stem]
    for g in games:
        make_clip(g, args.frames)


if __name__ == "__main__":
    main()
