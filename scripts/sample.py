"""Sample K alternative futures from a trained GenTac checkpoint.

Loads a Lightning checkpoint, picks a "decision moment" (frame index in a match),
runs K causal-rollouts, denormalizes back to meters, and writes a JSON the
tactics-board renderer can consume.

Output schema (the renderer expects this in M3):
{
  "metadata": {...},
  "history": {"<frame_id>": {...}},     # the H frames before the decision moment
  "samples":  [ {"<frame_id>": {...}}, ... ]    # K alternative futures (length-w * n_rollout_steps)
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.config import GenTacConfig
from src.model.dataset import _load_match_arrays
from src.model.lightning_module import GenTacTrajectoryModule
from src.model.physics import PhysicsConfig, apply_physics
from src.model.samplers import build_static_target_mask, causal_rollout


def _denormalize(arr: torch.Tensor, cfg: GenTacConfig) -> torch.Tensor:
    """Inverse of dataset normalization. arr: (..., 2) in [-1, 1] → meters."""
    scale = torch.tensor([cfg.pitch_x / 2.0, cfg.pitch_y / 2.0], device=arr.device, dtype=arr.dtype)
    return arr * scale


def _frame_to_dict(coords: torch.Tensor, valid: torch.Tensor, team0_slots, team1_slots, N: int, period: int):
    """coords (n_ent, 2) tensor in METERS, valid (n_ent,) bool → JSON frame dict."""
    out = {"period": int(period), "ball": None, "team0": {}, "team1": {}}
    for slot, pid in enumerate(team0_slots):
        if valid[slot]:
            out["team0"][pid] = [round(float(coords[slot, 0]), 2), round(float(coords[slot, 1]), 2)]
    for slot, pid in enumerate(team1_slots):
        if valid[N + slot]:
            out["team1"][pid] = [round(float(coords[N + slot, 0]), 2), round(float(coords[N + slot, 1]), 2)]
    if valid[2 * N]:
        out["ball"] = [round(float(coords[2 * N, 0]), 2), round(float(coords[2 * N, 1]), 2)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/smoke/smoke_final.ckpt")
    ap.add_argument("--match", default="data/processed/Sample_Game_1.json")
    ap.add_argument("--decision-frame", type=int, default=15000,
                    help="absolute frame number in the match; history = the preceding H frames")
    ap.add_argument("--horizon", type=int, default=25, help="future horizon in frames (multiple of w)")
    ap.add_argument("--k", type=int, default=20, help="number of alternative futures to sample")
    ap.add_argument("--mode", default="unconditioned",
                    choices=["unconditioned", "opp_conditioned_team0", "opp_conditioned_team1"])
    ap.add_argument("--out", default="data/processed/samples.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-physics", action="store_true", help="skip the physical-plausibility post-processor")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = (
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    print(f"device: {device}")

    # ── load checkpoint + config ──────────────────────────────────────────
    ckpt_path = root / args.ckpt
    mod = GenTacTrajectoryModule.load_from_checkpoint(str(ckpt_path), cfg=GenTacConfig(smoke=True))
    mod.to(device).eval()
    cfg = mod.cfg
    schedule = mod.schedule.to(device)
    model = mod.model
    N = cfg.n_players_per_team
    H = cfg.history_frames
    w = cfg.window_frames

    if args.horizon % w != 0:
        raise ValueError(f"--horizon must be multiple of window size w={w}")

    # ── load match arrays ─────────────────────────────────────────────────
    match_path = root / args.match
    arrs = _load_match_arrays(match_path, cfg)
    pos_all = torch.from_numpy(arrs["positions"]).to(device)                 # (F, 23, 2), normalized
    mask_all = torch.from_numpy(arrs["mask"]).to(device)                     # (F, 23)
    frame_ids = arrs["frame_ids"]
    periods = arrs["period"]
    team0_slots = arrs["team0_slots"]
    team1_slots = arrs["team1_slots"]

    # find the array index whose frame_id == decision_frame
    matches = (frame_ids == args.decision_frame).nonzero()[0]
    if len(matches) == 0:
        raise ValueError(f"frame {args.decision_frame} not in match")
    di = int(matches[0])
    if di < H:
        raise ValueError(f"need at least {H} preceding frames; have {di}")
    if di + args.horizon > len(frame_ids):
        raise ValueError(f"horizon {args.horizon} exceeds match length")

    history = pos_all[di - H : di].unsqueeze(0).expand(args.k, -1, -1, -1).contiguous()    # (K, H, 23, 2)
    valid_static = mask_all[di - 1].unsqueeze(0).expand(args.k, -1).contiguous()           # (K, 23)
    target_mask = build_static_target_mask(cfg, cfg.n_entities, args.mode, device).expand(args.k, cfg.n_entities)

    # opponent future: pulled from GT for the relevant slots (irrelevant for unconditioned)
    opp_future = None
    if args.mode != "unconditioned":
        opp_future = pos_all[di : di + args.horizon].unsqueeze(0).expand(args.k, -1, -1, -1).contiguous()

    print(f"history: {tuple(history.shape)}  horizon: {args.horizon} frames  k={args.k}  mode={args.mode}")

    # ── sample ────────────────────────────────────────────────────────────
    samples = causal_rollout(
        model, schedule, history, valid_static, args.horizon, target_mask,
        opponent_future_full=opp_future,
    )                                                                                      # (K, horizon, 23, 2)

    # for opp-cond modes, overlay the GT opponent future so JSON shows the real values for them
    if args.mode != "unconditioned":
        mix = target_mask.view(args.k, 1, cfg.n_entities, 1).expand_as(samples)
        samples = torch.where(mix, samples, opp_future)

    # denormalize to meters
    samples_m = _denormalize(samples, cfg).cpu()
    history_m = _denormalize(history[0], cfg).cpu()                                        # (H, 23, 2)
    valid_history = mask_all[di - H : di].cpu()                                            # (H, 23)
    valid_static_cpu = valid_static[0].cpu()                                               # (23,)

    # physical-plausibility post-processor
    if not args.no_physics:
        pre_stddev = samples_m[:, -1].std(dim=0).mean().item()
        samples_m = apply_physics(samples_m, history_m[-1], valid_static_cpu, cfg, PhysicsConfig())
        post_stddev = samples_m[:, -1].std(dim=0).mean().item()
        print(f"physics post-processor: final-frame stddev {pre_stddev:.2f} → {post_stddev:.2f} m")

    # ── build output JSON ────────────────────────────────────────────────
    history_frames = {}
    for i in range(H):
        fid = int(frame_ids[di - H + i])
        history_frames[str(fid)] = _frame_to_dict(
            history_m[i], valid_history[i], team0_slots, team1_slots, N, periods[di - H + i],
        )

    sample_list = []
    for k in range(args.k):
        out_k = {}
        for i in range(args.horizon):
            fid = int(frame_ids[di + i])
            out_k[str(fid)] = _frame_to_dict(
                samples_m[k, i], valid_static_cpu, team0_slots, team1_slots, N, periods[di + i],
            )
        sample_list.append(out_k)

    # ground-truth future, denormalized — the "actual" outcome we compare against
    actual_future_m = _denormalize(pos_all[di : di + args.horizon], cfg).cpu()
    valid_actual = mask_all[di : di + args.horizon].cpu()
    actual_future = {}
    for i in range(args.horizon):
        fid = int(frame_ids[di + i])
        actual_future[str(fid)] = _frame_to_dict(
            actual_future_m[i], valid_actual[i], team0_slots, team1_slots, N, periods[di + i],
        )

    payload = {
        "metadata": {
            "source": "gentac_sample",
            "ckpt": str(ckpt_path.relative_to(root)),
            "match": str(match_path.relative_to(root)),
            "decision_frame": args.decision_frame,
            "history_frames": H,
            "horizon_frames": args.horizon,
            "k": args.k,
            "mode": args.mode,
            "fps": cfg.fps,
            "pitch": {"x": cfg.pitch_x, "y": cfg.pitch_y, "origin": "center", "y_axis": "up"},
            "team0": {"name": "Home", "players": team0_slots},
            "team1": {"name": "Away", "players": team1_slots},
        },
        "history": history_frames,
        "actual_future": actual_future,
        "samples": sample_list,
    }

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")

    # quick sanity: per-sample range and inter-sample diversity
    coords_m = samples_m.flatten(0, 1)                                                      # (K*horizon, 23, 2)
    print(f"sample coords range x: [{coords_m[..., 0].min():.1f}, {coords_m[..., 0].max():.1f}]")
    print(f"sample coords range y: [{coords_m[..., 1].min():.1f}, {coords_m[..., 1].max():.1f}]")
    # diversity: stddev across the K dimension at the final frame, target slots only
    final = samples_m[:, -1]                                                                # (K, 23, 2)
    spread = final.std(dim=0).mean().item()
    print(f"inter-sample stddev at final frame (avg over entities): {spread:.2f} m")


if __name__ == "__main__":
    main()
