"""Quantitative eval for a GenTac checkpoint.

Picks N evenly-spaced decision frames from a match. For each one, samples K
alternatives (no arrows), then measures:

  ADE  — Average Displacement Error vs. ground-truth future, averaged over
         time and entities (lower = closer to GT).
  FDE  — Final Displacement Error at the last horizon frame.
  div  — inter-sample stddev of trajectories (a healthy diffusion model has
         div > 0; a collapsed model has div ≈ 0).
  vmax — max per-frame inter-frame speed across samples (m/s). Should be ≤
         player_max_speed (10.5) for plausibility.
  off  — fraction of (sample, frame, entity) cells where coords leave the
         pitch by > 2 m (the physics-clamp margin).

Then a second pass measures arrow-honoring: for each frame, draws one arrow at
GT's actual future position for a random entity and verifies the model's
generated trajectory hits within `arrow_tol_m` of the destination. CFG sweep
reports how arrow-honoring improves with guidance_scale.

Output: results table to stdout + machine-readable JSON to checkpoints/<name>/eval.json.

Usage:
    python scripts/eval.py                                   # default smoke ckpt
    python scripts/eval.py --ckpt checkpoints/full/last.ckpt
    python scripts/eval.py --n-frames 8 --k 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.config import GenTacConfig
from src.model.dataset import _load_match_arrays
from src.model.lightning_module import GenTacTrajectoryModule
from src.model.physics import PhysicsConfig, apply_physics
from src.model.samplers import build_static_target_mask, causal_rollout


PITCH_HALF_X = 105.0 / 2.0
PITCH_HALF_Y = 68.0 / 2.0
PLAYER_MAX_SPEED_MPS = 10.5
PITCH_OOB_MARGIN_M = 2.0


@dataclass
class MetricSummary:
    mean: float
    ci_low: float
    ci_high: float
    n: int


@dataclass
class EvalResult:
    ckpt: str
    n_frames: int
    k: int
    horizon_frames: int
    fps: int
    seed: int
    ade_m: MetricSummary
    fde_m: MetricSummary
    diversity_m: MetricSummary
    max_speed_mps: float                                # absolute max — no CI
    off_pitch_frac: MetricSummary
    arrow_honor: dict                                   # {guidance_scale: {...}}
    arrow_tol_m: float
    elapsed_s: float
    ade_by_role: dict = field(default_factory=dict)     # {role: mean ADE m} — above the paper
    realism_discriminator_acc: float = float("nan")     # 0.5 = indistinguishable from real — above the paper


@dataclass
class CompareResult:
    a: EvalResult
    b: EvalResult
    deltas: dict                                        # per-metric: {a_minus_b, p_value, n_paired}


def bootstrap_ci(samples: Iterable[float], n_resamples: int = 1000, alpha: float = 0.05,
                 rng: random.Random | None = None) -> MetricSummary:
    """Percentile bootstrap CI for the mean. Robust to small n + non-normal data."""
    xs = list(samples)
    n = len(xs)
    if n == 0:
        return MetricSummary(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)
    rng = rng or random.Random(0)
    means = []
    for _ in range(n_resamples):
        m = sum(rng.choice(xs) for _ in range(n)) / n
        means.append(m)
    means.sort()
    lo = means[int(n_resamples * alpha / 2)]
    hi = means[int(n_resamples * (1 - alpha / 2))]
    return MetricSummary(mean=sum(xs) / n, ci_low=lo, ci_high=hi, n=n)


def paired_bootstrap_p(a_samples: list[float], b_samples: list[float],
                       n_resamples: int = 2000, rng: random.Random | None = None) -> float:
    """Two-sided paired bootstrap p-value for H0: mean(a-b) = 0.

    Returns the fraction of resampled paired differences whose mean has the
    OPPOSITE sign of the observed difference, doubled (two-sided). For small n
    this beats a parametric t-test on non-normal metric distributions.
    """
    if len(a_samples) != len(b_samples) or not a_samples:
        return float("nan")
    rng = rng or random.Random(1)
    diffs = [a - b for a, b in zip(a_samples, b_samples)]
    observed_mean = sum(diffs) / len(diffs)
    if observed_mean == 0:
        return 1.0
    n = len(diffs)
    centered = [d - observed_mean for d in diffs]
    extreme = 0
    for _ in range(n_resamples):
        m = sum(rng.choice(centered) for _ in range(n)) / n
        if (observed_mean > 0 and m >= observed_mean) or (observed_mean < 0 and m <= observed_mean):
            extreme += 1
    return min(1.0, 2 * extreme / n_resamples)


def _denorm(arr: torch.Tensor) -> torch.Tensor:
    return arr * torch.tensor([PITCH_HALF_X, PITCH_HALF_Y], device=arr.device, dtype=arr.dtype)


def _sample_no_arrows(model, schedule, cfg, m, di, K, horizon, device):
    pos_all = torch.from_numpy(m["positions"]).to(device)
    mask_all = torch.from_numpy(m["mask"]).to(device)
    H = cfg.history_frames
    history = pos_all[di - H : di].unsqueeze(0).expand(K, -1, -1, -1).contiguous()
    valid_static = mask_all[di - 1].unsqueeze(0).expand(K, -1).contiguous()
    role_idx = torch.from_numpy(m["role_idx"]).to(device).unsqueeze(0).expand(K, -1)
    target_mask = build_static_target_mask(cfg, cfg.n_entities, "unconditioned", device).expand(K, cfg.n_entities)
    samples_norm = causal_rollout(model, schedule, history, valid_static, horizon, target_mask, role_idx=role_idx)
    samples_m = _denorm(samples_norm).cpu()
    history_m = _denorm(history[0]).cpu()
    valid_static_cpu = valid_static[0].cpu()
    samples_m = apply_physics(samples_m, history_m[-1], valid_static_cpu, cfg, PhysicsConfig())
    return samples_m, valid_static_cpu                            # (K, T, n_ent, 2), (n_ent,)


def _sample_with_arrow(model, schedule, cfg, m, di, K, horizon, device, ent, target_xy_m, scale):
    pos_all = torch.from_numpy(m["positions"]).to(device)
    mask_all = torch.from_numpy(m["mask"]).to(device)
    H = cfg.history_frames
    history = pos_all[di - H : di].unsqueeze(0).expand(K, -1, -1, -1).contiguous()
    valid_static = mask_all[di - 1].unsqueeze(0).expand(K, -1).contiguous()
    role_idx = torch.from_numpy(m["role_idx"]).to(device).unsqueeze(0).expand(K, -1)
    target_mask = build_static_target_mask(cfg, cfg.n_entities, "unconditioned", device).expand(K, cfg.n_entities)
    waypoint = (ent, horizon - 1, [target_xy_m[0] / PITCH_HALF_X, target_xy_m[1] / PITCH_HALF_Y])
    samples_norm = causal_rollout(
        model, schedule, history, valid_static, horizon, target_mask,
        waypoints=[waypoint],
        waypoint_fade_frames=0,             # measure the LEARNED signal cleanly, no hard-pin fade
        guidance_scale=scale,
        role_idx=role_idx,
    )
    samples_m = _denorm(samples_norm).cpu()
    history_m = _denorm(history[0]).cpu()
    valid_static_cpu = valid_static[0].cpu()
    samples_m = apply_physics(samples_m, history_m[-1], valid_static_cpu, cfg, PhysicsConfig())
    return samples_m, valid_static_cpu


def _displacement_errors(samples_m: torch.Tensor, actual_m: torch.Tensor, valid: torch.Tensor):
    """ADE/FDE in meters. samples (K,T,n_ent,2), actual (T,n_ent,2), valid (n_ent,)."""
    K, T, n_ent, _ = samples_m.shape
    valid_e = valid.unsqueeze(0).unsqueeze(0).expand(K, T, n_ent).float()
    diff = (samples_m - actual_m.unsqueeze(0)).norm(dim=-1)        # (K, T, n_ent)
    ade = (diff * valid_e).sum() / valid_e.sum().clamp(min=1)
    valid_final = valid.float().unsqueeze(0).expand(K, n_ent)      # (K, n_ent)
    fde = (diff[:, -1] * valid_final).sum() / valid_final.sum().clamp(min=1)
    return float(ade), float(fde)


# ── Per-role tactical breakdown (above the paper) ──────────────────────────
# Paper reports aggregate ADE/FDE. We additionally bucket players by tactical
# role inferred from their average X position in the history, and report
# per-role metrics. This surfaces "the model gets defenders right but loses
# attackers" patterns that aggregate metrics hide.
def _infer_roles(history_last_m: torch.Tensor, n_players_per_team: int) -> dict[str, list[int]]:
    """Return entity-slot indices grouped by tactical role.

    Heuristic: sort each team's players by their X position (long-axis of pitch).
    For team0 (attacking left-to-right): leftmost = GK, next 4 = defenders,
    middle 3 = midfielders, rightmost 3 = forwards. Team1 mirrored.
    """
    N = n_players_per_team
    roles: dict[str, list[int]] = {"gk": [], "defender": [], "midfielder": [], "forward": [], "ball": []}
    for team_offset, sign in [(0, 1), (N, -1)]:
        xs = history_last_m[team_offset:team_offset + N, 0]
        order = (xs * sign).argsort().tolist()       # ascending: nearest own goal first
        roles["gk"].append(team_offset + order[0])
        roles["defender"].extend(team_offset + i for i in order[1:5])
        roles["midfielder"].extend(team_offset + i for i in order[5:8])
        roles["forward"].extend(team_offset + i for i in order[8:11])
    roles["ball"].append(2 * N)
    return roles


# ── Realism discriminator (above the paper) ────────────────────────────────
# Train a tiny MLP to classify (real window) vs (generated window). Its test
# accuracy is a realism proxy:
#   • 0.50 → samples are indistinguishable from real (best)
#   • 1.00 → discriminator trivially separates them (worst)
# Paper has no equivalent — they report ADE/FDE only, which can be low even
# for samples that look obviously synthetic (e.g. straight-line motion).
class _RealismMLP(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h), torch.nn.GELU(),
            torch.nn.Linear(dim_h, dim_h), torch.nn.GELU(),
            torch.nn.Linear(dim_h, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def discriminator_realism(real_windows: torch.Tensor, fake_windows: torch.Tensor,
                          epochs: int = 8, seed: int = 0) -> float:
    """Train on first 80%, return test accuracy on last 20%. 0.5 = perfect realism."""
    torch.manual_seed(seed)
    K_real, T, n_ent, _ = real_windows.shape
    K_fake = fake_windows.shape[0]
    # Flatten each window into a feature vector (positions). Crude but effective.
    real_x = real_windows.reshape(K_real, -1)
    fake_x = fake_windows.reshape(K_fake, -1)
    X = torch.cat([real_x, fake_x], dim=0)
    y = torch.cat([torch.ones(K_real), torch.zeros(K_fake)], dim=0)
    perm = torch.randperm(len(y))
    X, y = X[perm], y[perm]
    n_test = max(2, len(y) // 5)
    X_tr, X_te = X[n_test:], X[:n_test]
    y_tr, y_te = y[n_test:], y[:n_test]
    if len(X_tr) < 4:
        return float("nan")
    # Per-feature normalization so MLP isn't dominated by absolute scale
    mu, sd = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp(min=1e-6)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    mlp = _RealismMLP(X_tr.shape[1])
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = mlp(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (mlp(X_te) > 0).float()
        acc = (pred == y_te).float().mean().item()
    return acc


def _per_role_ade(samples_m: torch.Tensor, actual_m: torch.Tensor,
                  valid: torch.Tensor, roles: dict[str, list[int]]) -> dict[str, float]:
    """Mean ADE per tactical role over the K, T axes."""
    out = {}
    K, T, _, _ = samples_m.shape
    diff = (samples_m - actual_m.unsqueeze(0)).norm(dim=-1)        # (K, T, n_ent)
    for role, ents in roles.items():
        if not ents:
            continue
        ent_t = torch.tensor(ents, dtype=torch.long)
        role_diff = diff[:, :, ent_t]                              # (K, T, |ents|)
        role_valid = valid[ent_t].float().view(1, 1, -1).expand_as(role_diff)
        denom = role_valid.sum().clamp(min=1)
        out[role] = float((role_diff * role_valid).sum() / denom)
    return out


def _max_speed(samples_m: torch.Tensor, fps: int) -> float:
    # entity 0..21 are players (skip ball at index 22) for the speed test
    n_players = samples_m.shape[2] - 1
    step_m = (samples_m[:, 1:, :n_players] - samples_m[:, :-1, :n_players]).norm(dim=-1)
    return float(step_m.max() * fps)


def _off_pitch_frac(samples_m: torch.Tensor) -> float:
    off = (samples_m[..., 0].abs() > PITCH_HALF_X + PITCH_OOB_MARGIN_M) | \
          (samples_m[..., 1].abs() > PITCH_HALF_Y + PITCH_OOB_MARGIN_M)
    return float(off.float().mean())


def evaluate(ckpt_path: Path, args: argparse.Namespace) -> tuple[EvalResult, dict[str, list[float]]]:
    """Run the full eval against one checkpoint. Returns the result plus per-frame
    raw values for ADE/FDE/diversity/off-pitch (needed for paired comparison).
    """
    root = Path(__file__).resolve().parents[1]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print(f"[eval] loading {ckpt_path} on {device}")
    cfg = GenTacTrajectoryModule.cfg_from_checkpoint(ckpt_path)
    mod = GenTacTrajectoryModule.load_from_checkpoint(str(ckpt_path), cfg=cfg).to(device).eval()
    schedule = mod.schedule.to(device)

    m = _load_match_arrays(root / args.match, cfg)
    pos_all = torch.from_numpy(m["positions"])
    mask_all = torch.from_numpy(m["mask"])
    n_frames_total = len(m["frame_ids"])
    H = cfg.history_frames

    lo, hi = H, n_frames_total - args.horizon - 1
    di_list = torch.linspace(lo, hi, args.n_frames).long().tolist()
    print(f"[eval] sampling at {len(di_list)} decision frames, K={args.k}, horizon={args.horizon}, seed={args.seed}")

    t0 = time.time()

    ades, fdes, divs, vmaxes, off_fracs = [], [], [], [], []
    per_role_acc: dict[str, list[float]] = {}
    history_meters_for_roles = (pos_all[di_list[0] - 1] *
                                torch.tensor([PITCH_HALF_X, PITCH_HALF_Y])).float()
    roles = _infer_roles(history_meters_for_roles, cfg.n_players_per_team)

    # Collect real and fake horizon windows for the discriminator.
    real_windows_list: list[torch.Tensor] = []
    fake_windows_list: list[torch.Tensor] = []

    for di in di_list:
        samples_m, valid = _sample_no_arrows(mod.model, schedule, cfg, m, di, args.k, args.horizon, device)
        actual = (pos_all[di : di + args.horizon] *
                  torch.tensor([PITCH_HALF_X, PITCH_HALF_Y])).float()
        ade, fde = _displacement_errors(samples_m, actual, valid)
        div = float(samples_m.std(dim=0).mean())
        vmax = _max_speed(samples_m, cfg.fps)
        off = _off_pitch_frac(samples_m)
        role_ades = _per_role_ade(samples_m, actual, valid, roles)
        for r, v in role_ades.items():
            per_role_acc.setdefault(r, []).append(v)
        ades.append(ade); fdes.append(fde); divs.append(div); vmaxes.append(vmax); off_fracs.append(off)
        real_windows_list.append(actual.unsqueeze(0))                       # (1, T, n_ent, 2)
        fake_windows_list.append(samples_m)                                 # (K, T, n_ent, 2)
        print(f"  frame {di:>6}  ADE={ade:.2f}m  FDE={fde:.2f}m  div={div:.3f}m  vmax={vmax:.1f}m/s  off={off:.3f}")

    ade_by_role = {r: sum(v) / len(v) for r, v in per_role_acc.items()}
    print("  per-role ADE: " + "  ".join(f"{r}={ade_by_role[r]:.2f}m" for r in ["gk","defender","midfielder","forward","ball"] if r in ade_by_role))

    real_w = torch.cat(real_windows_list, dim=0)
    fake_w = torch.cat(fake_windows_list, dim=0)
    realism_acc = discriminator_realism(real_w, fake_w, seed=args.seed)
    print(f"  realism discriminator test acc: {realism_acc:.3f}  (0.50 = indistinguishable, 1.00 = trivially separable)")

    ade_summary = bootstrap_ci(ades, args.n_bootstrap, args.alpha, rng)
    fde_summary = bootstrap_ci(fdes, args.n_bootstrap, args.alpha, rng)
    div_summary = bootstrap_ci(divs, args.n_bootstrap, args.alpha, rng)
    off_summary = bootstrap_ci(off_fracs, args.n_bootstrap, args.alpha, rng)
    vmax_overall = max(vmaxes) if vmaxes else float("nan")

    pct = int(100 * (1 - args.alpha))
    print(f"\n[eval] aggregate (no arrows)  [{pct}% CIs via {args.n_bootstrap}-resample bootstrap]")
    print(f"       ADE {ade_summary.mean:.2f} m  [{ade_summary.ci_low:.2f}, {ade_summary.ci_high:.2f}]")
    print(f"       FDE {fde_summary.mean:.2f} m  [{fde_summary.ci_low:.2f}, {fde_summary.ci_high:.2f}]")
    print(f"       diversity {div_summary.mean:.3f} m  [{div_summary.ci_low:.3f}, {div_summary.ci_high:.3f}]")
    print(f"       max speed {vmax_overall:.1f} m/s (player cap {PLAYER_MAX_SPEED_MPS})")
    print(f"       off-pitch {off_summary.mean:.4f}  [{off_summary.ci_low:.4f}, {off_summary.ci_high:.4f}]")

    scales = [float(s) for s in args.guidance_scales.split(",")]
    arrow_results: dict[str, dict] = {}
    for scale in scales:
        ades_to_target: list[float] = []
        hits = 0
        total = 0
        for di in di_list:
            valid_now = mask_all[di - 1]
            valid_idx = valid_now.nonzero().flatten().tolist()
            player_idx = [i for i in valid_idx if i < cfg.n_entities - 1]
            if not player_idx:
                continue
            ent = player_idx[di % len(player_idx)]
            target_xy_m = (pos_all[di + args.horizon - 1, ent] *
                           torch.tensor([PITCH_HALF_X, PITCH_HALF_Y])).tolist()
            samples_m, _ = _sample_with_arrow(mod.model, schedule, cfg, m, di, args.k,
                                              args.horizon, device, ent, target_xy_m, scale)
            final = samples_m[:, -1, ent]
            target_t = torch.tensor(target_xy_m)
            err = (final - target_t).norm(dim=-1)
            ades_to_target.extend(err.tolist())
            hits += int((err < args.arrow_tol_m).sum())
            total += err.numel()
        if total == 0:
            continue
        err_summary = bootstrap_ci(ades_to_target, args.n_bootstrap, args.alpha, rng)
        arrow_results[str(scale)] = {
            "ade_to_target": asdict(err_summary),
            "hit_rate_within_tol": hits / total,
            "n_attempts": total,
        }
        print(f"  arrow scale={scale:>4}: err {err_summary.mean:.2f} m "
              f"[{err_summary.ci_low:.2f}, {err_summary.ci_high:.2f}]  "
              f"hit@{args.arrow_tol_m}m {hits/total:.0%} ({hits}/{total})")

    elapsed = time.time() - t0
    root_rel = ckpt_path.relative_to(root) if ckpt_path.is_relative_to(root) else ckpt_path
    result = EvalResult(
        ckpt=str(root_rel), n_frames=len(di_list), k=args.k, horizon_frames=args.horizon,
        fps=cfg.fps, seed=args.seed,
        ade_m=ade_summary, fde_m=fde_summary, diversity_m=div_summary,
        max_speed_mps=vmax_overall, off_pitch_frac=off_summary,
        arrow_honor=arrow_results, arrow_tol_m=args.arrow_tol_m, elapsed_s=elapsed,
        ade_by_role=ade_by_role,
        realism_discriminator_acc=realism_acc,
    )
    raw = {"ade": ades, "fde": fdes, "diversity": divs, "off_pitch": off_fracs}
    return result, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/smoke/smoke_final.ckpt")
    ap.add_argument("--compare", default=None,
                    help="Optional second checkpoint to evaluate in parallel; reports paired deltas with p-values.")
    ap.add_argument("--match", default="data/processed/Sample_Game_1.json")
    ap.add_argument("--n-frames", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--arrow-tol-m", type=float, default=2.0)
    ap.add_argument("--guidance-scales", default="1.0,2.0,3.5")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05, help="2-sided CI alpha (0.05 → 95% CI)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_a = root / args.ckpt

    a, raw_a = evaluate(ckpt_a, args)
    out_a = ckpt_a.parent / "eval.json"
    out_a.parent.mkdir(parents=True, exist_ok=True)
    out_a.write_text(json.dumps(asdict(a), indent=2))
    print(f"\n[eval] wrote {out_a}")

    if args.compare:
        ckpt_b = root / args.compare
        b, raw_b = evaluate(ckpt_b, args)
        out_b = ckpt_b.parent / "eval.json"
        out_b.parent.mkdir(parents=True, exist_ok=True)
        out_b.write_text(json.dumps(asdict(b), indent=2))

        rng = random.Random(args.seed + 1)
        deltas = {}
        for k in ("ade", "fde", "diversity", "off_pitch"):
            d = sum(x - y for x, y in zip(raw_a[k], raw_b[k])) / len(raw_a[k])
            p = paired_bootstrap_p(raw_a[k], raw_b[k], args.n_bootstrap * 2, rng)
            deltas[k] = {"a_minus_b": d, "p_value": p, "n_paired": len(raw_a[k])}

        cmp = CompareResult(a=a, b=b, deltas=deltas)
        cmp_path = root / "checkpoints" / "compare.json"
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        cmp_path.write_text(json.dumps(asdict(cmp), indent=2))

        print("\n[compare]  A − B  (lower is better for ADE/FDE/off; higher is better for diversity)")
        for k, d in deltas.items():
            star = "***" if d["p_value"] < 0.01 else ("**" if d["p_value"] < 0.05 else ("*" if d["p_value"] < 0.1 else "ns"))
            print(f"  {k:>12}  Δ={d['a_minus_b']:+.4f}  p={d['p_value']:.3f}  [{star}]")
        print(f"[compare] wrote {cmp_path}")


if __name__ == "__main__":
    main()
