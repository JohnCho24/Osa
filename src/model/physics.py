"""Physical-plausibility post-processor for GenTac trajectory samples.

Real soccer players have physical limits. A raw diffusion sample can violate them in
ways that ruin both the analyst's trust and the downstream video generator's output.
We apply a thin, deterministic post-processing pipeline:

  1. Anchor the first sampled frame to the last observed history frame, so there's no
     teleport at the seam between history and prediction.
  2. Per-frame max-speed clip: cap |Δposition| / Δt at v_max (m/s) per entity class.
  3. Exponential smoothing along the time axis to remove jitter.
  4. On-pitch clamp: allow brief excursions but keep within (pitch + margin).
  5. (Optional) Pairwise inter-player repulsion: push apart any two players whose
     distance falls below r_min.

These constraints reflect realistic soccer kinematics — they make undertrained model
output legible without hiding model errors. ADE/FDE versus ground truth are still
the source of truth for model quality; this layer is presentation-only.

References used for constants:
  - Top sprint speed elite footballers: ~10–11 m/s (Salah, Mbappé measured ~12 m/s briefly)
  - Ball speed: shots can hit 35 m/s, normal play 5–20 m/s
  - Inter-player minimum spacing: ~0.6 m (no two players sharing a square meter)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GenTacConfig


@dataclass
class PhysicsConfig:
    v_max_player: float = 10.5          # m/s
    v_max_ball:   float = 35.0          # m/s
    smoothing_alpha: float = 0.6        # 0 = full smoothing, 1 = no smoothing
    pitch_margin: float = 2.0           # m past the boundary allowed
    enforce_collisions: bool = True
    r_min: float = 0.6                  # m  minimum inter-player distance
    collision_iters: int = 2            # repulsion passes


def apply_physics(
    samples: torch.Tensor,              # (K, T, n_ent, 2)  IN METERS
    history_last_frame: torch.Tensor,   # (n_ent, 2) IN METERS — for seam anchor
    valid: torch.Tensor,                # (n_ent,) bool
    cfg: GenTacConfig,
    pcfg: PhysicsConfig | None = None,
) -> torch.Tensor:
    """Apply max-speed, smoothing, pitch clamp, collision repulsion. Pure function."""
    pcfg = pcfg or PhysicsConfig()
    K, T, n_ent, _ = samples.shape
    N = cfg.n_players_per_team
    dt = 1.0 / cfg.fps

    # entity-specific max speed (players vs ball)
    v_max = torch.full((n_ent,), pcfg.v_max_player, device=samples.device, dtype=samples.dtype)
    v_max[2 * N] = pcfg.v_max_ball
    max_dx = (v_max * dt).view(1, 1, n_ent, 1)  # broadcast (1,1,n_ent,1) over xy

    # ── 1. seam anchor: glue the first sample frame to the last observed frame ──
    # We don't overwrite, we just compute the offset and apply it across the rollout
    # so motion still flows from where the player actually was.
    seam = history_last_frame.unsqueeze(0).unsqueeze(0)            # (1, 1, n_ent, 2)
    initial_offset = samples[:, 0:1] - seam                        # (K, 1, n_ent, 2)
    # Fade the offset out over the first ~0.4s (10 frames) so we don't snap.
    fade_frames = min(10, T)
    fade = torch.linspace(1.0, 0.0, fade_frames, device=samples.device, dtype=samples.dtype)
    fade = torch.cat([fade, torch.zeros(T - fade_frames, device=samples.device, dtype=samples.dtype)])
    samples = samples - initial_offset * fade.view(1, T, 1, 1)

    # ── 2. per-frame max-speed clip ─────────────────────────────────────────────
    # Walk t = 1..T-1 sequentially, clipping each step's velocity.
    out = [samples[:, 0]]
    for t in range(1, T):
        prev = out[-1]
        delta = samples[:, t] - prev
        norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)    # (K, n_ent, 1)
        scale = torch.minimum(norm, max_dx.squeeze(1)) / norm      # (K, n_ent, 1)
        out.append(prev + delta * scale)
    samples = torch.stack(out, dim=1)                              # (K, T, n_ent, 2)

    # ── 3. exponential smoothing (causal: forward pass only) ────────────────────
    alpha = pcfg.smoothing_alpha
    ema = [samples[:, 0]]
    for t in range(1, T):
        ema.append(alpha * samples[:, t] + (1 - alpha) * ema[-1])
    samples = torch.stack(ema, dim=1)

    # ── 4. on-pitch clamp ───────────────────────────────────────────────────────
    x_lim = cfg.pitch_x / 2.0 + pcfg.pitch_margin
    y_lim = cfg.pitch_y / 2.0 + pcfg.pitch_margin
    samples[..., 0] = samples[..., 0].clamp(-x_lim, x_lim)
    samples[..., 1] = samples[..., 1].clamp(-y_lim, y_lim)

    # ── 5. inter-player repulsion (player slots only, not ball) ────────────────
    if pcfg.enforce_collisions:
        player_slots = list(range(2 * N))                          # 0..21
        valid_players = valid[player_slots]                        # (2N,)
        for _ in range(pcfg.collision_iters):
            p = samples[:, :, player_slots]                        # (K, T, 2N, 2)
            # pairwise distance (K, T, 2N, 2N)
            diff = p.unsqueeze(-2) - p.unsqueeze(-3)               # (K, T, 2N, 2N, 2)
            dist = diff.norm(dim=-1).clamp(min=1e-9)               # (K, T, 2N, 2N)
            mask = (dist < pcfg.r_min).float()
            # don't push self vs self
            eye = torch.eye(len(player_slots), device=samples.device).bool()
            mask = mask.masked_fill(eye, 0)
            # push each pair apart by half the violation, along their unit vector
            push = (pcfg.r_min - dist).clamp(min=0).unsqueeze(-1) * 0.5
            unit = diff / dist.unsqueeze(-1)
            # mask out invalid players (don't push from / to missing slots)
            valid_pairs = (valid_players.unsqueeze(0) & valid_players.unsqueeze(1)).float()
            push = push * mask.unsqueeze(-1) * valid_pairs.view(1, 1, len(player_slots), len(player_slots), 1)
            delta = push * unit                                    # (K, T, 2N, 2N, 2)
            samples[:, :, player_slots] = p + delta.sum(dim=-2)

    # ── 6. RE-CLIP after collision-pass ────────────────────────────────────────
    # Repulsion shifts positions and can inflate per-frame velocity past v_max.
    # Without this second pass the eval shows ~12 m/s peaks even though step 2
    # capped at 10.5 m/s. We also re-anchor the seam since the clamp may have
    # moved samples[:, 0] off the history.
    samples[:, 0] = history_last_frame                              # absolute seam
    out = [samples[:, 0]]
    for t in range(1, T):
        prev = out[-1]
        delta = samples[:, t] - prev
        norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        scale = torch.minimum(norm, max_dx.squeeze(1)) / norm
        out.append(prev + delta * scale)
    samples = torch.stack(out, dim=1)

    # ── 7. final on-pitch re-clamp ─────────────────────────────────────────────
    # The re-clip in step 6 can re-introduce out-of-pitch positions if the seam
    # anchor (history_last_frame) is itself off-pitch — possible under adversarial
    # input. Final clamp guarantees the contract.
    samples[..., 0] = samples[..., 0].clamp(-x_lim, x_lim)
    samples[..., 1] = samples[..., 1].clamp(-y_lim, y_lim)

    return samples


if __name__ == "__main__":
    # Quick sanity: feed in a jittery sample, ensure max-speed is enforced + no NaN
    cfg = GenTacConfig(smoke=True)
    pcfg = PhysicsConfig()
    K, T = 4, 25
    samples = torch.randn(K, T, cfg.n_entities, 2) * 30        # huge jitter
    history_last = torch.randn(cfg.n_entities, 2) * 30
    valid = torch.ones(cfg.n_entities, dtype=torch.bool)

    out = apply_physics(samples, history_last, valid, cfg, pcfg)
    print(f"output shape: {tuple(out.shape)}")
    # verify max speed
    dt = 1.0 / cfg.fps
    step = (out[:, 1:] - out[:, :-1]).norm(dim=-1)             # (K, T-1, n_ent)
    v = step / dt
    print(f"max player speed (m/s): {v[:, :, :22].max().item():.2f}  (limit {pcfg.v_max_player})")
    print(f"max ball   speed (m/s): {v[:, :, 22].max().item():.2f}  (limit {pcfg.v_max_ball})")
    print(f"on-pitch clamp: x ∈ [{out[..., 0].min():.1f}, {out[..., 0].max():.1f}]  "
          f"y ∈ [{out[..., 1].min():.1f}, {out[..., 1].max():.1f}]")
    # collision check
    p = out[:, :, :2 * cfg.n_players_per_team]
    diff = p.unsqueeze(-2) - p.unsqueeze(-3)
    dist = diff.norm(dim=-1)
    eye = torch.eye(p.size(-2)).bool()
    dist_off_diag = dist.masked_fill(eye, float("inf"))
    print(f"min inter-player distance: {dist_off_diag.min().item():.3f} m  (target ≥ {pcfg.r_min})")
    assert not torch.isnan(out).any()
    print("no NaN ✓")
