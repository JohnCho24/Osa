"""Reverse-diffusion sampler + causal sliding-window rollout (paper §6.3.1).

DDPM reverse update (per step s):
  μ_θ(x_s, s) = (1/√α_s) · (x_s − (β_s / √(1−ᾱ_s)) · ε_θ(x_s, x_h, s, c))
  x_{s−1}    = μ_θ + σ_s · z   (z ∼ N(0, I); σ_s = √β_s)
  at s = 0 we set z = 0.

Rollout: forecast a long horizon by sampling consecutive w-step windows, appending
each sample to the history, sliding the window forward, and repeating.
"""
from __future__ import annotations

import torch

from .config import GenTacConfig
from .diffusion import GenTacDiffusion, LinearBetaSchedule, build_schedule


@torch.no_grad()
def sample_window(
    model: GenTacDiffusion,
    schedule: LinearBetaSchedule,
    history: torch.Tensor,         # (B, H, n_ent, 2)
    valid: torch.Tensor,           # (B, H + w, n_ent)
    target_mask: torch.Tensor,     # (B, w, n_ent) — True = denoise; False = use clean_future
    clean_future: torch.Tensor | None = None,   # (B, w, n_ent, 2) — values where target is False
    waypoint_target: torch.Tensor | None = None,    # (B, w, n_ent, 2)
    waypoint_present: torch.Tensor | None = None,   # (B, w, n_ent) bool
    guidance_scale: float = 1.0,                    # 1.0 = pure cond; >1 = CFG-amplified
    role_idx: torch.Tensor | None = None,           # (B, n_ent) long — per-player role
) -> torch.Tensor:
    """Sample one causal window of w future frames via S reverse-diffusion steps.

    If waypoints are provided AND guidance_scale != 1, we run the model twice per
    step (conditional on waypoints + unconditional) and combine the predicted noise
    as eps = eps_uncond + scale * (eps_cond - eps_uncond). This is standard
    classifier-free guidance, scoped to the waypoint signal only.
    """
    model.eval()
    B, H, n_ent, _ = history.shape
    w = target_mask.size(1)
    device = history.device

    if clean_future is None:
        clean_future = torch.zeros(B, w, n_ent, 2, device=device, dtype=history.dtype)

    has_waypoints = waypoint_target is not None and waypoint_present is not None and bool(waypoint_present.any())
    do_cfg = has_waypoints and guidance_scale != 1.0

    # initialize: noise for target positions, clean values elsewhere
    x = torch.randn(B, w, n_ent, 2, device=device, dtype=history.dtype)
    mix = target_mask.unsqueeze(-1)                                              # (B, w, n_ent, 1)
    x = torch.where(mix, x, clean_future)

    for s in reversed(range(schedule.n_steps)):
        step_t = torch.full((B,), s, dtype=torch.long, device=device)
        coords = torch.cat([history, x], dim=1)

        if has_waypoints:
            eps_cond = model(coords, valid, step_t, future_start=H,
                             waypoint_target=waypoint_target,
                             waypoint_present=waypoint_present,
                             role_idx=role_idx)
        else:
            eps_cond = model(coords, valid, step_t, future_start=H, role_idx=role_idx)

        if do_cfg:
            eps_uncond = model(coords, valid, step_t, future_start=H, role_idx=role_idx)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        alpha = schedule.alphas[s]
        alpha_bar = schedule.alpha_bar[s]
        alpha_bar_prev = schedule.alpha_bar[s - 1] if s > 0 else torch.ones_like(alpha_bar)
        beta = schedule.betas[s]

        # Predict x0 from eps and clip to the normalized data range [-1, 1].
        # Essential with a cosine schedule: the terminal β→0.999 makes 1/√α huge,
        # so unclipped reverse steps amplify ε errors and blow up. Clipping x0 to
        # the known pitch bounds keeps every step on the data manifold.
        x0 = (x - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        x0 = x0.clamp(-1.0, 1.0)
        # DDPM posterior mean q(x_{t-1} | x_t, x0)
        mu = (beta * alpha_bar_prev.sqrt() / (1.0 - alpha_bar)) * x0 \
             + ((1.0 - alpha_bar_prev) * alpha.sqrt() / (1.0 - alpha_bar)) * x
        if s > 0:
            post_var = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            x_next = mu + post_var.sqrt() * torch.randn_like(x)
        else:
            x_next = mu
        # keep non-target positions clamped to their clean values
        x = torch.where(mix, x_next, clean_future)

    return x


@torch.no_grad()
def causal_rollout(
    model: GenTacDiffusion,
    schedule: LinearBetaSchedule,
    history: torch.Tensor,                       # (B, H, n_ent, 2)
    valid_static: torch.Tensor,                  # (B, n_ent)  — assume validity is constant
    horizon_frames: int,
    target_mask_template: torch.Tensor,          # (B, n_ent) — True per entity = noise it
    opponent_future_full: torch.Tensor | None = None,   # (B, horizon_frames, n_ent, 2) — for opp-cond
    waypoints: list[tuple[int, int, list[float]]] | None = None,    # per-arrow pins
    waypoint_fade_frames: int = 0,               # hard-pin fade frames before the target (kept as a safety net for under-trained models)
    guidance_scale: float = 1.0,                 # CFG scale on the learned waypoint signal
    role_idx: torch.Tensor | None = None,        # (B, n_ent) long — per-player role
) -> torch.Tensor:
    """Predict `horizon_frames` ahead by autoregressively sliding w-step windows.

    Conditioning surface:
      • **learned waypoints** (our extension) — the arrow signal is also fed to the
        model as a per-(entity, time) embedding interpolated smoothly across the
        whole horizon. Combined with CFG (`guidance_scale > 1`) this is what makes
        the player bend toward the arrow over the full horizon instead of just
        snapping at the last frame.
      • **hard projection** at pinned positions — guarantees the arrow destination
        is literally honored regardless of model output. By default we hard-pin
        only the final (literal) pin frame and rely on the learned signal +
        `waypoint_fade_frames` for the path in between.

    Returns: (B, horizon_frames, n_ent, 2).
    """
    cfg: GenTacConfig = model.cfg
    H = cfg.history_frames
    w = cfg.window_frames
    if horizon_frames % w != 0:
        raise ValueError(f"horizon_frames ({horizon_frames}) must be a multiple of w={w}")
    n_steps = horizon_frames // w
    B, _, n_ent, _ = history.shape
    device = history.device

    # Two parallel tensors per arrow:
    #   pin_mask / pin_value     — hard projection during diffusion (guaranteed)
    #   learn_present / learn_xy — learned-waypoint signal fed to the model (soft)
    # The learned signal is smoothly interpolated across the WHOLE horizon so each
    # rollout window has signal, not just the one containing the literal pin frame.
    pin_mask = torch.zeros(B, horizon_frames, n_ent, dtype=torch.bool, device=device)
    pin_value = torch.zeros(B, horizon_frames, n_ent, 2, dtype=history.dtype, device=device)
    learn_present = torch.zeros(B, horizon_frames, n_ent, dtype=torch.bool, device=device)
    learn_xy = torch.zeros(B, horizon_frames, n_ent, 2, dtype=history.dtype, device=device)

    if waypoints:
        cur = history[:, -1]                                                     # (B, n_ent, 2)
        for ent, t_idx, xy in waypoints:
            if not (0 <= t_idx < horizon_frames):
                raise ValueError(f"waypoint time {t_idx} out of horizon {horizon_frames}")
            target = torch.tensor(xy, dtype=history.dtype, device=device)

            # learned signal: linear fade from `cur` to `target` over [0, t_idx],
            # then hold `target` thereafter. The model sees a smooth per-frame
            # destination on every window of the rollout.
            for ti in range(horizon_frames):
                if ti <= t_idx:
                    alpha = (ti + 1) / (t_idx + 1)
                    learn_xy[:, ti, ent] = (1 - alpha) * cur[:, ent] + alpha * target
                else:
                    learn_xy[:, ti, ent] = target
                learn_present[:, ti, ent] = True

            # hard pin at the literal target frame (always) and optionally fade
            # preceding K frames (for older clients passing fade_frames > 0).
            pin_mask[:, t_idx, ent] = True
            pin_value[:, t_idx, ent] = target
            if waypoint_fade_frames > 0:
                start = max(0, t_idx - waypoint_fade_frames)
                for ti in range(start, t_idx):
                    alpha = (ti - start + 1) / (t_idx - start + 1)
                    pin_mask[:, ti, ent] = True
                    pin_value[:, ti, ent] = (1 - alpha) * cur[:, ent] + alpha * target

    accumulated = []
    current_history = history.clone()

    for step_idx in range(n_steps):
        fs = step_idx * w
        valid = torch.cat(
            [
                valid_static.unsqueeze(1).expand(B, H, n_ent),
                valid_static.unsqueeze(1).expand(B, w, n_ent),
            ],
            dim=1,
        )
        target_mask = target_mask_template.unsqueeze(1).expand(B, w, n_ent).clone()    # (B, w, n_ent)

        if opponent_future_full is not None:
            clean_future = opponent_future_full[:, fs : fs + w].clone()
        else:
            clean_future = torch.zeros(B, w, n_ent, 2, device=device, dtype=history.dtype)

        # hard-projection pins (any in this window)
        win_pin_mask = pin_mask[:, fs : fs + w]                                        # (B, w, n_ent)
        win_pin_val = pin_value[:, fs : fs + w]
        if win_pin_mask.any():
            target_mask = target_mask & ~win_pin_mask
            clean_future = torch.where(win_pin_mask.unsqueeze(-1), win_pin_val, clean_future)

        # learned waypoint signal for this window
        win_learn_present = learn_present[:, fs : fs + w]
        win_learn_xy = learn_xy[:, fs : fs + w]

        sample = sample_window(
            model, schedule, current_history, valid, target_mask,
            clean_future=clean_future,
            waypoint_target=win_learn_xy if win_learn_present.any() else None,
            waypoint_present=win_learn_present if win_learn_present.any() else None,
            guidance_scale=guidance_scale,
            role_idx=role_idx,
        )                                                                              # (B, w, n_ent, 2)
        accumulated.append(sample)
        current_history = torch.cat([current_history[:, w:], sample], dim=1)

    return torch.cat(accumulated, dim=1)                                              # (B, horizon, n_ent, 2)


def build_static_target_mask(cfg: GenTacConfig, n_ent: int, mode: str, device) -> torch.Tensor:
    """Per-entity (1, n_ent) target mask matching the modes in build_target_mask()."""
    N = cfg.n_players_per_team
    if mode == "unconditioned":
        return torch.ones(1, n_ent, dtype=torch.bool, device=device)
    m = torch.zeros(1, n_ent, dtype=torch.bool, device=device)
    if mode == "opp_conditioned_team0":
        m[:, :N] = True
    elif mode == "opp_conditioned_team1":
        m[:, N:2 * N] = True
    else:
        raise ValueError(mode)
    return m


if __name__ == "__main__":
    cfg = GenTacConfig(smoke=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GenTacDiffusion(cfg).to(device)
    sched = build_schedule(cfg).to(device)

    B = 2
    H, w, n_ent = cfg.history_frames, cfg.window_frames, cfg.n_entities
    history = torch.randn(B, H, n_ent, 2, device=device).clamp(-1, 1)
    valid_static = torch.ones(B, n_ent, dtype=torch.bool, device=device)

    # 1 s = 25 frames at 25 fps; window is 5 frames → 5 rollout steps
    horizon = 25

    print(f"device {device}  horizon {horizon} frames  window {w}  rollout steps {horizon // w}")
    for mode in ["unconditioned", "opp_conditioned_team0", "opp_conditioned_team1"]:
        target_mask = build_static_target_mask(cfg, n_ent, mode, device).expand(B, n_ent)
        opp_future = torch.randn(B, horizon, n_ent, 2, device=device).clamp(-1, 1)
        out = causal_rollout(
            model, sched, history, valid_static, horizon, target_mask,
            opponent_future_full=opp_future if mode != "unconditioned" else None,
        )
        in_range = out.abs().max().item()
        print(f"  {mode:30s} → {tuple(out.shape)}  max|out|={in_range:.3f}")

    # waypoint pinning test: pin entity 3 to (0.5, -0.3) at frame 24 (last frame of horizon)
    print("\nwaypoint pinning test:")
    target_mask = build_static_target_mask(cfg, n_ent, "unconditioned", device).expand(B, n_ent)
    waypoints = [(3, horizon - 1, [0.5, -0.3])]
    out = causal_rollout(
        model, sched, history, valid_static, horizon, target_mask,
        waypoints=waypoints, waypoint_fade_frames=8,
    )
    pinned_final = out[0, horizon - 1, 3]
    print(f"  pinned entity 3 at t=24 → {pinned_final.tolist()}  (expected [0.5, -0.3])")
    err = (pinned_final - torch.tensor([0.5, -0.3], device=device)).norm().item()
    print(f"  error from target: {err:.4f}  (should be ~0)")
    print(f"  full output shape: {tuple(out.shape)}")
