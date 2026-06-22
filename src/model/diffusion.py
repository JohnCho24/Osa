"""DDPM noise schedule + GenTac diffusion network (paper §6.3.1).

Forward (training):
  x_s = √ᾱ_s · x_0 + √(1 − ᾱ_s) · ε,   ε ∼ N(0, I)
  L_traj = E[ || ε − ε_θ(x_s, x_h, s, c) ||² ]   over future-window positions.

Inference (sampling) lives in samplers.py.

Conditioning:
  - Unconditioned: every valid entity in the future window is noised.
  - Opponent-conditioned: opponent's future kept clean; only target team's future noised.
  Both are expressed via a single per-(t, entity) "target mask" tensor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SpatioTemporalBackbone
from .config import GenTacConfig
from .tokenizer import TrajectoryTokenizer


# ── noise schedule ──────────────────────────────────────────────────────────
def cosine_betas(n_steps: int, s: float = 0.008, max_beta: float = 0.999,
                 device: str | torch.device = "cpu") -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule.

    ᾱ_t = cos²(((t/T)+s)/(1+s) · π/2) (normalized so ᾱ_0 = 1), which drives
    ᾱ_T → 0 — i.e. the terminal state is (near-)pure noise, matching the N(0,I)
    the sampler starts from. Returns the per-step betas derived from that ᾱ.
    """
    t = torch.linspace(0, n_steps, n_steps + 1, device=device) / n_steps
    abar = torch.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2
    abar = abar / abar[0]
    betas = 1.0 - abar[1:] / abar[:-1]
    return betas.clamp(max=max_beta)


class LinearBetaSchedule:
    """Precompute β, α, ᾱ tables for a DDPM schedule.

    Despite the name (kept for back-compat), this holds any schedule: pass
    `betas` directly (e.g. cosine) or leave it None to build a linear schedule
    from (beta_start, beta_end). Use `build_schedule(cfg)` to pick by config.
    """

    def __init__(self, n_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str | torch.device = "cpu", betas: torch.Tensor | None = None):
        if betas is None:
            betas = torch.linspace(beta_start, beta_end, n_steps, device=device)    # (S,)
        self.n_steps = len(betas)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device):
        for k in ("betas", "alphas", "alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x0 (B, ...), step (B,) long, noise same shape as x0."""
        sa = self.sqrt_alpha_bar[step]
        soma = self.sqrt_one_minus_alpha_bar[step]
        while sa.dim() < x0.dim():
            sa = sa.unsqueeze(-1)
            soma = soma.unsqueeze(-1)
        return sa * x0 + soma * noise


def build_schedule(cfg: GenTacConfig, device: str | torch.device = "cpu") -> LinearBetaSchedule:
    """Construct the noise schedule chosen by cfg.schedule_type ("cosine" | "linear").

    Single source of truth — training (lightning_module) and inference (samplers)
    must use the same schedule the weights were trained on.
    """
    if getattr(cfg, "schedule_type", "linear") == "cosine":
        return LinearBetaSchedule(cfg.n_diffusion_steps,
                                  betas=cosine_betas(cfg.n_diffusion_steps, device=device))
    return LinearBetaSchedule(cfg.n_diffusion_steps, cfg.beta_start, cfg.beta_end, device=device)


# ── diffusion-step embedding (sinusoidal, transformer-style) ────────────────
class StepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        """step: (B,) long → (B, d_model)."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=step.device, dtype=torch.float32) / half
        )
        x = step.float().unsqueeze(-1) * freqs.unsqueeze(0)        # (B, half)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)        # (B, d_model)
        return self.mlp(x)


# ── GenTac diffusion model ──────────────────────────────────────────────────
class GenTacDiffusion(nn.Module):
    """ε-prediction network = tokenizer → step-embed inject → backbone → noise head."""

    def __init__(self, cfg: GenTacConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TrajectoryTokenizer(cfg)
        self.step_embed = StepEmbedding(cfg.d_model)
        self.backbone = SpatioTemporalBackbone(cfg)
        # noise head: (paper §6.3.3) LayerNorm + Linear(d → 2)
        self.noise_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, 2))

    def forward(
        self,
        coords: torch.Tensor,
        valid: torch.Tensor,
        step: torch.Tensor,
        future_start: int,
        waypoint_target: torch.Tensor | None = None,      # (B, w, n_ent, 2) — future only
        waypoint_present: torch.Tensor | None = None,     # (B, w, n_ent) bool
        role_idx: torch.Tensor | None = None,             # (B, n_ent) long — per-player role
    ) -> torch.Tensor:
        """
        coords: (B, L, n_ent, 2)  — history clean + future possibly corrupted
        valid:  (B, L, n_ent)     — True for real entities
        step:   (B,) long         — diffusion step index per sample
        future_start: int         — first index of the future window in L
        waypoint_target / waypoint_present: optional learned-waypoint signal,
            scoped to the FUTURE window. History rows are always treated as
            no-waypoint inside the tokenizer.
        role_idx: optional per-entity playing-position role (constant over time);
            None → tokenizer's UNK/BALL fallback.

        Returns predicted noise on the future window only: (B, w, n_ent, 2).
        """
        B, L, n_ent, _ = coords.shape

        # Pad future-only waypoints to full sequence length with no-waypoint on history.
        wp_target_full: torch.Tensor | None
        wp_present_full: torch.Tensor | None
        if waypoint_target is None:
            wp_target_full = None
            wp_present_full = None
        else:
            if waypoint_present is None:
                raise ValueError("waypoint_target given without waypoint_present")
            wp_target_full = torch.zeros(B, L, n_ent, self.cfg.waypoint_dim,
                                         dtype=coords.dtype, device=coords.device)
            wp_present_full = torch.zeros(B, L, n_ent, dtype=torch.bool, device=coords.device)
            wp_target_full[:, future_start:] = waypoint_target
            wp_present_full[:, future_start:] = waypoint_present

        # tokens + (optional) separate condition tokens. In "additive" mode the
        # waypoint is folded into `h` and cond is None; in "cross_attn" mode `h`
        # is waypoint-free and the backbone cross-attends it to `cond`.
        h, cond = self.tokenizer(coords, wp_target_full, wp_present_full, role_idx=role_idx)   # (B, L, n_ent, d)
        # Inject diffusion-step embedding into the future-window tokens only.
        step_h = self.step_embed(step)                             # (B, d)
        future_token_emb = step_h.view(coords.size(0), 1, 1, -1)   # (B, 1, 1, d)
        h_future = h[:, future_start:].clone()
        h_future = h_future + future_token_emb
        h = torch.cat([h[:, :future_start], h_future], dim=1)
        h = self.backbone(h, valid, cond)                          # (B, L, n_ent, d)
        eps = self.noise_head(h[:, future_start:])                 # (B, w, n_ent, 2)
        return eps


# ── synthetic-waypoint augmentation (our extension) ─────────────────────────
def generate_synthetic_waypoints(
    future: torch.Tensor,        # (B, w, n_ent, 2)
    target_mask: torch.Tensor,   # (B, w, n_ent)
    cfg: GenTacConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (waypoint_target, waypoint_present) tensors for training.

    Per-sample policy:
      • with prob cfg.p_waypoint_dropout → no waypoints (CFG dropout / uncond branch)
      • else                              → each (entity, time) in the target window
                                            gets a waypoint independently with
                                            prob cfg.p_waypoint_per_entity, using
                                            the ground-truth future position as the
                                            target (the model sees "this entity will
                                            be at xy at this t" and learns to honor it).

    Only entities that ARE being denoised (target_mask=True) can receive waypoints —
    otherwise the model could trivially read off the clean position from coords and
    we'd teach it a shortcut.
    """
    B, w, n_ent, _ = future.shape
    device = future.device

    drop_sample = torch.rand(B, device=device) < cfg.p_waypoint_dropout            # (B,) True = no arrows for whole sample
    per_slot = torch.rand(B, w, n_ent, device=device) < cfg.p_waypoint_per_entity  # (B, w, n_ent)
    waypoint_present = per_slot & target_mask & ~drop_sample.view(B, 1, 1)
    waypoint_target = future                                                       # raw future coords; gated by `present`
    return waypoint_target, waypoint_present


# ── training-step utility ───────────────────────────────────────────────────
def compute_diffusion_loss(
    model: GenTacDiffusion,
    schedule: LinearBetaSchedule,
    history: torch.Tensor,       # (B, H, n_ent, 2)
    future: torch.Tensor,        # (B, w, n_ent, 2)
    valid: torch.Tensor,         # (B, H+w, n_ent)
    target_mask: torch.Tensor,   # (B, w, n_ent) — True = noise this entity's future
    role_idx: torch.Tensor | None = None,   # (B, n_ent) long — per-player role
) -> tuple[torch.Tensor, dict]:
    """One training step: corrupt the target positions, predict noise, return MSE.

    Synthesizes random "arrows" from the ground-truth future via
    `generate_synthetic_waypoints` and feeds them through the model. With prob
    cfg.p_waypoint_dropout the whole sample is run unconditionally so that
    classifier-free guidance has a well-defined uncond branch at sample time.
    """
    B, H, n_ent, _ = history.shape
    w = future.size(1)
    device = history.device

    step = torch.randint(0, schedule.n_steps, (B,), device=device)
    noise = torch.randn_like(future)
    corrupted_future = schedule.q_sample(future, step, noise)
    mix_mask = target_mask.unsqueeze(-1)                                # (B, w, n_ent, 1)
    future_input = torch.where(mix_mask, corrupted_future, future)
    coords = torch.cat([history, future_input], dim=1)                  # (B, H+w, n_ent, 2)

    wp_target, wp_present = generate_synthetic_waypoints(future, target_mask, model.cfg)
    pred = model(coords, valid, step, future_start=H,
                 waypoint_target=wp_target, waypoint_present=wp_present,
                 role_idx=role_idx)                                        # (B, w, n_ent, 2)

    # only score positions that are both targeted AND have a valid entity
    score_mask = target_mask & valid[:, H:, :]                          # (B, w, n_ent)
    diff_sq = ((pred - noise) ** 2).sum(dim=-1)                         # (B, w, n_ent)
    n_targets = score_mask.sum().clamp(min=1)
    loss = (diff_sq * score_mask.float()).sum() / n_targets
    return loss, {
        "n_targets": n_targets.item(),
        "n_waypoints": int(wp_present.sum().item()),
    }


def build_target_mask(valid: torch.Tensor, mode: str, cfg: GenTacConfig) -> torch.Tensor:
    """Build (B, w, n_ent) target mask for one of the base modes.

    valid is the FULL (B, H+w, n_ent) validity mask; we use the future slice.

    Modes:
      unconditioned: noise every valid entity (both teams + ball) in the future
      opp_conditioned_team0: noise team0 future only (team1 + ball kept clean)
      opp_conditioned_team1: noise team1 future only (team0 + ball kept clean)
    """
    H = cfg.history_frames
    N = cfg.n_players_per_team
    fut_valid = valid[:, H:, :]                                         # (B, w, n_ent)

    if mode == "unconditioned":
        return fut_valid

    target = torch.zeros_like(fut_valid)
    if mode == "opp_conditioned_team0":
        target[:, :, :N] = fut_valid[:, :, :N]
    elif mode == "opp_conditioned_team1":
        target[:, :, N:2 * N] = fut_valid[:, :, N:2 * N]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return target


if __name__ == "__main__":
    cfg = GenTacConfig(smoke=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = GenTacDiffusion(cfg).to(device)
    sched = build_schedule(cfg).to(device)

    B = 2
    history = torch.randn(B, cfg.history_frames, cfg.n_entities, 2, device=device).clamp(-1, 1)
    future = torch.randn(B, cfg.window_frames, cfg.n_entities, 2, device=device).clamp(-1, 1)
    valid = torch.ones(B, cfg.history_frames + cfg.window_frames, cfg.n_entities, dtype=torch.bool, device=device)

    print(f"device         {device}")
    print(f"model params   {sum(p.numel() for p in model.parameters()):,}")

    for mode in ["unconditioned", "opp_conditioned_team0", "opp_conditioned_team1"]:
        target_mask = build_target_mask(valid, mode, cfg)
        loss, info = compute_diffusion_loss(model, sched, history, future, valid, target_mask)
        print(f"{mode:30s} loss={loss.item():.4f}  n_targets={info['n_targets']}")
