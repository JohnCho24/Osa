"""Physics post-processor invariants.

These tests are the safety net for the seam-vmax regression we fixed earlier:
a jittery raw sample fed in must come out with every per-frame speed ≤ the cap.
"""
import torch

from src.model.config import GenTacConfig
from src.model.physics import PhysicsConfig, apply_physics


def _jittery_input(K=4, T=25, seed=0, *, realistic_history=True):
    """Adversarial samples (huge jitter) but optionally a realistic on-pitch
    history. `realistic_history=False` lets us test the on-pitch clamp under
    an off-pitch seam."""
    cfg = GenTacConfig(smoke=True)
    g = torch.Generator().manual_seed(seed)
    samples = torch.randn(K, T, cfg.n_entities, 2, generator=g) * 30          # huge jitter
    if realistic_history:
        # Real tracking data is bounded; samples will get clipped down to it.
        history_last = torch.randn(cfg.n_entities, 2, generator=g) * torch.tensor([20.0, 12.0])
    else:
        history_last = torch.randn(cfg.n_entities, 2, generator=g) * 30
    valid = torch.ones(cfg.n_entities, dtype=torch.bool)
    return cfg, samples, history_last, valid


def test_speed_clip_holds_after_collision_pass():
    """The bug we shipped a fix for: collision-repulsion was inflating velocities
    past v_max because the speed clip only ran BEFORE it. Re-clipping at the end
    must hold even on adversarial jittery input."""
    cfg, samples, history_last, valid = _jittery_input()
    pcfg = PhysicsConfig()
    out = apply_physics(samples, history_last, valid, cfg, pcfg)

    dt = 1.0 / cfg.fps
    step = (out[:, 1:] - out[:, :-1]).norm(dim=-1)
    v = step / dt
    # Players (slots 0..21)
    max_player_v = v[:, :, : 2 * cfg.n_players_per_team].max().item()
    # Ball (slot 22)
    max_ball_v = v[:, :, 2 * cfg.n_players_per_team].max().item()
    # Allow tiny floating-point slop
    assert max_player_v <= pcfg.v_max_player + 1e-3, f"player speed {max_player_v} > {pcfg.v_max_player}"
    assert max_ball_v <= pcfg.v_max_ball + 1e-3, f"ball speed {max_ball_v} > {pcfg.v_max_ball}"


def test_on_pitch_clamp_holds_even_with_off_pitch_seam():
    """Output must be on-pitch even if the history (seam) is itself off-pitch
    — adversarial input must not punch a hole in the clamp contract."""
    cfg, samples, history_last, valid = _jittery_input(realistic_history=False)
    pcfg = PhysicsConfig()
    out = apply_physics(samples, history_last, valid, cfg, pcfg)
    x_lim = cfg.pitch_x / 2.0 + pcfg.pitch_margin
    y_lim = cfg.pitch_y / 2.0 + pcfg.pitch_margin
    assert out[..., 0].abs().max().item() <= x_lim + 1e-3
    assert out[..., 1].abs().max().item() <= y_lim + 1e-3


def test_seam_anchored_to_history():
    """First predicted frame should equal the (on-pitch) last history frame."""
    cfg, samples, history_last, valid = _jittery_input(realistic_history=True)
    out = apply_physics(samples, history_last, valid, cfg, PhysicsConfig())
    assert torch.allclose(out[:, 0], history_last.unsqueeze(0).expand_as(out[:, 0]), atol=1e-4)


def test_no_nan():
    cfg, samples, history_last, valid = _jittery_input()
    out = apply_physics(samples, history_last, valid, cfg, PhysicsConfig())
    assert not torch.isnan(out).any()
