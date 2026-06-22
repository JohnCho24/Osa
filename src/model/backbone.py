"""Spatiotemporal attention backbone (paper §6.2.2).

Factorized: each of the M layers applies
  1. spatial attention   — over entities at a fixed time step
  2. temporal attention  — over time steps for a fixed entity
followed by a position-wise MLP. Pre-norm residual blocks throughout.

Missing entities are excluded via key_padding_mask, with a guard so rows containing
zero valid keys do not produce NaN softmaxes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import GenTacConfig


def _safe_padding_mask(pad: torch.Tensor) -> torch.Tensor:
    """If a row is entirely padded, mark it all-valid so softmax doesn't NaN.

    Caller is responsible for masking the output of those rows downstream.
    """
    all_pad = pad.all(dim=-1, keepdim=True)
    return pad & ~all_pad


class FactorizedBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float, dropout: float, use_cross_attn: bool = False):
        super().__init__()
        self.norm_s = nn.LayerNorm(d)
        self.spatial_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(d)
        self.temporal_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        # Optional cross-attention to condition (waypoint) tokens — only built for
        # cfg.conditioning == "cross_attn"; otherwise the block is byte-identical to
        # the original (no extra params).
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm_cq = nn.LayerNorm(d)      # query = main tokens
            self.norm_ck = nn.LayerNorm(d)      # key/value = condition tokens
            self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, L, n_ent, d). valid: (B, L, n_ent) bool. cond: (B, L, n_ent, d) | None."""
        B, L, n_ent, d = x.shape

        # ── spatial: (B*L, n_ent, d) ─────────────────────────────────────
        x_in = self.norm_s(x).reshape(B * L, n_ent, d)
        pad_s = _safe_padding_mask((~valid).reshape(B * L, n_ent))
        attn_out, _ = self.spatial_attn(x_in, x_in, x_in, key_padding_mask=pad_s, need_weights=False)
        x = x + attn_out.reshape(B, L, n_ent, d)

        # ── temporal: (B*n_ent, L, d) ────────────────────────────────────
        # permute to put entity before time, then flatten.
        x_in = self.norm_t(x).permute(0, 2, 1, 3).reshape(B * n_ent, L, d)
        pad_t = _safe_padding_mask((~valid).permute(0, 2, 1).reshape(B * n_ent, L))
        attn_out, _ = self.temporal_attn(x_in, x_in, x_in, key_padding_mask=pad_t, need_weights=False)
        attn_out = attn_out.reshape(B, n_ent, L, d).permute(0, 2, 1, 3)
        x = x + attn_out

        # ── cross-attention to condition tokens (per-frame, entity axis) ──
        # Each token reads the waypoint/arrow conditions of all entities in its
        # frame, so players react to where others are instructed to go.
        if self.use_cross_attn and cond is not None:
            q = self.norm_cq(x).reshape(B * L, n_ent, d)
            kv = self.norm_ck(cond).reshape(B * L, n_ent, d)
            pad_c = _safe_padding_mask((~valid).reshape(B * L, n_ent))
            attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=pad_c, need_weights=False)
            x = x + attn_out.reshape(B, L, n_ent, d)

        # ── position-wise MLP ────────────────────────────────────────────
        x = x + self.mlp(self.norm_m(x))
        return x


class SpatioTemporalBackbone(nn.Module):
    def __init__(self, cfg: GenTacConfig):
        super().__init__()
        self.cfg = cfg
        use_cross_attn = getattr(cfg, "conditioning", "additive") == "cross_attn"
        self.blocks = nn.ModuleList([
            FactorizedBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, use_cross_attn=use_cross_attn)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, L, n_ent, d). valid: (B, L, n_ent) bool. cond: (B, L, n_ent, d) | None.
        returns (B, L, n_ent, d)."""
        for blk in self.blocks:
            tokens = blk(tokens, valid, cond)
        return self.final_norm(tokens)


if __name__ == "__main__":
    cfg = GenTacConfig(smoke=True)
    from .tokenizer import TrajectoryTokenizer
    tok = TrajectoryTokenizer(cfg)
    bb = SpatioTemporalBackbone(cfg)

    B = 2
    L = cfg.history_frames + cfg.window_frames
    n_ent = cfg.n_entities

    coords = torch.randn(B, L, n_ent, 2)
    valid = torch.ones(B, L, n_ent, dtype=torch.bool)
    valid[0, 50, 7] = False                       # simulate missing player
    valid[1, :, 22] = False                       # simulate missing ball
    # also exercise the all-masked-row guard
    valid[1, 30, :] = False

    h, cond = tok(coords)
    out = bb(h, valid, cond)
    print(f"tokens {tuple(h.shape)} → backbone {tuple(out.shape)}")
    print(f"backbone params: {sum(p.numel() for p in bb.parameters()):,}")
    print(f"total params (tok+bb): {sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in bb.parameters()):,}")
    assert not torch.isnan(out).any(), "NaN in backbone output"
    print("no NaN in output ✓")
