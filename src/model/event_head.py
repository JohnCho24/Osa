"""Tactical event recognition head (paper §6.2.3, §6.3.2).

Pipeline:
  backbone(x) → H_M  ∈ (B, L, n_ent, d)
  attention-pool over (L × n_ent) → z ∈ (B, d)
  classifier heads → (type_logits ∈ R^5, subtype_logits ∈ R^15)

Hierarchical loss:  L_event = L_type + λ · L_sub  (paper sets λ = 1)

For v0 we implement a flat subtype classifier (all 15 classes compete) plus a separate
5-class type classifier. The paper's stricter "incompatible-subtype suppression"
variant routes through a per-type sub-classifier; that lives in a TODO for later.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SpatioTemporalBackbone
from .config import GenTacConfig
from .tokenizer import TrajectoryTokenizer


class AttentionPool(nn.Module):
    """Learned scalar importance per (time, entity) token, softmax → weighted sum.

    Tokens corresponding to padded/missing entities receive −inf logits → zero weight.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """h: (B, L, n_ent, d). valid: (B, L, n_ent) bool. returns (B, d)."""
        B, L, n_ent, d = h.shape
        scores = self.score_mlp(h).squeeze(-1)                # (B, L, n_ent)
        scores = scores.masked_fill(~valid, float("-inf"))
        flat = scores.view(B, L * n_ent)
        # if a sample has zero valid tokens this softmax NaNs — guard by setting one valid
        all_pad = (~valid.view(B, L * n_ent)).all(dim=1, keepdim=True)
        if all_pad.any():
            sentinel = torch.zeros_like(flat).masked_fill(all_pad, 0.0)
            flat = torch.where(all_pad.expand_as(flat) & (torch.arange(L * n_ent, device=h.device) == 0).unsqueeze(0),
                               sentinel, flat)
        a = F.softmax(flat, dim=-1)                           # (B, L*n_ent)
        z = (a.unsqueeze(-1) * h.view(B, L * n_ent, d)).sum(dim=1)
        return z                                              # (B, d)


class EventHead(nn.Module):
    """Type classifier + subtype classifier on top of a pooled latent."""

    def __init__(self, cfg: GenTacConfig):
        super().__init__()
        self.pool = AttentionPool(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.type_clf = nn.Linear(cfg.d_model, cfg.n_event_types)
        self.sub_clf = nn.Linear(cfg.d_model, cfg.n_event_subtypes)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.norm(self.pool(h, valid))
        return self.type_clf(z), self.sub_clf(z)


@dataclass
class EventLoss:
    total: torch.Tensor
    type: torch.Tensor
    sub: torch.Tensor


def event_loss(
    type_logits: torch.Tensor, sub_logits: torch.Tensor,
    y_type: torch.Tensor, y_sub: torch.Tensor, lambda_sub: float = 1.0,
) -> EventLoss:
    lt = F.cross_entropy(type_logits, y_type)
    ls = F.cross_entropy(sub_logits, y_sub)
    return EventLoss(total=lt + lambda_sub * ls, type=lt, sub=ls)


class GenTacEventRecognizer(nn.Module):
    """End-to-end event-recognition model: tokenizer → backbone → event head.

    Shares NO weights with the diffusion network in v0; the paper trains them in
    separate stages so this is faithful.
    """

    def __init__(self, cfg: GenTacConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TrajectoryTokenizer(cfg, max_seq_len=cfg.event_seq_len)
        self.backbone = SpatioTemporalBackbone(cfg)
        self.head = EventHead(cfg)

    def forward(self, coords: torch.Tensor, valid: torch.Tensor):
        h, _ = self.tokenizer(coords)          # event head doesn't use waypoint conditioning
        h = self.backbone(h, valid)
        return self.head(h, valid)


if __name__ == "__main__":
    cfg = GenTacConfig(smoke=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GenTacEventRecognizer(cfg).to(device)
    B, L = 2, 60
    coords = torch.randn(B, L, cfg.n_entities, 2, device=device).clamp(-1, 1)
    valid = torch.ones(B, L, cfg.n_entities, dtype=torch.bool, device=device)
    valid[1, 30, :] = False                                    # all-pad row stress test

    t_logits, s_logits = model(coords, valid)
    print(f"type_logits {tuple(t_logits.shape)}  subtype_logits {tuple(s_logits.shape)}")

    y_type = torch.randint(0, cfg.n_event_types, (B,), device=device)
    y_sub = torch.randint(0, cfg.n_event_subtypes, (B,), device=device)
    loss = event_loss(t_logits, s_logits, y_type, y_sub, cfg.lambda_subtype)
    print(f"event loss total={loss.total.item():.4f}  type={loss.type.item():.4f}  sub={loss.sub.item():.4f}")
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
