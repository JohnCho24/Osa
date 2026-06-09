"""Within-team permutation invariance (schema_version 4).

Identity/role is fed as token *content* (role embedding) rather than by slot
index, so the spatial attention is permutation-equivariant over the players of a
team: relabel which slot holds which player (consistently across all frames, with
their role tag carried along) and the predicted noise comes back under the exact
same relabeling. These tests pin that property — and that role actually matters,
so the embedding isn't silently ignored.
"""
import torch

from src.model.config import GenTacConfig, ROLE_TO_IDX
from src.model.diffusion import GenTacDiffusion


def _inputs(cfg, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = cfg.history_frames + cfg.window_frames
    n, N = cfg.n_entities, cfg.n_players_per_team
    coords = torch.randn(B, L, n, 2, generator=g).clamp(-1, 1)
    valid = torch.ones(B, L, n, dtype=torch.bool)
    # a plausible role layout: GK + mix per team, ball = BALL
    base = ([ROLE_TO_IDX["GK"]] + [ROLE_TO_IDX["DEF"]] * 4 + [ROLE_TO_IDX["MID"]] * 3 + [ROLE_TO_IDX["FWD"]] * 3)
    role_idx = torch.tensor(base + base + [ROLE_TO_IDX["BALL"]], dtype=torch.long).unsqueeze(0).expand(B, n).contiguous()
    step = torch.randint(0, cfg.n_diffusion_steps, (B,), generator=g)
    return coords, valid, role_idx, step


def _team_perm(cfg, team: int, seed=1):
    """Full-length entity permutation that shuffles one team's 11 slots, identity elsewhere."""
    n, N = cfg.n_entities, cfg.n_players_per_team
    g = torch.Generator().manual_seed(seed)
    perm = torch.arange(n)
    block = slice(0, N) if team == 0 else slice(N, 2 * N)
    perm[block] = perm[block][torch.randperm(N, generator=g)]
    return perm


@torch.no_grad()
def _forward(model, cfg, coords, valid, role_idx, step):
    return model(coords, valid, step, future_start=cfg.history_frames, role_idx=role_idx)


def test_within_team_permutation_equivariance():
    cfg = GenTacConfig(smoke=True)
    model = GenTacDiffusion(cfg).eval()
    coords, valid, role_idx, step = _inputs(cfg)

    out = _forward(model, cfg, coords, valid, role_idx, step)            # (B, w, n_ent, 2)

    for team in (0, 1):
        perm = _team_perm(cfg, team)
        out_p = _forward(model, cfg, coords[:, :, perm], valid[:, :, perm], role_idx[:, perm], step)
        # permuting the input players must permute the output the same way
        assert torch.allclose(out_p, out[:, :, perm], atol=1e-5), f"team{team} not equivariant"


def test_role_actually_matters():
    """Changing a player's role must change the prediction — else the embedding is dead."""
    cfg = GenTacConfig(smoke=True)
    model = GenTacDiffusion(cfg).eval()
    coords, valid, role_idx, step = _inputs(cfg)

    out = _forward(model, cfg, coords, valid, role_idx, step)
    role_alt = role_idx.clone()
    role_alt[:, 0] = ROLE_TO_IDX["FWD"]            # slot 0 was GK → make it FWD
    out_alt = _forward(model, cfg, coords, valid, role_alt, step)
    assert not torch.allclose(out, out_alt, atol=1e-5), "role embedding has no effect"
