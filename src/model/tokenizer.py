"""Multi-agent trajectory tokenizer (paper §6.2.1 + waypoint extension).

Input:  raw coords (B, L, n_ent, 2)  — normalized to [-1, 1]
Output: token embeddings (B, L, n_ent, d)

Adds four learnable embeddings to the projected coords:
  1. temporal — one vector per time step, shared across entities at that step
  2. group    — one vector per group {team0, team1, ball}
  3. entity   — one vector per slot index (each player slot + the ball)
  4. waypoint — for each (t, entity) optionally project a target xy and add it,
                or fall back to a learned "no waypoint" null embedding. This is
                our extension on top of the paper; together with CFG dropout in
                training and CFG mixing at sample time it gives a tunable
                "trust the coach" knob.
"""
import torch
import torch.nn as nn

from .config import GenTacConfig


class TrajectoryTokenizer(nn.Module):
    def __init__(self, cfg: GenTacConfig, max_seq_len: int | None = None):
        super().__init__()
        self.cfg = cfg
        N = cfg.n_players_per_team
        self.n_entities = 2 * N + 1
        # Cover trajectory windows (H+w) AND event sequences (event_seq_len).
        self.L_max = max_seq_len or max(cfg.history_frames + cfg.window_frames, cfg.event_seq_len)

        # 1. coord projection
        self.coord_proj = nn.Linear(2, cfg.d_model)

        # 2. temporal embedding (L_max, d) — Parameter so it's learnable like any other
        self.temporal_emb = nn.Parameter(torch.zeros(self.L_max, cfg.d_model))
        nn.init.trunc_normal_(self.temporal_emb, std=0.02)

        # 3. group embedding (3, d) — team0=0, team1=1, ball=2
        self.group_emb = nn.Embedding(3, cfg.d_model)

        # 4. entity (slot) embedding (n_entities, d)
        self.entity_emb = nn.Embedding(self.n_entities, cfg.d_model)

        # 5. waypoint embedding (our extension)
        self.waypoint_proj = nn.Linear(cfg.waypoint_dim, cfg.d_model)
        # learned null vector added when no waypoint is provided for this (t, entity)
        self.waypoint_null_emb = nn.Parameter(torch.zeros(cfg.d_model))

        # precompute group index per entity slot
        group_idx = torch.zeros(self.n_entities, dtype=torch.long)
        group_idx[:N] = 0          # team0
        group_idx[N:2 * N] = 1     # team1
        group_idx[2 * N] = 2       # ball
        self.register_buffer("group_idx", group_idx, persistent=False)

        entity_idx = torch.arange(self.n_entities, dtype=torch.long)
        self.register_buffer("entity_idx", entity_idx, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.group_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.entity_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.waypoint_null_emb, std=0.02)
        nn.init.xavier_uniform_(self.coord_proj.weight)
        nn.init.zeros_(self.coord_proj.bias)
        nn.init.xavier_uniform_(self.waypoint_proj.weight)
        nn.init.zeros_(self.waypoint_proj.bias)

    def forward(
        self,
        coords: torch.Tensor,
        waypoint_target: torch.Tensor | None = None,        # (B, L, n_ent, 2)
        waypoint_present: torch.Tensor | None = None,       # (B, L, n_ent) bool
    ) -> torch.Tensor:
        """coords: (B, L, n_ent, 2)  →  tokens: (B, L, n_ent, d)

        If waypoint_target is None the unconditional (no-arrow) branch is used —
        the same null embedding the model sees during CFG dropout in training,
        so calling forward without waypoints stays in-distribution.
        """
        B, L, n_ent, _ = coords.shape
        if L > self.L_max:
            raise ValueError(f"seq len {L} > tokenizer L_max {self.L_max}")
        if n_ent != self.n_entities:
            raise ValueError(f"got n_ent={n_ent}, expected {self.n_entities}")

        h = self.coord_proj(coords)                            # (B, L, n_ent, d)
        h = h + self.temporal_emb[:L].view(1, L, 1, -1)        # broadcast over entity
        h = h + self.group_emb(self.group_idx).view(1, 1, n_ent, -1)
        h = h + self.entity_emb(self.entity_idx).view(1, 1, n_ent, -1)

        if waypoint_target is None:
            # Unconditional branch: every (t, entity) gets the null waypoint embedding.
            h = h + self.waypoint_null_emb.view(1, 1, 1, -1)
        else:
            if waypoint_present is None:
                raise ValueError("waypoint_target given without waypoint_present")
            wp_proj = self.waypoint_proj(waypoint_target)                       # (B, L, n_ent, d)
            wp_emb = torch.where(
                waypoint_present.unsqueeze(-1),
                wp_proj,
                self.waypoint_null_emb.view(1, 1, 1, -1).expand_as(wp_proj),
            )
            h = h + wp_emb
        return h


if __name__ == "__main__":
    cfg = GenTacConfig(smoke=True)
    tok = TrajectoryTokenizer(cfg)
    x = torch.randn(2, cfg.history_frames + cfg.window_frames, cfg.n_entities, 2)
    h = tok(x)
    print(f"input  {tuple(x.shape)}  → tokens {tuple(h.shape)}")
    print(f"params: {sum(p.numel() for p in tok.parameters()):,}")
