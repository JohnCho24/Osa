"""Multi-agent trajectory tokenizer (paper §6.2.1 + waypoint extension).

Input:  raw coords (B, L, n_ent, 2)  — normalized to [-1, 1]
Output: token embeddings (B, L, n_ent, d)

Adds four learnable embeddings to the projected coords:
  1. temporal — one vector per time step, shared across entities at that step
  2. group    — one vector per group {team0, team1, ball}
  3. role     — one vector per playing position {GK, DEF, MID, FWD, BALL, UNK},
                indexed by the *player's* role rather than their slot. Because role
                is content that travels with the player, the spatial attention is
                permutation-invariant within a team: any slot can hold any player
                and the prediction is unchanged (the old per-slot `entity_emb`
                broke this — see schema_version 4). Identity across time is carried
                by the backbone's per-slot temporal attention, not this embedding.
  4. waypoint — for each (t, entity) optionally project a target xy and add it,
                or fall back to a learned "no waypoint" null embedding. This is
                our extension on top of the paper; together with CFG dropout in
                training and CFG mixing at sample time it gives a tunable
                "trust the coach" knob.
"""
import torch
import torch.nn as nn

from .config import GenTacConfig, ROLE_TO_IDX


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

        # 4. role (playing-position) embedding (n_roles, d) — indexed by role, not slot
        self.role_emb = nn.Embedding(cfg.n_roles, cfg.d_model)

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

        # fallback role per slot when callers don't supply roles (e.g. the event
        # head, or synthetic inputs): players → UNK, ball → BALL. Keeps every
        # forward in-distribution without forcing every caller to pass roles.
        default_role_idx = torch.full((self.n_entities,), ROLE_TO_IDX["UNK"], dtype=torch.long)
        default_role_idx[2 * N] = ROLE_TO_IDX["BALL"]
        self.register_buffer("default_role_idx", default_role_idx, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.group_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.role_emb.weight, std=0.02)
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
        role_idx: torch.Tensor | None = None,               # (B, n_ent) long — per-player role
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """coords: (B, L, n_ent, 2)  →  (tokens, condition)

        Returns (tokens, condition):
          • additive mode   — the waypoint is added onto `tokens`; condition is None.
          • cross_attn mode — `tokens` carry NO waypoint; the waypoint embedding is
            returned as a separate `condition` (B, L, n_ent, d) for the backbone to
            cross-attend to. (Either way the waypoint embedding itself is identical.)

        If waypoint_target is None the unconditional (no-arrow) branch is used —
        the same null embedding the model sees during CFG dropout in training,
        so calling forward without waypoints stays in-distribution.

        role_idx gives each entity its playing-position role (constant over time);
        if None, the per-slot UNK/BALL fallback is used.
        """
        B, L, n_ent, _ = coords.shape
        if L > self.L_max:
            raise ValueError(f"seq len {L} > tokenizer L_max {self.L_max}")
        if n_ent != self.n_entities:
            raise ValueError(f"got n_ent={n_ent}, expected {self.n_entities}")

        h = self.coord_proj(coords)                            # (B, L, n_ent, d)
        h = h + self.temporal_emb[:L].view(1, L, 1, -1)        # broadcast over entity
        h = h + self.group_emb(self.group_idx).view(1, 1, n_ent, -1)
        if role_idx is None:
            role_h = self.role_emb(self.default_role_idx).view(1, 1, n_ent, -1)
        else:
            role_h = self.role_emb(role_idx).view(B, 1, n_ent, -1)   # broadcast over time
        h = h + role_h

        # Waypoint/condition embedding: projected target where present, else the
        # learned null vector. Same computation regardless of how it's consumed.
        null = self.waypoint_null_emb.view(1, 1, 1, -1)
        if waypoint_target is None:
            cond = null.expand(B, L, n_ent, -1)
        else:
            if waypoint_present is None:
                raise ValueError("waypoint_target given without waypoint_present")
            wp_proj = self.waypoint_proj(waypoint_target)                       # (B, L, n_ent, d)
            cond = torch.where(waypoint_present.unsqueeze(-1), wp_proj, null.expand_as(wp_proj))

        if self.cfg.conditioning == "cross_attn":
            return h, cond          # backbone cross-attends `h` to `cond`
        return h + cond, None       # additive (original): fold the waypoint into the token


if __name__ == "__main__":
    from .config import GenTacConfig
    cfg = GenTacConfig(smoke=True)
    tok = TrajectoryTokenizer(cfg)
    x = torch.randn(2, cfg.history_frames + cfg.window_frames, cfg.n_entities, 2)
    h, cond = tok(x)
    print(f"input  {tuple(x.shape)}  → tokens {tuple(h.shape)}  cond={None if cond is None else tuple(cond.shape)}")
    print(f"params: {sum(p.numel() for p in tok.parameters()):,}")
