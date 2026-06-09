"""Single source of truth for all GenTac hyperparameters.

Paper-faithful defaults (§ 6.3.3). Use cfg.smoke=True to shrink for Mac MPS sanity checks.
"""
from dataclasses import dataclass, field


# ── entity role vocabulary ──────────────────────────────────────────────────
# Collapsed playing-position groups used by the tokenizer's role embedding (our
# permutation-invariant replacement for the old per-slot entity embedding). Role
# is a property of the *player*, fed as token content so it travels with them
# under any within-team reordering. The ball gets its own BALL role; UNK is the
# fallback when a roster position is missing or unmapped.
ROLE_NAMES: tuple[str, ...] = ("GK", "DEF", "MID", "FWD", "BALL", "UNK")
ROLE_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ROLE_NAMES)}


@dataclass
class GenTacConfig:
    # ── geometry ───────────────────────────────────────────────────────────
    pitch_x: float = 105.0
    pitch_y: float = 68.0
    n_players_per_team: int = 11
    n_entities: int = 23                  # 2 * 11 + 1 (ball)
    n_roles: int = 6                      # len(ROLE_NAMES): GK/DEF/MID/FWD/BALL/UNK
    fps: int = 25

    # ── trajectory task ───────────────────────────────────────────────────
    history_frames: int = 100             # 4 s @ 25 fps
    window_frames: int = 5                # 0.2 s causal window
    # Total tokens per training sample: history_frames + window_frames

    # ── model ──────────────────────────────────────────────────────────────
    d_model: int = 256                    # paper §6.3.3
    n_heads: int = 8
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # ── diffusion ──────────────────────────────────────────────────────────
    n_diffusion_steps: int = 100          # paper §6.3.1
    # Noise schedule. "cosine" (Nichol & Dhariwal) drives ᾱ_T → 0 so the terminal
    # state is (near-)pure noise, matching the N(0,I) the sampler starts from. The
    # short linear schedule below only reaches ᾱ_T≈0.36 (60% signal retained at the
    # top step) → a train/inference mismatch, so cosine is the default.
    schedule_type: str = "cosine"         # "cosine" | "linear"
    beta_start: float = 1e-4              # linear schedule only
    beta_end: float = 0.02               # linear schedule only

    # ── event head ─────────────────────────────────────────────────────────
    n_event_types: int = 5
    n_event_subtypes: int = 15
    event_seq_len: int = 250              # 10 s @ 25 fps
    lambda_subtype: float = 1.0

    # ── training ───────────────────────────────────────────────────────────
    lr_traj: float = 1e-3
    lr_event: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 200                 # paper batch
    epochs: int = 60                      # paper

    # ── data ───────────────────────────────────────────────────────────────
    train_stride_frames: int = 25         # 1 sample per second from each match
    val_split: float = 0.1

    # ── learned waypoint conditioning + CFG (our extension on top of the paper) ─
    # The paper has no learned arrow signal — pinning is a hard projection on the
    # noisy state during sampling. We give the model an explicit per-(entity, time)
    # waypoint input and train it both with and without that signal so we can use
    # classifier-free guidance to dial how aggressively the model honors the coach.
    p_waypoint_dropout: float = 0.2        # train: prob of zeroing all waypoints (uncond branch)
    p_waypoint_per_entity: float = 0.35    # train: prob a given valid entity gets a synthetic waypoint
    waypoint_dim: int = 2                   # x,y of the target position

    # Inference defaults (overridable per-request).
    default_guidance_scale: float = 1.5     # 1.0 = pure cond, >1 = sharper toward arrows

    # ── training improvements ──────────────────────────────────────────────
    # EMA shadow weights for the model parameters. 0.999 is the canonical
    # diffusion-training default; raise toward 0.9999 for very long runs. Set
    # to 0 to disable. Saved in the checkpoint as buffers; the server's
    # inference path reads them via GenTacTrajectoryModule.ema_state_dict().
    ema_decay: float = 0.999

    # ── checkpoint schema (bump on incompatible cfg changes) ───────────────
    # 1 = no learned-waypoint; 2 = + learned-waypoint+CFG; 3 = + EMA buffers;
    # 4 = entity_emb (per-slot) replaced by role_emb (per-position) → permutation
    #     invariant within a team. Architecture change, so old ckpts are incompatible.
    # 5 = cosine noise schedule (full noising). Changes the diffusion process, so a
    #     model trained on the old linear schedule would sample wrong → incompatible.
    schema_version: int = 5

    # ── smoke-test overrides (for Mac MPS dev) ─────────────────────────────
    smoke: bool = False

    def __post_init__(self):
        if self.smoke:
            self.d_model = 128
            self.n_layers = 2
            self.n_heads = 4
            self.n_diffusion_steps = 20
            self.batch_size = 4
            self.epochs = 1
            self.train_stride_frames = 250  # ~12 samples per match
            self.ema_decay = 0.0             # EMA needs ≥ thousands of steps to converge
