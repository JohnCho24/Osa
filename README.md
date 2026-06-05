# B START UP — Tactical "What-If" Engine

> **For league readers (60-second pitch):** Every match has fifty decisions you
> wish you could replay. Today, you can only argue about them. **B** turns the
> coach's "what if this player had run there instead?" into a simulated
> alternative: every player and the ball react plausibly, first on a 2D
> bird's-eye pitch and later through any 3D/video layer that consumes the same
> trajectory output. The underlying data stays inside your league's existing
> tracking partnership. See [docs/product_vision.md](docs/product_vision.md) for
> the product goal and [docs/handoff_api.md](docs/handoff_api.md) for the video
> handoff contract. Pricing is bespoke per deployment — book a
> 15-minute demo via the landing page or email `leagues@bstartup.dev`.
>
> The rest of this file is for engineers.

---

A tactical simulation platform for football. A coach, analyst, broadcast team, or
player opens a moment from a match, **draws arrows on players or the ball** to show
what could have happened, hits Generate, and gets back a simulated alternative
future where the full game state reacts.

The big picture is not "move one dot where the user drew an arrow." The goal is
to resimulate the situation from trained game data: all players react, move, and
interact realistically based on the new tactical intent. The viewer should be
able to understand how to move or react because they can visually see the
alternative play unfold.

Our first checkpoint is a **2D bird's-eye soccer simulation**, not a hyper-realistic
video. The 2D model must prove the core behavior: review previous footage, draw
potential movements or situations, simulate all players and the ball reacting, and
compare actual vs. alternative on a pitch. Hyper-realistic 3D video is a later
visualization layer once the 2D tactical simulation is credible.

We own the brain: the **diffusion model** that turns "frozen moment + arrows" into
a physically plausible alternative trajectory for all 23 entities (22 players + ball).
If and when the product needs photorealistic 3D, that can ship to a downstream AI
video service through the handoff contract.

Built on a reimplementation of **GenTac** (Rao et al.,
[arXiv:2604.11786](https://arxiv.org/abs/2604.11786)), with a custom waypoint-
conditioning extension for the arrow-input UX.

**Long-term users:** coaches, analysts, broadcasters, and players who need to
understand tactical alternatives visually. **Initial buyer/channel:** leagues or
clubs with access to tracking data, because they can provide the match data needed
to train and validate the model.

**Status:** Full end-to-end pipeline works on Mac (renderer → arrow UI → inference
server → physics → response → side-by-side compare). Remaining gaps are external:
real league data partnership (M5), real cloud training run (M7), and validation
conversations with league staff (M0).

---

## How it works

```
Tracking JSON           Arrow JSON              Trajectory JSON         Visual Simulation
(frozen moment)  ──►   (user constraints)  ──► (alternative outcome) ──►  (downstream
                                                                          2D now,
                                                                          3D later)
                            │
                            ▼
                  ┌──────────────────────┐
                  │   OUR SYSTEM         │
                  │  ──────────────────  │
                  │  • GenTac diffusion  │
                  │  • Waypoint cond.    │
                  │  • Physics post-proc │
                  │  • 2D analyst UI     │
                  └──────────────────────┘
```

1. **Tracking ingest** — coach loads a moment (decision frame + 4s history). Today
   we support Metrica's open format; production will plug into Hawk-Eye / Sportec
   DFL / Stats Perform feeds depending on the league.
2. **Arrow drawing** — coach clicks a player on the 2D tactics board and drags
   an arrow to where they should run.
3. **Waypoint-conditioned diffusion** — model samples a trajectory where the
   arrow destinations are honored, every other entity moves plausibly, and
   physical constraints (max-speed, collision repulsion, on-pitch) are enforced.
4. **2D simulation checkpoint** — output JSON (one keypoint per player per frame,
   25 fps, meters, center-origin) is rendered on a bird's-eye soccer field so the
   user can compare actual vs. alternative immediately.
5. **Later visualization handoff** — the same trajectory output can be shipped to
   a downstream 3D/video service after the 2D tactical behavior is validated.

---

## Quick start

```bash
# 1. Raw data
git clone https://github.com/metrica-sports/sample-data.git data/metrica

# 2. Convert + extract dev clips
python3 src/data/metrica_to_gentac.py
python3 src/data/extract_clip.py

# 3. Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 4. Smoke-train the model (3 epochs on Mac MPS, ~80s)
python scripts/smoke_train.py

# 5. (optional) Sample K=20 alternatives without the server
python scripts/sample.py

# 6. Start the inference server (for arrow-driven generation)
uvicorn src.server.main:app --host 127.0.0.1 --port 8001 &

# 7. Start the renderer
python3 -m http.server 8000
open http://localhost:8000              # default clip view
open "http://localhost:8000/?samples=1" # actual vs model alternative (uses samples.json)
```

Project contribution and security requirements live in
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

### The arrow-driven flow (the actual product UX)

1. Open `http://localhost:8000`.
2. Check the **Draw** toggle in the toolbar.
3. Click on any player and drag to where you want them to run.
4. Repeat for any number of players (right-click a player to undo their arrow).
5. Click **Generate ▶**.
6. The right panel swaps to a model-generated alternative future that honors your arrows.
7. Hit Play; the two panels play in sync — left is actual, right is your alternative.

Keyboard:
- `Space` play/pause
- `←` `→` step frame
- `[` `]` step between samples (in samples mode)

URL params:
- `?frame=N` deep-link to a frame
- `?compare=1` open in compare mode
- `?samples=1` load `data/processed/samples.json` (model output)

---

## What's built

### M1 — 2D renderer (done)
Vanilla JS + Canvas tactics board. FIFA-spec pitch markings, color-coded teams,
jersey numbers, ball, fading trails. Per-panel architecture (factory pattern)
ready for grid layouts. Compare mode shows two clips side-by-side at half-scale
on a synced scrubber. This is the **first checkpoint product surface**: the
bird's-eye simulation where users can see whether arrow-driven counterfactuals
produce realistic movement before any hyper-realistic video layer exists.

### M2 — GenTac model (smoke complete)
Paper-faithful PyTorch + Lightning reimplementation:
- TimeSformer-style factorized spatiotemporal attention backbone (M=4, d=256
  in paper config; M=2, d=128 in smoke)
- DDPM diffusion over future trajectories (100 steps, linear β)
- Causal sliding-window rollout for long-horizon forecasting
- Event head: attention pool + hierarchical classifier (code complete; labels
  not yet aligned, parked until needed)
- Pretraining modes: unconditioned + opponent-conditioned

Smoke run on Mac MPS: train loss 1.380 → 0.378 → 0.144 over 3 epochs.
~26s/epoch on MPS. Real training run goes to cloud (M7).

### M3.1 — Samples-mode renderer (done)
URL `?samples=1` loads `data/processed/samples.json` instead of raw clips.
Left panel shows ground-truth future, right panel cycles through K=20 model
alternatives via `◀ ▶`. Same scrubber drives both panels — analyst sees
actual vs. alternative side-by-side.

### M6 — Physics post-processor (done)
Deterministic post-pass on raw model output:
- Seam anchor (no teleport between history end and prediction start)
- Per-frame max-speed clip (players ≤ 10.5 m/s, ball ≤ 35 m/s)
- Exponential smoothing along time axis
- On-pitch clamp with 2 m margin
- Pairwise inter-player repulsion (min 0.6 m spacing)

Without this layer the smoke-trained model produced visibly jittery output
that wouldn't survive 30 seconds in a league review. With it, the same
checkpoint produces smooth, physically plausible trajectories.

### M8 — Arrow-drawing UI (done)
Click a player, drag to set their target destination, repeat for any number
of players. Right-click on a player clears their arrow. Hit Generate to ship
the arrow set to the inference layer. This is the **primary user input
mechanism** and the killer UX for the league pitch because coaches already
think in arrows.

### M9 — Waypoint-conditioned sampling (done)
`causal_rollout` accepts a list of `(entity_slot, time_idx, target_xy)`
waypoints. Pinned positions stay clean throughout the reverse-diffusion
process. Optional `fade_frames` parameter linearly interpolates the preceding
K frames toward the pin so the trajectory bends smoothly rather than snapping.
Smoke test: pin error = 0.0000 against target.

### M10 — Inference server (done)
FastAPI on `:8001`. `POST /api/generate` accepts arrow JSON, runs
arrow-conditioned `causal_rollout`, applies the M6 physics post-processor,
re-enforces arrow destinations as a safety net, returns the same JSON shape
the renderer's samples mode already consumes. CORS open for browser fetch.
Holds the smoke checkpoint in memory at startup.

### M11 — Handoff API spec (done)
`docs/handoff_api.md` defines the contract for a later downstream 3D/video layer
that consumes the same trajectory output as the 2D renderer. Coordinate system,
validity rules, time alignment, physical guarantees, error codes, versioning, and
the deliberate non-responsibilities (we don't own video, audio, graphics, or
refereeing).

---

## Roadmap

| ID  | Milestone | Status |
|-----|-----------|--------|
| M0  | Validate wedge — 1–2 league/club analyst conversations | not started |
| M1  | 2D pitch renderer (Metrica) | done |
| M2  | Paper-faithful GenTac (smoke level) | done |
| M3.1 | Samples-mode in renderer | done |
| M6  | Physical-plausibility post-processor | done |
| M8  | Arrow-drawing UI on renderer | done |
| M9  | Waypoint-conditioned sampling (generalize causal_rollout) | done |
| M10 | Inference server (FastAPI: arrow JSON → trajectory JSON) | done |
| M11 | Handoff API spec to downstream 3D video generator | done |
| M5  | Onboard real league tracking data (Bundesliga DFL) + team-conditioning | pending (needs data partnership) |
| M7  | Cloud-training run on real Bundesliga/SkillCorner data | pending (needs ~$30 GPU budget) |
| M3.2 | Grid view of 20 alternatives (deprioritized — not primary UX) | pending |

---

## Project layout

```
B START UP/
├── index.html                       ← renderer entry
├── src/
│   ├── data/
│   │   ├── metrica_to_gentac.py     ← Metrica CSV → GenTac JSON
│   │   └── extract_clip.py          ← slice 30s dev clips
│   ├── render/
│   │   ├── pitch.js                 ← renderer (Panel factory, anim loop,
│   │   │                              samples mode, draw mode)
│   │   └── style.css
│   └── model/
│       ├── config.py                ← all hyperparameters
│       ├── dataset.py               ← trajectory window Dataset
│       ├── tokenizer.py             ← coord proj + temporal/group/entity embeddings
│       ├── backbone.py              ← factorized spatiotemporal attention
│       ├── diffusion.py             ← DDPM schedule + ε-prediction network
│       ├── samplers.py              ← reverse-diffusion + causal rollout
│       ├── physics.py               ← physical-plausibility post-processor
│       ├── event_head.py            ← attention pool + classifier (parked)
│       └── lightning_module.py      ← LightningModule + DataModule
│   └── server/
│       └── main.py                  ← FastAPI inference server
├── scripts/
│   ├── smoke_train.py               ← 3-epoch smoke training on MPS
│   └── sample.py                    ← K-sample inference + physics + JSON dump
├── docs/
│   ├── product_vision.md            ← big picture + 2D first checkpoint
│   ├── mvp.md                       ← first-checkpoint scope discipline
│   ├── operations.md                ← deployment/runbook gaps
│   ├── cloud_training.md            ← Modal / Lambda / RunPod instructions
│   └── handoff_api.md               ← contract for downstream 3D video generator
├── data/
│   ├── metrica/                     ← raw clone of Metrica sample-data
│   └── processed/                   ← converted GenTac JSON + samples.json
├── checkpoints/                     ← Lightning checkpoints
├── .claude/{agents,skills}/         ← installed automation helpers
├── gentac.pdf  /  gentac.txt        ← the paper + extracted text
└── README.md
```

---

## Data format

All converted matches and all sample outputs share a single JSON schema. Coords
are in **meters from pitch center, y positive "up"**, x ∈ [-52.5, 52.5],
y ∈ [-34, 34].

### Match clip (ground truth)
```json
{
  "metadata": {
    "source": "metrica", "game_id": "Sample_Game_1", "fps": 25,
    "pitch": {"x": 105.0, "y": 68.0, "origin": "center", "y_axis": "up"},
    "team0": {"name": "Home", "players": ["Player1", "..."]},
    "team1": {"name": "Away", "players": ["Player15", "..."]},
    "n_frames": 145006
  },
  "frames": {
    "<frame_id>": {
      "period": 1,
      "ball":  [6.50, 4.20],
      "team0": {"Player1": [-0.74, -30.28]},
      "team1": {"Player15": [-25.95, 10.47]}
    }
  }
}
```

### Samples output (model inference)
```json
{
  "metadata": { ..., "mode": "unconditioned", "k": 20, "decision_frame": 15000, ... },
  "history":        {"<frame_id>": { ... }},
  "actual_future":  {"<frame_id>": { ... }},
  "samples":        [{"<frame_id>": { ... }}, ...]
}
```

The 2D renderer uses this per-frame structure as the first checkpoint output. A
downstream 3D video generator can use the same structure later through the M11
handoff contract.

---

## Open questions (need real answers before this becomes a product)

- **Which league(s) first?** Bundesliga is the most technically convenient
  (paper uses DFL data). Premier League is the biggest brand. MLS is the most
  accessible first conversation.
- **Which video generation service** might we plug into later? This only matters
  after the 2D counterfactual behavior is good enough to validate.
- **Data licensing.** Production needs a real tracking-data partnership
  (Hawk-Eye / Sportec / Stats Perform / SkillCorner). Hours-of-conversation
  with their data ops team are required before we can ship a paid product.
- **Compliance posture.** League deals require SOC 2, GDPR (EU leagues),
  player likeness rights (FIFPro alignment). Most of this kicks in *after*
  first revenue but the code we write today shouldn't block it.

---

## References

- GenTac paper: [arXiv:2604.11786](https://arxiv.org/abs/2604.11786)
- Metrica Sports sample data: [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)
- TimeSformer (backbone architecture): [arXiv:2102.05095](https://arxiv.org/abs/2102.05095)
- DDPM (diffusion math): [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
