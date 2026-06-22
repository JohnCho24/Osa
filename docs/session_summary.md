# GenTac — Training & Architecture Session Summary

_Worked on: trained the waypoint-conditioned diffusion model on real 2022 World Cup
tracking data, fixed several training/architecture issues, evaluated the result, wired
up the live GUI, and added two opt-in model variants._

---

## 1. Data — PFF FC 2022 World Cup → GenTac format

**What it is:** the free **PFF FC** release for the **2022 FIFA World Cup** — all **64 matches**.
Two zips in `~/Downloads/` (~2.9 GB): tracking (`.jsonl.bz2`, 29.97 fps, all 22 players + ball,
raw & smoothed), event data, metadata, and rosters.

**Converter built:** [src/data/pff_to_gentac.py](../src/data/pff_to_gentac.py) — streams the
`.bz2` (no full unzip) → `data/processed/<gid>.json`. Transforms:

- `homePlayers→team0`, `awayPlayers→team1` (jersey # as id), `balls[0]→ball`
- **smoothed** coords by default; resample **29.97 → 25 fps**
- rotate flipped periods 180° so **team0 always attacks +x** (uses `homeTeamStartLeft`)
- clamp to the pitch (keeps the `[-1, 1]` normalization invariant)
- join `Rosters/<gid>.json` on `shirtNumber` → per-player **roles** (GK/DEF/MID/FWD) in metadata

**Result:** 64/64 matches converted — **4.7 GB, 9.88 M frames (~110 match-hours)**, every match
has roles. Held-out val match = `data/processed/3859.json` (Cameroon vs Brazil).

---

## 2. Model architecture changes (all schema-versioned & backward-compatible on load)

### schema 4 — permutation invariance within a team
Replaced the per-slot `entity_emb` (an arbitrary jersey-number "name tag") with a per-position
**`role_emb`** (GK/DEF/MID/FWD/BALL/UNK) in [tokenizer.py](../src/model/tokenizer.py). Because role
is *content that travels with the player*, the spatial attention is now permutation-equivariant:
any team slot can hold any player and the prediction follows. Identity-across-time is carried by the
backbone's axial temporal attention, not the embedding — so dropping `entity_emb` cost nothing.
Locked by [tests/test_permutation_invariance.py](../tests/test_permutation_invariance.py).

### schema 5 — cosine noise schedule (full noising)
The original linear schedule (β 1e-4→0.02, 100 steps) only reached **ᾱ_T ≈ 0.36** — the "most-noised"
state still kept **60 %** of the real positions, so training denoised signal-bearing states while the
sampler started from pure `N(0, I)` (a train/inference mismatch). Switched the default to a **cosine
schedule** (`cfg.schedule_type="cosine"`, [diffusion.py](../src/model/diffusion.py)) → **ᾱ_T ≈ 0**
(terminal state is pure noise, matching the sampler). Added **x₀-clipping** to the reverse step
(samplers.py) — essential because cosine's large terminal β makes `1/√α` huge and unclipped steps
blow up (smoke went from `max|out|=1630` → `1.000`).

### Two opt-in variants (both schema 5, distinguished by persisted cfg → load side-by-side)
| Knob | Flag | Original default | New option |
|---|---|---|---|
| Frames predicted per diffusion pass | `--window` | `5` (0.2 s, autoregressive rollout) | `25` (1 s single-shot, less rollout drift) |
| Waypoint/arrow conditioning | `--conditioning` | `additive` (waypoint added onto the token) | `cross_attn` (separate condition tokens the future cross-attends to; future always denoises from noise) |

The cross-attention sublayer is built **only** when `conditioning="cross_attn"`, so the additive
model is byte-identical to the original (695,682 params vs 828,802 for cross-attn). Verified: all 4
combos do forward→loss→backward, cross-attn params receive gradients, permutation-invariance holds for
both conditioning modes, and the full test suite passes.

---

## 3. Training

**Environment:** the **`RoboPRO`** conda env (CUDA torch 2.7 + lightning/fastapi added) on an
**RTX 5080 Laptop GPU (16 GB)**. Batch capped at ~96 (factorized attention is ~140 MB/sample);
throughput is launch-bound (~300 samp/s with `torch.compile`, ~22 min/epoch).

**The LR story (why the first runs failed):**
1. **lr 1e-3 (paper, batch 200) → diverged.** Loss trained well during warm-up (0.08) then climbed to
   exactly **2.0** (the "predict-zero-noise" floor) as LR hit peak. Cause: 1e-3 is too hot for **batch 64**
   (3× noisier gradients than batch 200). Added a `--lr` flag.
2. **lr 3e-4 → healthy but spiked.** Recovered from one spike, then a second, growing spike (val 0.0044 → 0.27).
3. **lr 1e-4 → stable.** Smooth monotonic decrease, val tracked train, no spikes.

**Final original model:** `checkpoints/full/best-09-0.0003.ckpt` (val/loss 0.00029, ~epoch 9).
Note: with the cosine schedule the absolute loss is small *because* ε-prediction is easy here
(strong history/opponent context + waypoint leakage + high-noise≈copy) — judge by **eval**, not loss.

---

## 4. Evaluation (held-out match 3859, best-09)

| Horizon | ADE | FDE | note |
|---|---|---|---|
| 1 s | ~0.4 m | ~1.0 m | excellent short-term |
| 2 s | 1.25 m | 3.02 m | solid |
| 4 s | 4.27 m | 11.37 m | autoregressive drift (expected) |

- **Realism discriminator: 0.57** (0.5 = indistinguishable from real PFF tracking) → motion looks real.
- **Diversity** opens up with horizon (0.10 m @ 1 s → 0.71 m @ 4 s) — samples fork, not collapsed.
- **off-pitch = 0** everywhere (x₀-clip holds); per-role ADE consistent (ball hardest).
- **Arrow adherence (2 s):** distance-to-target & hit-rate@2 m by guidance scale —
  1.0: 67 % · **2.0: 77 % (sweet spot)** · 3.5: 67 % · 5.0: 70 % (over-pushing). CFG behaves exactly as designed.
  > The earlier scary "0 % hit" was an artifact of testing at a 4 s horizon with guidance 1.0 only.

---

## 5. How a drawn arrow becomes conditioning

- **Tokens:** a "token" = one **(frame, entity)** cell. Input = `(history 100 + window 5) × 23 = 2,415`
  tokens; output (noise head) = `window × 23 = 115` (original window).
- **Arrow → waypoint:** the GUI sends only the arrow's **endpoint** `{player, to:[x,y]}`; the server makes
  one waypoint `(entity, last_frame, target)`; the sampler **interpolates** it into a per-frame straight-line
  path (current→target). So the model receives "sampled points" — but they're synthesized server-side from the
  two endpoints, not from your drawing.
- **Two channels:** the arrowed player's position is **clamped** in `x` (hard pin, GUI default `fade_frames=horizon-1`)
  **and** its intent is broadcast as a **waypoint embedding** so the other 21 players react. In hard-pin mode the
  model's output for the arrowed player is overwritten; in soft mode (`fade_frames=0`, what the eval used) the model
  predicts the arrowed player itself (the 77 %-on-target number).
- **Horizon:** 5 frames = 0.2 s is just the per-pass chunk; `causal_rollout` chains windows autoregressively to any
  horizon (`horizon_frames/25 = seconds`). The window-25 variant predicts a full second per pass to avoid that drift.

---

## 6. GUI

- **Stack:** [index.html](../index.html) (toolbar + two canvas panels) → [src/render/pitch.js](../src/render/pitch.js)
  (all logic: render, playback, pointer-drawn arrows, `fetch` to the model) → FastAPI server
  [src/server/main.py](../src/server/main.py) on `:8001`. Static page served by `http.server :8000`.
- **Run it:**
  ```bash
  # 1. inference server, pointed at the trained checkpoint
  GENTAC_CKPT=checkpoints/full/best-09-0.0003.ckpt PYTHONPATH=. \
    /home/joshua/miniconda3/envs/RoboPRO/bin/python -m uvicorn src.server.main:app --host 127.0.0.1 --port 8001
  # 2. static server (from repo root)
  python3 -m http.server 8000 --bind 127.0.0.1
  # 3. open http://localhost:8000  → check Draw → click+drag a player → Generate ▶
  ```
  A 300-frame PFF clip (game 3859) was written to `data/processed/Sample_Game_1_clip.json` (the
  frontend's hardcoded default path) so it loads real World-Cup data.
- **Guidance scale** in the request changed `1.0 → 2.0` (eval's sweet spot).
- **Other branches** (`datas-to-train`, `draft/demo-video-section`) have prettier boards (curved arrows,
  11 pitch themes, demo videos) but are **frontend simulations / marketing — not wired to the model**.
  The current branch's `pitch.js` is the only GUI that shows real predictions.

---

## 7. Key files changed

- **New:** [src/data/pff_to_gentac.py](../src/data/pff_to_gentac.py), [src/data/__init__.py](../src/data/__init__.py),
  [tests/test_permutation_invariance.py](../tests/test_permutation_invariance.py)
- **Model:** [config.py](../src/model/config.py) (roles, cosine schedule, conditioning flag, schema 4→5),
  [tokenizer.py](../src/model/tokenizer.py) (role_emb, condition split), [backbone.py](../src/model/backbone.py)
  (cross-attn sublayer), [diffusion.py](../src/model/diffusion.py) (cosine schedule + factory, condition routing),
  [dataset.py](../src/model/dataset.py) (role_idx), [lightning_module.py](../src/model/lightning_module.py),
  [samplers.py](../src/model/samplers.py) (role_idx, x₀-clip)
- **Scripts/server:** [train_full.py](../scripts/train_full.py) (`--lr/--window/--conditioning`),
  [eval.py](../scripts/eval.py) (CUDA device fix, role_idx), [sample.py](../scripts/sample.py),
  [server/main.py](../src/server/main.py), [render/pitch.js](../src/render/pitch.js) (guidance 2.0)

---

## 8. Command cheat-sheet

```bash
RP=/home/joshua/miniconda3/envs/RoboPRO/bin/python

# Convert all 64 matches (after unzipping into data/pff/)
PYTHONPATH=. python3 -m src.data.pff_to_gentac --pff-root "data/pff/FIFA World Cup 2022" --games all

# Train ORIGINAL (additive, 0.2 s window) — this produced best-09
PYTHONPATH=. $RP scripts/train_full.py --batch-size 64 --precision bf16-mixed --num-workers 8 --lr 1e-4

# Train v2 (cross-attention + 1 s window), separate ckpt dir
PYTHONPATH=. $RP scripts/train_full.py --batch-size 64 --precision bf16-mixed --num-workers 8 --lr 1e-4 \
  --window 25 --conditioning cross_attn --ckpt-dir checkpoints/full_v2

# Evaluate (ADE/FDE/diversity/arrow adherence)
PYTHONPATH=. $RP scripts/eval.py --ckpt checkpoints/full/best-09-0.0003.ckpt \
  --match data/processed/3859.json --k 6 --n-frames 5 --horizon 50 --guidance-scales 1.0,2.0,3.5,5.0
```

---

## 9. Open items / next steps

- **Run the v2 training** and A/B its 1 s/2 s ADE + arrow adherence vs `best-09` (expect lower drift).
- For cross-attn inference, use request `fade_frames: 0` (let cross-attention drive, not the hard pin).
- Nothing is committed — all changes are in the working tree on branch `model_training`.
- Possible future work: min-SNR / v-prediction loss (so the loss number tracks quality), scheduled
  sampling (fix autoregressive exposure bias), wiring the pretty `product.js` board to the real server.
