# Project Functionality and Gap Analysis

Last reviewed: 2026-06-05.

## Project Goal

**B** is a tactical counterfactual engine for football. The long-term user can be
a coach, analyst, broadcaster, or player. They review previous footage, draw
arrows for possible movements or situations, and watch the whole game react from
trained game data so the right movement is visually obvious.

The first checkpoint is not hyper-realistic video. It is a 2D bird's-eye soccer
simulation that proves the same core behavior on a pitch view. The core workflow
is:

1. Load a frozen decision frame with preceding tracking history.
2. Draw an arrow showing what a player or ball should have done.
3. Generate an alternative future for all 22 players plus the ball.
4. Compare actual vs. generated trajectory in the 2D renderer.
5. Later, hand the same trajectory JSON to a downstream 3D/video generator if
   the 2D simulation proves useful.

The repo owns the trajectory brain and the first 2D simulation surface. It does
not need to own photorealistic video generation for checkpoint 1.

## Implemented Functionality

- **2D renderer:** `index.html` and `src/render/` provide a static Canvas tactics
  board with playback, trails, compare mode, arrow drawing, ball-pass arrows,
  sample browsing, and API-driven generation.
- **Inference API:** `src/server/main.py` provides FastAPI endpoints for health,
  generation, request validation, API-key gating, request IDs, structured logs,
  CORS allowlisting, queue bounds, and a streaming response variant.
- **Model implementation:** `src/model/` includes GenTac-style config, dataset
  loading, tokenizer, factorized spatiotemporal backbone, diffusion wrapper,
  samplers, event head, Lightning module, and physics post-processing.
- **Training and evaluation:** `scripts/` includes smoke training, full training,
  Modal validation/training, sampling, and quantitative eval helpers.
- **Tests:** `tests/` covers physics behavior, config/checkpoint schema
  regressions, server request validation, and eval helper behavior.
- **Business/product docs:** README plus `docs/product_vision.md`,
  `docs/mvp.md`, `docs/operations.md`, `docs/cloud_training.md`, and
  `docs/handoff_api.md` describe product vision, scope, operational posture,
  cloud training, and downstream trajectory handoff.
- **Landing page:** `site/` contains a static marketing site deployable to Vercel.

## Missing or Inconsistent Points

- **Data conversion code is missing from checkout.** README references
  `src/data/metrica_to_gentac.py` and `src/data/extract_clip.py`, but
  `src/data/` is absent. Without this, a fresh clone cannot regenerate
  `data/processed/*.json`.
- **No committed sample data or checkpoint.** This is correct for large artifacts,
  but the quick start and end-to-end local flows depend on regenerated
  `data/processed/` and `checkpoints/` files. The missing converter makes this a
  larger blocker.
- **CI covers pytest but not full local artifact regeneration.**
  `.github/workflows/tests.yml` now installs from `requirements.txt` and uses
  deterministic test fixtures, so it does not validate data conversion,
  checkpoint creation, or renderer/server artifact bootstrapping.
- **README project tree is stale.** It lists `src/data/`, `data/`, `checkpoints/`,
  and `.claude/`; those are not present in the current tracked checkout.
- **Handoff schema versions are inconsistent.** `docs/handoff_api.md` is titled
  v0.2, but the response example still says `"version": "0.1"` and examples mix
  v0.1/v0.2 arrow shapes.
- **Production deployment artifact is missing.** `docs/operations.md` correctly
  names the missing Dockerfile, environment promotion path, TLS setup, metrics,
  artifact capture, alerts, and rollback playbook.
- **Real model/data validation is missing.** The project still depends on a real
  trained model, real tracking data partnership, and analyst validation.
- **2D simulation quality is the first product gate.** Before 3D video matters,
  the bird's-eye output must show believable reactions from all entities after
  the user draws arrows.
- **No documented local artifact bootstrap.** There is no single doc that states
  exactly how to recreate `data/processed/`, smoke checkpoints, sample outputs,
  and the local renderer/server loop from a clean clone.
- **No security/compliance follow-through beyond code-level basics.** Request
  bounds, API key checks, CORS, queue caps, and path traversal tests exist, but
  TLS, key rotation, request body size limits, retention policy, data residency,
  and pen-test posture remain open.

## Recommended Markdown Additions

- **`CODEX.md`** - added. Gives future Codex sessions a compact project map,
  priorities, anti-scope, hard gaps, commands, and change guidance.
- **`docs/gap_analysis.md`** - added. Captures this functionality inventory and
  missing-points list in a durable project document.
- **`docs/product_vision.md`** - added. Captures the big picture and the first
  2D checkpoint in one short source-of-truth document.
- **`docs/local_bootstrap.md`** - still recommended. Should define a clean-clone
  path for dependencies, data conversion, smoke checkpoint creation, sample
  generation, server startup, renderer startup, and expected health checks.
- **`docs/decision_log.md`** - recommended once M0/M7 choices start changing
  direction. Keep dated decisions around target customer, data provider, model
  mode, schema changes, and deployment posture.

## Highest-Leverage Next Fixes

1. Restore or recreate `src/data/metrica_to_gentac.py` and
   `src/data/extract_clip.py`.
2. Add `docs/local_bootstrap.md` after the data converter path is restored.
3. Reconcile `docs/handoff_api.md` to one schema version and one arrow format.
4. Run the CI setup locally from a clean clone path and update docs to match
   what actually works.
