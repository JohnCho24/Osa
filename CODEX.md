# Codex Working Guide

This repo is an internal prototype for **B**, a tactical football "what-if"
engine. The big picture is:

- A coach, analyst, broadcast team, or player reviews previous match footage.
- They draw arrows for potential movements, passes, presses, drops, or tactical
  situations.
- The system resimulates the game from that point, based on trained tracking
  data, so all players and the ball react realistically.
- Viewers learn how to move or react because they visually see the alternative
  play unfold.

The first checkpoint is a **2D bird's-eye soccer simulation**, not a
hyper-realistic video. The near-term product loop is:

1. Load a tracked match moment.
2. Let an analyst draw player or ball-pass arrows on the 2D pitch.
3. Generate a physically plausible alternative trajectory with a GenTac-style
   diffusion model.
4. Return `samples.json`-shaped trajectory JSON for side-by-side 2D review.

The highest-priority product question is still validation: can the 2D simulation
make the counterfactual clear enough that a real analyst can use it in a
coaching or player-review workflow?

## Where Things Live

- `index.html`, `src/render/` - local 2D tactics board and compare UI.
- `src/server/main.py` - FastAPI inference API.
- `src/model/` - GenTac reimplementation, waypoint conditioning, sampling,
  physics post-processing, Lightning training module.
- `scripts/` - smoke/full/cloud training, sampling, and evaluation entrypoints.
- `tests/` - focused regression tests for config schema, server validation,
  physics, and eval helpers.
- `site/` - static public landing page.
- `docs/` - product vision, product scope, operations, cloud training, handoff
  API, reports, and gap analysis.

## Do Not Drift

- Do not turn this into a general ML platform. The MVP is one analyst drawing
  one or more arrows on one football moment and getting one useful alternative.
- Do not build photorealistic video generation in this repo. The repo owns
  trajectory generation and the handoff contract.
- Do not optimize the later broadcaster/player/fan packaging before the 2D
  counterfactual simulator works.
- Keep the output schema compatible with the renderer and `docs/handoff_api.md`.

## Current Hard Gaps

- No committed `data/` or `checkpoints/` artifacts. They are intentionally
  gitignored, but most end-to-end flows require regenerated local files.
- `src/data/` converters are referenced by README and CI, but are not present in
  this checkout. Restore or recreate them before relying on CI or quick-start
  setup.
- No dependency manifest exists. Use the install commands in README/CI until a
  `requirements.txt` or `pyproject.toml` is added.
- No Dockerfile or deploy artifact exists.
- No real trained model or real league data partnership exists.

## Useful Commands

Set up local Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch pytorch-lightning einops numpy fastapi 'uvicorn[standard]' pytest
```

Run tests:

```bash
pytest -q
```

Start the local inference server after a checkpoint exists:

```bash
uvicorn src.server.main:app --host 127.0.0.1 --port 8001
```

Start the renderer from the repo root:

```bash
python3 -m http.server 8000
```

## Change Guidance

- Prefer narrow, testable changes around the current MVP loop.
- Add tests when touching request validation, checkpoint schema behavior,
  sampling, physics, or eval metrics.
- Keep server request bounds strict. Avoid reopening CORS to `*`.
- Treat checkpoint schema changes as breaking: bump `GenTacConfig.schema_version`
  and add/update config-schema tests.
- For renderer changes, keep the static-file workflow intact unless a build
  system is deliberately introduced.
