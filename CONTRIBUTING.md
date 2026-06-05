# Contributing

This repo is a tactical football simulation prototype. Changes should keep the
MVP loop intact: one analyst loads one moment, draws tactical intent, and gets a
plausible trajectory response that the 2D renderer can compare against reality.

## Development Setup

Use Python 3.11, matching GitHub Actions.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run the required local checks before opening or merging a change:

```bash
ruff check .
bandit -r src scripts -c pyproject.toml
pytest -q
```

## Coding Standards

- Keep changes scoped to the current product loop. Do not introduce a framework,
  service, database, build system, or training abstraction unless the feature
  directly needs it.
- Preserve the existing JSON contracts used by `src/server/main.py`,
  `scripts/sample.py`, `src/render/`, and `docs/handoff_api.md`.
- Treat checkpoint schema changes as breaking: bump
  `GenTacConfig.schema_version`, update config-schema tests, and document the
  retraining impact.
- Add or update tests when changing request validation, path handling,
  checkpoint loading, data parsing, sampling behavior, physics constraints, or
  evaluation metrics.
- Keep generated data, checkpoints, model weights, secrets, and customer files
  out of git.
- Prefer explicit bounds, typed request models, and deterministic fixtures over
  implicit assumptions or external network setup in tests.

## Security Requirements

- Production-like deployments must set `GENTAC_API_KEY`; unauthenticated mode is
  only for local development.
- Do not widen CORS to `*`. Add exact origins through `GENTAC_CORS_ORIGINS`.
- Any endpoint that reads a file path must resolve it under an explicit
  allowlisted directory and must have path traversal tests.
- Any request that can trigger model inference must have bounded input sizes and
  concurrency/queue controls.
- Treat checkpoints and match data as untrusted unless they came from a known
  pipeline. Do not load arbitrary user-supplied checkpoint paths.
- Never log API keys, raw customer tracking feeds, or full request bodies by
  default. Debug artifact capture must be opt-in and retention-limited.
- New dependencies must be added to `requirements.txt` or
  `requirements-dev.txt`, reviewed for necessity, and covered by dependency
  audit checks.

## Pull Request Checklist

- Tests or deterministic fixtures cover the changed behavior.
- `ruff`, `bandit`, and `pytest` pass locally or the PR explains why a check
  cannot run locally.
- Public API or JSON schema changes are reflected in docs.
- Security-sensitive changes preserve auth, CORS, path, request-bound, and
  queue/concurrency protections.
