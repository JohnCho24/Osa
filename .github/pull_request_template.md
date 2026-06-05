## Summary

-

## Verification

- [ ] `ruff check .`
- [ ] `bandit -r src scripts -c pyproject.toml`
- [ ] `pytest -q`

## Security / Data Checklist

- [ ] No secrets, checkpoints, generated data, or customer artifacts committed.
- [ ] Request bounds, path guards, auth, CORS, and queue/concurrency controls remain intact.
- [ ] Public API or JSON schema changes are reflected in docs.
