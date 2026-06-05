# Security Policy

## Supported Status

This project is an internal prototype and has no production deployment yet. The
security baseline below is still mandatory for any customer demo, design-partner
deployment, or externally reachable environment.

## Reporting Vulnerabilities

Do not open public issues for vulnerabilities, secrets, customer data exposure,
or model/checkpoint integrity concerns. Report them privately to the repository
owner and include:

- affected file, endpoint, workflow, or deployment surface
- reproduction steps or proof of concept
- expected impact and affected data
- suggested fix, if known

## Required Controls Before External Deployment

- Enable API key auth by setting `GENTAC_API_KEY`.
- Serve only over HTTPS behind a trusted reverse proxy or platform TLS layer.
- Set `GENTAC_CORS_ORIGINS` to exact customer/frontend origins.
- Keep generation request bounds, path traversal guards, concurrency caps, and
  queue limits enabled.
- Keep `GENTAC_MAX_BODY_BYTES` enabled and mirror the same or a lower limit at
  the reverse proxy.
- Store checkpoints and match data in access-controlled storage, not git.
- Keep API keys and cloud credentials in secret storage, not environment files
  committed to the repo.
- Retain logs and debug artifacts only as long as required for support or
  customer contract terms.

## Data Handling

Football tracking data can identify players, team tactics, and proprietary
match strategy. Treat all customer tracking feeds, generated trajectories,
checkpoints trained on customer data, request artifacts, and logs as confidential.

Minimum handling rules:

- no customer data in tests, docs, screenshots, or public artifacts
- no raw request/response body logging by default
- no cross-customer data mixing without written approval
- delete local customer artifacts when an evaluation ends
- choose deployment regions that match customer data-residency requirements

## Dependency And Supply Chain

- Runtime dependencies live in `requirements.txt`.
- Development and security tooling lives in `requirements-dev.txt`.
- Dependabot tracks Python and GitHub Actions updates.
- The standards workflow runs Ruff, Bandit, and dependency audit checks on
  pushes and pull requests.
- Checkpoint files are executable deserialization inputs in practice; only load
  checkpoints produced by this repository's trusted training pipeline.

## Incident Response

For a suspected security incident:

1. Rotate affected API keys and cloud credentials.
2. Disable external traffic or put the deployment behind an allowlist.
3. Preserve relevant logs and request IDs.
4. Identify affected data, customers, checkpoints, and generated artifacts.
5. Patch, test, redeploy, and document the root cause.
