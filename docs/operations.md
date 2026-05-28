# Operations runbook

*What "production" actually looks like and what's missing from getting there.
Honest accounting, not aspirational.*

---

## Current state (2026-05-28)

There is no production. The inference server runs on the founder's laptop. This
document exists so the gap to a real deployment is named, not hidden.

| Layer | Status | Where it lives |
|---|---|---|
| Inference server | Built, hardened, **not deployed anywhere** | `src/server/main.py` |
| Renderer | Built, **served by `python -m http.server`** | `index.html` + `src/render/` |
| Landing page | Built, **deployable to Vercel**, not yet deployed | `site/` |
| Model checkpoint | Smoke only (3 epochs, 2 matches), on local disk | `checkpoints/smoke/` |
| CI | **Shipped** as of 2026-05-28 | `.github/workflows/tests.yml` |
| Monitoring / alerting | **None** | — |
| Authentication | API-key middleware ready, **not enforced** (env var unset) | server lifespan |
| HTTPS / TLS | Not configured | — |

---

## What "first paid customer" deployment looks like

The minimum viable production environment for one design partner. Designed
to be operable by a single founder, not an SRE team.

### Topology

```
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│  Customer       │ HTTPS │  Cloudflare      │ HTTPS │  Single VM (A10G)    │
│  browser /      │ ────► │  (TLS, WAF,     │ ────► │  Ubuntu 22, systemd  │
│  API client     │       │   rate limit)   │       │   • uvicorn (port    │
│                 │       │                  │       │     127.0.0.1:8001) │
└─────────────────┘       └──────────────────┘       │   • caddy (TLS)     │
                                                     │   • model in RAM    │
                                                     └──────────────────────┘
                                                              │
                                                     ┌──────────────────────┐
                                                     │  S3 bucket           │
                                                     │   • checkpoints/     │
                                                     │   • request artifacts│
                                                     │   • access logs      │
                                                     └──────────────────────┘
```

### Cost envelope (per month)

- 1× A10G VM (Lambda, RunPod, or comparable): **~$200–$400** depending on uptime
- Cloudflare: free tier
- S3 bucket + transfer: **~$10**
- Domain + monitoring (UptimeRobot free or Better Stack ~$30): **~$30**
- **Total: ~$250–$450/month** per single-tenant customer deployment

### What's NOT in this topology (and why)

- **GPU autoscaling.** A single A10G handles the design-partner workload alone. Autoscaling adds 2 weeks of work and a Kubernetes dependency we don't need yet.
- **Multi-tenancy.** Single VM per customer until customer #3. Until then the operational simplicity of "one VM, one customer, one keypair" is worth the unit-economics hit.
- **DR / backup region.** S3 is cross-AZ. Checkpoint corruption is recoverable from the training pipeline. We accept downtime up to 4h for a regional outage.
- **Custom domain CDN for the renderer.** Vercel handles it.

---

## Observability

### Shipped
- Per-request access log line: `{"ts","level","logger","msg","req_id","method","path","status","ms"}` ([src/server/main.py](../src/server/main.py))
- Per-request UUID propagated as `X-Request-Id` (request + response header)
- Schema/version logged at server boot
- `/api/health` endpoint (200 OK + device name)

### Missing (in priority order)
1. **Request artifact capture.** When a generation is weird, we currently have no way to inspect the inputs. **Fix:** wrap `_generate_sync` to dump request body + first/last frames of output to S3 under `artifacts/<req_id>.json` whenever `?debug=1` is set or when latency > P95.
2. **Metrics export.** No `/metrics` endpoint. **Fix:** add `prometheus_client` + scrape from Better Stack or Grafana Cloud. Top counters: requests-by-status, queue depth, generation latency histogram, model load time.
3. **Per-customer rate limiting.** API key bounded by request count, not concurrent. **Fix:** wrap auth middleware with `slowapi` keyed on `X-API-Key`.
4. **Alerts.** Nothing pages anyone. **Fix:** Better Stack rules — health check fails > 5 min, P95 latency > 30s for 10 min, 5xx rate > 1%.

---

## Deploy / rollback

### Shipped
- `pl.seed_everything` makes training reproducible.
- Schema versioning refuses incompatible checkpoint loads.
- `train_full.py --dry-run` and `modal_train.py --validate-local` validate the code path before cloud spend.

### Missing (in priority order)
1. **Dockerfile.** Reproducible deploy artifact. The current "deploy" is `git pull && systemctl restart`.
2. **Environment promotion.** dev → staging → prod. Today there's only "founder's laptop." **Fix:** copy the single-VM topology twice (staging on a smaller spot instance) and promote checkpoints by S3 prefix.
3. **Blue-green for checkpoint swaps.** Loading a new checkpoint requires a server restart; existing in-flight requests die. **Fix:** load new checkpoint in a sibling process, swap traffic via Cloudflare load balancer, drain old.
4. **Rollback playbook.** If a new checkpoint is bad, how do we revert in < 5 min? **Fix:** S3-prefixed checkpoints with a `current` symlink; rollback = repoint the symlink, restart.

---

## On-call posture

There is no on-call rotation. The founder is on-call by default 24/7 for any
design-partner customer. This is fine for the first 1-2 customers. By the third,
hire someone or set explicit SLA hours (e.g. business hours Mon-Fri, best-effort
otherwise) in the contract.

**Pager:** PagerDuty free tier or Better Stack alerts → personal phone via SMS.

**First-response targets:** P95 latency alert: 30 min. 5xx alert: 15 min. Health
check fail: 5 min. (These are aspirations; nothing enforces them.)

---

## Security checklist before first paying customer

- [x] Path-traversal guard ([test](../tests/test_server_validation.py))
- [x] Pydantic request bounds ([test](../tests/test_server_validation.py))
- [x] Concurrency semaphore + queue cap (no DoS via flood)
- [x] CORS allowlist (not `*`)
- [x] API key authentication code path (must be enabled by setting `GENTAC_API_KEY`)
- [x] Schema-versioned checkpoint loading (no silent misload)
- [x] Constant-time API key comparison (`secrets.compare_digest`)
- [ ] **HTTPS / TLS termination** (Caddy or Cloudflare) — required for any non-localhost deployment
- [ ] **Per-customer API key rotation procedure** — currently no rotation flow
- [ ] **Request body size limit** (FastAPI default is generous; cap at 1 MB)
- [ ] **Audit log retention policy** — currently logs to stdout, no retention
- [ ] **Data residency posture** (EU customer = EU region; not handled)
- [ ] **Pen test** before any contract over $50K ARR
- [ ] **SOC 2 Type I** — defer until first customer requires it; ~$30K + 6 months when triggered

---

## Cost-at-scale envelope

If we have 10 paying league/club customers each generating 500 alternatives/day:

- Requests: 5,000/day = ~58/hr peak (assume 4× peak hour multiplier)
- Per request: ~8s on A10G at K=1
- Concurrent peak: ~58 × 8 / 3600 = ~0.13 (effectively 1 GPU is fine)
- **Single A10G handles 10 customers** with room to spare
- If usage grows to 50 customers: 2× A10Gs in load balance, < $1K/month
- The unit economics work cleanly up to ~100 customers per A10G **for the trajectory layer**. The 3D video vendor cost dominates beyond that; not our line item.

---

## Hiring profile (when, what, why)

| Trigger | Hire | Why |
|---|---|---|
| First signed evaluation agreement (M0) | None — founder runs it | Validates demand |
| First $25K+ contract | Part-time ML engineer (contract) | Run M7 properly + maintain checkpoint pipeline |
| Second paying customer | Full-time ML engineer + part-time BD | Customer ops + sales motion |
| $500K ARR | Full-time BD + part-time SRE | Sales reach + infra maturation |
| $1.5M ARR | SOC 2 audit + compliance hire | Procurement gate |

---

## What this document does NOT cover

- **Database schema** — we have none. Match data is on-disk JSON; checkpoints on disk or S3. Stays that way until the first customer demands per-user state.
- **Disaster recovery RTO/RPO** — for design-partner-tier customers, 4h RTO and 24h RPO are acceptable.
- **Multi-region failover** — premature.
- **Federated learning across leagues** — interesting future tech, not relevant to operations today.
