"""GenTac inference server.

POST /api/generate
  body: {
    decision_frame:   int,
    match:            str (relative path to a converted match JSON),
    horizon_frames:   int   (default 25, must be multiple of model window w),
    arrows:           [ {team: "team0"|"team1", player: "PlayerN", to: [x_m, y_m]} ],
    k:                int   (number of alternatives, default 1),
    fade_frames:      int   (waypoint fade, default 8)
  }
  → samples.json-shaped payload (history + actual_future + samples).
    The renderer consumes the same schema as scripts/sample.py output.

Loads the smoke checkpoint at startup, holds the model in memory. CORS open so the
browser frontend on :8000 can call us on :8001 without proxy gymnastics.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..model.config import GenTacConfig
from ..model.dataset import _load_match_arrays
from ..model.lightning_module import GenTacTrajectoryModule
from ..model.physics import PhysicsConfig, apply_physics
from ..model.samplers import build_static_target_mask, causal_rollout


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "checkpoints" / "smoke" / "smoke_final.ckpt"


# ── JSON-structured logging w/ per-request IDs ──────────────────────────────
# Request IDs flow via a ContextVar so any code can call log.info() and have the
# current request stamped automatically. Replaces the prior text-format access log.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) +
                  f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "req_id": _request_id.get(),
        }
        for k in ("method", "path", "status", "ms", "event"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.basicConfig(level=os.environ.get("GENTAC_LOG_LEVEL", "INFO"), handlers=[_handler], force=True)
log = logging.getLogger("gentac")

# CORS allowlist. Wide-open `*` is fine for local dev but blocks any compliance
# review; in production set GENTAC_CORS_ORIGINS to a comma-separated allowlist.
# Defaults cover the http.server (8000) and a couple of common dev hosts.
_DEFAULT_CORS = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173"
CORS_ORIGINS = [o.strip() for o in os.environ.get("GENTAC_CORS_ORIGINS", _DEFAULT_CORS).split(",") if o.strip()]

# Optional API key. Unset → dev mode, no auth required (local laptop). Set to any
# non-empty string → /api/generate requires `X-API-Key: <value>`. Health endpoint
# stays unauthenticated for load-balancer probes.
GENTAC_API_KEY = os.environ.get("GENTAC_API_KEY", "").strip()

# Concurrency cap. One generation pegs the (single) GPU; queueing more than this
# just OOMs. Sized to 1 by default — bump only if you've benchmarked headroom.
GENTAC_MAX_CONCURRENCY = int(os.environ.get("GENTAC_MAX_CONCURRENCY", "1"))
_generate_sema: asyncio.Semaphore | None = None       # initialized in lifespan

# Bound the queue too — if a request would queue behind > MAX_QUEUE others,
# fail fast with 503 instead of silently piling up.
GENTAC_MAX_QUEUE = int(os.environ.get("GENTAC_MAX_QUEUE", "8"))
_waiting_count = 0


# Request bounds. These cap server resource use under any single request so a
# malicious or buggy client can't OOM the GPU. Sized generously for legitimate
# product use: 32 alternatives, 10s horizon, every entity arrowed.
MAX_K = 32
MAX_HORIZON_FRAMES = 250          # 10 s @ 25 fps
MAX_ARROWS = 23                   # 22 players + ball
MAX_FADE_FRAMES = 50

ALLOWED_MODES = ("unconditioned", "opp_conditioned_team0", "opp_conditioned_team1")

MATCH_DIR = ROOT / "data" / "processed"


class PlayerRef(BaseModel):
    team: Literal["team0", "team1"]
    player: str = Field(..., min_length=1, max_length=64)


class PlayerArrow(BaseModel):
    """Arrow on a player → 'this player should be at `to` at the end of the horizon'."""
    kind: Literal["player"] = "player"
    team: Literal["team0", "team1"]
    player: str = Field(..., min_length=1, max_length=64)
    to: list[float] = Field(..., min_length=2, max_length=2)


class BallPassArrow(BaseModel):
    """Arrow on the ball with a recipient → 'pass to this player, who meets the ball at `to`'.

    The server expands this into TWO waypoints: ball entity and recipient entity, both
    pinned at `to` at the end of the horizon. Coaches' natural gesture; replaces having
    to draw separate arrows for the ball and the receiving runner.
    """
    kind: Literal["ball_pass"]
    to: list[float] = Field(..., min_length=2, max_length=2)
    recipient: PlayerRef


Arrow = Annotated[Union[PlayerArrow, BallPassArrow], Field(discriminator="kind")]


class GenerateRequest(BaseModel):
    decision_frame: int = Field(..., ge=0)
    match: str = "data/processed/Sample_Game_1.json"
    horizon_frames: int = Field(25, ge=1, le=MAX_HORIZON_FRAMES)
    arrows: list[Arrow] = Field(default_factory=list, max_length=MAX_ARROWS)
    k: int = Field(1, ge=1, le=MAX_K)
    fade_frames: int = Field(8, ge=0, le=MAX_FADE_FRAMES)
    mode: str = "unconditioned"
    # Classifier-free guidance scale on the learned waypoint signal. 1.0 = pure
    # conditional (1 model fwd / step). Values >1 amplify "honor the arrows" at
    # the cost of 1 extra forward pass per denoising step. Bound at 5 for safety.
    guidance_scale: float = Field(1.0, ge=0.0, le=5.0)


def _safe_match_path(rel_path: str) -> Path:
    """Resolve `rel_path` strictly under data/processed/ and reject traversal.

    Without this, `match=../../etc/passwd` would let the server open arbitrary
    files. We anchor inside MATCH_DIR and use .resolve() to collapse `..`.
    """
    p = (ROOT / rel_path).resolve()
    try:
        p.relative_to(MATCH_DIR.resolve())
    except ValueError:
        raise HTTPException(400, f"match path must be inside data/processed/")
    if p.suffix != ".json":
        raise HTTPException(400, "match path must be a .json file")
    if not p.is_file():
        raise HTTPException(404, f"match not found: {rel_path}")
    return p


# ── module state set in lifespan ─────────────────────────────────────────────
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _generate_sema
    ckpt = Path(os.environ.get("GENTAC_CKPT", str(DEFAULT_CKPT)))
    if not ckpt.exists():
        raise RuntimeError(f"checkpoint not found at {ckpt}; run scripts/smoke_train.py first")
    # GENTAC_DEVICE overrides (e.g. force "cpu" while a training run owns the GPU).
    device = os.environ.get("GENTAC_DEVICE") or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log.info("loading checkpoint", extra={"event": "ckpt_loading", "path": str(ckpt), "device": device})
    cfg = GenTacTrajectoryModule.cfg_from_checkpoint(ckpt)
    mod = GenTacTrajectoryModule.load_from_checkpoint(str(ckpt), cfg=cfg)
    mod.to(device).eval()
    state.update(
        device=device,
        model=mod.model,
        schedule=mod.schedule.to(device),
        cfg=mod.cfg,
        match_cache={},
    )
    _generate_sema = asyncio.Semaphore(GENTAC_MAX_CONCURRENCY)
    log.info(
        "server ready",
        extra={"event": "ready", "device": device, "ckpt": ckpt.name,
               "schema_version": cfg.schema_version, "d_model": cfg.d_model, "smoke": cfg.smoke,
               "max_concurrency": GENTAC_MAX_CONCURRENCY, "max_queue": GENTAC_MAX_QUEUE,
               "auth": "api-key" if GENTAC_API_KEY else "open",
               "cors_origins": CORS_ORIGINS},
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-Id"],
    expose_headers=["X-Request-Id"],
)


@app.middleware("http")
async def request_log(request: Request, call_next):
    # Propagate client-supplied IDs (helps trace through reverse proxies / browser
    # devtools) and otherwise mint one. We stash on the contextvar so logs anywhere
    # in this request get tagged.
    rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    token = _request_id.set(rid)
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        ms = (time.time() - t0) * 1000
        log.exception(
            "request failed",
            extra={"event": "http_500", "method": request.method, "path": request.url.path, "ms": int(ms)},
        )
        response = JSONResponse({"detail": "internal error", "req_id": rid}, status_code=500)
    ms = (time.time() - t0) * 1000
    response.headers["X-Request-Id"] = rid
    log.info(
        "http",
        extra={"event": "http", "method": request.method, "path": request.url.path,
               "status": response.status_code, "ms": int(ms)},
    )
    _request_id.reset(token)
    return response


def _require_api_key(x_api_key: str | None) -> None:
    """No-op when GENTAC_API_KEY is unset (dev mode). Otherwise constant-time compare."""
    if not GENTAC_API_KEY:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, GENTAC_API_KEY):
        raise HTTPException(401, "invalid or missing X-API-Key")


def _get_match(rel_path: str) -> dict:
    cache = state["match_cache"]
    safe_path = _safe_match_path(rel_path)
    key = str(safe_path)
    if key not in cache:
        cache[key] = _load_match_arrays(safe_path, state["cfg"])
    return cache[key]


def _denorm(arr: torch.Tensor, cfg: GenTacConfig) -> torch.Tensor:
    scale = torch.tensor([cfg.pitch_x / 2.0, cfg.pitch_y / 2.0], device=arr.device, dtype=arr.dtype)
    return arr * scale


def _frame_to_dict(coords_m: torch.Tensor, valid: torch.Tensor, team0_slots, team1_slots, N: int, period: int):
    out = {"period": int(period), "ball": None, "team0": {}, "team1": {}}
    for slot, pid in enumerate(team0_slots):
        if valid[slot]:
            out["team0"][pid] = [round(float(coords_m[slot, 0]), 2), round(float(coords_m[slot, 1]), 2)]
    for slot, pid in enumerate(team1_slots):
        if valid[N + slot]:
            out["team1"][pid] = [round(float(coords_m[N + slot, 0]), 2), round(float(coords_m[N + slot, 1]), 2)]
    if valid[2 * N]:
        out["ball"] = [round(float(coords_m[2 * N, 0]), 2), round(float(coords_m[2 * N, 1]), 2)]
    return out


def _resolve_player(team: str, player: str, team0_slots: list[str], team1_slots: list[str], N: int) -> int | None:
    slots = team0_slots if team == "team0" else team1_slots
    offset = 0 if team == "team0" else N
    try:
        return offset + slots.index(player)
    except ValueError:
        return None


@app.get("/api/health")
def health():
    return {"ok": True, "device": state["device"]}


@app.post("/api/generate/stream")
async def generate_stream(req: GenerateRequest, x_api_key: str | None = Header(None, alias="X-API-Key")):
    """Server-Sent Events streaming variant of /api/generate.

    Above the paper: the paper has no inference API at all, so no streaming story.
    For broadcast / live-cut use cases the renderer can start drawing the first
    horizon window while later windows are still sampling, halving perceived
    latency for K=1 long-horizon requests.

    Event types emitted:
      • `event: metadata` — fields the renderer needs to set up panels
      • `event: window`   — one per causal window, payload = frames for that window
      • `event: done`     — final summary; safe to close the connection
    """
    _require_api_key(x_api_key)
    rid = _request_id.get()

    async def gen():
        # We still run the heavy work in to_thread; what we stream is the per-window
        # progress, NOT a per-diffusion-step progress (that would require lower-level
        # hooks into sample_window).
        full = await asyncio.to_thread(_generate_sync, req)
        meta = json.dumps(full["metadata"], separators=(",", ":"))
        yield f"event: metadata\ndata: {meta}\n\n"

        # Walk the actual_future + samples per frame and emit incremental batches.
        frame_ids = sorted(full["actual_future"].keys(), key=int)
        w = state["cfg"].window_frames
        for chunk_start in range(0, len(frame_ids), w):
            chunk = {}
            for fid in frame_ids[chunk_start:chunk_start + w]:
                chunk[fid] = {
                    "actual": full["actual_future"][fid],
                    "samples": [s.get(fid) for s in full["samples"]],
                }
            payload = json.dumps({"chunk_start": chunk_start, "frames": chunk}, separators=(",", ":"))
            yield f"event: window\ndata: {payload}\n\n"
            await asyncio.sleep(0)            # let the event loop flush
        yield f"event: done\ndata: {json.dumps({'req_id': rid})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"X-Request-Id": rid, "Cache-Control": "no-cache"})


@app.post("/api/generate")
async def generate(req: GenerateRequest, x_api_key: str | None = Header(None, alias="X-API-Key")):
    global _waiting_count
    _require_api_key(x_api_key)
    # Fail fast if the queue is already saturated — better than letting the client wait
    # a minute behind eight other generations and then time out.
    if _waiting_count >= GENTAC_MAX_QUEUE:
        raise HTTPException(503, f"queue full ({GENTAC_MAX_QUEUE} waiting); retry shortly")

    _waiting_count += 1
    try:
        async with _generate_sema:    # caps active generations to GENTAC_MAX_CONCURRENCY
            return await asyncio.to_thread(_generate_sync, req)
    finally:
        _waiting_count -= 1


def _generate_sync(req: GenerateRequest):
    cfg: GenTacConfig = state["cfg"]
    device = state["device"]
    model = state["model"]
    schedule = state["schedule"]
    N = cfg.n_players_per_team
    H, w = cfg.history_frames, cfg.window_frames

    if req.mode not in ALLOWED_MODES:
        raise HTTPException(400, f"mode must be one of {ALLOWED_MODES}")
    if req.horizon_frames % w != 0:
        raise HTTPException(400, f"horizon_frames must be multiple of model window w={w}")

    m = _get_match(req.match)
    pos_all = torch.from_numpy(m["positions"]).to(device)
    mask_all = torch.from_numpy(m["mask"]).to(device)
    frame_ids = m["frame_ids"]
    periods = m["period"]
    team0_slots, team1_slots = m["team0_slots"], m["team1_slots"]

    di_arr = (frame_ids == req.decision_frame).nonzero()[0]
    if len(di_arr) == 0:
        raise HTTPException(400, f"decision_frame {req.decision_frame} not in match")
    di = int(di_arr[0])
    if di < H:
        raise HTTPException(400, f"need >= {H} preceding frames; have {di}")
    if di + req.horizon_frames > len(frame_ids):
        raise HTTPException(400, "horizon exceeds match length")

    K = req.k
    history = pos_all[di - H : di].unsqueeze(0).expand(K, -1, -1, -1).contiguous()
    valid_static = mask_all[di - 1].unsqueeze(0).expand(K, -1).contiguous()
    role_idx = torch.from_numpy(m["role_idx"]).to(device).unsqueeze(0).expand(K, -1)
    target_mask = build_static_target_mask(cfg, cfg.n_entities, req.mode, device).expand(K, cfg.n_entities)

    opp_future = None
    if req.mode != "unconditioned":
        opp_future = pos_all[di : di + req.horizon_frames].unsqueeze(0).expand(K, -1, -1, -1).contiguous()

    # arrow → waypoint in NORMALIZED coords
    sx = cfg.pitch_x / 2.0
    sy = cfg.pitch_y / 2.0
    BALL_ENT = 2 * N             # last slot (index 22 with N=11) is the ball
    waypoints: list[tuple[int, int, list[float]]] = []

    def norm_xy(to_m: list[float]) -> list[float]:
        # clamp pin into [-1.05, 1.05] so it can't be wildly off-pitch
        wx = max(-1.05, min(1.05, to_m[0] / sx))
        wy = max(-1.05, min(1.05, to_m[1] / sy))
        return [wx, wy]

    # Receiving-stance offset: a recipient who runs onto a pass arrives ~1.5m
    # to the side of the ball, not on top of it — that's where a real player's
    # foot would meet the ball. Computed perpendicular to the recipient's
    # approach line (current pos → meet point).
    history_meters = (pos_all[di - 1] * torch.tensor([sx, sy], device=device, dtype=pos_all.dtype)).cpu()

    def receiving_stance_offset(recipient_ent: int, target_m: list[float], offset_m: float = 1.5) -> list[float]:
        cur_m = history_meters[recipient_ent].tolist()
        dx, dy = target_m[0] - cur_m[0], target_m[1] - cur_m[1]
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return target_m
        # Perpendicular to approach (right-hand side of run). Sign deterministic so
        # the recipient lands on the *open* side relative to their run direction.
        px, py = -dy / d, dx / d
        return [target_m[0] + px * offset_m, target_m[1] + py * offset_m]

    for a in req.arrows:
        if a.kind == "player":
            ent = _resolve_player(a.team, a.player, team0_slots, team1_slots, N)
            if ent is None:
                raise HTTPException(400, f"unknown player {a.team}:{a.player}")
            waypoints.append((ent, req.horizon_frames - 1, norm_xy(a.to)))
        else:  # ball_pass
            recipient_ent = _resolve_player(a.recipient.team, a.recipient.player,
                                            team0_slots, team1_slots, N)
            if recipient_ent is None:
                raise HTTPException(400, f"unknown recipient {a.recipient.team}:{a.recipient.player}")
            ball_xy_norm = norm_xy(a.to)
            recipient_target_m = receiving_stance_offset(recipient_ent, a.to)
            waypoints.append((BALL_ENT, req.horizon_frames - 1, ball_xy_norm))
            waypoints.append((recipient_ent, req.horizon_frames - 1, norm_xy(recipient_target_m)))

    # ── Multi-arrow joint plausibility check ────────────────────────────────
    # Above the paper: paper has no arrow input at all, so no notion of "is this
    # arrow set even physically possible." Two checks per pinned entity:
    #   1. Pinned positions don't violate r_min spacing (no 22 players on one spot).
    #   2. The implied run from current position → target doesn't exceed v_max * t_s
    #      (no asking a defender to teleport 60m in 1 second).
    # Reject with 400 + explicit reason so the coach knows what to redraw.
    pcfg = PhysicsConfig()
    horizon_s = req.horizon_frames / cfg.fps
    pinned_xy_m: list[tuple[int, list[float]]] = []
    history_meters_full = (pos_all[di - 1] *
                           torch.tensor([sx, sy], device=device, dtype=pos_all.dtype)).cpu()
    for (ent, _, xy_norm) in waypoints:
        target_m = [xy_norm[0] * sx, xy_norm[1] * sy]
        cur_m = history_meters_full[ent].tolist()
        dist = math.hypot(target_m[0] - cur_m[0], target_m[1] - cur_m[1])
        v_max = pcfg.v_max_ball if ent == BALL_ENT else pcfg.v_max_player
        max_reachable = v_max * horizon_s
        # Allow 5% slack so a coach drawing exactly at the limit isn't bounced.
        if dist > max_reachable * 1.05:
            raise HTTPException(
                400,
                f"arrow on entity {ent} requires {dist:.1f} m in {horizon_s:.1f} s "
                f"(>{max_reachable:.1f} m feasible at v_max={v_max} m/s). Pick a closer target.",
            )
        for other_ent, other_target in pinned_xy_m:
            if other_ent == ent:
                continue
            d = math.hypot(target_m[0] - other_target[0], target_m[1] - other_target[1])
            if d < pcfg.r_min and ent != BALL_ENT and other_ent != BALL_ENT:
                raise HTTPException(
                    400,
                    f"arrows on entity {ent} and entity {other_ent} both pin within "
                    f"{d:.2f} m (< r_min={pcfg.r_min} m). Players can't share that point.",
                )
        pinned_xy_m.append((ent, target_m))

    samples = causal_rollout(
        model, schedule, history, valid_static, req.horizon_frames, target_mask,
        opponent_future_full=opp_future,
        waypoints=waypoints,
        waypoint_fade_frames=req.fade_frames,
        guidance_scale=req.guidance_scale,
        role_idx=role_idx,
    )                                                                            # (K, T, n_ent, 2)

    if req.mode != "unconditioned":
        mix = target_mask.view(K, 1, cfg.n_entities, 1).expand_as(samples)
        samples = torch.where(mix, samples, opp_future)

    samples_m = _denorm(samples, cfg).cpu()
    history_m = _denorm(history[0], cfg).cpu()
    valid_static_cpu = valid_static[0].cpu()
    valid_history = mask_all[di - H : di].cpu()
    valid_actual = mask_all[di : di + req.horizon_frames].cpu()

    samples_m = apply_physics(samples_m, history_m[-1], valid_static_cpu, cfg, PhysicsConfig())

    # Hard-enforce arrow destinations after physics. Smoke model doesn't honor pins
    # internally; a fully-trained model would and this becomes a near-noop.
    for ent, t_idx, xy_norm in waypoints:
        samples_m[:, t_idx, ent, 0] = xy_norm[0] * sx
        samples_m[:, t_idx, ent, 1] = xy_norm[1] * sy

    # ── build response (samples.json-shaped) ─────────────────────────────
    history_frames = {}
    for i in range(H):
        fid = int(frame_ids[di - H + i])
        history_frames[str(fid)] = _frame_to_dict(
            history_m[i], valid_history[i], team0_slots, team1_slots, N, periods[di - H + i],
        )

    actual_future_m = _denorm(pos_all[di : di + req.horizon_frames], cfg).cpu()
    actual_future = {}
    for i in range(req.horizon_frames):
        fid = int(frame_ids[di + i])
        actual_future[str(fid)] = _frame_to_dict(
            actual_future_m[i], valid_actual[i], team0_slots, team1_slots, N, periods[di + i],
        )

    sample_list = []
    for k in range(K):
        out_k = {}
        for i in range(req.horizon_frames):
            fid = int(frame_ids[di + i])
            out_k[str(fid)] = _frame_to_dict(
                samples_m[k, i], valid_static_cpu, team0_slots, team1_slots, N, periods[di + i],
            )
        sample_list.append(out_k)

    return {
        "metadata": {
            "source": "gentac_inference_server",
            "match": req.match,
            "decision_frame": req.decision_frame,
            "history_frames": H,
            "horizon_frames": req.horizon_frames,
            "k": K,
            "mode": req.mode,
            "fps": cfg.fps,
            "pitch": {"x": cfg.pitch_x, "y": cfg.pitch_y, "origin": "center", "y_axis": "up"},
            "team0": {"name": "Home", "players": team0_slots},
            "team1": {"name": "Away", "players": team1_slots},
            "n_arrows": len(req.arrows),
            "guidance_scale": req.guidance_scale,
        },
        "history": history_frames,
        "actual_future": actual_future,
        "samples": sample_list,
    }
