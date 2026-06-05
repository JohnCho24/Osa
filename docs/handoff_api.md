# GenTac → Video Generator Handoff Spec  (v0.2)

This document is the **contract** between our trajectory inference service and the
downstream 3D / photorealistic video generator. Any video gen that implements this
schema can consume our output without further integration work.

We produce **trajectory keypoints**. We do not produce video, 3D models, broadcast
graphics, or audio. The video generator is responsible for everything the human eye
sees in the final clip.

Current checkpoint: the same trajectory keypoints first drive a **2D bird's-eye
soccer simulation**. The product must prove that a user can review footage, draw
arrows, and see all players and the ball react realistically in 2D before this
handoff becomes the primary output path.

---

## 1. Transport

The video generator calls our HTTPS endpoint:

```
POST  https://api.gentac.example/v1/generate
X-API-Key:    <api-key>
X-Request-Id: <uuid, optional — propagated to response>
Content-Type: application/json
```

We respond with a single JSON document (schema below). The response includes
`X-Request-Id` matching the request (or one we minted if you omitted it) for
distributed tracing.

Latency target: P95 under 10 s per request at K=1; under 25 s at K=20. (Production —
not the dev server in `src/server/main.py`, which runs on Mac MPS.)

Idempotency: include `idempotency_key` in the request; we cache the response for
24 h so retries return byte-identical output.

Concurrency: the dev server defaults to one in-flight generation per process plus
a queue of 8 (returns 503 past that). Production sizing is per-deployment.

---

## 2. Request

```json
{
  "version": "0.2",
  "idempotency_key": "uuid-v4",
  "match_id": "BUN-2025-3046",
  "decision_frame": 47382,
  "history_frames": 100,
  "horizon_frames": 100,
  "fps": 25,
  "k": 1,
  "mode": "unconditioned" | "opp_conditioned_team0" | "opp_conditioned_team1",
  "arrows": [
    { "kind": "player",    "team": "team0", "player": "Player7", "to": [12.5, -8.3] },
    { "kind": "ball_pass", "to": [25.0, 14.0],
      "recipient": { "team": "team0", "player": "Player9" } }
  ],
  "fade_frames": 24,
  "guidance_scale": 2.0,
  "video_hints": {
    "camera": "broadcast_main" | "tactical_top_down" | "follow_ball",
    "focus_player": "team0:Player7",
    "highlight_players": ["team0:Player7", "team1:Player18"]
  }
}
```

### Field semantics

- `match_id` — opaque identifier our system can resolve to ingested tracking data.
- `decision_frame` — absolute frame index in the match where the alternative
  branches. The 100 frames before this are clean history; the prediction extends
  forward from this frame.
- `arrows[i]` — a discriminated union on `kind`:
  - `kind:"player"` — pin a single player to `to` at the end of the horizon.
  - `kind:"ball_pass"` — pin the ball to `to`; the recipient runs onto the ball
    and arrives at a receiving-stance offset (~1.5 m to the side, perpendicular
    to their approach line). Coaches' natural "pass to X" gesture.
- `arrows[i].to` — destination in pitch meters (origin centre, x ∈ [-52.5, 52.5],
  y ∈ [-34, 34], y positive "up" toward team1's goal).
- `fade_frames` — over how many preceding frames the trajectory should bend
  smoothly toward the arrow destination via hard projection. Set to
  `horizon_frames - 1` for a strict straight-line follow; lower values let the
  model's learned waypoint signal own more of the path. Default 24 (~1.0 s at 25 fps).
- `guidance_scale` — classifier-free guidance scale on the learned waypoint
  signal. 1.0 = pure conditional (1 model forward per denoising step). Values >1
  amplify "honor my arrow" semantics at the cost of one extra forward pass per
  step. Bounded [0.0, 5.0]. Default 1.0.
- `mode` — which entities are noised vs. pinned to ground truth (see paper §6.3.1).
  Default `unconditioned` (the model samples all 23 entities; the coach's arrow
  bends them around the new state).
- `video_hints` — non-binding suggestions to the video generator. We don't enforce
  these; the generator decides whether to use them.

---

## 3. Response

```json
{
  "version": "0.1",
  "idempotency_key": "uuid-v4",
  "metadata": {
    "match_id": "BUN-2025-3046",
    "decision_frame": 47382,
    "history_frames": 100,
    "horizon_frames": 100,
    "fps": 25,
    "k": 1,
    "mode": "unconditioned",
    "n_arrows": 1,
    "pitch": { "x": 105.0, "y": 68.0, "origin": "center", "y_axis": "up" },
    "team0": { "name": "FC Bayern",   "players": ["Player1", "Player2", "..."] },
    "team1": { "name": "Hamburger SV", "players": ["Player15", "..."] },
    "generated_at": "2026-05-28T13:41:00Z",
    "model_version": "gentac-bundesliga-v0.3"
  },
  "history": {
    "47282": {
      "period": 1,
      "ball":  [6.50, 4.20],
      "team0": { "Player1": [-0.74, -30.28], "...": "..." },
      "team1": { "Player15": [-25.95, 10.47], "...": "..." }
    },
    "...": "..."
  },
  "actual_future": {
    "47382": { "period": 1, "ball": [6.51, 4.22], "team0": { "...": "..." }, "team1": { "...": "..." } },
    "...": "..."
  },
  "samples": [
    {
      "47382": { "period": 1, "ball": [6.51, 4.22], "team0": { "...": "..." }, "team1": { "...": "..." } },
      "...": "..."
    }
  ]
}
```

### Coordinate system (unambiguous)

- Units: **meters**, two decimal places.
- Origin: **centre spot** of the pitch.
- x-axis: long axis of the pitch, range nominal [-52.5, 52.5]; up to ±2 m
  excursion permitted (players chasing balls past the goal line).
- y-axis: short axis, positive **toward team1's goal**, range nominal [-34, 34];
  up to ±2 m excursion permitted.
- Pitch dimensions (`pitch.x`, `pitch.y`) are 105 × 68 m.
- Frames are sampled at `fps` (always 25 in v0.1).

### Validity

- A player that is **missing from a frame's `team0`/`team1` dict** is not on the
  pitch at that moment (substitution, red card, tracking failure). The video
  generator should not render them. If a player appears for some frames and
  disappears later, the generator may interpolate or freeze at the disappearance
  frame — its choice.
- `ball: null` means the ball position is unknown for that frame. Video gen
  should fall back to physics extrapolation or hide the ball.

### Samples ordering

- `samples` is an array of length `k`. Each sample is an alternative future
  trajectory generated from the same history + arrows + mode.
- When `k=1` and `arrows` is non-empty, the single sample is the user's
  arrow-conditioned alternative.
- When `k>1`, samples are independent draws from the same conditional
  distribution. The video generator may render them in any order, or render
  only a subset; there's no semantic ranking.

### Time alignment

- `history` covers absolute frames `[decision_frame - history_frames,
  decision_frame)`.
- `actual_future` covers `[decision_frame, decision_frame + horizon_frames)`
  with ground-truth positions.
- Each `samples[i]` covers `[decision_frame, decision_frame + horizon_frames)`
  with generated positions.
- All three keyed on the same absolute frame numbers as strings.

### Physical guarantees (post-processed)

- Player speeds ≤ 10.5 m/s in any single frame transition (paper-conditioned
  values may temporarily allow up to 12 m/s; v0.1 enforces 10.5).
- Ball speed ≤ 35 m/s.
- No two players within 0.6 m of each other in any frame.
- All positions within (pitch + 2 m margin).
- The seam frame `decision_frame` is anchored to the last history frame
  (no teleport at the boundary).

These guarantees are **honored after the deterministic physics post-processor**
runs. They are the contract we expose to the video gen; the generator can rely on
them and skip its own sanity checks.

---

## 4. Errors

```
400 Bad Request           — schema violation, unknown player, horizon not multiple of window
404 Not Found             — match_id unknown
409 Conflict              — idempotency_key reuse with mismatched body
429 Too Many Requests     — rate limit
500 Internal Server Error — model failure (retry safe; uses idempotency cache)
503 Service Unavailable   — model loading / cold start (Retry-After header)
```

Error body:
```json
{ "error": { "code": "ARROW_PLAYER_UNKNOWN", "message": "team0:Player99 not in match BUN-2025-3046" } }
```

---

## 5. Versioning

`version` is the schema version, not the model version. Schema-breaking changes
bump `version` and we run both versions in parallel for at least 90 days. Model
weight changes are tracked separately in `metadata.model_version`.

---

## 6. Example: minimal arrow-driven call

Request:

```json
{
  "version": "0.1",
  "idempotency_key": "8b7d1a..-..-..-..",
  "match_id": "BUN-2025-3046",
  "decision_frame": 47382,
  "horizon_frames": 100,
  "k": 1,
  "arrows": [
    { "team": "team0", "player": "Player7", "to": [12.5, -8.3] }
  ]
}
```

Response: the schema above with `samples[0]` containing 100 frames where Player7
ends at exactly `[12.5, -8.3]` (clamped to pitch bounds) and the other 22 entities
are sampled by the model.

---

## 7. Example: ball-pass (the new v0.2 capability)

```json
{
  "version": "0.2",
  "idempotency_key": "ca91...",
  "match_id": "BUN-2025-3046",
  "decision_frame": 47382,
  "horizon_frames": 100,
  "k": 1,
  "fade_frames": 99,
  "guidance_scale": 2.0,
  "arrows": [{
    "kind": "ball_pass",
    "to":   [25.0, 14.0],
    "recipient": { "team": "team0", "player": "Player9" }
  }]
}
```

In the response, `samples[0]` at the final frame contains:
- `ball:  [25.00, 14.00]` — exactly the requested target (hard-projected).
- `team0.Player9: [25.00 ± ~1.5, 14.00 ± ~1.5]` — perpendicular receiving-stance
  offset from the ball, computed from Player9's approach angle.

All other 21 entities are sampled by the model conditional on this new state.

## 8. Calling from Python

```python
import httpx, uuid

resp = httpx.post(
    "https://api.gentac.example/v1/generate",
    headers={"X-API-Key": API_KEY, "X-Request-Id": uuid.uuid4().hex},
    json={
        "version": "0.2",
        "idempotency_key": uuid.uuid4().hex,
        "match_id": "BUN-2025-3046",
        "decision_frame": 47382,
        "horizon_frames": 100,
        "k": 1,
        "arrows": [{"kind": "player", "team": "team0", "player": "Player7",
                    "to": [12.5, -8.3]}],
    },
    timeout=30.0,
)
resp.raise_for_status()
payload = resp.json()
print(f"req_id={resp.headers['X-Request-Id']}  samples={len(payload['samples'])}")
```

## 9. Vendor compatibility notes (informational)

The handoff schema is video-vendor-agnostic. A few concrete integration shapes
we've evaluated:

- **Runway / Pika / Luma** — Per-frame trajectory JSON → motion-conditioned
  diffusion video. Camera and player likeness are vendor-side. We supply
  `samples[k]` as the motion conditioning.
- **Unity / Unreal "broadcast" pipelines** — Trajectories are consumed as
  animation curves on rigged player models. The vendor handles camera, lighting,
  crowd, broadcast graphics. Best for live-broadcast augmentation where 3D scene
  control matters.
- **In-house 2D renderer** — For internal competition research, the league may
  prefer our `src/render/pitch.js` tactical view (or their own equivalent) and
  skip 3D video entirely.

We do not endorse a vendor. The integration cost is mostly per-vendor video
SLAs and creative direction, not per-vendor trajectory adapters.

## 10. Things this spec deliberately does NOT include

- Player names, jersey numbers, biometric data, or likeness rights. The video
  generator is responsible for skin / face / number assignment based on
  `match_id` and its own roster source.
- Camera angles, lighting, weather, crowd, broadcast graphics. All video
  generator decisions.
- Audio of any kind.
- Refereeing decisions, fouls, offside flags. Trajectories are kinematic only.
- Ball spin, possession state, pass intent, dribble vs. pass classification.
  These are derivable from trajectories by the consumer if needed.

If a downstream video generator needs any of these, they're out-of-band concerns
and need a separate negotiation.
