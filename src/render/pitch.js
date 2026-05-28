// GenTac tactics board renderer
// Coords in meters from pitch center; x ∈ [-52.5, 52.5], y ∈ [-34, 34], y positive "up"
// Canvas pixel buffer is always 1050×680; CSS scales it for compare mode.

const PITCH = { x: 105, y: 68, scale: 10 };           // 1 m = 10 px (in canvas-buffer space)
const CANVAS_W = PITCH.x * PITCH.scale;
const CANVAS_H = PITCH.y * PITCH.scale;
const NATIVE_FRAME_MS = 1000 / 25;
const TRAIL_FRAMES = 20;

const COLORS = {
  line:  "#ffffff",
  team0: "#3aa0ff",
  team1: "#ff5c5c",
  ball:  "#ffd93d",
};

// ── coord transforms ────────────────────────────────────────────────────────
const mx = (x) => (x + PITCH.x / 2) * PITCH.scale;
const my = (y) => (PITCH.y / 2 - y) * PITCH.scale;

// ── pitch markings ──────────────────────────────────────────────────────────
function drawPitch(ctx) {
  // Gradient turf + mowing stripes (ported from the marketing hero demo).
  const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
  grad.addColorStop(0, "#0e2a1c");
  grad.addColorStop(1, "#082015");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  const stripeH = CANVAS_H / 12;
  for (let i = 0; i < 12; i++) {
    ctx.fillStyle = i % 2 === 0 ? "rgba(255,255,255,0.018)" : "rgba(0,0,0,0.05)";
    ctx.fillRect(0, i * stripeH, CANVAS_W, stripeH);
  }

  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.82)";
  ctx.lineWidth = 2;
  ctx.strokeRect(mx(-52.5), my(34), PITCH.x * PITCH.scale, PITCH.y * PITCH.scale);
  ctx.beginPath();
  ctx.moveTo(mx(0), my(34));
  ctx.lineTo(mx(0), my(-34));
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(mx(0), my(0), 9.15 * PITCH.scale, 0, 2 * Math.PI);
  ctx.stroke();
  dot(ctx, mx(0), my(0), 2, COLORS.line);

  const halfArc = Math.acos(5.5 / 9.15);
  for (const side of [-1, 1]) {
    const goalLineX = 52.5 * side;
    rectAt(ctx, goalLineX, side, 16.5, 40.32);
    rectAt(ctx, goalLineX, side, 5.5, 18.32);
    const spotX = goalLineX - 11 * side;
    dot(ctx, mx(spotX), my(0), 2, COLORS.line);
    ctx.beginPath();
    if (side === 1) ctx.arc(mx(spotX), my(0), 9.15 * PITCH.scale, Math.PI - halfArc, Math.PI + halfArc);
    else ctx.arc(mx(spotX), my(0), 9.15 * PITCH.scale, -halfArc, halfArc);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(mx(goalLineX), my(3.66));
    ctx.lineTo(mx(goalLineX + 2 * side), my(3.66));
    ctx.lineTo(mx(goalLineX + 2 * side), my(-3.66));
    ctx.lineTo(mx(goalLineX), my(-3.66));
    ctx.stroke();
  }

  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    ctx.beginPath();
    ctx.arc(mx(52.5 * sx), my(34 * sy), 1 * PITCH.scale, 0, 2 * Math.PI);
    ctx.stroke();
  }
  ctx.restore();
}

function dot(ctx, px, py, r, color) {
  ctx.beginPath();
  ctx.fillStyle = color;
  ctx.arc(px, py, r, 0, 2 * Math.PI);
  ctx.fill();
}

function rectAt(ctx, goalLineX, side, depth, width) {
  const x0 = goalLineX - depth * side;
  ctx.strokeRect(mx(Math.min(goalLineX, x0)), my(width / 2), depth * PITCH.scale, width * PITCH.scale);
}

// ── entity / trails drawing ─────────────────────────────────────────────────
function drawTrails(ctx, perPlayerHistory, color) {
  if (!perPlayerHistory) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  for (const path of Object.values(perPlayerHistory)) {
    if (path.length < 2) continue;
    for (let i = 1; i < path.length; i++) {
      ctx.globalAlpha = (i / path.length) * 0.75;
      ctx.beginPath();
      ctx.moveTo(mx(path[i - 1][0]), my(path[i - 1][1]));
      ctx.lineTo(mx(path[i][0]), my(path[i][1]));
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawBallTrail(ctx, path) {
  if (!path || path.length < 2) return;
  ctx.save();
  ctx.strokeStyle = COLORS.ball;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  for (let i = 1; i < path.length; i++) {
    ctx.globalAlpha = (i / path.length) * 0.6;
    ctx.beginPath();
    ctx.moveTo(mx(path[i - 1][0]), my(path[i - 1][1]));
    ctx.lineTo(mx(path[i][0]), my(path[i][1]));
    ctx.stroke();
  }
  ctx.restore();
}

function drawFrame(ctx, frame, history, highlight) {
  if (history) {
    drawTrails(ctx, history.team0, COLORS.team0);
    drawTrails(ctx, history.team1, COLORS.team1);
    drawBallTrail(ctx, history.ball);
  }
  const isHi = (team, pid) =>
    highlight && !highlight.ball && highlight.team === team && highlight.pid === pid;
  for (const [pid, xy] of Object.entries(frame.team0 || {})) {
    drawPlayer(ctx, xy, COLORS.team0, pid, isHi("team0", pid));
  }
  for (const [pid, xy] of Object.entries(frame.team1 || {})) {
    drawPlayer(ctx, xy, COLORS.team1, pid, isHi("team1", pid));
  }
  if (frame.ball) {
    const [x, y] = frame.ball;
    const ballHi = !!(highlight && highlight.ball);
    if (ballHi) {
      const g = ctx.createRadialGradient(mx(x), my(y), 0, mx(x), my(y), 22);
      g.addColorStop(0, COLORS.ball + "88");
      g.addColorStop(1, COLORS.ball + "00");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(mx(x), my(y), 22, 0, 2 * Math.PI);
      ctx.fill();
    }
    dot(ctx, mx(x), my(y), 5, COLORS.ball);
    ctx.beginPath();
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 1;
    ctx.arc(mx(x), my(y), 5, 0, 2 * Math.PI);
    ctx.stroke();
  }
}

function drawPlayer(ctx, [x, y], color, pid, highlighted = false) {
  const px = mx(x);
  const py = my(y);
  if (highlighted) {
    const g = ctx.createRadialGradient(px, py, 0, px, py, 26);
    g.addColorStop(0, color + "88");
    g.addColorStop(1, color + "00");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(px, py, 26, 0, 2 * Math.PI);
    ctx.fill();
  }
  dot(ctx, px, py, 8, color);
  if (highlighted) {
    ctx.save();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px, py, 12, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }
  const num = String(pid).replace(/^Player/, "");
  ctx.fillStyle = "#0a0a0a";
  ctx.font = "bold 10px -apple-system, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(num, px, py + 0.5);
}

// ── Panel ───────────────────────────────────────────────────────────────────
function createPanel({ root, data }) {
  const canvas = root.querySelector("canvas.pitch");
  const ctx = canvas.getContext("2d");
  const metaEl = root.querySelector('[data-bind="meta"]');
  const frameKeys = Object.keys(data.frames).map(Number).sort((a, b) => a - b).map(String);

  function buildHistory(idx) {
    const start = Math.max(0, idx - TRAIL_FRAMES + 1);
    const team0 = {}, team1 = {};
    const ball = [];
    for (let i = start; i <= idx; i++) {
      const f = data.frames[frameKeys[i]];
      for (const [pid, xy] of Object.entries(f.team0 || {})) (team0[pid] = team0[pid] || []).push(xy);
      for (const [pid, xy] of Object.entries(f.team1 || {})) (team1[pid] = team1[pid] || []).push(xy);
      if (f.ball) ball.push(f.ball);
    }
    return { team0, team1, ball };
  }

  function render(idx, withTrails, highlight) {
    const clampedIdx = Math.min(idx, frameKeys.length - 1);
    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    drawPitch(ctx);
    const key = frameKeys[clampedIdx];
    const frame = data.frames[key];
    drawFrame(ctx, frame, withTrails ? buildHistory(clampedIdx) : null, highlight);

    const tSec = (clampedIdx / data.metadata.fps).toFixed(1);
    const ball = frame.ball ? `ball [${frame.ball[0].toFixed(1)}, ${frame.ball[1].toFixed(1)}]` : "ball —";
    metaEl.textContent = `${data.metadata.game_id} · f${key} · ${tSec}s · ${ball}`;
  }

  function currentFrame(idx) {
    return data.frames[frameKeys[Math.min(idx, frameKeys.length - 1)]];
  }
  function getCanvas() { return canvas; }

  function currentFrameId(idx) {
    return parseInt(frameKeys[Math.min(idx, frameKeys.length - 1)], 10);
  }

  return {
    render, length: frameKeys.length, gameId: data.metadata.game_id, fps: data.metadata.fps,
    currentFrame, currentFrameId, getCanvas,
  };
}

const SERVER_URL = "http://127.0.0.1:8001";

// ── draw-mode: pixel ↔ meter conversion (canvas buffer space) ───────────────
function canvasToMeters(px, py) {
  return [px / PITCH.scale - PITCH.x / 2, PITCH.y / 2 - py / PITCH.scale];
}

function eventToCanvasXY(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  // map CSS-pixel mouse position to canvas-buffer pixel space
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
}

function nearestPlayer(frame, mxMeters, myMeters, maxDistM = 3.5) {
  let best = null;
  let bestD = maxDistM;
  for (const [team, dict] of [["team0", frame.team0 || {}], ["team1", frame.team1 || {}]]) {
    for (const [pid, xy] of Object.entries(dict)) {
      const d = Math.hypot(xy[0] - mxMeters, xy[1] - myMeters);
      if (d < bestD) { bestD = d; best = { team, pid, xy }; }
    }
  }
  return best;
}

function ballHit(frame, mxMeters, myMeters, maxDistM = 2.0) {
  if (!frame.ball) return null;
  const [bx, by] = frame.ball;
  return Math.hypot(bx - mxMeters, by - myMeters) < maxDistM ? { xy: frame.ball } : null;
}

function drawArrow(ctx, fromMeters, toMeters, color, opacity = 0.95) {
  const [x1, y1] = [mx(fromMeters[0]), my(fromMeters[1])];
  const [x2, y2] = [mx(toMeters[0]), my(toMeters[1])];
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const headLen = 18;
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  // shaft
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2 - Math.cos(angle) * headLen * 0.6, y2 - Math.sin(angle) * headLen * 0.6);
  ctx.stroke();
  // head
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - Math.cos(angle - 0.4) * headLen, y2 - Math.sin(angle - 0.4) * headLen);
  ctx.lineTo(x2 - Math.cos(angle + 0.4) * headLen, y2 - Math.sin(angle + 0.4) * headLen);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

// ── App state + control wiring ──────────────────────────────────────────────
const app = {
  panels: [],
  idx: 0,
  playing: false,
  speed: 1,
  trails: true,
  compare: false,
  lastTick: 0,
  accum: 0,
  samplesData: null,
  sampleIdx: 0,
  draw: true,
  hoverPulseQueued: false,
  // arrows: Map keyed by:
  //   "p:team0:Player2"  → {kind:"player", team, pid, from:[x,y], to:[x,y]}
  //   "ball"             → {kind:"ball_pass", from:[bx,by], to:[x,y], recipient:{team, pid}}
  // Only one ball arrow at a time (the ball is a single entity).
  arrows: new Map(),
  drag: null,            // {kind, ...} while mouse is held
  hover: null,           // {team, pid} OR {ball:true} of entity under cursor in draw mode
  pendingBallPass: null, // {from, to} — waiting for recipient click
  overlayDiff: false,    // when true + compare mode, paints actual-future as a ghost on the alt panel
};

function updateSampleLabel() {
  if (!app.samplesData) return;
  els.sampleLabel.textContent = `Sample ${app.sampleIdx + 1} / ${app.samplesData.metadata.k}`;
  els.samplePrev.disabled = app.sampleIdx === 0;
  els.sampleNext.disabled = app.sampleIdx === app.samplesData.metadata.k - 1;
}

function showSample(newIdx) {
  if (!app.samplesData) return;
  const K = app.samplesData.metadata.k;
  app.sampleIdx = Math.max(0, Math.min(K - 1, newIdx));
  rebindPanel(1, els.panelB, buildVirtualClip(app.samplesData, app.sampleIdx));
  // sync scrubber max in case clip length differs
  els.scrubber.max = maxIdx();
  if (app.idx > maxIdx()) app.idx = maxIdx();
  updateSampleLabel();
  renderAll();
}

const els = {};

function maxIdx() {
  return Math.min(...app.panels.map(p => p.length)) - 1;
}

function renderAll() {
  app.panels.forEach((p, i) => {
    if (i === 0 || app.compare) {
      // Only the primary (drawable) panel reflects the cursor highlight.
      p.render(app.idx, app.trails, i === 0 ? app.hover : null);
    }
  });
  // Overlay-diff: paint the actual-future onto the alternative panel in white at low
  // alpha, so divergence between actual and alternative reads visually as colored
  // dots vs ghost dots.
  if (app.compare && app.overlayDiff && app.samplesData && app.panels[1]) {
    const actualClip = buildVirtualClip(app.samplesData, null);
    const fids = Object.keys(actualClip.frames).map(Number).sort((a, b) => a - b).map(String);
    const ghostKey = fids[Math.min(app.idx, fids.length - 1)];
    const ghostFrame = actualClip.frames[ghostKey];
    if (ghostFrame) {
      const ctx = app.panels[1].getCanvas().getContext("2d");
      ctx.save();
      ctx.globalAlpha = 0.35;
      for (const [, xy] of Object.entries(ghostFrame.team0 || {})) {
        dot(ctx, mx(xy[0]), my(xy[1]), 8, "#ffffff");
      }
      for (const [, xy] of Object.entries(ghostFrame.team1 || {})) {
        dot(ctx, mx(xy[0]), my(xy[1]), 8, "#ffffff");
      }
      if (ghostFrame.ball) {
        dot(ctx, mx(ghostFrame.ball[0]), my(ghostFrame.ball[1]), 5, "#ffffff");
      }
      ctx.restore();
    }
  }
  if (app.draw || app.arrows.size > 0 || app.drag) {
    drawArrowOverlay();
  }
  els.scrubber.value = app.idx;
  els.timeLabel.textContent = `${(app.idx / app.panels[0].fps).toFixed(1)}s`;
}

function drawDashedArrow(ctx, fromMeters, toMeters, color, opacity = 0.95) {
  ctx.save();
  ctx.setLineDash([6, 5]);
  drawArrow(ctx, fromMeters, toMeters, color, opacity);
  ctx.restore();
}

function drawRecipientRing(ctx, xy, color, opacity = 0.95) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.arc(mx(xy[0]), my(xy[1]), 16, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.restore();
}

function drawArrowOverlay() {
  const panel = app.panels[0];
  if (!panel) return;
  const ctx = panel.getCanvas().getContext("2d");
  const frame = panel.currentFrame(app.idx);

  for (const a of app.arrows.values()) {
    if (a.kind === "player") {
      const color = a.team === "team0" ? COLORS.team0 : COLORS.team1;
      drawArrow(ctx, a.from, a.to, color, 0.95);
    } else if (a.kind === "ball_pass") {
      // Ball arrow: dashed yellow shaft to the meet point + dashed ring on the recipient.
      drawDashedArrow(ctx, a.from, a.to, COLORS.ball, 0.95);
      const rxy = (frame[a.recipient.team] || {})[a.recipient.pid];
      if (rxy) {
        drawRecipientRing(ctx, rxy, COLORS.ball);
        // Tiny connector from meet-point to recipient so the relationship reads at a glance.
        const ctx2 = ctx;
        ctx2.save();
        ctx2.globalAlpha = 0.5;
        ctx2.strokeStyle = COLORS.ball;
        ctx2.lineWidth = 1.5;
        ctx2.setLineDash([2, 3]);
        ctx2.beginPath();
        ctx2.moveTo(mx(a.to[0]), my(a.to[1]));
        ctx2.lineTo(mx(rxy[0]), my(rxy[1]));
        ctx2.stroke();
        ctx2.restore();
      }
    }
  }

  if (app.drag) {
    if (app.drag.kind === "player") {
      const color = app.drag.team === "team0" ? COLORS.team0 : COLORS.team1;
      drawArrow(ctx, app.drag.from, app.drag.to, color, 0.55);
    } else if (app.drag.kind === "ball_pass") {
      drawDashedArrow(ctx, app.drag.from, app.drag.to, COLORS.ball, 0.55);
    }
  }

  // Pending ball-pass: arrow committed, waiting for recipient click. Pulse all players.
  if (app.pendingBallPass) {
    drawDashedArrow(ctx, app.pendingBallPass.from, app.pendingBallPass.to, COLORS.ball, 0.95);
    const allPlayers = [...Object.entries(frame.team0 || {}), ...Object.entries(frame.team1 || {})];
    ctx.save();
    ctx.strokeStyle = COLORS.ball;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 3]);
    ctx.globalAlpha = 0.5 + 0.4 * (0.5 + 0.5 * Math.sin(performance.now() * 0.005));
    for (const [, xy] of allPlayers) {
      ctx.beginPath();
      ctx.arc(mx(xy[0]), my(xy[1]), 12, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Hover highlight — pulsing dashed ring on the hovered entity. The player
  // dot itself is also drawn with a soft glow upstream (see drawPlayer), so
  // this ring is the second, more obvious affordance.
  if (app.draw && app.hover) {
    let xy = null;
    let color = "#ffffff";
    if (app.hover.ball) {
      xy = frame.ball;
      color = COLORS.ball;
    } else {
      xy = (frame[app.hover.team] || {})[app.hover.pid];
      if (app.pendingBallPass) color = COLORS.ball;
      else if (app.hover.team === "team0") color = COLORS.team0;
      else if (app.hover.team === "team1") color = COLORS.team1;
    }
    if (xy) {
      const pulse = 0.7 + 0.3 * Math.sin(performance.now() * 0.006);
      ctx.save();
      ctx.globalAlpha = pulse;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(mx(xy[0]), my(xy[1]), 16, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();
      // Keep the overlay animating while a hover is active (single rAF chain).
      if (!app.playing && !app.hoverPulseQueued) {
        app.hoverPulseQueued = true;
        requestAnimationFrame(() => {
          app.hoverPulseQueued = false;
          if (app.hover && !app.playing) renderAll();
        });
      }
    }
  }
}

function tick(ts) {
  if (!app.playing) return;
  if (app.lastTick === 0) app.lastTick = ts;
  const dt = ts - app.lastTick;
  app.lastTick = ts;
  app.accum += dt * app.speed;
  while (app.accum >= NATIVE_FRAME_MS) {
    app.accum -= NATIVE_FRAME_MS;
    app.idx++;
    if (app.idx >= maxIdx()) { app.idx = maxIdx(); pause(); break; }
  }
  renderAll();
  if (app.playing) requestAnimationFrame(tick);
}

function play() {
  if (app.idx >= maxIdx()) app.idx = 0;
  app.playing = true;
  app.lastTick = 0;
  app.accum = 0;
  els.playBtn.textContent = "❚❚ Pause";
  els.playBtn.classList.add("paused");
  requestAnimationFrame(tick);
}

function pause() {
  app.playing = false;
  els.playBtn.textContent = "▶︎ Play";
  els.playBtn.classList.remove("paused");
}

async function loadJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}

// ── samples-mode helpers ────────────────────────────────────────────────────
function buildVirtualClip(samplesData, sampleIdx) {
  // sampleIdx === null → ground-truth "actual" future; otherwise samples[sampleIdx]
  const futureFrames = sampleIdx === null
    ? samplesData.actual_future
    : samplesData.samples[sampleIdx];
  const frames = { ...samplesData.history, ...futureFrames };
  const label = sampleIdx === null
    ? `${samplesData.metadata.match.split("/").pop().replace(".json", "")} · actual`
    : `Sample ${sampleIdx + 1} / ${samplesData.metadata.k} · ${samplesData.metadata.mode}`;
  return {
    metadata: {
      ...samplesData.metadata,
      game_id: label,
      n_frames: Object.keys(frames).length,
    },
    frames,
  };
}

function rebindPanel(slotIdx, root, data) {
  app.panels[slotIdx] = createPanel({ root, data });
}

async function main() {
  console.log("[pitch] main() starting, v=11");
  els.status = document.getElementById("status");
  els.playBtn = document.getElementById("play-btn");
  els.scrubber = document.getElementById("scrubber");
  els.speedSelect = document.getElementById("speed-select");
  els.trailsToggle = document.getElementById("trails-toggle");
  els.compareToggle = document.getElementById("compare-toggle");
  els.samplePicker = document.getElementById("sample-picker");
  els.samplePrev = document.getElementById("sample-prev");
  els.sampleNext = document.getElementById("sample-next");
  els.sampleLabel = document.getElementById("sample-label");
  els.timeLabel = document.getElementById("time-label");
  els.panelA = document.querySelector('[data-panel="A"]');
  els.panelB = document.querySelector('[data-panel="B"]');
  els.panelLabelA = els.panelA.querySelector(".panel-label");
  els.panelLabelB = els.panelB.querySelector(".panel-label");

  const params = new URLSearchParams(location.search);
  const samplesMode = params.get("samples") === "1";

  els.status.textContent = samplesMode ? "Loading samples…" : "Loading clips…";

  if (samplesMode) {
    let samplesData;
    try {
      samplesData = await loadJson("data/processed/samples.json");
    } catch (e) {
      els.status.textContent = `Load failed: ${e.message} — run \`python scripts/sample.py\` first`;
      return;
    }
    app.samplesData = samplesData;
    app.sampleIdx = 0;

    els.panelLabelA.textContent = "Actual";
    els.panelLabelB.textContent = "Alternative";
    els.samplePicker.hidden = false;

    rebindPanel(0, els.panelA, buildVirtualClip(samplesData, null));
    rebindPanel(1, els.panelB, buildVirtualClip(samplesData, 0));

    els.status.textContent = `samples.json · ${samplesData.metadata.k} alternatives · mode=${samplesData.metadata.mode} · decision frame ${samplesData.metadata.decision_frame}`;
    updateSampleLabel();
  } else {
    let data;
    try {
      data = await loadJson("data/processed/Sample_Game_1_clip.json");
    } catch (e) {
      els.status.textContent = `Load failed: ${e.message} — run \`python3 -m http.server 8000\` from project root`;
      return;
    }
    // Both panels start with the SAME clip — left is the actual play, right is
    // a copy of the actual play. The right panel only diverges once the user
    // draws arrow(s) and hits Generate. Until then the two panels are identical
    // on purpose, so the comparison is meaningful when an alternative DOES arrive.
    rebindPanel(0, els.panelA, data);
    rebindPanel(1, els.panelB, data);
    els.panelLabelA.textContent = "▶ ACTUAL — DRAW ARROWS HERE";
    els.panelLabelB.textContent = "ALTERNATIVE — appears after Generate";
    els.status.textContent = `${data.metadata.n_frames} frames @ ${data.metadata.fps} FPS — ✏ Draw ON · click & drag a player on the LEFT panel (or drag the ball → click recipient for a pass)`;
    console.log("[pitch] panels bound, frames:", data.metadata.n_frames);
  }

  els.scrubber.max = maxIdx();
  els.scrubber.disabled = false;
  els.playBtn.disabled = false;

  // optional ?frame=N deep link
  const requested = parseInt(params.get("frame") ?? "0", 10);
  app.idx = Math.max(0, Math.min(requested, maxIdx()));
  // In samples mode, default-enable compare so the analyst sees actual vs alt at once.
  const wantCompare = params.get("compare") === "1" || samplesMode;
  if (wantCompare) {
    els.compareToggle.checked = true;
    app.compare = true;
    document.body.classList.add("compare");
    els.panelB.hidden = false;
  }

  els.playBtn.addEventListener("click", () => (app.playing ? pause() : play()));
  els.scrubber.addEventListener("input", (e) => { pause(); app.idx = +e.target.value; renderAll(); });
  els.speedSelect.addEventListener("change", (e) => { app.speed = +e.target.value; });
  els.trailsToggle.addEventListener("change", (e) => { app.trails = e.target.checked; renderAll(); });
  els.compareToggle.addEventListener("change", (e) => {
    app.compare = e.target.checked;
    els.panelB.hidden = !app.compare;
    document.body.classList.toggle("compare", app.compare);
    renderAll();
  });
  els.samplePrev.addEventListener("click", () => showSample(app.sampleIdx - 1));
  els.sampleNext.addEventListener("click", () => showSample(app.sampleIdx + 1));
  document.addEventListener("keydown", (e) => {
    if (!app.samplesData) return;
    if (e.code === "BracketLeft")  { e.preventDefault(); showSample(app.sampleIdx - 1); }
    if (e.code === "BracketRight") { e.preventDefault(); showSample(app.sampleIdx + 1); }
  });

  // ── M8: draw mode (arrows) ────────────────────────────────────────────
  els.drawToggle = document.getElementById("draw-toggle");
  els.clearArrows = document.getElementById("clear-arrows");
  els.generateBtn = document.getElementById("generate-btn");

  function refreshArrowControls() {
    const has = app.arrows.size > 0;
    els.clearArrows.disabled = !has;
    els.generateBtn.disabled = !has;
    document.body.classList.toggle("draw", app.draw);
  }

  // Draw mode defaults ON so the cursor immediately interacts with the pitch.
  els.drawToggle.checked = app.draw;
  els.drawToggle.addEventListener("change", (e) => {
    app.draw = e.target.checked;
    if (app.draw) {
      els.status.textContent = "✏ Draw ON — click and drag a player on the LEFT panel. (Drag the ball → click recipient for a pass.)";
    } else {
      els.status.textContent = "Draw OFF.";
      app.pendingBallPass = null;
      app.hover = null;
    }
    refreshArrowControls();
    renderAll();
  });
  if (app.draw) document.body.classList.add("draw");

  // debug: ?demo_arrows=1 injects two arrows so we can visually verify rendering
  if (params.get("demo_arrows") === "1") {
    const frame = app.panels[0].currentFrame(app.idx);
    const team0Entries = Object.entries(frame.team0 || {});
    const team1Entries = Object.entries(frame.team1 || {});
    if (team0Entries.length > 0) {
      const [pid, xy] = team0Entries[Math.floor(team0Entries.length / 2)];
      app.arrows.set(`p:team0:${pid}`, { kind: "player", team: "team0", pid, from: [...xy], to: [xy[0] + 15, xy[1] + 6] });
    }
    if (team1Entries.length > 0) {
      const [pid, xy] = team1Entries[Math.floor(team1Entries.length / 2)];
      app.arrows.set(`p:team1:${pid}`, { kind: "player", team: "team1", pid, from: [...xy], to: [xy[0] - 12, xy[1] - 8] });
    }
    refreshArrowControls();
  }

  // debug: ?auto_generate=1 fires Generate on load — for screenshot verification.
  if (params.get("auto_generate") === "1" && app.arrows.size > 0) {
    setTimeout(() => els.generateBtn.click(), 200);
  }

  els.clearArrows.addEventListener("click", () => {
    app.arrows.clear();
    app.drag = null;
    refreshArrowControls();
    renderAll();
  });

  els.generateBtn.addEventListener("click", async () => {
    const payload = buildArrowPayload();
    if (payload.arrows.length === 0) return;
    els.generateBtn.disabled = true;
    const prevLabel = els.generateBtn.textContent;
    els.generateBtn.textContent = "Generating…";
    els.status.textContent = `Generating with ${payload.arrows.length} arrow${payload.arrows.length > 1 ? "s" : ""}…`;
    try {
      const res = await fetch(`${SERVER_URL}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
      const samplesData = await res.json();

      // Swap into samples mode: panel A = actual, panel B = generated sample 0.
      app.samplesData = samplesData;
      app.sampleIdx = 0;
      els.panelLabelA.textContent = "Actual";
      els.panelLabelB.textContent = "Alternative";
      els.samplePicker.hidden = samplesData.metadata.k <= 1;

      rebindPanel(0, els.panelA, buildVirtualClip(samplesData, null));
      rebindPanel(1, els.panelB, buildVirtualClip(samplesData, 0));

      // Force compare mode so analyst sees the side-by-side immediately.
      if (!app.compare) {
        els.compareToggle.checked = true;
        app.compare = true;
        els.panelB.hidden = false;
        document.body.classList.add("compare");
      }

      // Note: mouse handlers reference app.panels[0] dynamically, so the existing
      // canvas listeners pick up the rebound panel automatically.
      els.scrubber.max = maxIdx();
      app.idx = 0;
      updateSampleLabel();
      renderAll();
      els.status.textContent = `Generated · ${samplesData.metadata.k} alternative${samplesData.metadata.k > 1 ? "s" : ""} · ${samplesData.metadata.n_arrows} arrows`;
    } catch (e) {
      console.error(e);
      els.status.textContent = `Generate failed: ${e.message}`;
    } finally {
      els.generateBtn.textContent = prevLabel;
      refreshArrowControls();
    }
  });

  // ── Overlay-diff toggle ──────────────────────────────────────────────────
  // Ghosts the actual-future as a faint underlay on the alternative panel, so
  // the analyst can read the divergence at a glance instead of A/B-flipping.
  els.overlayToggle = document.createElement("label");
  els.overlayToggle.className = "toggle";
  els.overlayToggle.innerHTML = '<input id="overlay-toggle" type="checkbox" /> Overlay diff';
  // Both compare and draw checkboxes live inside <label> elements that are
  // siblings under #controls — insertBefore must be called on the shared
  // parent (#controls), not on the compare label.
  document.getElementById("controls").insertBefore(els.overlayToggle, els.drawToggle.parentElement);
  els.overlayCheckbox = els.overlayToggle.querySelector("input");
  els.overlayCheckbox.addEventListener("change", (e) => {
    app.overlayDiff = e.target.checked;
    renderAll();
  });

  // Canvas mouse handlers on the primary panel
  const primaryCanvas = app.panels[0].getCanvas();

  // ── Touch event normalization ────────────────────────────────────────────
  // Browsers fire touchstart/move/end on tablets (the iPad-on-the-touchline
  // use case). We synthesize equivalent mouse events so the existing drag
  // handlers below "just work" without a parallel codepath.
  function touchToMouse(e, type) {
    if (e.touches.length > 1) return;
    e.preventDefault();
    const t = e.changedTouches[0];
    const me = new MouseEvent(type, {
      bubbles: true, cancelable: true, view: window,
      clientX: t.clientX, clientY: t.clientY, button: 0,
    });
    primaryCanvas.dispatchEvent(me);
  }
  primaryCanvas.addEventListener("touchstart", (e) => touchToMouse(e, "mousedown"), { passive: false });
  primaryCanvas.addEventListener("touchmove",  (e) => touchToMouse(e, "mousemove"), { passive: false });
  primaryCanvas.addEventListener("touchend",   (e) => touchToMouse(e, "mouseup"),   { passive: false });

  primaryCanvas.addEventListener("mousemove", (e) => {
    if (!app.draw && !app.drag) return;
    const [cx, cy] = eventToCanvasXY(primaryCanvas, e);
    const [xm, ym] = canvasToMeters(cx, cy);
    if (app.drag) {
      app.drag.to = [xm, ym];
      renderAll();
      return;
    }
    if (app.draw) {
      const frame = app.panels[0].currentFrame(app.idx);
      // While selecting a recipient, only players are hover targets.
      const hit = nearestPlayer(frame, xm, ym, 2.5);
      const ball = app.pendingBallPass ? null : ballHit(frame, xm, ym, 2.0);
      const newHover = ball
        ? { ball: true }
        : (hit ? { team: hit.team, pid: hit.pid } : null);
      const same = newHover && app.hover &&
        ((newHover.ball && app.hover.ball) ||
         (newHover.team === app.hover.team && newHover.pid === app.hover.pid));
      if (!same) { app.hover = newHover; renderAll(); }
    }
  });

  primaryCanvas.addEventListener("mousedown", (e) => {
    if (!app.draw) return;
    if (e.button !== 0) return;
    const [cx, cy] = eventToCanvasXY(primaryCanvas, e);
    const [xm, ym] = canvasToMeters(cx, cy);
    const frame = app.panels[0].currentFrame(app.idx);

    // If we're waiting for a recipient, this click resolves it instead of starting a drag.
    if (app.pendingBallPass) {
      const playerHit = nearestPlayer(frame, xm, ym, 3.0);
      if (playerHit) {
        app.arrows.set("ball", {
          kind: "ball_pass",
          from: [...app.pendingBallPass.from],
          to: [...app.pendingBallPass.to],
          recipient: { team: playerHit.team, pid: playerHit.pid },
        });
        app.pendingBallPass = null;
        els.status.textContent = `Ball pass to ${playerHit.team} ${playerHit.pid} queued. Hit Generate.`;
        refreshArrowControls();
        renderAll();
        e.preventDefault();
      }
      return;
    }

    // Ball drag has priority (the ball can sit under a player in some frames).
    const ball = ballHit(frame, xm, ym, 2.0);
    if (ball) {
      app.drag = { kind: "ball_pass", from: [...ball.xy], to: [xm, ym] };
      e.preventDefault();
      return;
    }

    const hit = nearestPlayer(frame, xm, ym, 2.5);
    if (hit) {
      app.drag = { kind: "player", team: hit.team, pid: hit.pid, from: [...hit.xy], to: [xm, ym] };
      e.preventDefault();
    }
  });

  primaryCanvas.addEventListener("mouseup", () => {
    if (!app.drag) return;
    const len = Math.hypot(app.drag.to[0] - app.drag.from[0], app.drag.to[1] - app.drag.from[1]);
    if (len < 1.0) { app.drag = null; renderAll(); return; }

    if (app.drag.kind === "player") {
      const key = `p:${app.drag.team}:${app.drag.pid}`;
      app.arrows.set(key, { ...app.drag });
    } else if (app.drag.kind === "ball_pass") {
      // Don't commit yet — wait for the user to click a recipient.
      app.pendingBallPass = { from: [...app.drag.from], to: [...app.drag.to] };
      els.status.textContent = "Click the recipient player to complete the pass (Esc to cancel).";
    }
    app.drag = null;
    refreshArrowControls();
    renderAll();
  });

  primaryCanvas.addEventListener("mouseleave", () => {
    if (app.drag) { app.drag = null; renderAll(); }
    if (app.hover) { app.hover = null; renderAll(); }
  });

  primaryCanvas.addEventListener("contextmenu", (e) => {
    if (!app.draw) return;
    e.preventDefault();
    const [cx, cy] = eventToCanvasXY(primaryCanvas, e);
    const [xm, ym] = canvasToMeters(cx, cy);
    const frame = app.panels[0].currentFrame(app.idx);
    // Right-click on ball → clear ball arrow; on a player → clear player arrow OR clear
    // the ball arrow if this player was the pending recipient.
    if (ballHit(frame, xm, ym, 2.0)) {
      if (app.arrows.delete("ball")) { refreshArrowControls(); renderAll(); }
      return;
    }
    const hit = nearestPlayer(frame, xm, ym, 2.5);
    if (!hit) return;
    const playerKey = `p:${hit.team}:${hit.pid}`;
    const ballArrow = app.arrows.get("ball");
    if (ballArrow && ballArrow.recipient.team === hit.team && ballArrow.recipient.pid === hit.pid) {
      app.arrows.delete("ball");
      refreshArrowControls(); renderAll();
      return;
    }
    if (app.arrows.delete(playerKey)) { refreshArrowControls(); renderAll(); }
  });

  // Esc cancels a pending recipient selection
  document.addEventListener("keydown", (e) => {
    if (e.code === "Escape" && app.pendingBallPass) {
      app.pendingBallPass = null;
      els.status.textContent = "Ball pass cancelled.";
      renderAll();
    }
  });

  function buildArrowPayload() {
    const decisionFrame = app.samplesData
      ? app.samplesData.metadata.decision_frame
      : app.panels[0].currentFrameId(app.idx);
    const match = app.samplesData
      ? app.samplesData.metadata.match
      : `data/processed/${app.panels[0].gameId}.json`;

    // Dynamic horizon: pick the shortest horizon (in 5-frame windows = the
    // model's window_frames) that's long enough to make every drawn arrow
    // physically feasible at v_max. Player v_max = 10.5 m/s, ball = 25 m/s
    // (server enforces; see src/server/main.py). Floor at 25 frames (1s),
    // cap at server's MAX_HORIZON_FRAMES = 250 (10s).
    const FPS = 25, W = 5, V_PLAYER = 10.5, V_BALL = 25.0, FLOOR = 25, CAP = 250;
    let maxNeededSec = FLOOR / FPS;
    for (const a of app.arrows.values()) {
      if (a.kind === "player") {
        const d = Math.hypot(a.to[0] - a.from[0], a.to[1] - a.from[1]);
        maxNeededSec = Math.max(maxNeededSec, (d / V_PLAYER) * 1.1); // 10% slack
      } else if (a.kind === "ball_pass") {
        const d = Math.hypot(a.to[0] - a.from[0], a.to[1] - a.from[1]);
        maxNeededSec = Math.max(maxNeededSec, (d / V_BALL) * 1.1);
      }
    }
    const neededFrames = Math.ceil(maxNeededSec * FPS);
    const horizon = Math.min(CAP, Math.max(FLOOR, Math.ceil(neededFrames / W) * W));
    return {
      decision_frame: decisionFrame,
      match,
      horizon_frames: horizon,
      k: 1,
      // Pin-fade across most of the horizon so the arrowed entities visibly trace the
      // arrow path instead of only snapping at the end. The model still runs and
      // predicts every other player's reaction; fade only affects arrowed slots.
      fade_frames: horizon - 1,
      // Smoke checkpoint is undertrained — CFG > 1 amplifies noise into visible
      // player/ball "crashes" (overlaps, jittery paths). Keep at 1.0 for now;
      // bump back up once a full-trained checkpoint (M7) lands.
      guidance_scale: 1.0,
      mode: "unconditioned",
      arrows: Array.from(app.arrows.values()).map(a => {
        if (a.kind === "ball_pass") {
          return {
            kind: "ball_pass",
            to: a.to,
            recipient: { team: a.recipient.team, player: a.recipient.pid },
          };
        }
        return { kind: "player", team: a.team, player: a.pid, to: a.to };
      }),
    };
  }
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); app.playing ? pause() : play(); }
    if (e.code === "ArrowRight") { pause(); app.idx = Math.min(app.idx + 1, maxIdx()); renderAll(); }
    if (e.code === "ArrowLeft") { pause(); app.idx = Math.max(app.idx - 1, 0); renderAll(); }
    // D toggles draw mode — the onboarding tells the user about this.
    if (e.code === "KeyD" && document.activeElement === document.body) {
      e.preventDefault();
      els.drawToggle.checked = !els.drawToggle.checked;
      els.drawToggle.dispatchEvent(new Event("change"));
    }
  });

  // Auto-dismiss the onboarding overlay on first meaningful interaction.
  // (The overlay was set up by setupOnboarding() before main() ran, so its
  // dismiss handlers are already attached — these are just opportunistic.)
  const _scrim = document.getElementById("onboarding");
  if (_scrim && els.generateBtn) {
    els.generateBtn.addEventListener("click", () => { _scrim.hidden = true; }, { once: true });
  }
  if (_scrim && els.drawToggle) {
    els.drawToggle.addEventListener("change", () => { _scrim.hidden = true; }, { once: true });
  }

  renderAll();
}

// ── First-run onboarding overlay ────────────────────────────────────────────
// Runs BEFORE main() so the dismiss handlers attach even if main() fails (e.g.
// data files missing). Coaches will not discover the arrow mechanic unattended,
// so we surface the 3-step flow on first visit and persist dismissal in
// localStorage. ?onboarding=1 forces it back for testing.
function setupOnboarding() {
  const scrim = document.getElementById("onboarding");
  if (!scrim) return;
  const params = new URLSearchParams(location.search);
  const forceOnboard = params.get("onboarding") === "1";
  let onboarded = false;
  try { onboarded = !forceOnboard && localStorage.getItem("gentac.onboarded") === "1"; } catch (_) {}

  function dismiss() {
    scrim.hidden = true;
    try { localStorage.setItem("gentac.onboarded", "1"); } catch (_) {}
  }

  // Always wire the buttons (idempotent if overlay stays hidden).
  const startBtn = document.getElementById("ob-start");
  const skipBtn = document.getElementById("ob-skip");
  if (startBtn) startBtn.addEventListener("click", dismiss);
  if (skipBtn)  skipBtn.addEventListener("click", dismiss);

  // Click on the scrim background (outside the card) also dismisses.
  scrim.addEventListener("click", (e) => { if (e.target === scrim) dismiss(); });

  // Esc dismisses.
  document.addEventListener("keydown", (e) => {
    if (e.code === "Escape" && !scrim.hidden) dismiss();
  });

  if (!onboarded) {
    scrim.hidden = false;
    // Focus the primary button so Enter / Space also dismisses.
    if (startBtn) try { startBtn.focus(); } catch (_) {}
  }
}

// Onboarding overlay removed — was blocking, not helping.
main().catch((e) => {
  console.error("[pitch] main() crashed:", e);
  const status = document.getElementById("status");
  if (status) status.textContent = `Renderer crashed: ${e.message}. See console.`;
});
