/* ============================================================================
   Use Product — interactive 2D what-if board (front-end only)

   The real trajectory engine is NOT connected yet. This page is a faithful
   product shell: footage -> simulated "convert to 2D" -> an accurate top-down
   board in REAL PITCH COORDINATES (FIFA 105 x 68 m), freehand curved-arrow
   drawing, a playback timeline, and a macOS-style dock. "Alternative play"
   animates players along the arrows you draw as a SIMULATED preview — swap the
   engine into buildPlan('alt') later for real generations.
   ========================================================================== */
(() => {
  "use strict";
  const DEBUG = location.search.includes("debug");

  // ─── Pitch geometry (real metres) ─────────────────────────────────────────
  const PITCH_L = 105;          // length, m  (touchline to touchline along x)
  const PITCH_W = 68;           // width,  m  (along y)
  const PX_PER_M = 10;          // logical px per metre
  const PAD = 60;               // logical px margin around the pitch (goals/air)
  const LOG_W = PITCH_L * PX_PER_M + PAD * 2;   // 1170
  const LOG_H = PITCH_W * PX_PER_M + PAD * 2;   // 800

  const FPS = 25;
  const DURATION_S = 5;
  const TOTAL_FRAMES = FPS * DURATION_S;

  // metre <-> logical-px transforms (origin at pitch centre, y up positive)
  const mx = (x) => PAD + (x + PITCH_L / 2) * PX_PER_M;
  const my = (y) => PAD + (PITCH_W / 2 - y) * PX_PER_M;

  // ─── DOM ───────────────────────────────────────────────────────────────────
  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  const stage = document.getElementById("stage");

  const elStatusChip = document.getElementById("status-chip");
  const elStatusText = document.getElementById("status-text");
  const elCoord = document.getElementById("coord-readout");
  const elClear = document.getElementById("clear-arrows");
  const elHint = document.getElementById("draw-hint");
  const elControls = document.getElementById("controls");
  const elEngineFlag = document.getElementById("engine-flag");

  const elTrack = document.getElementById("tl-track");
  const elCur = document.getElementById("tl-cur");
  const elTotal = document.getElementById("tl-total");
  const elFrame = document.getElementById("tl-frame");

  const btnPlay = document.getElementById("dock-play");
  const btnPause = document.getElementById("dock-pause");
  const btnReset = document.getElementById("dock-reset");
  const btnAlt = document.getElementById("dock-alt");

  const modal = document.getElementById("up-modal");
  const viewDrop = document.getElementById("up-view-drop");
  const viewProc = document.getElementById("up-view-proc");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("up-input");
  const upError = document.getElementById("up-error");
  const procSteps = document.getElementById("proc-steps");
  const procFill = document.getElementById("proc-fill");

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const COLOR = {
    a: css("--team-a") || "#ff5722",
    b: css("--team-b") || "#4fc3f7",
    ball: css("--ball") || "#fafafa",
    line: css("--pitch-line") || "#20543a",
  };

  // ─── State ─────────────────────────────────────────────────────────────────
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const view = { fit: 1, ox: 0, oy: 0 };

  const board = {
    ready: false,
    players: [],            // { id, team:'a'|'b', x, y, bx, by }  (metres)
    ball: { x: 0, y: 0, bx: 0, by: 0 },
    arrows: [],             // { id, ownerId, team, path:[{x,y}…] }  (metres)
  };
  let hoveredId = null;
  let draft = null;         // { ownerId, team, team, raw:[{x,y}…] }

  const play = { running: false, mode: null, t: 0, plan: null, planMode: null, startWall: 0, startT: 0 };
  let rafId = 0;

  // ─── Math helpers ────────────────────────────────────────────────────────
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const lerp = (a, b, t) => a + (b - a) * t;
  const lerpPt = (a, b, t) => ({ x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t) });
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hexRGB(hex) {
    const h = hex.replace("#", "");
    return [0, 2, 4].map((i) => parseInt(h.substr(i, 2), 16)).join(",");
  }

  // Catmull-Rom through points -> dense smooth polyline (unlimited curvature).
  function smooth(pts, perSeg = 14) {
    if (pts.length <= 2) return pts.map((p) => ({ ...p }));
    const out = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[i + 2] || pts[i + 1];
      for (let j = 0; j < perSeg; j++) {
        const t = j / perSeg, t2 = t * t, t3 = t2 * t;
        out.push({
          x: 0.5 * (2 * p1.x + (-p0.x + p2.x) * t + (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 + (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
          y: 0.5 * (2 * p1.y + (-p0.y + p2.y) * t + (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 + (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
        });
      }
    }
    out.push({ ...pts[pts.length - 1] });
    return out;
  }
  // Arc-length table for constant-speed traversal of a polyline.
  function arcTable(path) {
    const cum = [0];
    for (let i = 1; i < path.length; i++) cum[i] = cum[i - 1] + dist(path[i - 1], path[i]);
    return { cum, total: cum[cum.length - 1] };
  }
  function pointAtArc(path, tbl, u) {
    if (path.length === 1) return { ...path[0] };
    const target = clamp01(u) * tbl.total;
    let i = 1;
    while (i < tbl.cum.length && tbl.cum[i] < target) i++;
    if (i >= path.length) return { ...path[path.length - 1] };
    const segLen = tbl.cum[i] - tbl.cum[i - 1] || 1;
    return lerpPt(path[i - 1], path[i], (target - tbl.cum[i - 1]) / segLen);
  }

  // ─── Canvas sizing ─────────────────────────────────────────────────────────
  function fit() {
    const cssW = canvas.clientWidth || stage.clientWidth;
    const cssH = canvas.clientHeight || stage.clientHeight;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const s = Math.min(cssW / LOG_W, cssH / LOG_H) * 0.97;
    view.fit = s;
    view.ox = (cssW - LOG_W * s) / 2;
    view.oy = (cssH - LOG_H * s) / 2;
  }
  function applyTransform() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.translate(view.ox, view.oy);
    ctx.scale(view.fit, view.fit);
  }
  function eventToMetres(e) {
    const r = canvas.getBoundingClientRect();
    const logX = (e.clientX - r.left - view.ox) / view.fit;
    const logY = (e.clientY - r.top - view.oy) / view.fit;
    return { x: (logX - PAD) / PX_PER_M - PITCH_L / 2, y: PITCH_W / 2 - (logY - PAD) / PX_PER_M };
  }

  // ─── Drawing ───────────────────────────────────────────────────────────────
  function drawPitch() {
    const left = mx(-PITCH_L / 2), right = mx(PITCH_L / 2);
    const top = my(PITCH_W / 2), bot = my(-PITCH_W / 2);

    const g = ctx.createLinearGradient(0, top, 0, bot);
    g.addColorStop(0, "#0e2a1c"); g.addColorStop(1, "#082015");
    ctx.fillStyle = g;
    ctx.fillRect(left, top, right - left, bot - top);

    // mowing stripes
    const bands = 12, bw = (right - left) / bands;
    for (let i = 0; i < bands; i++) {
      ctx.fillStyle = i % 2 === 0 ? "rgba(255,255,255,0.014)" : "rgba(0,0,0,0.05)";
      ctx.fillRect(left + i * bw, top, bw, bot - top);
    }

    ctx.strokeStyle = COLOR.line;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";

    ctx.strokeRect(left, top, right - left, bot - top);            // outer
    line(mx(0), top, mx(0), bot);                                   // halfway
    circle(mx(0), my(0), 9.15 * PX_PER_M);                          // centre circle
    dot(mx(0), my(0), 2.4);                                         // centre spot

    // penalty + goal areas (both ends)
    [-1, 1].forEach((s) => {
      const goalX = s * PITCH_L / 2;
      // penalty area: 16.5 m deep, 40.32 m wide
      rect(mx(goalX), my(20.16), mx(goalX - s * 16.5), my(-20.16));
      // goal area: 5.5 m deep, 18.32 m wide
      rect(mx(goalX), my(9.16), mx(goalX - s * 5.5), my(-9.16));
      // penalty spot 11 m from goal line
      const spotX = goalX - s * 11;
      dot(mx(spotX), my(0), 2.4);
      // penalty arc (portion outside the box)
      const a = Math.acos(5.5 / 9.15);
      ctx.beginPath();
      if (s === 1) ctx.arc(mx(spotX), my(0), 9.15 * PX_PER_M, Math.PI - a, Math.PI + a);
      else ctx.arc(mx(spotX), my(0), 9.15 * PX_PER_M, -a, a);
      ctx.stroke();
      // goal (7.32 m, posts behind the line)
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      rect(mx(goalX), my(3.66), mx(goalX + s * 2), my(-3.66));
      ctx.strokeStyle = COLOR.line;
    });
  }
  function line(x1, y1, x2, y2) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
  function rect(x1, y1, x2, y2) { ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1)); }
  function circle(cx, cy, r) { ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke(); }
  function dot(cx, cy, r) { ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fillStyle = COLOR.line; ctx.fill(); }

  function drawPlayer(p, opts = {}) {
    const x = mx(p.x), y = my(p.y), c = p.team === "a" ? COLOR.a : COLOR.b;
    if (opts.glow || opts.hover) {
      const rg = ctx.createRadialGradient(x, y, 0, x, y, 26);
      rg.addColorStop(0, c + "55"); rg.addColorStop(1, c + "00");
      ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(x, y, 26, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = c;
    ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.45)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.stroke();
    if (opts.hover) {
      ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.setLineDash([4, 5]);
      ctx.beginPath(); ctx.arc(x, y, 17, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  function drawBall(b, opts = {}) {
    const x = mx(b.x), y = my(b.y);
    if (opts.glow || opts.hover) {
      const rg = ctx.createRadialGradient(x, y, 0, x, y, 20);
      rg.addColorStop(0, "rgba(250,250,250,0.5)"); rg.addColorStop(1, "rgba(250,250,250,0)");
      ctx.fillStyle = rg; ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = COLOR.ball; ctx.shadowColor = "rgba(0,0,0,0.6)"; ctx.shadowBlur = 6;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill(); ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke();
    if (opts.hover) {
      ctx.strokeStyle = COLOR.ball; ctx.lineWidth = 2; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  function drawCurve(path, color, alpha = 1, head = true, dashed = false) {
    if (path.length < 2) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.shadowColor = color; ctx.shadowBlur = 8;
    if (dashed) ctx.setLineDash([11, 7]);
    ctx.beginPath();
    ctx.moveTo(mx(path[0].x), my(path[0].y));
    for (let i = 1; i < path.length; i++) ctx.lineTo(mx(path[i].x), my(path[i].y));
    ctx.stroke();
    ctx.setLineDash([]);
    if (head) {
      const a = path[path.length - 2], b = path[path.length - 1];
      const ang = Math.atan2(my(b.y) - my(a.y), mx(b.x) - mx(a.x));
      const tipX = mx(b.x), tipY = my(b.y), hl = 16;
      ctx.fillStyle = color; ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - hl * Math.cos(ang - 0.42), tipY - hl * Math.sin(ang - 0.42));
      ctx.lineTo(tipX - hl * Math.cos(ang + 0.42), tipY - hl * Math.sin(ang + 0.42));
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }
  function drawTrail(samples, color) {
    if (samples.length < 2) return;
    ctx.save(); ctx.lineCap = "round";
    for (let i = 1; i < samples.length; i++) {
      ctx.strokeStyle = `rgba(${hexRGB(color)},${(i / samples.length) * 0.5})`;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(mx(samples[i - 1].x), my(samples[i - 1].y));
      ctx.lineTo(mx(samples[i].x), my(samples[i].y));
      ctx.stroke();
    }
    ctx.restore();
  }

  function render() {
    applyTransform();
    ctx.clearRect(-PAD * 2, -PAD * 2, LOG_W + PAD * 4, LOG_H + PAD * 4);
    drawPitch();

    // committed arrows (dim during playback so motion reads on top)
    const arrowAlpha = play.running ? 0.5 : 0.95;
    board.arrows.forEach((ar) => drawCurve(ar.path, arrowColor(ar), arrowAlpha, true, ar.kind === "ball"));

    // live draft
    if (draft && draft.raw.length >= 2) {
      drawCurve(smooth(draft.raw), arrowColor(draft), 0.6, true, draft.kind === "ball");
    }

    // trails during playback
    if (play.running && play.plan) {
      board.players.forEach((p) => {
        const fn = play.plan.posFns.get(p.id);
        if (!fn || !play.plan.moving.has(p.id)) return;
        const s = [];
        for (let k = 5; k >= 0; k--) s.push(fn(clamp01(play.t - k * 0.045)));
        drawTrail(s, p.team === "a" ? COLOR.a : COLOR.b);
      });
    }

    board.players.forEach((p) => drawPlayer(p, { hover: !play.running && p.id === hoveredId, glow: !!arrowOf(p.id) }));
    drawBall(board.ball, { hover: !play.running && hoveredId === "ball", glow: !!arrowOf("ball") });
  }
  function arrowColor(ar) { return ar.kind === "ball" ? COLOR.ball : ar.team === "a" ? COLOR.a : COLOR.b; }

  // ─── Formation seeding ─────────────────────────────────────────────────────
  const FORMATION = {
    a: [[-49, 0], [-36, -18], [-36, -6], [-36, 6], [-36, 18], [-20, -12], [-18, 2], [-20, 14], [-4, -20], [-2, 2], [-4, 20]],
    b: [[49, 0], [36, 18], [36, 6], [36, -6], [36, -18], [20, 12], [18, -2], [20, -14], [4, 20], [2, -2], [4, -20]],
  };
  function seedBoard(seed) {
    const rnd = mulberry32(seed || 1);
    const jit = () => (rnd() - 0.5) * 2.2;       // ±1.1 m organic offset
    board.players = [];
    ["a", "b"].forEach((team) => {
      FORMATION[team].forEach(([x, y], i) => {
        const px = Math.max(-52, Math.min(52, x + jit()));
        const py = Math.max(-33, Math.min(33, y + jit()));
        board.players.push({ id: `${team}${i}`, team, x: px, y: py, bx: px, by: py });
      });
    });
    board.ball = { x: -17 + jit() * 0.4, y: 2 + jit() * 0.4, bx: 0, by: 0 };
    board.ball.bx = board.ball.x; board.ball.by = board.ball.y;
    board.arrows = [];
    invalidatePlan();
  }

  // ─── Footage → coordinates (in-browser detector) ───────────────────────────
  // Dependency-free and fully client-side, so the page stays a static site.
  // Segments the green field, finds coloured/dark player blobs inside it, splits
  // them into two teams by jersey colour, and projects image positions onto the
  // pitch using each row's field width as a cheap perspective correction. It's an
  // estimate (not ±1 cm) — swap in the real homography engine here later.
  function frameFromMedia(media, w, h) {
    const scale = Math.min(1, 960 / Math.max(1, w));
    const cw = Math.max(1, Math.round(w * scale));
    const ch = Math.max(1, Math.round(h * scale));
    const c = document.createElement("canvas");
    c.width = cw; c.height = ch;
    c.getContext("2d", { willReadFrequently: true }).drawImage(media, 0, 0, cw, ch);
    return c;
  }
  function fileToFrame(file) {
    const url = URL.createObjectURL(file);
    const done = (frame) => { URL.revokeObjectURL(url); return frame; };
    if (file.type.startsWith("video/")) {
      return new Promise((resolve, reject) => {
        const v = document.createElement("video");
        v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
        v.addEventListener("loadeddata", () => { v.currentTime = Math.min(0.6, (v.duration || 2) * 0.25); });
        v.addEventListener("seeked", () => resolve(done(frameFromMedia(v, v.videoWidth, v.videoHeight))), { once: true });
        v.addEventListener("error", () => { URL.revokeObjectURL(url); reject(new Error("video")); });
      });
    }
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(done(frameFromMedia(img, img.naturalWidth, img.naturalHeight)));
      img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("image")); };
      img.src = url;
    });
  }
  function urlToFrame(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(frameFromMedia(img, img.naturalWidth, img.naturalHeight));
      img.onerror = () => reject(new Error("image"));
      img.src = url;
    });
  }

  const _d2 = (p, c) => { const a = p[0]-c[0], b = p[1]-c[1], e = p[2]-c[2]; return a*a + b*b + e*e; };
  function kmeans2(pts) {                            // split blob colours into 2 teams
    if (!pts.length) return [];
    let c0 = pts[0], c1 = pts[0], lo = Infinity, hi = -Infinity;
    pts.forEach((p) => { const s = p[0]+p[1]+p[2]; if (s < lo) { lo = s; c0 = p; } if (s > hi) { hi = s; c1 = p; } });
    c0 = c0.slice(); c1 = c1.slice();
    const lab = new Array(pts.length).fill(0);
    for (let it = 0; it < 8; it++) {
      for (let i = 0; i < pts.length; i++) lab[i] = _d2(pts[i], c0) <= _d2(pts[i], c1) ? 0 : 1;
      const s0 = [0,0,0], s1 = [0,0,0]; let n0 = 0, n1 = 0;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        if (lab[i] === 0) { s0[0]+=p[0]; s0[1]+=p[1]; s0[2]+=p[2]; n0++; }
        else { s1[0]+=p[0]; s1[1]+=p[1]; s1[2]+=p[2]; n1++; }
      }
      if (n0) c0 = [s0[0]/n0, s0[1]/n0, s0[2]/n0];
      if (n1) c1 = [s1[0]/n1, s1[1]/n1, s1[2]/n1];
    }
    const aCluster = (c0[0]-c0[2]) >= (c1[0]-c1[2]) ? 0 : 1;   // warmer kit → team a (orange)
    return lab.map((l) => (l === aCluster ? "a" : "b"));
  }

  function detectFormation(frame) {
    const W = frame.width, H = frame.height;
    const { data } = frame.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, W, H);
    const at = (x, y) => y * W + x;

    const field = new Uint8Array(W * H);
    const player = new Uint8Array(W * H);
    const rowL = new Int32Array(H).fill(W);
    const rowR = new Int32Array(H).fill(-1);
    let fTop = H, fBot = -1, gMinX = W, gMaxX = -1, fCount = 0;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = at(x, y) * 4, r = data[i], g = data[i+1], b = data[i+2];
        const mx_ = Math.max(r, g, b), val = mx_ / 255;
        if (g >= mx_ && g - Math.max(r, b) > 12 && val > 0.12 && val < 0.95) {   // green pitch
          field[at(x, y)] = 1;
          if (x < rowL[y]) rowL[y] = x;
          if (x > rowR[y]) rowR[y] = x;
          if (x < gMinX) gMinX = x; if (x > gMaxX) gMaxX = x;
          if (y < fTop) fTop = y; if (y > fBot) fBot = y;
          fCount++;
        }
      }
    }
    if (fCount < W * H * 0.03 || fBot <= fTop) return null;     // no recognizable pitch

    for (let y = fTop; y <= fBot; y++) {                        // players: inside field, not green
      const L = rowL[y], R = rowR[y];
      if (R - L < 4) continue;
      for (let x = L; x <= R; x++) {
        const p = at(x, y);
        if (field[p]) continue;
        const i = p * 4, r = data[i], g = data[i+1], b = data[i+2];
        const mx_ = Math.max(r, g, b), mn = Math.min(r, g, b);
        const val = mx_ / 255, sat = mx_ ? (mx_ - mn) / mx_ : 0;
        if ((sat > 0.32 && val > 0.25) || val < 0.34) player[p] = 1;  // coloured kit OR dark kit
      }
    }

    const comp = new Int32Array(W * H).fill(-1);                // connected components (8-conn)
    const blobs = [], stack = [];
    for (let y = fTop; y <= fBot; y++) {
      for (let x = 0; x < W; x++) {
        const s0 = at(x, y);
        if (!player[s0] || comp[s0] !== -1) continue;
        const id = blobs.length;
        let n = 0, sx = 0, sy = 0, sr = 0, sg = 0, sb = 0, minx = x, maxx = x, miny = y, maxy = y;
        stack.length = 0; stack.push(s0); comp[s0] = id;
        while (stack.length) {
          const q = stack.pop(), qx = q % W, qy = (q / W) | 0, i = q * 4;
          n++; sx += qx; sy += qy; sr += data[i]; sg += data[i+1]; sb += data[i+2];
          if (qx < minx) minx = qx; if (qx > maxx) maxx = qx;
          if (qy < miny) miny = qy; if (qy > maxy) maxy = qy;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const nx = qx + dx, ny = qy + dy;
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const np = at(nx, ny);
            if (player[np] && comp[np] === -1) { comp[np] = id; stack.push(np); }
          }
        }
        blobs.push({ n, cx: sx / n, cy: sy / n, r: sr / n, g: sg / n, b: sb / n, w: maxx - minx + 1, h: maxy - miny + 1 });
      }
    }

    const areaMin = Math.max(8, W * H * 0.00004);
    const areaMax = W * H * 0.02;
    let cands = blobs.filter((bl) => bl.n >= areaMin && bl.n <= areaMax && bl.w < W * 0.16 && bl.h < H * 0.24);
    if (DEBUG) console.log(`[detect] ${W}x${H} field=${fCount} blobs=${blobs.length} cands=${cands.length}`);
    if (cands.length < 6) return null;
    cands.sort((a, b) => b.n - a.n);
    cands = cands.slice(0, 24);

    const teamLab = kmeans2(cands.map((c) => [c.r, c.g, c.b]));
    const toMetres = (cx, cy) => {                              // image px → pitch metres
      const ry = Math.min(fBot, Math.max(fTop, Math.round(cy)));
      let L = rowL[ry], R = rowR[ry];
      if (R - L < 8) { L = gMinX; R = gMaxX; }                  // per-row de-perspective, w/ fallback
      const u = Math.min(1, Math.max(0, (cx - L) / ((R - L) || 1)));
      const v = Math.min(1, Math.max(0, (cy - fTop) / ((fBot - fTop) || 1)));
      return { x: +((u - 0.5) * PITCH_L * 0.94).toFixed(2), y: +((0.5 - v) * PITCH_W * 0.9).toFixed(2) };
    };

    const players = [];
    let ca = 0, cb = 0;
    cands.forEach((c, k) => {
      const team = teamLab[k];
      if (team === "a" ? ca >= 11 : cb >= 11) return;           // cap 11 per side
      team === "a" ? ca++ : cb++;
      players.push({ team, ...toMetres(c.cx, c.cy) });
    });
    if (players.length < 6) return null;

    const n = players.length;                                   // ball ≈ centroid of detected players
    const ball = {
      x: +(players.reduce((s, p) => s + p.x, 0) / n).toFixed(2),
      y: +(players.reduce((s, p) => s + p.y, 0) / n).toFixed(2),
    };
    return { players, ball, count: n };
  }

  function setBoardPlayers(players, ball) {
    let ca = 0, cb = 0;
    board.players = players.map((p) => {
      const id = p.team === "a" ? `a${ca++}` : `b${cb++}`;
      return { id, team: p.team, x: p.x, y: p.y, bx: p.x, by: p.y };
    });
    board.ball = { x: ball.x, y: ball.y, bx: ball.x, by: ball.y };
    board.arrows = [];
    invalidatePlan();
  }
  function hashFrame(frame) {                                   // deterministic seed for the fallback
    const { data } = frame.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, frame.width, frame.height);
    let h = 0x811c9dc5;
    for (let i = 0; i < data.length; i += 997) { h ^= data[i]; h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  }
  function setEngineFlag(text) { if (elEngineFlag) elEngineFlag.textContent = text; }

  const playerById = (id) => board.players.find((p) => p.id === id);
  const arrowOf = (id) => board.arrows.find((a) => a.ownerId === id);
  function nearestPlayer(pt, radius = 2.4) {
    let best = null, bd = radius;
    board.players.forEach((p) => { const d = dist(pt, { x: p.bx, y: p.by }); if (d < bd) { bd = d; best = p; } });
    return best;
  }
  // Hit-test the ball OR a player; whichever is physically closer wins (the ball
  // must be within 1.8 m so an overlapping player stays grabbable).
  function hitTarget(pt) {
    const db = dist(pt, { x: board.ball.bx, y: board.ball.by });
    const p = nearestPlayer(pt);
    const dp = p ? dist(pt, { x: p.bx, y: p.by }) : Infinity;
    if (db <= 1.8 && db <= dp) return { ball: true };
    if (p) return { player: p };
    if (db <= 1.8) return { ball: true };
    return null;
  }

  // ─── Freehand curved-arrow drawing ────────────────────────────────────────
  function onPointerDown(e) {
    if (!board.ready || play.running || modalOpen()) return;
    const m = eventToMetres(e);
    const h = hitTarget(m);
    if (!h) return;
    resetToBase();                             // arrows anchor to rest positions
    draft = h.ball
      ? { ownerId: "ball", kind: "ball", team: null, raw: [{ x: board.ball.bx, y: board.ball.by }] }
      : { ownerId: h.player.id, kind: "run", team: h.player.team, raw: [{ x: h.player.bx, y: h.player.by }] };
    canvas.setPointerCapture(e.pointerId);
    render();
  }
  function onPointerMove(e) {
    const m = eventToMetres(e);
    updateCoord(m);
    if (draft) {
      const last = draft.raw[draft.raw.length - 1];
      if (dist(m, last) >= 0.35) { draft.raw.push(m); render(); }
      return;
    }
    if (board.ready && !play.running) {
      const h = hitTarget(m);
      const id = h ? (h.ball ? "ball" : h.player.id) : null;
      if (id !== hoveredId) { hoveredId = id; render(); }
    }
  }
  function onPointerUp(e) {
    if (!draft) return;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
    const raw = draft.raw;
    let len = 0; for (let i = 1; i < raw.length; i++) len += dist(raw[i - 1], raw[i]);
    if (len >= 1.0 && raw.length >= 2) {
      board.arrows = board.arrows.filter((a) => a.ownerId !== draft.ownerId);
      board.arrows.push({ id: `ar${Date.now()}`, ownerId: draft.ownerId, kind: draft.kind, team: draft.team, path: smooth(raw) });
      onArrowsChanged();
    }
    draft = null;
    render();
  }
  function onContextMenu(e) {
    if (!board.ready) return;
    e.preventDefault();
    const h = hitTarget(eventToMetres(e));
    if (!h) return;
    const id = h.ball ? "ball" : h.player.id;
    if (arrowOf(id)) { board.arrows = board.arrows.filter((a) => a.ownerId !== id); onArrowsChanged(); render(); }
  }
  function clearArrows() { board.arrows = []; onArrowsChanged(); render(); }
  function onArrowsChanged() {
    invalidatePlan();
    const has = board.arrows.length > 0;
    btnAlt.disabled = !has;
    elClear.disabled = !has;
    if (has && elHint) { elHint.style.opacity = "0"; setTimeout(() => (elHint.hidden = true), 400); }
  }

  function updateCoord(m) {
    const inside = Math.abs(m.x) <= 52.5 && Math.abs(m.y) <= 34;
    elCoord.innerHTML = inside
      ? `x <b>${m.x.toFixed(2)}</b> m · y <b>${m.y.toFixed(2)}</b> m`
      : "x — · y —";
  }

  // ─── Playback plan (SIMULATED — swap the real engine in here) ──────────────
  function invalidatePlan() { play.plan = null; play.planMode = null; }
  function tacticalTarget(p, focus, sameTeam) {
    const dx = focus.x - p.bx, dy = focus.y - p.by;
    const L = Math.hypot(dx, dy) || 1;
    const falloff = Math.max(0.15, Math.min(1, 28 / L));
    const mag = (sameTeam ? 2.2 : 3.0) * falloff;
    return { x: p.bx + (dx / L) * mag, y: p.by + (dy / L) * mag };
  }
  function buildPlan(mode) {
    const posFns = new Map();
    const moving = new Set();

    if (mode === "alt" && board.arrows.length) {
      const tables = new Map();
      board.arrows.forEach((a) => tables.set(a.ownerId, arcTable(a.path)));
      const runArrows = board.arrows.filter((a) => a.kind !== "ball");
      // off-ball players drift toward the mean destination of the drawn arrows
      const dests = board.arrows.map((a) => a.path[a.path.length - 1]);
      const focus = dests.reduce((s, d) => ({ x: s.x + d.x / dests.length, y: s.y + d.y / dests.length }), { x: 0, y: 0 });
      const refTeam = (runArrows[0] && runArrows[0].team) || "a";
      board.players.forEach((p) => {
        const ar = arrowOf(p.id);
        if (ar) {
          moving.add(p.id);
          const tbl = tables.get(p.id);
          posFns.set(p.id, (t) => pointAtArc(ar.path, tbl, ease(t)));
        } else {
          const tgt = tacticalTarget(p, focus, p.team === refTeam);
          posFns.set(p.id, (t) => lerpPt({ x: p.bx, y: p.by }, tgt, ease(t) * 0.5));
        }
      });
      // ball: an explicit drawn pass wins; otherwise the ball is carried by
      // whichever arrowed player starts nearest it.
      const ballAr = arrowOf("ball");
      if (ballAr) {
        return finalize(mode, posFns, moving, (t) => pointAtArc(ballAr.path, tables.get("ball"), ease(t)), true);
      }
      let passAr = null, bd = 3.0;
      runArrows.forEach((a) => {
        const o = playerById(a.ownerId);
        const d = dist({ x: o.bx, y: o.by }, { x: board.ball.bx, y: board.ball.by });
        if (d < bd) { bd = d; passAr = a; }
      });
      if (passAr) {
        return finalize(mode, posFns, moving, (t) => pointAtArc(passAr.path, tables.get(passAr.ownerId), ease(t)), true);
      }
    } else {
      // base scene: everyone breathes toward the ball
      const focus = { x: board.ball.bx, y: board.ball.by };
      board.players.forEach((p) => {
        moving.add(p.id);
        const tgt = tacticalTarget(p, focus, p.team === "a");
        posFns.set(p.id, (t) => lerpPt({ x: p.bx, y: p.by }, tgt, ease(t)));
      });
    }
    // default ball: gentle idle bob
    const bbob = (t) => ({ x: board.ball.bx + Math.sin(t * Math.PI * 2) * 0.3, y: board.ball.by + Math.cos(t * Math.PI * 3) * 0.2 });
    return finalize(mode, posFns, moving, bbob, false);
  }
  function finalize(mode, posFns, moving, ballFn, ballMoving) {
    if (ballMoving) moving.add("ball");
    return { mode, posFns, moving, ballFn };
  }
  function ensurePlan(mode) {
    if (play.plan && play.planMode === mode) return;
    play.plan = buildPlan(mode);
    play.planMode = mode;
  }
  function applyFrame(t) {
    play.t = clamp01(t);
    if (play.plan) {
      board.players.forEach((p) => { const fn = play.plan.posFns.get(p.id); if (fn) { const q = fn(play.t); p.x = q.x; p.y = q.y; } });
      const b = play.plan.ballFn(play.t); board.ball.x = b.x; board.ball.y = b.y;
    }
    syncTimeline();
    render();
  }
  function resetToBase() {
    stopRaf();
    play.running = false; play.t = 0;
    board.players.forEach((p) => { p.x = p.bx; p.y = p.by; });
    board.ball.x = board.ball.bx; board.ball.y = board.ball.by;
    setStatus("ready");
    setPressed(null);
    syncTimeline();
  }

  // ─── Playback transport ────────────────────────────────────────────────────
  function startPlay(mode) {
    if (!board.ready) return;
    if (mode === "alt" && !board.arrows.length) return;
    ensurePlan(mode);
    play.mode = mode;
    if (play.t >= 1) play.t = 0;
    play.running = true;
    play.startWall = performance.now();
    play.startT = play.t;
    setStatus("playing", mode);
    setPressed(mode === "alt" ? btnAlt : btnPlay);
    loop();
  }
  function loop() {
    stopRaf();
    const step = () => {
      const elapsed = (performance.now() - play.startWall) / 1000;
      const t = play.startT + elapsed / DURATION_S;
      if (t >= 1) { applyFrame(1); pausePlay(true); return; }
      applyFrame(t);
      rafId = requestAnimationFrame(step);
    };
    rafId = requestAnimationFrame(step);
  }
  function pausePlay(ended = false) {
    stopRaf();
    play.running = false;
    setPressed(null);
    setStatus(ended ? "ready" : "paused");
    render();
  }
  function resetPlay() { resetToBase(); render(); }
  function stopRaf() { if (rafId) cancelAnimationFrame(rafId); rafId = 0; }

  function onScrub() {
    const t = clamp01(parseInt(elTrack.value, 10) / 1000);
    if (play.running) pausePlay();
    ensurePlan(play.mode || (board.arrows.length ? "alt" : "base"));
    applyFrame(t);
  }
  function syncTimeline() {
    const v = Math.round(play.t * 1000);
    if (parseInt(elTrack.value, 10) !== v) elTrack.value = v;
    elTrack.style.setProperty("--fill", `${play.t * 100}%`);
    elCur.textContent = `${(play.t * DURATION_S).toFixed(1)}s`;
    elFrame.textContent = `frame ${Math.round(play.t * TOTAL_FRAMES)} / ${TOTAL_FRAMES}`;
  }
  function setPressed(btn) {
    [btnPlay, btnAlt].forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
  }

  // ─── Status chip ───────────────────────────────────────────────────────────
  function setStatus(mode, sub) {
    const map = {
      idle: ["idle", "Awaiting footage"],
      proc: ["proc", "Converting to 2D…"],
      ready: ["ready", "2D board ready · draw an arrow"],
      paused: ["ready", "Paused"],
      playing: ["playing", sub === "alt" ? "Playing alternative (simulated)" : "Playing base scene"],
    };
    const [m, text] = map[mode] || map.idle;
    elStatusChip.dataset.mode = m;
    elStatusText.textContent = text;
  }

  // ─── Upload + conversion ───────────────────────────────────────────────────
  function modalOpen() { return !modal.hidden; }
  function openUpload() {
    upError.textContent = "";
    viewProc.hidden = true; viewDrop.hidden = false;
    modal.hidden = false;
  }
  function closeUpload() { modal.hidden = true; }

  let pendingFrame = null;          // the decoded frame awaiting detection

  function handleFile(file) {
    if (!file) return;
    const kind = file.type.startsWith("image/") ? "image" : file.type.startsWith("video/") ? "video" : null;
    if (!kind) { upError.textContent = "Please drop an image or video file."; return; }
    upError.textContent = "";
    fileToFrame(file)
      .then((frame) => { pendingFrame = frame; runConversion(); })
      .catch(() => { upError.textContent = "Couldn't read that file — try another image or clip."; });
  }
  function runConversion() {
    viewDrop.hidden = true; viewProc.hidden = false;
    setStatus("proc");
    const steps = [...procSteps.querySelectorAll("li")];
    steps.forEach((s) => s.classList.remove("active", "done"));
    procFill.style.width = "0%";
    let i = 0;
    const total = steps.length;
    const tick = () => {
      if (i > 0) steps[i - 1].classList.add("done");
      if (i < total) {
        steps[i].classList.add("active");
        procFill.style.width = `${Math.round(((i + 1) / total) * 100)}%`;
        i++;
        setTimeout(tick, 620);
      } else {
        finishConversion();
      }
    };
    tick();
  }
  function finishConversion() {
    let detected = null;
    try { if (pendingFrame) detected = detectFormation(pendingFrame); } catch { detected = null; }
    if (detected) {
      setBoardPlayers(detected.players, detected.ball);                  // ← coordinates from the upload
      setEngineFlag(`Detected ${detected.count} players from your footage — positions estimated in-browser`);
    } else {
      seedBoard(pendingFrame ? hashFrame(pendingFrame) : 7);            // graceful, honest fallback
      setEngineFlag(pendingFrame
        ? "Couldn't lock onto players in that frame — showing an estimated formation"
        : "Engine not connected — simulated preview");
    }
    pendingFrame = null;
    board.ready = true;
    closeUpload();
    elControls.hidden = false;
    elHint.hidden = false; elHint.style.opacity = "1";
    canvas.classList.add("is-drawable");
    btnAlt.disabled = true; elClear.disabled = true;
    setStatus("ready");
    syncTimeline();
    render();
  }

  // ─── Wiring ────────────────────────────────────────────────────────────────
  function bind() {
    window.addEventListener("resize", () => { fit(); render(); });
    if (window.ResizeObserver) new ResizeObserver(() => { fit(); render(); }).observe(stage);

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("contextmenu", onContextMenu);
    canvas.addEventListener("pointerleave", () => { if (!draft) { hoveredId = null; updateCoord({ x: 99, y: 99 }); render(); } });

    btnPlay.addEventListener("click", () => startPlay("base"));
    btnPause.addEventListener("click", () => pausePlay());
    btnReset.addEventListener("click", resetPlay);
    btnAlt.addEventListener("click", () => startPlay("alt"));
    elTrack.addEventListener("input", onScrub);
    elClear.addEventListener("click", clearArrows);

    document.getElementById("open-upload").addEventListener("click", openUpload);
    document.getElementById("use-sample").addEventListener("click", () => {
      urlToFrame("/public/demo-poster.jpg")
        .then((frame) => { pendingFrame = frame; runConversion(); })
        .catch(() => { pendingFrame = null; runConversion(); });
    });
    fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
    modal.querySelectorAll("[data-close]").forEach((el) => el.addEventListener("click", () => { if (board.ready) closeUpload(); }));

    // drag & drop
    ["dragover", "dragenter"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("is-dragging"); }));
    ["dragleave", "dragend"].forEach((t) => dropzone.addEventListener(t, () => dropzone.classList.remove("is-dragging")));
    dropzone.addEventListener("drop", (e) => { e.preventDefault(); dropzone.classList.remove("is-dragging"); handleFile(e.dataTransfer.files[0]); });

    // keyboard
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modalOpen() && board.ready) closeUpload();
      if (e.code === "Space" && board.ready && !modalOpen()) {
        e.preventDefault();
        play.running ? pausePlay() : startPlay(play.mode || "base");
      }
    });
  }

  // ─── Init ──────────────────────────────────────────────────────────────────
  fit();
  elTotal.textContent = `${DURATION_S.toFixed(1)}s`;
  seedBoard(7);                       // default formation = the starting point
  board.ready = true;
  elControls.hidden = false;
  elHint.hidden = false; elHint.style.opacity = "1";
  canvas.classList.add("is-drawable");
  setStatus("ready");
  syncTimeline();
  bind();
  render();                           // board visible on load; upload via "New upload"

  // Optional automation hook (only when the page is opened with ?debug).
  if (location.search.includes("debug")) {
    window.__gentac = {
      get board() { return board; },
      screenOf: (xm, ym) => {
        const r = canvas.getBoundingClientRect();
        return { x: r.left + view.ox + mx(xm) * view.fit, y: r.top + view.oy + my(ym) * view.fit };
      },
    };
  }
})();
