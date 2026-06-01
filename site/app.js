// ─── B / landing-page interactions ─────────────────────────────────────────
// Vanilla JS, no build step, no framework. Drop the directory into Vercel/Netlify
// and it ships.

// ─── Scroll progress bar ──────────────────────────────────────────────────
{
  const bar = document.querySelector(".scroll-progress");
  const update = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const pct = max > 0 ? window.scrollY / max : 0;
    bar.style.transform = `scaleX(${pct})`;
  };
  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
}

// ─── Reveal-on-scroll ─────────────────────────────────────────────────────
{
  // Mark anything that should reveal. Sections + the elements that benefit from a
  // staggered child appearance.
  document.querySelectorAll(".section-head, .section, .hero-copy, .visual-frame").forEach(el => {
    if (!el.hasAttribute("data-reveal")) el.setAttribute("data-reveal", "");
  });
  document.querySelectorAll(".problem-grid, .pillars, .usecases, .how-steps, .tech-grid, .hero-credibility, .buyers-row").forEach(el => {
    el.setAttribute("data-reveal-children", "");
  });

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("is-visible");
        io.unobserve(e.target);
      }
    }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });

  document.querySelectorAll("[data-reveal], [data-reveal-children]").forEach(el => io.observe(el));
}

// ─── Contact form (no backend — just a mailto handoff + UX polish) ─────────
{
  const form = document.getElementById("contact-form");
  const note = document.getElementById("form-note");
  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const data = new FormData(form);
      const name = (data.get("name") || "").toString().trim();
      const email = (data.get("email") || "").toString().trim();
      const org = (data.get("org") || "").toString().trim();
      const message = (data.get("message") || "").toString().trim();
      if (!name || !email || !org || !message) {
        note.textContent = "Please fill every field.";
        return;
      }
      const subject = encodeURIComponent(`B — League inquiry from ${org}`);
      const body = encodeURIComponent(
        `Name: ${name}\nEmail: ${email}\nOrganization: ${org}\n\n${message}\n\n—\nSent from bstartup.dev`
      );
      window.location.href = `mailto:leagues@bstartup.dev?subject=${subject}&body=${body}`;
      note.textContent = "Opening your mail client…";
    });
  }
}

// ─── Hero pitch animation ─────────────────────────────────────────────────
// A self-contained looping demo: an arrow is "drawn" from a player to a target,
// the target player moves smoothly along the path (with a trail), and the rest
// of the team adjusts. Loops every ~7 s.
{
  const canvas = document.getElementById("hero-pitch");
  if (!canvas) {
    /* no canvas — skip */
  } else {
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    function fitCanvas() {
      // CSS already gives us the layout size; bump the backing-store for sharpness.
      const cssW = canvas.clientWidth || 900;
      const cssH = Math.round(cssW * (canvas.height / canvas.width));
      canvas.style.height = `${cssH}px`;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    fitCanvas();
    window.addEventListener("resize", fitCanvas);

    // Pitch geometry in canvas units (we draw in the original 900x600 space and
    // ctx.scale by dpr*scale at render time so the layout adapts.)
    const PITCH_W = 900;
    const PITCH_H = 600;
    const COLOR_TEAM_A = getCss("--team-a");
    const COLOR_TEAM_B = getCss("--team-b");
    const COLOR_BALL = getCss("--ball");
    const COLOR_LINE = getCss("--pitch-line");
    const COLOR_ACCENT = getCss("--accent");

    function getCss(name) {
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }

    // Initial 4-3-3-ish formation for team A (left) and team B (right),
    // chosen to look like a real half-pitch transition moment.
    const baseFormation = {
      a: [
        { x: 80,  y: 300 },  // GK
        { x: 220, y: 140 }, { x: 220, y: 280 }, { x: 220, y: 360 }, { x: 220, y: 480 },
        { x: 380, y: 200 }, { x: 380, y: 320 }, { x: 380, y: 440 },
        { x: 540, y: 180 }, { x: 540, y: 340 }, { x: 540, y: 460 },
      ],
      b: [
        { x: 820, y: 300 },
        { x: 680, y: 160 }, { x: 680, y: 280 }, { x: 680, y: 360 }, { x: 680, y: 460 },
        { x: 540, y: 220 }, { x: 540, y: 360 }, { x: 540, y: 480 },
        { x: 400, y: 200 }, { x: 400, y: 340 }, { x: 400, y: 460 },
      ],
    };

    const ball = { x: 540, y: 340 };

    // Animation state. Re-seeded on every loop.
    let scenario = null;

    function tacticalReaction(player, focus, sameTeam, pressureScale = 0.18) {
      // Each non-arrowed player gets a small reactive displacement toward the focus
      // (the ball-future or arrow destination). Same-team players move to support /
      // open angles; opposing players move to apply pressure. Magnitude is bounded
      // so the formation doesn't collapse onto a single point.
      const dx = focus.x - player.x;
      const dy = focus.y - player.y;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      // Players further from the action react less (attention falloff)
      const falloff = Math.max(0.15, Math.min(1.0, 350 / len));
      const mag = pressureScale * 60 * falloff;
      // Add slight perpendicular jitter so movements don't look mechanical
      const perpAngle = Math.atan2(uy, ux) + Math.PI / 2;
      const jitter = (Math.random() - 0.5) * 18;
      const teamBias = sameTeam ? 0.6 : 0.9;       // opponents press harder than teammates support
      return {
        x: player.x + ux * mag * teamBias + Math.cos(perpAngle) * jitter,
        y: player.y + uy * mag * teamBias + Math.sin(perpAngle) * jitter,
      };
    }

    function pickScenario() {
      // 60% player run, 40% ball pass — both feel like real moments.
      const isBallPass = Math.random() < 0.4;

      let actorEnt, actorStart, target, recipientEnt, recipientStart, ballTarget;
      if (isBallPass) {
        // Holder: a midfielder closish to the ball
        actorEnt = 5 + Math.floor(Math.random() * 3);          // midfield
        actorStart = { ...baseFormation.a[actorEnt] };
        // Recipient: an attacker ahead of the holder
        const recipientCandidates = [8, 9, 10];
        recipientEnt = recipientCandidates[Math.floor(Math.random() * 3)];
        recipientStart = { ...baseFormation.a[recipientEnt] };
        // Meet point ahead of the recipient, deep in opposition half
        target = {
          x: 700 + Math.random() * 160,
          y: Math.max(100, Math.min(500, recipientStart.y + (Math.random() - 0.5) * 200)),
        };
        ballTarget = target;
      } else {
        const candidates = [5, 6, 7, 8, 9, 10];
        actorEnt = candidates[Math.floor(Math.random() * candidates.length)];
        actorStart = { ...baseFormation.a[actorEnt] };
        target = {
          x: 720 + Math.random() * 140,
          y: 120 + Math.random() * 360,
        };
        ballTarget = null;
      }

      // Compute reaction destinations for EVERY other player (this is "all 22 have
      // their own brain" — they see the new focus point and shift accordingly).
      const focus = ballTarget || target;
      const aTargets = baseFormation.a.map((p, i) => {
        if (i === actorEnt) return { x: target.x, y: target.y };
        if (isBallPass && i === recipientEnt) return { x: target.x, y: target.y };
        return tacticalReaction(p, focus, true, 0.22);
      });
      const bTargets = baseFormation.b.map((p) => tacticalReaction(p, focus, false, 0.30));

      return {
        type: isBallPass ? "ball_pass" : "player_run",
        actorEnt, actorStart,
        recipientEnt, recipientStart,
        ballTarget,
        target,
        aStart: baseFormation.a.map(p => ({ ...p })),
        bStart: baseFormation.b.map(p => ({ ...p })),
        aTargets,
        bTargets,
        ballStart: { ...ball },
        duration: 7200,
        drawDuration: 1100,
        moveStart: 1300,
        moveDuration: 4200,
        startedAt: performance.now(),
        latencyShownAt: 1300,
      };
    }
    scenario = pickScenario();

    function easeOutCubic(x) { return 1 - Math.pow(1 - x, 3); }
    function easeInOutCubic(x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; }
    function clamp01(x) { return Math.max(0, Math.min(1, x)); }

    function drawPitch(scale) {
      ctx.save();
      ctx.scale(scale, scale);

      // background gradient
      const grad = ctx.createLinearGradient(0, 0, 0, PITCH_H);
      grad.addColorStop(0, "#0e2a1c");
      grad.addColorStop(1, "#082015");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, PITCH_W, PITCH_H);

      // mowing stripes (5 horizontal bands)
      const stripeH = PITCH_H / 10;
      for (let i = 0; i < 10; i++) {
        ctx.fillStyle = i % 2 === 0 ? "rgba(255,255,255,0.012)" : "rgba(0,0,0,0.04)";
        ctx.fillRect(0, i * stripeH, PITCH_W, stripeH);
      }

      // outer rect + halfway
      ctx.strokeStyle = COLOR_LINE;
      ctx.lineWidth = 1.5;
      const pad = 20;
      ctx.strokeRect(pad, pad, PITCH_W - pad * 2, PITCH_H - pad * 2);
      ctx.beginPath();
      ctx.moveTo(PITCH_W / 2, pad);
      ctx.lineTo(PITCH_W / 2, PITCH_H - pad);
      ctx.stroke();

      // center circle
      ctx.beginPath();
      ctx.arc(PITCH_W / 2, PITCH_H / 2, 60, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(PITCH_W / 2, PITCH_H / 2, 2, 0, Math.PI * 2);
      ctx.fillStyle = COLOR_LINE;
      ctx.fill();

      // penalty areas (left + right)
      const paW = 110, paH = 240;
      ctx.strokeRect(pad, (PITCH_H - paH) / 2, paW, paH);
      ctx.strokeRect(PITCH_W - pad - paW, (PITCH_H - paH) / 2, paW, paH);

      // goal areas
      const gaW = 36, gaH = 100;
      ctx.strokeRect(pad, (PITCH_H - gaH) / 2, gaW, gaH);
      ctx.strokeRect(PITCH_W - pad - gaW, (PITCH_H - gaH) / 2, gaW, gaH);

      ctx.restore();
    }

    function drawPlayer(scale, x, y, color, ringColor, glow = false) {
      ctx.save();
      ctx.scale(scale, scale);
      if (glow) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, 28);
        g.addColorStop(0, color + "60");
        g.addColorStop(1, color + "00");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, 28, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fill();

      if (ringColor) {
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();
    }

    function drawTrail(scale, points, color) {
      if (points.length < 2) return;
      ctx.save();
      ctx.scale(scale, scale);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 1; i < points.length; i++) {
        const a = points[i - 1];
        const b = points[i];
        const alpha = i / points.length;
        ctx.strokeStyle = `rgba(${hexRGB(color)},${alpha * 0.6})`;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    function hexRGB(hex) {
      const h = hex.replace("#", "");
      const r = parseInt(h.substring(0, 2), 16);
      const g = parseInt(h.substring(2, 4), 16);
      const b = parseInt(h.substring(4, 6), 16);
      return `${r},${g},${b}`;
    }

    function drawArrow(scale, from, to, progress, color = COLOR_ACCENT, dashed = false) {
      ctx.save();
      ctx.scale(scale, scale);
      const headLen = 18;
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const len = Math.hypot(dx, dy);
      const visibleLen = len * progress;
      const ux = dx / len;
      const uy = dy / len;
      const tipX = from.x + ux * visibleLen;
      const tipY = from.y + uy * visibleLen;

      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 3 : 3.5;
      ctx.lineCap = "round";
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      if (dashed) ctx.setLineDash([10, 6]);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      ctx.setLineDash([]);

      if (progress > 0.8) {
        const headAlpha = (progress - 0.8) / 0.2;
        ctx.globalAlpha = headAlpha;
        ctx.fillStyle = color;
        const angle = Math.atan2(uy, ux);
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - headLen * Math.cos(angle - 0.4), tipY - headLen * Math.sin(angle - 0.4));
        ctx.lineTo(tipX - headLen * Math.cos(angle + 0.4), tipY - headLen * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

    function drawRing(scale, x, y, color, opacity, radius = 16) {
      ctx.save();
      ctx.scale(scale, scale);
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    function lerpPoint(a, b, t) {
      return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    }

    function render(now) {
      const elapsed = now - scenario.startedAt;
      const scale = canvas.width / PITCH_W;

      ctx.setTransform(1, 0, 0, 1, 0, 0);
      drawPitch(scale);

      // Phase progresses
      const drawT = clamp01(elapsed / scenario.drawDuration);
      const drawProgress = easeOutCubic(drawT);
      const moveT = clamp01((elapsed - scenario.moveStart) / scenario.moveDuration);
      const moveProgress = easeInOutCubic(moveT);

      // ── ARROW (drawn first so dots render over the head) ─────────────────
      const isBallPass = scenario.type === "ball_pass";
      if (isBallPass) {
        drawArrow(scale, scenario.ballStart, scenario.ballTarget, drawProgress, COLOR_BALL, true);
        // After draw: highlight the recipient with a pulsing ring
        if (drawT >= 1) {
          const pulse = 0.6 + 0.3 * Math.sin(elapsed * 0.006);
          const r = scenario.recipientStart;
          drawRing(scale, r.x, r.y, COLOR_BALL, pulse, 18);
        }
      } else {
        drawArrow(scale, scenario.actorStart, scenario.target, drawProgress);
      }

      // ── All 22 players animate from start to their reaction target ───────
      // (this is the "every player has its own brain" semantic for the canvas demo)
      for (let i = 0; i < scenario.bStart.length; i++) {
        const cur = lerpPoint(scenario.bStart[i], scenario.bTargets[i], moveProgress);
        drawPlayer(scale, cur.x, cur.y, COLOR_TEAM_B);
      }
      for (let i = 0; i < scenario.aStart.length; i++) {
        const cur = lerpPoint(scenario.aStart[i], scenario.aTargets[i], moveProgress);
        const isActor = i === scenario.actorEnt;
        const isRecipient = isBallPass && i === scenario.recipientEnt;
        if (isActor || isRecipient) {
          // Trail for the highlighted entity
          if (moveT > 0) {
            const samples = 14;
            const trail = [];
            for (let s = 0; s < samples; s++) {
              const tt = Math.max(0, moveProgress - s * 0.04);
              trail.push(lerpPoint(scenario.aStart[i], scenario.aTargets[i], tt));
            }
            drawTrail(scale, trail, COLOR_TEAM_A);
          }
          drawPlayer(scale, cur.x, cur.y, COLOR_TEAM_A, COLOR_ACCENT, true);
        } else {
          drawPlayer(scale, cur.x, cur.y, COLOR_TEAM_A);
        }
      }

      // ── Ball ─────────────────────────────────────────────────────────────
      let ballX, ballY;
      if (isBallPass && moveT > 0) {
        // The ball arcs along the dashed arrow; same easing as the recipient
        const bp = lerpPoint(scenario.ballStart, scenario.ballTarget, moveProgress);
        // Small parabolic lift (visualises the ball arcing) — peaks at moveProgress=0.5
        const arc = Math.sin(moveProgress * Math.PI) * 22;
        ballX = bp.x;
        ballY = bp.y - arc;
        // Ball trail
        if (moveT > 0.1) {
          const samples = 12;
          const trail = [];
          for (let s = 0; s < samples; s++) {
            const tt = Math.max(0, moveProgress - s * 0.045);
            const p = lerpPoint(scenario.ballStart, scenario.ballTarget, tt);
            const a = Math.sin(tt * Math.PI) * 22;
            trail.push({ x: p.x, y: p.y - a });
          }
          drawTrail(scale, trail, COLOR_BALL);
        }
      } else {
        ballX = ball.x + Math.sin(elapsed * 0.001) * 4;
        ballY = ball.y + Math.cos(elapsed * 0.0013) * 4;
      }
      drawPlayer(scale, ballX, ballY, COLOR_BALL);

      updateStats(elapsed, drawT, moveT, isBallPass);

      if (elapsed >= scenario.duration) {
        scenario = pickScenario();
      }

      requestAnimationFrame(render);
    }

    const timeEl = document.querySelector("[data-bind='time']");
    const arrowEl = document.querySelector("[data-bind='arrow']");
    const latencyEl = document.querySelector("[data-bind='latency']");

    function updateStats(elapsed, drawT, moveT, isBallPass) {
      if (!timeEl) return;
      const secs = elapsed / 1000;
      const m = String(Math.floor(secs / 60)).padStart(2, "0");
      const s = (secs % 60).toFixed(2).padStart(5, "0");
      timeEl.textContent = `${m}:${s}`;

      if (drawT < 1) {
        arrowEl.textContent = isBallPass
          ? `ball pass · drawing… ${Math.round(drawT * 100)}%`
          : `arrow · drawing… ${Math.round(drawT * 100)}%`;
      } else {
        if (isBallPass) {
          const dx = scenario.ballTarget.x - scenario.ballStart.x;
          const dy = scenario.ballTarget.y - scenario.ballStart.y;
          const distM = (Math.hypot(dx, dy) / 9.0).toFixed(1);
          arrowEl.textContent = `pass → entity ${scenario.recipientEnt + 1} · +${distM} m`;
        } else {
          const dx = scenario.target.x - scenario.actorStart.x;
          const dy = scenario.target.y - scenario.actorStart.y;
          const distM = (Math.hypot(dx, dy) / 9.0).toFixed(1);
          arrowEl.textContent = `entity ${scenario.actorEnt + 1} → +${distM} m`;
        }
      }

      if (elapsed > scenario.latencyShownAt) {
        if (!scenario._latencyMs) scenario._latencyMs = 720 + Math.round(Math.random() * 380);
        latencyEl.textContent = `${scenario._latencyMs} ms`;
      } else {
        latencyEl.textContent = "—";
      }
    }

    requestAnimationFrame(render);
  }
}

// ─── Defensive: pause heavy stuff on tab hide (saves laptop battery) ──────
document.addEventListener("visibilitychange", () => {
  // requestAnimationFrame already pauses on hidden tabs in modern browsers;
  // this is just an explicit signal hook for future expansion.
});

// ─── Demo sequencer: play → rewind → draw arrows → alternative ──────────────
// Stage 1 plays the real clip; stage 2 plays a pre-rendered 1.5x reverse clip
// (smooth & frame-accurate — browsers can't reverse-play reliably); stage 3
// draws tactical arrows over the frozen decision frame; stage 4 crossfades to
// the generated alternative.
(() => {
  const stage = document.getElementById("demo-stage");
  const orig = document.getElementById("demo-original");
  const rew = document.getElementById("demo-rewind");
  const alt = document.getElementById("demo-alt");
  const label = document.getElementById("demo-stage-label");
  const playBtn = document.getElementById("demo-play");
  if (!stage || !orig || !rew || !alt || !playBtn) return;

  const ARROW_HOLD_MS = 2200; // arrow choreography runs ~1.8s, then advance

  const setStage = (s) => stage.setAttribute("data-stage", s);
  const setLabel = (t) => { if (label) label.textContent = t; };

  // Pre-measure each run/defender arrow's length so the draw animation is exact.
  stage.querySelectorAll(".demo-arrow").forEach((p) => {
    try { p.style.setProperty("--len", Math.ceil(p.getTotalLength())); }
    catch { /* getTotalLength unsupported — CSS fallback covers it */ }
  });

  const showButton = (show, text) => {
    playBtn.style.display = show ? "" : "none";
    if (text) playBtn.querySelector(".demo-play-label").textContent = text;
  };

  // Stage 2 — smooth reverse clip; its last frame is the decision moment.
  const rewind = () => {
    setStage("rewind");
    setLabel("Rewinding to the decision");
    rew.currentTime = 0;
    rew.onended = drawArrows;
    rew.play().catch(drawArrows); // clip missing → skip to arrows
  };

  // Stage 3 — arrows draw via CSS over the held decision frame, then advance.
  const drawArrows = () => {
    rew.pause();
    setStage("arrows");
    setLabel("Coach draws the alternative");
    setTimeout(playAlternative, ARROW_HOLD_MS);
  };

  // Stage 4 — crossfade to the alternative (or finish if it's not added yet).
  const playAlternative = () => {
    setStage("alternative");
    setLabel("The generated what-if");
    alt.currentTime = 0;
    alt.onended = finish;
    alt.play().then(() => {}).catch(finish);
  };

  const finish = () => { setStage("done"); setLabel(""); showButton(true, "Replay"); };

  const start = () => {
    showButton(false);
    setStage("original");
    setLabel("The play as it happened");
    orig.currentTime = 0;
    orig.onended = rewind;
    orig.play().catch(() => {
      // Playback blocked / clip missing — recover to idle so the poster shows.
      setStage("idle");
      showButton(true, "Play demo");
    });
  };

  playBtn.addEventListener("click", start);
})();
