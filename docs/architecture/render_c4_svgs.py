#!/usr/bin/env python3
"""Render hand-laid-out C4 workflow SVGs.

The first version of these diagrams used Mermaid auto-layout. That made the
Markdown view technically renderable, but dense C4 views ended up with cramped
labels and crossing connectors. This script keeps the diagrams editable as data
while producing fixed-layout SVGs with predictable lanes and spacing.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

STYLE = {
    "person": ("#fff7ed", "#ea580c", "#431407"),
    "system": ("#e0f2fe", "#0369a1", "#0f172a"),
    "ui": ("#dcfce7", "#15803d", "#052e16"),
    "api": ("#e0f2fe", "#0369a1", "#0f172a"),
    "model": ("#f3e8ff", "#7e22ce", "#1f102e"),
    "process": ("#fef9c3", "#a16207", "#422006"),
    "store": ("#fee2e2", "#b91c1c", "#450a0a"),
    "external": ("#f8fafc", "#64748b", "#0f172a"),
    "risk": ("#ffedd5", "#c2410c", "#431407"),
}


def lines(text: str, width: int) -> list[str]:
    out: list[str] = []
    for part in text.split("\n"):
        if not part:
            out.append("")
        else:
            out.extend(wrap(part, width=width, break_long_words=False, break_on_hyphens=False))
    return out


class Svg:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
            "<defs>",
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">',
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/>',
            "</marker>",
            '<filter id="shadow" x="-15%" y="-20%" width="130%" height="150%">',
            '<feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#0f172a" flood-opacity="0.12"/>',
            "</filter>",
            "</defs>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
        ]

    def title(self, text: str, subtitle: str | None = None) -> None:
        self.parts.append(f'<text x="40" y="50" font-size="30" font-weight="800" fill="#0f172a">{escape(text)}</text>')
        if subtitle:
            self.parts.append(f'<text x="40" y="82" font-size="17" fill="#475569">{escape(subtitle)}</text>')

    def group(self, x: int, y: int, w: int, h: int, title: str, stroke: str = "#cbd5e1", fill: str = "#f8fafc") -> None:
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        self.parts.append(f'<text x="{x + 20}" y="{y + 34}" font-size="18" font-weight="800" fill="#334155">{escape(title)}</text>')

    def card(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        body: str,
        kind: str,
        *,
        number: str | None = None,
    ) -> None:
        fill, stroke, ink = STYLE[kind]
        self.parts.append(f'<g filter="url(#shadow)">')
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="8" rx="4" fill="{stroke}"/>')
        title_right_padding = 18
        if number:
            badge_w = max(32, 20 + len(number) * 10)
            bx = x + w - badge_w - 14
            self.parts.append(f'<rect x="{bx}" y="{y + 20}" width="{badge_w}" height="28" rx="14" fill="{stroke}"/>')
            self.parts.append(f'<text x="{bx + badge_w / 2:.1f}" y="{y + 39}" text-anchor="middle" font-size="15" font-weight="800" fill="#ffffff">{escape(number)}</text>')
            title_right_padding = badge_w + 28
        tx = x + 18
        max_chars = max(16, int((w - (tx - x) - title_right_padding) / 8.1))
        ty = y + 32
        for i, line in enumerate(lines(title, max_chars)):
            self.parts.append(f'<text x="{tx}" y="{ty + i * 20}" font-size="18" font-weight="800" fill="{ink}">{escape(line)}</text>')
        body_y = ty + 26 + max(0, len(lines(title, max_chars)) - 1) * 20
        for i, line in enumerate(lines(body, max_chars)):
            self.parts.append(f'<text x="{x + 18}" y="{body_y + i * 18}" font-size="14.5" fill="{ink}">{escape(line)}</text>')
        self.parts.append("</g>")

    def edge(self, points: list[tuple[int, int]], label: str | None = None, *, dashed: bool = False) -> None:
        # Keep connector lanes clean. The diagrams use numbered nodes and card
        # labels for semantics; floating edge captions caused text overlaps in
        # dense C4 views.
        label = None
        d = f"M {points[0][0]} {points[0][1]} " + " ".join(f"L {x} {y}" for x, y in points[1:])
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="#475569" stroke-width="2.4"{dash} marker-end="url(#arrow)"/>')
        if label:
            mid = points[len(points) // 2]
            label_lines = lines(label, 28)
            label_w = max(120, min(280, max(len(l) for l in label_lines) * 8 + 22))
            label_h = 22 + (len(label_lines) - 1) * 16
            lx = mid[0] - label_w // 2
            ly = mid[1] - label_h - 8
            self.parts.append(f'<rect x="{lx}" y="{ly}" width="{label_w}" height="{label_h}" rx="7" fill="#ffffff" stroke="#cbd5e1"/>')
            for i, line in enumerate(label_lines):
                self.parts.append(f'<text x="{mid[0]}" y="{ly + 15 + i * 16}" text-anchor="middle" font-size="12.5" fill="#334155">{escape(line)}</text>')

    def note(self, x: int, y: int, text: str, color: str = "#475569") -> None:
        for i, line in enumerate(lines(text, 80)):
            self.parts.append(f'<text x="{x}" y="{y + i * 18}" font-size="14" fill="{color}">{escape(line)}</text>')

    def save(self, name: str) -> None:
        self.parts.append("</svg>")
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        (ASSET_DIR / name).write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def system_context() -> None:
    s = Svg(1800, 980, "C4 Level 1 - System Context")
    s.title("C4 Level 1 - System Context", "Who uses Osa, what Osa owns, and which systems remain outside the boundary.")
    s.card(680, 330, 440, 230, "Osa Tactical What-If Engine", "Resimulates football moments from tracking history plus tactical arrows. Owns the 2D renderer, GenTac diffusion model, physics post-processing, and trajectory API.", "system")
    left = [
        (90, 170, "Coach / Analyst / Player", "Draws tactical intent and compares actual vs alternative movement."),
        (90, 390, "League / Club Buyer", "Provides data access, validates fit, and buys deployment."),
        (90, 610, "Founder / ML Operator", "Runs training, evaluates checkpoints, and operates design-partner deployments."),
    ]
    right = [
        (1320, 130, "Tracking Data Providers", "Metrica today; Hawk-Eye, Sportec DFL, Stats Perform, SkillCorner later."),
        (1320, 340, "GPU Training Providers", "Mac MPS for smoke tests; Modal, Lambda, RunPod, or vast.ai for full runs."),
        (1320, 550, "Downstream 3D / Video Generator", "Future consumer of trajectory keypoints through the handoff API."),
        (1320, 760, "Mail Client", "Landing-page inquiry via mailto to leagues@osa.dev."),
    ]
    for i, (x, y, t, b) in enumerate(left, 1):
        s.card(x, y, 350, 150, t, b, "person", number=str(i))
        s.edge([(440, y + 75), (610, y + 75), (610, 445), (680, 445)])
    for i, (x, y, t, b) in enumerate(right, 1):
        s.card(x, y, 390, 150, t, b, "external", number=str(i))
        s.edge([(1120, 445), (1210, 445), (1210, y + 75), (1320, y + 75)], ["Tracking feed", "Training jobs", "Trajectory JSON", "Demo inquiry"][i - 1])
    s.note(680, 610, "Boundary rule: Osa owns trajectory generation and 2D tactical validation. Photorealistic video is a downstream consumer, not this repo.", "#0369a1")
    s.save("c4-01-system-context.svg")


def containers() -> None:
    s = Svg(2200, 1240, "C4 Level 2 - Container View")
    s.title("C4 Level 2 - Container View", "Runnable containers, artifact stores, and the major workflows between them.")
    s.card(70, 130, 300, 130, "Coach / Analyst Browser", "Uses the static tactics board and reviews generated alternatives.", "person", number="1")
    s.card(70, 310, 300, 130, "League Buyer Browser", "Reads the landing site and starts the demo inquiry.", "person", number="2")
    s.card(70, 500, 300, 130, "Raw Tracking Data", "Provider files or feeds before conversion to GenTac JSON.", "external", number="3")
    s.card(70, 690, 300, 130, "Cloud GPU Runner", "Modal, Lambda Labs, RunPod, or vast.ai training execution.", "external", number="4")
    s.group(430, 115, 1650, 960, "Osa Tactical What-If Engine")
    s.group(470, 175, 430, 250, "Browser-facing")
    s.group(940, 175, 430, 250, "Inference runtime")
    s.group(1410, 175, 610, 250, "Offline ML and QA")
    s.group(470, 515, 1550, 500, "Artifact stores")
    s.card(500, 245, 330, 120, "Landing Website", "site/index.html, site/app.js, site/style.css. Static product story and mailto handoff.", "ui")
    s.card(980, 245, 330, 120, "GenTac Model Package", "src/model/*. Config, dataset, tokenizer, backbone, diffusion, sampler, physics.", "model")
    s.card(500, 385, 330, 120, "2D Tactics Board", "index.html and src/render/*. Canvas, draw mode, compare mode, samples mode.", "ui")
    s.card(980, 385, 330, 120, "FastAPI Inference Server", "src/server/main.py. /api/health, /api/generate, /api/generate/stream.", "api")
    s.card(1450, 245, 250, 120, "Training Scripts", "Smoke, dry-run, full, and Modal training paths.", "process")
    s.card(1725, 245, 250, 120, "Sampling / Eval Scripts", "samples.json generation plus ADE/FDE/diversity checks.", "process")
    stores = [
        (500, 610, "Processed Match JSON", "data/processed/*.json. Runtime frames, masks, metadata."),
        (870, 610, "Sample Output JSON", "data/processed/samples.json. Offline alternatives for samples mode."),
        (1240, 610, "Model Checkpoints", "checkpoints/smoke and checkpoints/full. Schema-versioned Lightning artifacts."),
        (1610, 610, "Handoff Contract", "docs/handoff_api.md. Trajectory contract for future video."),
    ]
    for i, (x, y, t, b) in enumerate(stores, 1):
        s.card(x, y, 300, 150, t, b, "store", number=str(i))
    s.card(1740, 850, 280, 120, "Downstream Video Generator", "Future 3D or photorealistic renderer that consumes trajectory JSON.", "external")
    s.edge([(370, 195), (410, 195), (410, 445), (500, 445)], "opens board")
    s.edge([(370, 375), (450, 375), (450, 305), (500, 305)], "reads product site")
    s.edge([(370, 565), (470, 565), (470, 685), (500, 685)], "offline conversion")
    s.edge([(370, 755), (395, 755), (395, 1115), (1575, 1115), (1575, 365)], "runs full training")
    s.edge([(830, 445), (980, 445)], "POST generate")
    s.edge([(1145, 385), (1145, 365)], "loads model")
    s.edge([(1145, 505), (1145, 545), (650, 545), (650, 610)], "reads matches")
    s.edge([(1310, 305), (1365, 305), (1365, 585), (1310, 585), (1310, 610)], "reads checkpoint")
    s.edge([(1555, 365), (1555, 785), (1465, 785), (1465, 760)], "writes checkpoint")
    s.edge([(1940, 365), (1940, 805), (1020, 805), (1020, 760)], "writes samples")
    s.edge([(830, 475), (880, 475), (880, 575), (1020, 575), (1020, 610)], "GET samples")
    s.edge([(1910, 760), (1910, 850)], "handoff")
    s.save("c4-02-container-view.svg")


def inference_components() -> None:
    s = Svg(2200, 1320, "C4 Level 3 - Inference And Model Components")
    s.title("C4 Level 3 - Inference And Model Components", "Request path through FastAPI and into GenTac sampling internals.")
    s.group(70, 120, 950, 1180, "FastAPI container: src/server/main.py")
    s.group(1090, 120, 1040, 1180, "GenTac model package: src/model/*")
    pipeline = [
        ("HTTP Middleware", "CORS allowlist, body-size guard, JSON logs, X-Request-Id.", "api"),
        ("Auth + Pydantic Schemas", "Optional API key plus GenerateRequest, PlayerArrow, BallPassArrow bounds.", "api"),
        ("Queue Guard", "Semaphore caps active generations; queue cap returns 503 early.", "api"),
        ("Match Resolver", "_safe_match_path, _get_match, match_cache, processed JSON arrays.", "api"),
        ("Arrow Mapper", "Player pins, ball_pass pins, recipient receiving stance offset.", "api"),
        ("Plausibility Gate", "Max reachable distance and minimum player spacing checks.", "api"),
        ("_generate_sync", "Builds history, valid_static, target_mask, opponent future, K samples.", "api"),
        ("Response Builder", "Denormalizes meters and returns history, actual_future, samples.", "api"),
    ]
    y = 190
    for i, (t, b, kind) in enumerate(pipeline, 1):
        s.card(130, y, 360, 105, t, b, kind, number=str(i))
        if i < len(pipeline):
            s.edge([(310, y + 105), (310, y + 140)])
        y += 140
    s.card(560, 190, 360, 115, "Lifespan Loader", "Loads checkpoint, restores cfg/model/schedule, chooses MPS or CPU.", "api")
    s.card(560, 370, 360, 115, "SSE Stream Wrapper", "/api/generate/stream emits metadata, window chunks, done.", "api")
    s.card(560, 550, 360, 115, "Processed Match JSON", "data/processed/*.json cached as positions, masks, periods, frame IDs.", "store")
    s.card(560, 730, 360, 115, "Model Checkpoint", "checkpoints/*/*.ckpt loaded through GenTacTrajectoryModule.", "store")
    s.edge([(490, 662), (525, 662), (525, 607), (560, 607)], "safe read")
    s.edge([(920, 248), (1000, 248), (1000, 788), (920, 788)], "loads")
    model_nodes = [
        (1150, 190, "GenTacConfig", "Pitch, 23 entities, 25 FPS, H=100, w=5, diffusion steps, waypoint CFG.", "model"),
        (1540, 190, "TrajectoryDataset", "Stable 11-player slots, normalized coordinates, validity masks.", "model"),
        (1150, 370, "LinearBetaSchedule", "beta, alpha, alpha_bar tables and q_sample.", "model"),
        (1540, 370, "GenTacTrajectoryModule", "cfg roundtrip, schema validation, EMA buffers, Lightning shell.", "model"),
        (1150, 550, "TrajectoryTokenizer", "Coordinate, temporal, group, entity, and waypoint/null embeddings.", "model"),
        (1540, 550, "GenTacDiffusion", "Epsilon-prediction network for future-window noise.", "model"),
        (1150, 730, "SpatioTemporalBackbone", "Factorized temporal and entity attention blocks.", "model"),
        (1540, 730, "causal_rollout", "w-frame reverse diffusion windows, hard pins, learned waypoints.", "model"),
        (1150, 910, "apply_physics", "Seam anchor, speed caps, smoothing, pitch clamp, repulsion, final re-clip.", "model"),
        (1540, 910, "Frame Serializer", "Rounds meter-space team/player/ball frames into response JSON.", "model"),
    ]
    for x, y, t, b, kind in model_nodes:
        s.card(x, y, 330, 125, t, b, kind)
    s.edge([(490, 1082), (1045, 1082), (1045, 885), (1705, 885), (1705, 855)], "calls sampler")
    s.edge([(1315, 315), (1315, 370)])
    s.edge([(1705, 495), (1705, 550)])
    s.edge([(1315, 675), (1315, 730)])
    s.edge([(1705, 675), (1705, 730)])
    s.edge([(1705, 855), (1705, 885), (1315, 885), (1315, 910)], "raw samples")
    s.edge([(1480, 972), (1540, 972)], "clean samples")
    s.edge([(1870, 972), (2050, 972), (2050, 1240), (490, 1240), (490, 1222)], "response")
    s.save("c4-03-inference-model-components.svg")


def browser_components() -> None:
    s = Svg(1900, 1040, "C4 Level 3 - Browser Renderer Components")
    s.title("C4 Level 3 - Browser Renderer Components", "How the static Canvas app loads data, captures arrows, calls inference, and compares samples.")
    s.card(70, 155, 300, 125, "Coach / Analyst", "Chooses a frame, draws tactical intent, and reviews playback.", "person")
    s.group(430, 120, 1340, 800, "2D tactics board: index.html + src/render/pitch.js")
    nodes = [
        (500, 190, "main Bootstrapping", "Binds DOM controls, URL params, panel labels, status text.", "ui"),
        (860, 190, "JSON Loader", "Loads clip JSON or samples.json; reports missing-data status.", "ui"),
        (1220, 190, "Panel Factory", "createPanel, drawPitch, drawFrame, trails, frame helpers.", "ui"),
        (500, 390, "Playback Controls", "Play/pause, scrubber, speed, trails, compare mode.", "ui"),
        (860, 390, "Draw Mode State", "Hover, drag, right-click clear, Esc cancel, touch-to-mouse.", "ui"),
        (1220, 390, "Arrow Overlay", "Player arrows, ball-pass shaft, recipient ring, hover pulse.", "ui"),
        (500, 590, "Payload Builder", "decision_frame, match, dynamic horizon, fade, mode, arrows.", "ui"),
        (860, 590, "Generate API Client", "fetch POST /api/generate and swaps panels to actual vs alternative.", "ui"),
        (1220, 590, "Sample / Diff View", "buildVirtualClip, sample picker, overlay ghost actual future.", "ui"),
    ]
    for x, y, t, b, kind in nodes:
        s.card(x, y, 300, 125, t, b, kind)
    s.card(500, 800, 300, 95, "Clip JSON", "data/processed/Sample_Game_1_clip.json", "store")
    s.card(860, 800, 300, 95, "samples.json", "Offline alternatives for ?samples=1.", "store")
    s.card(1220, 800, 300, 95, "FastAPI Server", "http://127.0.0.1:8001/api/generate", "api")
    s.edge([(370, 217), (500, 217)], "opens page")
    s.edge([(800, 252), (860, 252)])
    s.edge([(1160, 252), (1220, 252)])
    s.edge([(1370, 315), (1370, 390)])
    s.edge([(650, 315), (650, 390)])
    s.edge([(1010, 515), (650, 590)], "arrows")
    s.edge([(800, 652), (860, 652)], "payload")
    s.edge([(1160, 652), (1220, 652)], "response")
    s.edge([(1010, 315), (1010, 350), (460, 350), (460, 848), (500, 848)], "normal mode")
    s.edge([(1010, 315), (1010, 350), (1580, 350), (1580, 848), (1160, 848)], "samples mode")
    s.edge([(1010, 715), (1010, 760), (1370, 760), (1370, 800)], "POST")
    s.edge([(1370, 800), (1370, 780), (1010, 780), (1010, 715)], "JSON", dashed=True)
    s.save("c4-04-browser-renderer-components.svg")


def generation_workflow() -> None:
    s = Svg(2300, 760, "C4 Dynamic View - Arrow-Driven Generation Workflow")
    s.title("C4 Dynamic View - Arrow-Driven Generation Workflow", "The end-to-end request path, arranged as a two-row sequence to keep labels readable.")
    top = [
        ("Load Board", "Clip or samples mode initializes panels.", "ui"),
        ("Select Frame", "Scrubber or URL frame parameter sets decision point.", "ui"),
        ("Draw Intent", "Player run or ball-pass recipient captured.", "ui"),
        ("Build Payload", "Dynamic feasible horizon and arrow JSON.", "ui"),
        ("API Ingress", "CORS, size guard, request ID, schema bounds.", "api"),
        ("Load Match", "Safe path under data/processed and match cache.", "api"),
        ("Map Waypoints", "Player, ball, and recipient pins.", "api"),
    ]
    bottom = [
        ("Plausibility Gate", "Reject impossible speed or spacing.", "api"),
        ("Causal Rollout", "H=100 history, w=5 windows, DDPM reverse steps.", "model"),
        ("Learned Waypoints", "Soft signal plus hard projection pins.", "model"),
        ("Physics Post-Pass", "Seam, speed caps, smoothing, clamp, repulsion.", "model"),
        ("Build Response", "metadata, history, actual_future, samples.", "api"),
        ("Compare Playback", "Actual left, alternative right, synced controls.", "ui"),
        ("Future Handoff", "Same trajectory JSON can feed 3D/video.", "external"),
    ]
    xs = [70, 390, 710, 1030, 1350, 1670, 1990]
    for i, ((t, b, kind), x) in enumerate(zip(top, xs), 1):
        s.card(x, 150, 250, 125, t, b, kind, number=str(i))
        if i < 7:
            s.edge([(x + 250, 212), (xs[i], 212)])
    for i, ((t, b, kind), x) in enumerate(zip(bottom, reversed(xs)), 8):
        s.card(x, 455, 250, 125, t, b, kind, number=str(i))
    s.edge([(2240, 212), (2240, 360), (2240, 455)], "continues")
    reversed_xs = list(reversed(xs))
    for idx in range(6):
        s.edge([(reversed_xs[idx], 517), (reversed_xs[idx + 1] + 250, 517)])
    s.note(70, 690, "Numbering is the interaction order. Colors follow C4 categories: UI, API, model, and external consumer.", "#475569")
    s.save("c4-05-dynamic-generation-workflow.svg")


def training_workflow() -> None:
    s = Svg(2400, 1260, "C4 Dynamic View - Training, Sampling, And Evaluation Workflow")
    s.title("C4 Dynamic View - Training, Sampling, And Evaluation Workflow", "Separate lanes for data preparation, training, evaluation, and guardrails.")
    s.group(60, 120, 2280, 210, "Data preparation")
    s.group(60, 390, 2280, 300, "Training")
    s.group(60, 750, 2280, 250, "Sampling and evaluation")
    s.group(60, 1030, 2280, 170, "Automated guardrails")
    data = [
        (120, 185, "Raw Tracking Data", "Metrica sample data today; provider data later.", "external"),
        (520, 185, "Expected Conversion", "Tracking feed to GenTac JSON and dev clips. README documents this stage.", "process"),
        (920, 185, "Processed JSON Store", "data/processed/*.json with frames, ball, teams, pitch metadata.", "store"),
    ]
    for x, y, t, b, kind in data:
        s.card(x, y, 320, 105, t, b, kind)
    train = [
        (120, 470, "GenTacDataModule", "Discovers matches and creates match-level validation split.", "process"),
        (500, 470, "TrajectoryDataset", "H=100 history plus w=5 future windows, normalized coords, masks.", "process"),
        (880, 470, "GenTacTrajectoryModule", "Random pretrain mode, optimizer, cosine schedule, EMA.", "model"),
        (1260, 470, "DDPM Training Step", "Synthetic waypoints, CFG dropout, q_sample, epsilon loss.", "model"),
        (1640, 435, "Smoke Training", "scripts/smoke_train.py. Tiny Mac MPS/CPU run.", "process"),
        (1640, 560, "Full Training", "scripts/train_full.py and modal_train.py on A100/H100.", "process"),
    ]
    for x, y, t, b, kind in train:
        s.card(x, y, 320, 105, t, b, kind)
    s.card(2020, 500, 280, 120, "Checkpoint Store", "schema-versioned ckpt files", "store")
    evals = [
        (120, 815, "scripts/sample.py", "Loads checkpoint, samples K alternatives, applies physics.", "process"),
        (520, 815, "samples.json", "Renderer-ready actual vs generated alternatives.", "store"),
        (920, 815, "2D Renderer", "?samples=1 compares offline samples.", "ui"),
        (1320, 815, "scripts/eval.py", "ADE/FDE, diversity, speed, off-pitch, role ADE, arrow honor.", "process"),
        (1720, 815, "eval.json", "Machine-readable checkpoint metrics.", "store"),
    ]
    for x, y, t, b, kind in evals:
        s.card(x, y, 320, 105, t, b, kind)
    tests = [
        (120, 1095, "Server validation tests: schema bounds, path traversal, body size.", "api"),
        (820, 1095, "Physics tests: speed caps, seam anchor, pitch clamp, no NaN.", "model"),
        (1520, 1095, "Schema tests: cfg roundtrip and checkpoint mismatch refusal.", "process"),
    ]
    for x, y, t, kind in tests:
        s.card(x, y, 620, 80, t, "", kind)
    s.edge([(440, 237), (520, 237)])
    s.edge([(840, 237), (920, 237)])
    s.edge([(1080, 290), (1080, 360), (280, 360), (280, 470)], "feeds training")
    s.edge([(440, 522), (500, 522)])
    s.edge([(820, 522), (880, 522)])
    s.edge([(1200, 522), (1260, 522)])
    s.edge([(1580, 522), (1640, 487)], "smoke")
    s.edge([(1580, 522), (1640, 612)], "full")
    s.edge([(1960, 487), (2020, 535)])
    s.edge([(1960, 612), (2020, 585)])
    s.edge([(2160, 620), (2160, 725), (280, 725), (280, 815)], "load")
    s.edge([(440, 867), (520, 867)])
    s.edge([(840, 867), (920, 867)])
    s.edge([(2160, 620), (2160, 725), (1480, 725), (1480, 815)], "load")
    s.edge([(1640, 867), (1720, 867)])
    s.save("c4-06-training-sampling-evaluation.svg")


def deployment_view() -> None:
    s = Svg(1900, 980, "C4 Deployment View - Current Local State And First-Customer Target")
    s.title("C4 Deployment View", "Current local development topology and the first-customer deployment target from docs/operations.md.")
    s.group(70, 140, 820, 720, "Current local development")
    s.group(1010, 140, 820, 720, "First paid customer target")
    local = [
        (140, 230, "Developer Browser", "http://localhost:8000 or 127.0.0.1.", "person"),
        (470, 230, "Python http.server", "Serves index.html and src/render/*.", "ui"),
        (470, 410, "Uvicorn + FastAPI", "127.0.0.1:8001 inference server.", "api"),
        (140, 590, "Local Filesystem", "data/processed, checkpoints, docs.", "store"),
        (470, 590, "Smoke Model In Memory", "Mac MPS if available, otherwise CPU.", "model"),
    ]
    for x, y, t, b, kind in local:
        s.card(x, y, 300, 120, t, b, kind)
    target = [
        (1080, 230, "Customer Browser / API Client", "Uses static UI and HTTPS generation endpoint.", "person"),
        (1410, 230, "Vercel / Static Host", "Landing site and renderer assets.", "ui"),
        (1080, 410, "Cloudflare", "TLS, WAF, rate limiting.", "external"),
        (1410, 410, "Caddy On GPU VM", "Reverse proxy and TLS termination.", "api"),
        (1080, 590, "Uvicorn + FastAPI", "systemd process, model loaded in RAM.", "api"),
        (1410, 590, "A10G/A100 Runtime", "One active generation by default.", "model"),
        (1080, 760, "S3 Bucket", "checkpoints, request artifacts, access logs.", "store"),
        (1410, 760, "Planned Observability", "metrics, alerts, artifact capture, key rotation.", "risk"),
    ]
    for x, y, t, b, kind in target:
        s.card(x, y, 300, 120, t, b, kind)
    s.edge([(440, 290), (470, 290)], "static files")
    s.edge([(290, 350), (290, 470), (470, 470)], "POST generate")
    s.edge([(620, 530), (620, 590)], "loads")
    s.edge([(440, 650), (470, 650)])
    s.edge([(1380, 290), (1410, 290)], "static UI")
    s.edge([(1230, 350), (1230, 410)], "HTTPS API")
    s.edge([(1380, 470), (1410, 470)])
    s.edge([(1560, 530), (1560, 560), (1230, 560), (1230, 590)])
    s.edge([(1380, 650), (1410, 650)])
    s.edge([(1230, 710), (1230, 760)])
    s.edge([(1230, 710), (1230, 830), (1410, 830)], "missing ops", dashed=True)
    s.edge([(890, 500), (1010, 500)], "production gap")
    s.note(70, 910, "The target keeps deployment intentionally simple: a single GPU VM plus static hosting until design-partner demand proves otherwise.", "#475569")
    s.save("c4-07-deployment-view.svg")


def main() -> None:
    system_context()
    containers()
    inference_components()
    browser_components()
    generation_workflow()
    training_workflow()
    deployment_view()
    print(f"Rendered SVGs into {ASSET_DIR}")


if __name__ == "__main__":
    main()
