"""Detect, team-classify, and circle players in the Veo decision frame.

Reads the demo's decision frame (1280x720, the same frame the landing-page SVG
arrows draw onto — viewBox "0 0 1280 720", so every coordinate here maps 1:1 to
that overlay). Runs a COCO-pretrained Faster R-CNN at several scales/tiles to
catch the small far-side players, NMS-merges the detections, classifies each box
into the yellow team vs the dark team by shirt hue, and writes:

  analysis/veo/players_annotated.png  — circled + labelled frame
  analysis/veo/players.json           — [{id, team, cx, cy, box, score}]

This is a first automated pass. Tiny/occluded players and team calls on the
snow-washed far side will need a human cleanup — tune THRESHOLDS or hand-edit
players.json.

Usage:
    .venv/bin/python scripts/detect_players.py [path/to/frame.jpg]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME = ROOT / "site" / "public" / "demo-poster.jpg"
OUT_DIR = ROOT / "analysis" / "veo"

PERSON = 1                 # COCO class id
BALL = 37                  # COCO "sports ball"
BALL_THR = 0.50            # keep ball detections above this (one true ball expected)
SCORE_THR = 0.40           # keep person detections above this
NMS_IOU = 0.40
# Geometry sanity for a wide tactical shot (foot-point box, 1280x720):
MIN_H, MAX_H = 9, 170      # player box height in px
EXCLUDE = (1130, 650, 1280, 720)   # veo watermark, bottom-right — drop boxes here

TEAM_COLORS = {            # RGBA for circles/labels
    "yellow": (255, 150, 0, 255),
    "dark":   (60, 190, 255, 255),
    "other":  (235, 70, 220, 255),
}


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run(model, img: Image.Image, dev) -> tuple[np.ndarray, np.ndarray]:
    """Run detector on one PIL image → (boxes Nx4, scores N) for the person class."""
    t = to_tensor(img).to(dev)
    with torch.no_grad():
        out = model([t])[0]
    keep = (out["labels"] == PERSON) & (out["scores"] >= SCORE_THR)
    boxes = out["boxes"][keep].cpu().numpy()
    scores = out["scores"][keep].cpu().numpy()
    return boxes, scores


def detect(model, base: Image.Image, dev) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale + tiled detection, boxes returned in base (1280x720) coords."""
    W, H = base.size
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    # full frame at 1x and 2x
    for scale in (1.0, 2.0):
        img = base if scale == 1.0 else base.resize((int(W * scale), int(H * scale)))
        b, s = _run(model, img, dev)
        if len(b):
            all_boxes.append(b / scale)
            all_scores.append(s)

    # 2x2 tiles with 12% overlap, each upscaled 2x (best recall on far players)
    ox, oy = int(W * 0.12), int(H * 0.12)
    for cx in (0, 1):
        for cy in (0, 1):
            x0 = max(0, cx * W // 2 - ox)
            y0 = max(0, cy * H // 2 - oy)
            x1 = min(W, (cx + 1) * W // 2 + ox)
            y1 = min(H, (cy + 1) * H // 2 + oy)
            tile = base.crop((x0, y0, x1, y1))
            tile2 = tile.resize((tile.width * 2, tile.height * 2))
            b, s = _run(model, tile2, dev)
            if len(b):
                b = b / 2.0
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                all_boxes.append(b)
                all_scores.append(s)

    if not all_boxes:
        return np.empty((0, 4)), np.empty((0,))
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nms(torch.tensor(boxes), torch.tensor(scores), NMS_IOU).numpy()
    return boxes[keep], scores[keep]


def detect_ball(model, base: Image.Image, dev) -> dict | None:
    """Find the ball: 3x3 tiles upscaled 3x (it's ~5px in this wide shot).

    Returns the highest-scoring sports-ball detection above BALL_THR, or None.
    Low-score candidates tend to land on a player's torso — the size + score
    filter keeps the real ball.
    """
    W, H = base.size
    cands: list[tuple[float, float, float, float]] = []   # (cx, cy, size, score)
    for cx in range(3):
        for cy in range(3):
            x0, y0 = cx * W // 3, cy * H // 3
            x1, y1 = (cx + 1) * W // 3, (cy + 1) * H // 3
            tile = base.crop((x0, y0, x1, y1))
            t3 = tile.resize((tile.width * 3, tile.height * 3))
            t = to_tensor(t3).to(dev)
            with torch.no_grad():
                out = model([t])[0]
            keep = (out["labels"] == BALL) & (out["scores"] >= BALL_THR)
            for box, sc in zip(out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()):
                bx = (box[0] + box[2]) / 6 + x0     # /2 for center, /3 for scale
                by = (box[1] + box[3]) / 6 + y0
                size = float(box[2] - box[0]) / 3
                cands.append((float(bx), float(by), size, float(sc)))
    if not cands:
        return None
    bx, by, size, sc = max(cands, key=lambda c: c[3])
    return {"cx": round(bx, 1), "cy": round(by, 1), "size": round(size, 1), "score": round(sc, 3)}


def classify_team(crop: np.ndarray) -> str:
    """Classify a torso crop into yellow / dark / other from its mean colour.

    Calibrated on this clip's torsos: yellow kit is bright & blue-deficient
    ((R+G)/2 - B large, G high); the dark kit is dark green (green-dominant, low
    brightness). Greys/reds (referee, keepers) fall through to "other".
    """
    if crop.size == 0:
        return "other"
    r, g, b = crop.reshape(-1, 3).astype(np.float32).mean(0)
    mx = max(r, g, b)
    yellowness = (r + g) / 2 - b
    if yellowness >= 42 and g >= 92 and mx >= 88:
        return "yellow"
    if mx <= 118 and g >= r and yellowness < 42:
        return "dark"
    if mx < 70:                      # near-black (very dark kit / shadowed)
        return "dark"
    return "other"


def dedupe(players: list[dict], radius: float = 14.0) -> list[dict]:
    """Drop near-duplicate detections (same player from overlapping tiles).

    Two boxes for one player can survive NMS when their IoU is low; collapse any
    whose foot points are within `radius` px, keeping the higher-scoring one.
    """
    kept: list[dict] = []
    for p in sorted(players, key=lambda p: -p["score"]):
        if all((p["cx"] - q["cx"]) ** 2 + (p["cy"] - q["cy"]) ** 2 > radius ** 2 for q in kept):
            kept.append(p)
    return kept


def main() -> None:
    frame_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = Image.open(frame_path).convert("RGB")
    arr = np.asarray(base)
    dev = device()

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(dev)

    boxes, scores = detect(model, base, dev)
    ball = detect_ball(model, base, dev)

    players = []
    for (x0, y0, x1, y1), sc in zip(boxes, scores):
        h = y1 - y0
        if not (MIN_H <= h <= MAX_H):
            continue
        cx, foot = (x0 + x1) / 2, y1
        ex0, ey0, ex1, ey1 = EXCLUDE
        if ex0 <= cx <= ex1 and ey0 <= foot <= ey1:
            continue
        # torso crop: upper-middle of the box, central 60% width
        ty0, ty1 = int(y0 + 0.15 * h), int(y0 + 0.55 * h)
        bw = x1 - x0
        tx0, tx1 = int(x0 + 0.20 * bw), int(x1 - 0.20 * bw)
        crop = arr[max(0, ty0):max(1, ty1), max(0, tx0):max(1, tx1)]
        team = classify_team(crop)
        players.append({
            "team": team, "cx": round(float(cx), 1), "cy": round(float(foot), 1),
            "box": [round(float(x0), 1), round(float(y0), 1),
                    round(float(x1), 1), round(float(y1), 1)],
            "score": round(float(sc), 3),
        })

    players = dedupe(players)

    # stable IDs: per team, left-to-right
    prefix = {"yellow": "Y", "dark": "D", "other": "?"}
    counts: dict[str, int] = {}
    for p in sorted(players, key=lambda p: (p["team"], p["cx"])):
        counts[p["team"]] = counts.get(p["team"], 0) + 1
        p["id"] = f"{prefix[p['team']]}{counts[p['team']]}"

    # ── annotate ──
    canvas = base.copy()
    d = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    for p in players:
        col = TEAM_COLORS[p["team"]]
        cx, cy = p["cx"], p["cy"]
        rx = max(11, (p["box"][2] - p["box"][0]) * 0.9)
        ry = rx * 0.55
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=col, width=3)
        label = p["id"]
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx, ly = cx - tw / 2, cy - ry - th - 6
        d.rectangle([lx - 3, ly - 2, lx + tw + 3, ly + th + 4], fill=(0, 0, 0, 170))
        d.text((lx, ly), label, fill=col, font=font)

    if ball:
        bx, by = ball["cx"], ball["cy"]
        d.ellipse([bx - 13, by - 13, bx + 13, by + 13], outline=(255, 255, 255, 255), width=3)
        d.ellipse([bx - 13, by - 13, bx + 13, by + 13], outline=(255, 40, 40, 255), width=1)
        d.text((bx + 15, by - 8), "BALL", fill=(255, 255, 255, 255), font=font)

    out_png = OUT_DIR / "players_annotated.png"
    canvas.save(out_png)
    (OUT_DIR / "players.json").write_text(json.dumps({"players": players, "ball": ball}, indent=2))

    n_y = sum(p["team"] == "yellow" for p in players)
    n_d = sum(p["team"] == "dark" for p in players)
    n_o = sum(p["team"] == "other" for p in players)
    b = f"({ball['cx']},{ball['cy']}) score={ball['score']}" if ball else "NOT FOUND"
    print(f"device={dev.type}  players={len(players)}  "
          f"yellow={n_y} dark={n_d} other={n_o}  ball={b}")
    print(f"wrote {out_png.relative_to(ROOT)} and players.json")


if __name__ == "__main__":
    main()
