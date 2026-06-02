"""Convert Metrica Sports tracking CSV → GenTac JSON.

Metrica format:
  - Coords normalized [0, 1], origin top-left
  - 25 FPS, pitch 105 x 68 m
  - One CSV per team; 3 header rows (team / jersey / "PlayerN x, PlayerN y")
  - NaN = player not on pitch

GenTac format (matches paper § Supplementary 12.1.2):
  {
    "<frame_id>": {
      "ball":  [x_m, y_m],
      "team0": {"Player1": [x_m, y_m], ...},
      "team1": {"PlayerN": [x_m, y_m], ...}
    },
    ...
  }
Coords in meters, origin at pitch center, x∈[-52.5, 52.5], y∈[-34, 34], y positive "up".
"""

import csv
import json
import sys
from pathlib import Path

PITCH_X = 105.0
PITCH_Y = 68.0


def metrica_to_meters(x_norm: str, y_norm: str):
    """Map Metrica [0,1] top-left coords → meters from center, y-up."""
    if x_norm in ("NaN", "") or y_norm in ("NaN", ""):
        return None
    x = (float(x_norm) - 0.5) * PITCH_X
    y = (0.5 - float(y_norm)) * PITCH_Y
    return [round(x, 2), round(y, 2)]


def parse_tracking_csv(path: Path):
    """Return (player_ids, rows). rows = list of (period, frame, time, {player_id: [x,y] or None}, ball or None)."""
    with path.open() as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Row 2 (0-indexed: rows[2]) has the column names: Period, Frame, Time[s], PlayerN, '', PlayerN, '', ..., Ball, ''
    header = rows[2]
    # Each player and the ball occupy 2 columns (x, y). Build (name, x_col, y_col) tuples.
    entities = []
    i = 3  # first data column
    while i < len(header):
        name = header[i].strip()
        if not name:
            i += 1
            continue
        entities.append((name, i, i + 1))
        i += 2

    player_ids = [n for (n, _, _) in entities if n != "Ball"]

    out = []
    for row in rows[3:]:
        if not row or not row[0].strip():
            continue
        period = int(row[0])
        frame = int(row[1])
        time_s = float(row[2])
        coords = {}
        ball = None
        for name, xc, yc in entities:
            xy = metrica_to_meters(row[xc], row[yc]) if xc < len(row) and yc < len(row) else None
            if name == "Ball":
                ball = xy
            else:
                coords[name] = xy
        out.append((period, frame, time_s, coords, ball))
    return player_ids, out


def validate_payload(payload: dict, strict: bool = True) -> list[str]:
    """Return a list of validation warnings; raise ValueError on hard violations if strict.

    Catches the kind of silent corruption that would propagate downstream and break the
    model with a confusing error 200 ms before a deadline:
      • coords outside [-pitch/2, pitch/2] (sign-flip bugs, wrong axis convention)
      • frames with zero players on either side (parse failure rather than 'subs')
      • inconsistent player ID sets across frames (frame join bug)
      • ball missing in > 20% of frames (ball column drift)
    """
    md = payload.get("metadata", {})
    frames = payload.get("frames", {})
    warnings: list[str] = []
    errors: list[str] = []
    if not frames:
        errors.append("zero frames in payload")
    px = md.get("pitch", {}).get("x", 105.0)
    py = md.get("pitch", {}).get("y", 68.0)
    pad = 1.5                           # play happens within ~1.5 m of the lines
    soft_x, soft_y = px / 2 + pad, py / 2 + pad
    # "Insane" = clearly a scale / axis / unit bug, not a ball in the stands. Real
    # tracking lets the ball/keepers a few metres past the lines (throw-ins, behind
    # the goal); a full pitch-length past centre is corruption.
    insane_x, insane_y = px, py

    n_no_ball = 0
    n_empty_team0 = 0
    n_empty_team1 = 0
    n_oob_frames = 0                     # frames with any entity past the touchline
    sum_abs = [0.0, 0.0]
    n_pts = 0
    prev_ks0 = None
    prev_ks1 = None
    drift = 0                           # roster-change transitions (subs / tracking dropouts)

    for fid, f in frames.items():
        b = f.get("ball")
        if b is None:
            n_no_ball += 1
        t0, t1 = f.get("team0", {}), f.get("team1", {})
        if not t0:
            n_empty_team0 += 1
        if not t1:
            n_empty_team1 += 1
        # NB: iterate the teams separately — a dict merge {**t0, **t1} would dedupe
        # shared keys, and real rosters key players by shirt number (both teams have
        # a #6), silently dropping half the players from the bounds/mean checks.
        entities = list(t0.items()) + list(t1.items())
        if b is not None:
            entities.append(("ball", b))
        oob = False
        for pid, xy in entities:
            sum_abs[0] += abs(xy[0])
            sum_abs[1] += abs(xy[1])
            n_pts += 1
            if abs(xy[0]) > soft_x or abs(xy[1]) > soft_y:
                oob = True
            if abs(xy[0]) > insane_x or abs(xy[1]) > insane_y:
                errors.append(f"frame {fid}: {pid} {xy} grossly off-pitch (>±{insane_x:g},{insane_y:g}) — scale/axis bug")
        if oob:
            n_oob_frames += 1
        # Roster changes are real (subs) but should be a handful of transitions, not a
        # per-frame flicker (which would mean a parse/join bug).
        ks0, ks1 = frozenset(t0.keys()), frozenset(t1.keys())
        if prev_ks0 is not None and ks0 and ks0 != prev_ks0:
            drift += 1
        if prev_ks1 is not None and ks1 and ks1 != prev_ks1:
            drift += 1
        prev_ks0 = ks0 or prev_ks0
        prev_ks1 = ks1 or prev_ks1

    n = len(frames)
    # Un-converted coords (left normalized [0,1]) slip past the max-bounds because
    # they're too SMALL — catch them by their near-origin mean instead.
    if n_pts and sum_abs[0] / n_pts < 1.0 and sum_abs[1] / n_pts < 1.0:
        errors.append(f"mean |coord| ({sum_abs[0]/n_pts:.2f},{sum_abs[1]/n_pts:.2f}) ≈ 0 — coords look un-converted (normalized, not metres)")
    # A little out-of-play is expected; a high fraction means a systematic axis/scale issue.
    oob_frac = n_oob_frames / max(n, 1)
    if oob_frac > 0.40:
        errors.append(f"{oob_frac:.1%} of frames have entities past the touchline (>40% ⇒ systematic axis/scale issue, not out-of-play)")
    elif oob_frac > 0.03:
        warnings.append(f"{oob_frac:.1%} of frames have an entity past the touchline (out-of-play; normal for real tracking)")
    if n_no_ball / max(n, 1) > 0.2:
        warnings.append(f"ball missing in {n_no_ball/n:.1%} of frames (threshold 20%)")
    if n_empty_team0 / max(n, 1) > 0.05:
        warnings.append(f"team0 empty in {n_empty_team0/n:.1%} of frames")
    if n_empty_team1 / max(n, 1) > 0.05:
        warnings.append(f"team1 empty in {n_empty_team1/n:.1%} of frames")
    # 11 starters + up to ~5 subs/team → expect ≲20 roster transitions for a full game.
    if drift > 40:
        warnings.append(f"player roster changes {drift}× (>40 ⇒ likely tracking flicker / join bug)")

    if errors and strict:
        msgs = "\n  ".join(errors[:10])
        raise ValueError(f"validation failed ({len(errors)} errors):\n  {msgs}")
    return warnings + errors


def convert_match(match_dir: Path, out_path: Path):
    game_id = match_dir.name  # e.g. "Sample_Game_1"
    home_csv = match_dir / f"{game_id}_RawTrackingData_Home_Team.csv"
    away_csv = match_dir / f"{game_id}_RawTrackingData_Away_Team.csv"

    home_players, home_rows = parse_tracking_csv(home_csv)
    away_players, away_rows = parse_tracking_csv(away_csv)

    # Build frame index for fast join
    home_by_frame = {(p, f): (t, c, b) for (p, f, t, c, b) in home_rows}
    away_by_frame = {(p, f): (t, c, b) for (p, f, t, c, b) in away_rows}

    all_keys = sorted(set(home_by_frame) | set(away_by_frame))

    frames = {}
    for key in all_keys:
        period, frame = key
        h = home_by_frame.get(key)
        a = away_by_frame.get(key)
        # Prefer ball from whichever side has it (Metrica duplicates ball across both files)
        ball = (h[2] if h else None) or (a[2] if a else None)
        team0 = {pid: xy for pid, xy in (h[1] if h else {}).items() if xy is not None}
        team1 = {pid: xy for pid, xy in (a[1] if a else {}).items() if xy is not None}
        frames[str(frame)] = {
            "period": period,
            "ball": ball,
            "team0": team0,
            "team1": team1,
        }

    payload = {
        "metadata": {
            "source": "metrica",
            "game_id": game_id,
            "fps": 25,
            "pitch": {"x": PITCH_X, "y": PITCH_Y, "origin": "center", "y_axis": "up"},
            "team0": {"name": "Home", "players": home_players},
            "team1": {"name": "Away", "players": away_players},
            "n_frames": len(frames),
        },
        "frames": frames,
    }

    warnings = validate_payload(payload, strict=True)
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return len(frames), out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    src_root = root / "data" / "metrica" / "data"
    dst_root = root / "data" / "processed"

    matches = sorted(p for p in src_root.iterdir() if p.is_dir() and (p / f"{p.name}_RawTrackingData_Home_Team.csv").exists())
    if not matches:
        print("No CSV-format matches found.", file=sys.stderr)
        sys.exit(1)

    for m in matches:
        out = dst_root / f"{m.name}.json"
        n, p = convert_match(m, out)
        print(f"{m.name}: {n} frames → {p.relative_to(root)}")
