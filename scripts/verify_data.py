"""Read-only verification of converted GenTac match data.

Runs the full check battery over data/processed/*.json and writes a report to
docs/data_verification.md. Exits non-zero if any match fails a hard check, so it
doubles as a CI gate after (re)converting data.

Checks per match:
  1. Schema + integrity   — validate_payload(strict=True)  [src/data/metrica_to_gentac.py]
  2. Coordinate sanity    — pitch 105×68 centre origin; coord ranges in metres
  3. Team split + names   — slotted XI jerseys ∈ official DFL roster (sportec only)
  4. Model-dataset load   — TrajectoryDataset builds windows; shapes/range/mask sane

Usage:  python scripts/verify_data.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.metrica_to_gentac import validate_payload          # noqa: E402
from src.model.config import GenTacConfig                        # noqa: E402
from src.model.dataset import TrajectoryDataset, _build_slot_map  # noqa: E402

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "sportec" / "raw"


def parse_matchinfo_roster(dfl_id: str) -> dict[str, dict[str, str]] | None:
    """{team_name: {shirt: shortname}} from the official DFL matchinfo XML, or None."""
    p = RAW / f"matchinfo_{dfl_id}.xml"
    if not p.exists():
        return None
    xml = p.read_text(encoding="utf-8")
    teams: dict[str, dict[str, str]] = {}
    for tm in re.finditer(r'<Team\b[^>]*\bTeamName="([^"]+)"[^>]*>(.*?)</Team>', xml, re.S):
        name, body, js = tm.group(1), tm.group(2), {}
        for pl in re.finditer(r"<Player\b([^>]*?)/?>", body):
            a = pl.group(1)
            sn = re.search(r'ShirtNumber="(\d+)"', a)
            nm = re.search(r'Shortname="([^"]*)"', a) or re.search(r'LastName="([^"]*)"', a)
            if sn:
                js[sn.group(1)] = nm.group(1) if nm else "?"
        if js:
            teams[name] = js
    return teams


def is_match_file(d: dict) -> bool:
    md = d.get("metadata", {})
    return bool(d.get("frames")) and "team0" in md and "players" in md.get("team0", {})


def coord_stats(frames: dict, px: float, py: float):
    """One pass: ball-missing %, out-of-play %, coord min/max/mean|.| in metres."""
    n = len(frames)
    no_ball = oob = npts = 0
    mn = [1e9, 1e9]
    mx = [-1e9, -1e9]
    s_abs = [0.0, 0.0]
    sx, sy = px / 2 + 1.5, py / 2 + 1.5
    for f in frames.values():
        ents = list(f.get("team0", {}).values()) + list(f.get("team1", {}).values())
        b = f.get("ball")
        if b is None:
            no_ball += 1
        else:
            ents.append(b)
        hit = False
        for xy in ents:
            npts += 1
            for k in (0, 1):
                v = xy[k]
                mn[k] = min(mn[k], v); mx[k] = max(mx[k], v); s_abs[k] += abs(v)
            if abs(xy[0]) > sx or abs(xy[1]) > sy:
                hit = True
        if hit:
            oob += 1
    return {
        "ball_missing_pct": 100 * no_ball / max(n, 1),
        "oob_pct": 100 * oob / max(n, 1),
        "x_range": (mn[0], mx[0]),
        "y_range": (mn[1], mx[1]),
        "mean_abs": (s_abs[0] / max(npts, 1), s_abs[1] / max(npts, 1)),
    }


def check_team_split(d: dict):
    """sportec only: are the slotted XI jerseys in each team's official roster?"""
    md = d["metadata"]
    if md.get("source") != "sportec":
        return None
    dfl_id = md["game_id"].split("_")[-1]
    roster = parse_matchinfo_roster(dfl_id)
    if not roster:
        return ("?", "no matchinfo to cross-check")
    results = []
    for tk in ("team0", "team1"):
        tname = md[tk]["name"]
        slots = _build_slot_map(d["frames"], tk, 11)
        # Match converted team name to a roster team name (exact, else substring).
        rkey = tname if tname in roster else next((k for k in roster if k in tname or tname in k), None)
        rs = roster.get(rkey, {})
        n_in = sum(1 for j in slots if j in rs)
        results.append((tname, n_in, rkey is not None))
    ok = all(n >= 10 and matched for _, n, matched in results)
    summary = " | ".join(f"{t}:{n}/11" for t, n, _ in results)
    return ("PASS" if ok else "FAIL", summary)


def verify_match(path: Path, cfg: GenTacConfig):
    d = json.loads(path.read_text())
    if not is_match_file(d):
        return {"file": path.name, "skip": "not a match file"}
    md = d["metadata"]
    frames = md["n_frames"]
    fps = md.get("fps", 25)
    errors: list[str] = []
    warns: list[str] = []

    # 1. schema + integrity
    try:
        warns += validate_payload(d, strict=True)
    except ValueError as e:
        errors.append(f"validate_payload: {e}".replace("\n", " "))

    # 2. coordinate sanity
    pit = md.get("pitch", {})
    if (pit.get("x"), pit.get("y"), pit.get("origin")) != (105.0, 68.0, "center"):
        errors.append(f"unexpected pitch {pit}")
    cs = coord_stats(d["frames"], pit.get("x", 105.0), pit.get("y", 68.0))
    if cs["mean_abs"][0] < 1.0 and cs["mean_abs"][1] < 1.0:
        errors.append("coords look un-converted (mean ≈ 0)")

    # 3. team split + names (sportec)
    split = check_team_split(d)
    if split and split[0] == "FAIL":
        errors.append(f"team split: {split[1]}")

    # 4. model-dataset load
    windows = 0
    shape_ok = False
    try:
        ds = TrajectoryDataset([path], cfg)
        windows = len(ds)
        if windows:
            s = ds[0]
            shape_ok = (
                tuple(s["history"].shape) == (cfg.history_frames, 23, 2)
                and tuple(s["future"].shape) == (cfg.window_frames, 23, 2)
                and tuple(s["mask"].shape) == (cfg.history_frames + cfg.window_frames, 23)
            )
            rng = float(s["history"].abs().max())
            if not shape_ok:
                errors.append("dataset tensor shapes wrong")
            if rng > 1.5:
                errors.append(f"normalized coord |{rng:.2f}| > 1.5 (scale issue)")
        else:
            errors.append("zero training windows")
    except Exception as e:  # noqa: BLE001
        errors.append(f"dataset load: {type(e).__name__}: {e}")

    return {
        "file": path.name,
        "game_id": md["game_id"],
        "source": md.get("source", "?"),
        "teams": f"{md['team0']['name']} vs {md['team1']['name']}",
        "frames": frames,
        "minutes": frames / fps / 60,
        "fps": fps,
        "ball_missing_pct": cs["ball_missing_pct"],
        "oob_pct": cs["oob_pct"],
        "x_range": cs["x_range"],
        "y_range": cs["y_range"],
        "windows": windows,
        "split": split[1] if split else "n/a (not sportec)",
        "errors": errors,
        "warns": warns,
        "ok": not errors,
    }


def main() -> int:
    cfg = GenTacConfig(smoke=True)
    paths = sorted(p for p in PROC.glob("*.json") if "_clip" not in p.name)
    rows = []
    skipped = []
    for p in paths:
        r = verify_match(p, cfg)
        if r.get("skip"):
            skipped.append(f"{r['file']} ({r['skip']})")
            continue
        rows.append(r)

    # ── console report ──
    print(f"\nVerified {len(rows)} match files in {PROC.relative_to(ROOT)}\n")
    hdr = f"{'match':<22} {'teams':<42} {'frames':>8} {'min':>5} {'oob%':>5} {'noball%':>7} {'windows':>7}  split"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        flag = "" if r["ok"] else "  ✗ FAIL"
        print(f"{r['game_id']:<22} {r['teams'][:42]:<42} {r['frames']:>8,} {r['minutes']:>5.1f} "
              f"{r['oob_pct']:>5.1f} {r['ball_missing_pct']:>7.1f} {r['windows']:>7,}  {r['split']}{flag}")
        for e in r["errors"]:
            print(f"    ✗ {e}")
        for w in r["warns"]:
            print(f"    ⚠ {w}")
    if skipped:
        print("\nskipped:", ", ".join(skipped))

    n_pass = sum(r["ok"] for r in rows)
    verdict = "ALL PASS" if n_pass == len(rows) else f"{len(rows) - n_pass} FAILED"
    print(f"\n=== {n_pass}/{len(rows)} matches PASS — {verdict} ===")

    # ── markdown report ──
    write_report(rows, skipped, n_pass)
    return 0 if n_pass == len(rows) else 1


def write_report(rows, skipped, n_pass):
    lines = [
        "# Data verification report",
        "",
        f"Generated by `scripts/verify_data.py` (read-only). **{n_pass}/{len(rows)} matches PASS.**",
        "",
        "| Match | Teams | Frames | Min | OOB% | NoBall% | Windows | Split check | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        status = "✅ PASS" if r["ok"] else "❌ FAIL"
        lines.append(
            f"| `{r['game_id']}` | {r['teams']} | {r['frames']:,} | {r['minutes']:.1f} | "
            f"{r['oob_pct']:.1f} | {r['ball_missing_pct']:.1f} | {r['windows']:,} | {r['split']} | {status} |"
        )
    lines += [
        "",
        "**Columns:** OOB% = frames with an entity past the touchline (out-of-play, "
        "normal for real tracking); NoBall% = frames missing the ball; Windows = "
        "trainable (history+future) slices at the smoke config; Split check = slotted "
        "XI jerseys found in the official DFL roster (sportec only).",
        "",
        "**Checks run per match:** schema/integrity (`validate_payload` strict), "
        "coordinate sanity (105×68 centre origin, metres), team-split vs official DFL "
        "matchinfo roster, and a `TrajectoryDataset` load (window count + tensor "
        "shape + normalized range + mask).",
    ]
    for r in rows:
        if r["errors"] or r["warns"]:
            lines.append(f"\n**{r['game_id']}** notes:")
            for e in r["errors"]:
                lines.append(f"- ❌ {e}")
            for w in r["warns"]:
                lines.append(f"- ⚠ {w}")
    if skipped:
        lines.append("\n_Skipped (not match files):_ " + ", ".join(skipped))
    (ROOT / "docs" / "data_verification.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
