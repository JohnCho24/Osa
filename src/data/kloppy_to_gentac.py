"""Convert ANY kloppy-supported tracking source → GenTac JSON.

This is the provider-agnostic successor to `metrica_to_gentac.py`. kloppy
(https://kloppy.pysport.org) parses Metrica, Sportec/DFL, SkillCorner, PFF,
Tracab, Second Spectrum and Stats Perform into one normalized data model, so a
single emitter here turns all of them into the exact schema the GenTac dataset/
server/renderer already consume — and any future *paid* feed (Sportec license,
SkillCorner full, Opta Vision) drops into the same path with zero new parsing.

kloppy normalization (verified, kloppy 3.18):
  - Default coordinate system = KloppyCoordinateSystem: x,y ∈ [0,1],
    origin = top-left, vertical = TOP_TO_BOTTOM, pitch_length/width in meters.
  - => meters-from-center, y-up (our schema) is the SAME transform we already
    use for Metrica: x_m=(x-0.5)*L, y_m=(0.5-y)*W.

GenTac schema (identical to metrica_to_gentac output):
  {
    "metadata": {"source","game_id","fps","pitch":{x,y,origin,y_axis},
                 "team0":{name,players:[id...]}, "team1":{...},
                 "player_names":{id:full_name}, "n_frames"},
    "frames": {"<frame_id>": {"period", "ball":[x,y]|null,
                              "team0":{id:[x,y]}, "team1":{id:[x,y]}}}
  }

Player keys = shirt number strings (unique within a team, numeric so the model's
`_build_slot_map` shirt-order slotting works exactly as it does for Metrica).
Real names are preserved out-of-band in metadata.player_names for display.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the strict payload validator (sign-flip / parse / drift guards).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.metrica_to_gentac import validate_payload  # type: ignore


def _assert_normalized(cs) -> tuple[float, float]:
    """Fail loud unless coords are kloppy's normalized top-left [0,1] system.

    We convert against a known convention rather than guessing per provider — if a
    future kloppy default ever changes, this raises at *convert* time (cheap to
    fix) instead of silently feeding mirrored/scaled coords into training.
    """
    origin = str(getattr(cs, "origin", "")).lower()
    vert = str(getattr(cs, "vertical_orientation", "")).lower()
    L = float(getattr(cs, "pitch_length", 105.0) or 105.0)
    W = float(getattr(cs, "pitch_width", 68.0) or 68.0)
    if "top-left" not in origin or "top_to_bottom" not in vert:
        raise ValueError(
            f"unexpected kloppy coordinate system (origin={origin!r}, vertical={vert!r}); "
            "expected top-left / TOP_TO_BOTTOM. Load with coordinates=None."
        )
    return L, W


def _to_meters(pt, L: float, W: float):
    if pt is None or pt.x is None or pt.y is None:
        return None
    try:
        x = (float(pt.x) - 0.5) * L
        y = (0.5 - float(pt.y)) * W
    except (TypeError, ValueError):
        return None
    return [round(x, 2), round(y, 2)]


def _full_name(p) -> str:
    name = getattr(p, "name", None)
    if name:
        return str(name)
    parts = [getattr(p, "first_name", None), getattr(p, "last_name", None)]
    joined = " ".join(s for s in parts if s)
    return joined or str(p.player_id)


def _player_key(p, used: set[str]) -> str:
    """Shirt number string (numeric → slot ordering works); de-dup defensively."""
    j = getattr(p, "jersey_no", None)
    key = str(j) if j not in (None, "") else str(p.player_id)
    if key in used:  # shouldn't happen within a team, but never silently merge
        suffix = 2
        while f"{key}_{suffix}" in used:
            suffix += 1
        key = f"{key}_{suffix}"
    used.add(key)
    return key


def dataset_to_payload(ds, game_id: str, source: str) -> dict:
    md = ds.metadata
    L, W = _assert_normalized(md.coordinate_system)
    fps = int(md.frame_rate or 25)

    # Map each Player -> (team0|team1, key); team0 = home, team1 = away.
    home = next((t for t in md.teams if str(t.ground).lower() == "home"), md.teams[0])
    away = next((t for t in md.teams if str(t.ground).lower() == "away"), md.teams[1])
    team_of = {id(home): "team0", id(away): "team1"}

    key_by_player: dict[str, str] = {}
    # Names are keyed per team: shirt numbers collide across teams (both have a #6),
    # so a single flat {key: name} dict would let one team's roster clobber the other's.
    names = {"team0": {}, "team1": {}}
    used_keys = {"team0": set(), "team1": set()}
    members = {"team0": [], "team1": []}
    for team in (home, away):
        tk = team_of[id(team)]
        for p in team.players:
            k = _player_key(p, used_keys[tk])
            key_by_player[p.player_id] = k
            names[tk][k] = _full_name(p)
            members[tk].append(k)

    frames: dict[str, dict] = {}
    for fr in ds.frames:
        team0: dict[str, list] = {}
        team1: dict[str, list] = {}
        for player, pt in fr.players_coordinates.items():
            xy = _to_meters(pt, L, W)
            if xy is None:
                continue
            tk = team_of.get(id(player.team))
            if tk is None:
                tk = "team0" if str(player.team.ground).lower() == "home" else "team1"
            (team0 if tk == "team0" else team1)[key_by_player.get(player.player_id, _full_name(player))] = xy
        period = getattr(getattr(fr, "time", None), "period", None)
        frames[str(fr.frame_id)] = {
            "period": int(period.id) if period is not None else 0,
            "ball": _to_meters(fr.ball_coordinates, L, W),
            "team0": team0,
            "team1": team1,
        }

    payload = {
        "metadata": {
            "source": source,
            "game_id": game_id,
            "fps": fps,
            "pitch": {"x": round(L, 1), "y": round(W, 1), "origin": "center", "y_axis": "up"},
            "team0": {"name": home.name, "players": members["team0"], "player_names": names["team0"]},
            "team1": {"name": away.name, "players": members["team1"], "player_names": names["team1"]},
            "n_frames": len(frames),
        },
        "frames": frames,
    }
    return payload


def write_payload(payload: dict, out_path: Path) -> Path:
    warnings = validate_payload(payload, strict=True)
    for w in warnings:
        print(f"  ⚠ {w}", file=sys.stderr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return out_path


# ── provider loaders ────────────────────────────────────────────────────────
def load_sportec(args):
    from kloppy import sportec
    # only_alive=False keeps every 25 fps frame so training windows stay
    # temporally contiguous (dead-ball frames are still valid positions).
    return sportec.load_tracking(
        meta_data=args.meta, raw_data=args.raw,
        coordinates=None, only_alive=False,
        sample_rate=(1.0 / args.downsample if args.downsample else None),
        limit=args.limit,
    )


def load_skillcorner(args):
    from kloppy import skillcorner
    return skillcorner.load(
        meta_data=args.meta, raw_data=args.raw,
        coordinates=None, include_empty_frames=False,
        sample_rate=(1.0 / args.downsample if args.downsample else None),
        limit=args.limit,
    )


def load_pff(args):
    from kloppy import pff
    return pff.load_tracking(
        meta_data=args.meta, roster_meta_data=args.roster, raw_data=args.raw,
        coordinates=None,
        sample_rate=(1.0 / args.downsample if args.downsample else None),
        limit=args.limit,
    )


def load_metrica_open(args):
    from kloppy import metrica
    return metrica.load_open_data(match_id=args.match_id, coordinates=None, limit=args.limit)


LOADERS = {
    "sportec": load_sportec,
    "skillcorner": load_skillcorner,
    "pff": load_pff,
    "metrica-open": load_metrica_open,
}


def main():
    ap = argparse.ArgumentParser(description="Convert any kloppy tracking source → GenTac JSON")
    ap.add_argument("provider", choices=LOADERS.keys())
    ap.add_argument("--meta", help="match-info / metadata file")
    ap.add_argument("--raw", help="raw positions/tracking file")
    ap.add_argument("--roster", help="(pff) roster metadata file")
    ap.add_argument("--match-id", default="1", help="(metrica-open) match id")
    ap.add_argument("--game-id", required=True, help="output game id / filename stem")
    ap.add_argument("--out", help="output path (default data/processed/<game-id>.json)")
    ap.add_argument("--downsample", type=int, default=0, help="keep 1 of every N frames")
    ap.add_argument("--limit", type=int, default=None, help="cap frames (smoke test)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = Path(args.out) if args.out else root / "data" / "processed" / f"{args.game_id}.json"

    print(f"loading {args.provider} …", file=sys.stderr)
    ds = LOADERS[args.provider](args)
    payload = dataset_to_payload(ds, game_id=args.game_id, source=args.provider)
    write_payload(payload, out)
    md = payload["metadata"]
    print(f"{args.game_id}: {md['n_frames']} frames | {md['team0']['name']} vs {md['team1']['name']} "
          f"→ {out.relative_to(root) if out.is_relative_to(root) else out}")


if __name__ == "__main__":
    main()
