"""Download the IDSSE open Bundesliga tracking dataset from figshare.

IDSSE = "An integrated dataset of spatiotemporal and event data in elite soccer"
(German Sport University Cologne + DFL), the only FREE source of *official* DFL
position data. CC-BY 4.0. figshare DOI 10.6084/m9.figshare.28196177.

7 matches: 2 Bundesliga (1st div, DFL-COM-000001) + 5 2.Bundesliga (DFL-COM-000002).
Each match = 3 XML files: matchinformation, events, positions (~400 MB).

Usage:
    python scripts/fetch_sportec.py --div 1            # the 2 Bundesliga matches (default)
    python scripts/fetch_sportec.py --div all          # all 7 matches (~2.6 GB)
    python scripts/fetch_sportec.py --match J03WMX      # one match by id
Then convert each with:
    python -m src.data.kloppy_to_gentac sportec \
        --meta data/sportec/raw/matchinfo_<ID>.xml \
        --raw  data/sportec/raw/positions_<ID>.xml \
        --game-id Bundesliga_<ID>
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "sportec" / "raw"
API = "https://api.figshare.com/v2/articles/28196177"

# figshare file-id map (article version 1). matchinfo + positions per match; events
# are small and pulled too. Keyed by DFL match id; div tags which competition.
FILES = {
    # div 1 — Bundesliga
    "J03WMX": {"div": 1, "matchinfo": 51643475, "events": 51643493, "positions": 51643514},
    "J03WN1": {"div": 1, "matchinfo": 51643472, "events": 51643496, "positions": 51643517},
    # div 2 — 2.Bundesliga
    "J03WOH": {"div": 2, "matchinfo": 51643478, "events": 51643499, "positions": 51643520},
    "J03WOY": {"div": 2, "matchinfo": 51643481, "events": 51643502, "positions": 51643523},
    "J03WPY": {"div": 2, "matchinfo": 51643487, "events": 51643505, "positions": 51643526},
    "J03WQQ": {"div": 2, "matchinfo": 51643484, "events": 51643508, "positions": 51643529},
    "J03WR9": {"div": 2, "matchinfo": 51643490, "events": 51643511, "positions": 51643532},
}


def _download(file_id: int, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ✓ {dest.name} (cached, {dest.stat().st_size/1e6:.0f} MB)")
        return
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    print(f"  ↓ {dest.name} …", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"    {dest.stat().st_size/1e6:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--div", choices=["1", "2", "all"], default="1")
    ap.add_argument("--match", help="single DFL match id, e.g. J03WMX")
    ap.add_argument("--skip-positions", action="store_true", help="metadata + events only (small)")
    args = ap.parse_args()

    if args.match:
        ids = [args.match]
    elif args.div == "all":
        ids = list(FILES)
    else:
        ids = [m for m, v in FILES.items() if v["div"] == int(args.div)]

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"fetching {len(ids)} match(es) → {RAW.relative_to(ROOT)}")
    for mid in ids:
        f = FILES.get(mid)
        if not f:
            print(f"unknown match id {mid}", file=sys.stderr); continue
        print(f"\n{mid} (div {f['div']}):")
        _download(f["matchinfo"], RAW / f"matchinfo_{mid}.xml")
        _download(f["events"], RAW / f"events_{mid}.xml")
        if not args.skip_positions:
            _download(f["positions"], RAW / f"positions_{mid}.xml")
    print("\ndone. convert with: python -m src.data.kloppy_to_gentac sportec --meta ... --raw ... --game-id Bundesliga_<ID>")


if __name__ == "__main__":
    main()
