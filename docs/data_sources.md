# Data sources — training data for GenTac

GenTac trains on **2D positional tracking**: per frame, the `[x, y]` (metres,
pitch-centre origin, y-up) of all 22 players + the ball, at 25 fps. The model
never sees video — only coordinates.

> **The data files are NOT in this repo** (`/data/` is git-ignored — ~650 MB of
> converted JSON, ~2.5 GB of raw XML, and a third-party CC-BY licence). They are
> **fully regenerable** with two commands (see [Reproduce](#reproduce)).

## Pipeline

Everything converts to one **GenTac JSON** schema via a single provider-agnostic path:

```
raw provider data  →  kloppy  →  src/data/kloppy_to_gentac.py  →  GenTac JSON
 (DFL/Sportec, …)   (normalizes      (our emitter)              (data/processed/)
                    every provider)
```

`kloppy` normalizes every provider (Sportec/DFL, SkillCorner, PFF, Metrica,
Tracab, Second Spectrum, Stats Perform) into one model, so a future **paid** feed
(licensed Sportec, SkillCorner full, Opta Vision) drops into the same converter
with zero new parsing.

## What we have (verified)

**7 official Bundesliga matches** (IDSSE open release) + 2 anonymized Metrica
sample games. Full report: [data_verification.md](data_verification.md).

| Match file | Division | Teams | Frames | Min |
|---|---|---|---:|---:|
| `Bundesliga_J03WMX` | Bundesliga | 1. FC Köln vs FC Bayern München | 145,967 | 97.3 |
| `Bundesliga_J03WN1` | Bundesliga | VfL Bochum vs Bayer 04 Leverkusen | 141,561 | 94.4 |
| `Bundesliga2_J03WOH` | 2. Bundesliga | Fortuna Düsseldorf vs Jahn Regensburg | 137,214 | 91.5 |
| `Bundesliga2_J03WOY` | 2. Bundesliga | Fortuna Düsseldorf vs Hansa Rostock | 142,536 | 95.0 |
| `Bundesliga2_J03WPY` | 2. Bundesliga | Fortuna Düsseldorf vs 1. FC Nürnberg | 146,211 | 97.5 |
| `Bundesliga2_J03WQQ` | 2. Bundesliga | Fortuna Düsseldorf vs FC St. Pauli | 142,345 | 94.9 |
| `Bundesliga2_J03WR9` | 2. Bundesliga | Fortuna Düsseldorf vs 1. FC Kaiserslautern | 146,810 | 97.9 |

**Total: 1,002,644 frames** (matches the IDSSE paper exactly → complete set),
~4,000 trainable windows. Official full-pitch optical tracking, 25 fps, real teams
and players. 2.Bundesliga diversity is limited (all 5 are Fortuna Düsseldorf home games).

## Schema / data dictionary

```jsonc
{
  "metadata": {
    "source": "sportec",                 // provider tag
    "game_id": "Bundesliga_J03WMX",
    "fps": 25,
    "pitch": {"x":105.0, "y":68.0, "origin":"center", "y_axis":"up"},
    "team0": {                            // = home
      "name": "1. FC Köln",
      "players": ["2","4","6", ...],      // shirt-number ids present in the match
      "player_names": {"2":"B. Schmitz", ...}   // per-team (numbers collide across teams)
    },
    "team1": { ... },                    // = away
    "n_frames": 145967
  },
  "frames": {
    "<frame_id>": {                      // original provider frame number (string key)
      "period": 1,                       // 1 or 2
      "ball":  [x, y],                   // metres, centre origin; null if not tracked
      "team0": {"<shirt>": [x, y], ...}, // x∈[-52.5,52.5], y∈[-34,34] (+ small out-of-play)
      "team1": {"<shirt>": [x, y], ...}
    }
  }
}
```

The model dataset ([src/model/dataset.py](../src/model/dataset.py)) collapses each
match into a fixed 11-slot-per-team roster (the most-present players) and
normalizes coords to `[-1, 1]` by dividing by half-pitch.

## Reproduce

```bash
# 1. download (the 2 first-division Bundesliga matches, ~0.8 GB)
python scripts/fetch_sportec.py --div 1        # --div 2 → the 5 2.BL; --div all → all 7 (~2.6 GB)

# 2. convert each match
python -m src.data.kloppy_to_gentac sportec \
    --meta data/sportec/raw/matchinfo_J03WMX.xml \
    --raw  data/sportec/raw/positions_J03WMX.xml \
    --game-id Bundesliga_J03WMX

# 3. verify everything (writes docs/data_verification.md, exits non-zero on failure)
python scripts/verify_data.py
```

## Verification

`scripts/verify_data.py` runs, per match: schema/integrity
(`validate_payload` strict — gross scale/axis corruption, un-converted coords,
out-of-play %, ball-missing %), coordinate sanity (105×68 centre origin, metres),
**team-split cross-check vs the official DFL matchinfo roster** (slotted XI jerseys
must appear in the real roster — guards against swapped/merged teams), and a
`TrajectoryDataset` load (window count + tensor shape + normalized range + mask).
Current status: **9/9 PASS**.

## Provenance & licence (IDSSE)

The Bundesliga data is the **IDSSE** open release of official DFL position + event
data, used here under **CC-BY 4.0** for research. Attribution:

> Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of
> spatiotemporal and event data in elite soccer.* Scientific Data.
> figshare DOI [10.6084/m9.figshare.28196177](https://doi.org/10.6084/m9.figshare.28196177).
> Data © DFL Deutsche Fußball Liga / Sportec Solutions, released CC-BY 4.0.

> ⚠️ **Commercial caveat:** this open release is for **research / open science**,
> **not** a commercial data licence. It de-risks the tech and is fine for training
> and demos; a shipping product still needs a real Sportec deal (roadmap M5).

## More free data (same converter, when wanted)

- **PFF FC 2022 World Cup** — all 64 games, broadcast tracking + events, free on
  request. `python -m src.data.kloppy_to_gentac pff --meta <metadata> --roster
  <roster> --raw <tracking> --game-id WC_<id>`. (Broadcast = players tracked only
  while on camera → gaps; fine with the dataset's per-frame masking.)
- **SkillCorner open** — 10 A-League 2024/25 matches (Git LFS).
  `kloppy_to_gentac.py skillcorner --meta <match.json> --raw <tracking.jsonl>`.
