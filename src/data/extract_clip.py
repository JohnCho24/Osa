"""Slice a small JSON clip out of a converted GenTac match file (for fast browser dev)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.metrica_to_gentac import validate_payload     # type: ignore


def extract(src: Path, dst: Path, start_frame: int, n_frames: int):
    if not src.exists():
        sys.exit(f"source not found: {src}")
    if n_frames <= 0:
        sys.exit(f"n_frames must be positive (got {n_frames})")
    data = json.load(src.open())
    frames = data["frames"]
    end_frame = start_frame + n_frames
    clip = {k: v for k, v in frames.items() if start_frame <= int(k) < end_frame}
    if not clip:
        sys.exit(f"No frames in range [{start_frame}, {end_frame}) — match has frames "
                 f"{min(int(k) for k in frames)}–{max(int(k) for k in frames)}")
    out = {
        "metadata": {**data["metadata"], "clip_start": start_frame, "clip_end": end_frame, "n_frames": len(clip)},
        "frames": clip,
    }
    # Re-validate the clipped slice — clipping can expose a previously-rare frame whose
    # bad data we never noticed in the aggregate. Warnings only (don't fail dev clips).
    for w in validate_payload(out, strict=False):
        print(f"  ⚠ {w}", file=sys.stderr)

    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, dst.open("w"), separators=(",", ":"))
    print(f"{src.name} [{start_frame}:{end_frame}] → {dst} ({len(clip)} frames, {dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    # 30s window starting ~10 min into match 1 (frame 15000 @ 25fps = 600s)
    extract(
        root / "data" / "processed" / "Sample_Game_1.json",
        root / "data" / "processed" / "Sample_Game_1_clip.json",
        start_frame=15000,
        n_frames=750,
    )
    # Game 2 clip — placeholder "alternative scenario" for compare-mode dev
    extract(
        root / "data" / "processed" / "Sample_Game_2.json",
        root / "data" / "processed" / "Sample_Game_2_clip.json",
        start_frame=15000,
        n_frames=750,
    )
