"""Generate the demo's "alternative play" clip via Higgsfield Cloud.

Takes the decision frame (site/public/demo-poster.jpg), sends it to Higgsfield's
image-to-video model with a tactical-alternative prompt, and saves the result to
site/public/demo-alternative.mp4 — the file the landing-page demo sequencer plays
in its final stage.

Setup:
    1. cp .env.example .env   and fill in HF_API_KEY / HF_API_SECRET
       (from https://cloud.higgsfield.ai/ → API section). .env is gitignored.
    2. In your dashboard, open the API example for the DoP / image-to-video
       model and confirm HF_APPLICATION + the start-image argument key below.
    3. .venv/bin/python3 scripts/higgsfield_alt.py

This call SPENDS Higgsfield credits. Run it deliberately.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START_FRAME = ROOT / "site" / "public" / "demo-poster.jpg"
OUT_FILE = ROOT / "site" / "public" / "demo-alternative.mp4"

# The alternative the model should render from the same decision frame. Keep the
# stadium / kits / camera identical — only the play diverges.
PROMPT = (
    "Same wide tactical broadcast view of the same snowy football pitch, "
    "identical stadium, kits, lighting, snow patches and camera angle, one "
    "continuous shot with no cuts. The alternative play: the yellow player on "
    "the ball plays a long diagonal through-ball into the right channel; a "
    "yellow attacker sprints in behind the defensive line to run onto the pass "
    "while a dark-shirted defender scrambles across to cover. Smooth, realistic "
    "player and ball motion, broadcast quality, no text, no graphics, no arrows."
)


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv(ROOT / ".env")

    # The SDK reads HF_KEY or HF_API_KEY + HF_API_SECRET from the environment.
    if not (os.getenv("HF_KEY") or (os.getenv("HF_API_KEY") and os.getenv("HF_API_SECRET"))):
        sys.exit(
            "Missing credentials. Copy .env.example to .env and set "
            "HF_API_KEY / HF_API_SECRET (from https://cloud.higgsfield.ai/)."
        )
    if not START_FRAME.exists():
        sys.exit(f"Start frame not found: {START_FRAME}")

    application = os.getenv("HF_APPLICATION", "/v1/image2video/dop")
    model = os.getenv("HF_MODEL", "dop-turbo")   # dop-lite | dop-turbo | dop-preview

    import higgsfield_client as hf
    from PIL import Image

    print(f"→ uploading start frame: {START_FRAME.name}")
    start_url = hf.upload_image(Image.open(START_FRAME), format="jpeg")
    print(f"  uploaded: {start_url}")

    # DoP image-to-video body (github.com/higgsfield-ai/higgsfield-js):
    # the server expects a top-level `params` object; input_images are
    # {type:"image_url", image_url:url} entries. Start frame = decision moment.
    arguments = {
        "params": {
            "model": model,
            "prompt": PROMPT,
            "input_images": [{"type": "image_url", "image_url": start_url}],
            # "seed": 12345,   # optional, for reproducibility
        }
    }

    print(f"→ submitting to {application} (model={model}, this spends credits)…")
    result = hf.subscribe(
        application,
        arguments,
        on_queue_update=lambda s: print(f"  status: {s}"),
    )

    # Result shape varies by model; pull the first video-ish URL we can find.
    video_url = _find_video_url(result)
    if not video_url:
        sys.exit(f"No video URL in result. Raw result:\n{result}")

    print(f"→ downloading result → {OUT_FILE.relative_to(ROOT)}")
    urllib.request.urlretrieve(video_url, OUT_FILE)
    print(f"✓ done: {OUT_FILE} ({OUT_FILE.stat().st_size} bytes)")
    print("  Reload the site — stage 4 of the demo now plays the alternative.")
    return 0


def _find_video_url(obj) -> str | None:
    """Walk a nested dict/list result and return the first .mp4/video URL."""
    if isinstance(obj, str):
        return obj if (".mp4" in obj or "video" in obj) and obj.startswith("http") else None
    if isinstance(obj, dict):
        for v in obj.values():
            if (u := _find_video_url(v)):
                return u
    if isinstance(obj, list):
        for v in obj:
            if (u := _find_video_url(v)):
                return u
    return None


if __name__ == "__main__":
    raise SystemExit(main())
