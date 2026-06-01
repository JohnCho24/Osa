"""Headless screenshots of each demo-video stage, to verify FX + arrows render.

Drives the live site (localhost:8080), forces each data-stage on the demo
player, and screenshots the player element. Uses the decision-frame poster as
the backdrop for the rewind/arrows/alt stages (their video frames aren't loaded
headlessly) so the arrows and rewind sign are shown over the real pitch.
"""
from __future__ import annotations

import os
from playwright.sync_api import sync_playwright

URL = "http://localhost:8080/"
OUT = "analysis/veo/shots"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800}, device_scale_factor=2)
        page.goto(URL, wait_until="load")

        player = page.locator("#demo-stage")
        player.scroll_into_view_if_needed()

        # Give rewind/alt a poster so stages show the decision frame, and hide
        # the play button so it doesn't cover the screenshots.
        page.evaluate(
            """() => {
                for (const id of ['demo-rewind', 'demo-alt']) {
                    const v = document.getElementById(id);
                    if (v) v.setAttribute('poster', '/public/demo-poster.jpg');
                }
                const pb = document.getElementById('demo-play');
                if (pb) pb.style.display = 'none';
            }"""
        )
        page.wait_for_timeout(400)

        def shot(stage: str, name: str, wait: int = 400, retrigger: bool = False) -> None:
            if retrigger:
                page.evaluate("() => document.getElementById('demo-stage').setAttribute('data-stage','original')")
                page.wait_for_timeout(120)
            page.evaluate(f"() => document.getElementById('demo-stage').setAttribute('data-stage','{stage}')")
            page.wait_for_timeout(wait)
            player.screenshot(path=f"{OUT}/{name}.png")
            print(f"  shot {name} (stage={stage})")

        shot("original", "1_original", 500)
        shot("rewind", "2_rewind", 700)              # ⏪ REW badge + scanlines
        shot("arrows", "3_arrows", 2300, retrigger=True)  # wait out the draw animation
        shot("alternative", "4_alternative", 500)    # no clip yet → decision frame held

        browser.close()
    print(f"wrote 4 stage screenshots to {OUT}/")


if __name__ == "__main__":
    main()
