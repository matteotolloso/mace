#!/usr/bin/env python3
"""Render every slide of talk/index.html in its first and final state, for visual checks.

    PLAYWRIGHT_BROWSERS_PATH=<browsers> <venv>/bin/python talk/scripts/screenshots.py [OUT] [SLIDE ...]

OUT defaults to /tmp/talk-shots. SLIDE numbers are 1-based (default: all). Writes
sNNa.png (just after the slide opens: entry animation partly played) and sNNz.png (all
fragments shown, animations finished), plus contact sheets sheet-K.png (left: first state,
right: final state, four slides per sheet) that are quick to review. Prints any browser
console errors. Needs Playwright + Chromium and Pillow (not in the mace env).
"""
import asyncio
import sys
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

TALK = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].isdigit() else Path("/tmp/talk-shots")
ONLY = [int(a) for a in sys.argv[1:] if a.isdigit()]


async def shoot():
    OUT.mkdir(parents=True, exist_ok=True)
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1600, "height": 900})
        page.on("console", lambda m: errors.append(m.text) if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto((TALK / "index.html").as_uri())
        await page.wait_for_timeout(1500)
        total = await page.evaluate("Reveal.getTotalSlides()")
        slides = ONLY or list(range(1, total + 1))
        for n in slides:
            await page.evaluate(f"Reveal.slide({n - 1}, 0, -1)")
            await page.wait_for_timeout(3000)
            await page.screenshot(path=str(OUT / f"s{n:02d}a.png"))
            await page.evaluate(f"Reveal.slide({n - 1}, 0, 999)")
            await page.wait_for_timeout(5500)  # longest jump animation: epoch sweeps (~5 s)
            await page.screenshot(path=str(OUT / f"s{n:02d}z.png"))
        await browser.close()
    return slides, errors


def sheets(slides):
    w, h = 800, 450
    for k in range(0, len(slides), 4):
        group = slides[k:k + 4]
        sheet = Image.new("RGB", (2 * w + 10, len(group) * (h + 10)), "#888")
        for r, n in enumerate(group):
            for c, s in enumerate("az"):
                im = Image.open(OUT / f"s{n:02d}{s}.png").convert("RGB").resize((w, h), Image.LANCZOS)
                sheet.paste(im, (c * (w + 10), r * (h + 10)))
        sheet.save(OUT / f"sheet-{k // 4 + 1}.png")


if __name__ == "__main__":
    shown, errs = asyncio.run(shoot())
    sheets(shown)
    print(f"wrote {OUT} ({len(shown)} slides); console errors: {errs or 'none'}")
