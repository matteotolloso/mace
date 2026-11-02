#!/usr/bin/env python3
"""Export talk/index.html to talk/slides.pdf: one page per slide, every animation in its final state.

Needs Playwright with Chromium (not in the mace env), e.g.
    PLAYWRIGHT_BROWSERS_PATH=<browsers> <venv>/bin/python talk/scripts/export_pdf.py
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

TALK = Path(__file__).resolve().parents[1]


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto((TALK / "index.html").as_uri() + "?print-pdf")
        await page.wait_for_timeout(3000)
        await page.pdf(path=str(TALK / "slides.pdf"), width="1600px", height="900px", print_background=True)
        await browser.close()
    print("wrote", TALK / "slides.pdf")


asyncio.run(main())
