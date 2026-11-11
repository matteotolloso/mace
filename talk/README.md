# RPS @ NeurIPS 2026 oral: slides

Talk for *Multi-Fidelity Transfer Learning Reshapes Uncertainty Quantification in Atomistic Models*,
oral at the Representations for the Physical Sciences (RPS) workshop, NeurIPS 2026 (20 minutes):
13 slides + 7 backup slides. The result slides redraw the paper's Figs. 2–6,
animated, from the same numbers. Their structure follows the PhD-review deck: the title states
the finding, the RQ is asked in plain words, numbered findings (teal = good news, orange =
caveat) appear one per click next to the figure, and a "So:" box closes the slide. Each click
also plays the part of the plot that the finding refers to (a panel appears, a signal is
highlighted, training advances).

## Presenting

Open `index.html` in Chrome or Firefox (Safari should work but was not tested). Everything is local:
the folder works offline and can be copied as a whole.

| Key | Action |
|---|---|
| → / Space / clicker | next step (plot animation or text), then next slide |
| `S` | speaker view: notes, next slide, timer and pacing (20:00 over the main slides) |
| `R` | replay the plot animation of the current slide |
| `F` / `B` / `Esc` | full screen / black screen / overview |

Plot animations start by themselves when a slide opens (the training-dynamics slides
play the epochs as a sweep); each click then shows the next finding and its plot step. `slides.pdf` is the
static fallback: every slide in its final state.

## Folder

| Path | Content |
|---|---|
| `CLAUDE.md` (`AGENTS.md` points to it) | full build and maintenance guide for agents and humans |
| `index.html` | the slides: text, speaker notes, timings (edit directly; no build step) |
| `js/plots.js` | D3 drawing and animation of the paper figures and the three schematics |
| `css/theme.css` | layout and typography |
| `data/paper_figures.{json,js}` | every value plotted on the result slides |
| `assets/` | crops of the paper's Fig. 1 (`make_assets.py`), QR code, appendix figures for the backup slides |
| `lib/` | reveal.js 5.2.1, D3 7.9.0, KaTeX 0.16.11, Inter and Source Serif 4 fonts |
| `scripts/extract_data.py` | regenerates `data/` (mace env, CPU, ~1 min) |
| `scripts/make_assets.py` | regenerates the Fig. 1 crops and the QR code (needs `pymupdf`, `qrcode`) |
| `scripts/export_pdf.py` | regenerates `slides.pdf` (needs Playwright + Chromium) |
| `scripts/screenshots.py` | screenshots of every slide (first and final state) for visual checks |

## Where the numbers come from

`scripts/extract_data.py` imports the paper's own figure code read-only
(`papers/iclr27/figures/common.py`, `epoch_data.py`, `fig5_decomposition.py`,
`fig6_acquisition.py`) and saves exactly what Figs. 2–6 plot: five dataset splits,
training-support filter, no trim, 95% Student-t intervals (geometric for RMSE and ENCE),
same smoothing of the epoch curves. The browser only draws them. Spot checks against
`results/tables/iclr27_main_text_numbers.txt` (ledger entry "2026-09-23 ICLR27 main-text
numbers") match to all printed digits; the numbers quoted in slide text and notes come
from that table.

Differences from the paper figures are layout only (sizes, fonts, legend position); panels
and their arrangement are the paper's. The ranking/calibration,
ensemble and mechanism plots are schematics with synthetic numbers and say so on the slide.
The ensemble formula uses 1/M (as the code); the paper's Fig. 1 artwork shows 1/(M−1) and
its equation part is not used.
