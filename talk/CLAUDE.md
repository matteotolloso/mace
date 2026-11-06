# Agent guide: ICLR 2027 oral talk

Self-contained folder for the 20-minute oral of *Multi-Fidelity Training Reshapes Uncertainty
Decomposition in Atomistic Models* (Tolloso, Bacciu, Bonati). Everything needed to present,
edit and rebuild the talk is here; the only inputs from outside are the paper's figure code
(read-only, for `data/`) and the paper's Fig. 1 PDF (for `assets/`). The parent
`../CLAUDE.md` rules still apply (ledger, no claims without provenance, do not touch
`papers/`). `README.md` is the short user-facing version of this file.

## What it is

- reveal.js deck (`index.html`), 1600 x 900, 13 main slides + 7 backup, timed to 20:00.
- Result slides redraw paper Figs. 2–6 live with D3 (`js/plots.js`) from exported numbers
  (`data/paper_figures.js`), animated step by step.
- Structure and wording follow the author's PhD-review deck (`PhD_Year2_Tolloso_v5`, pages
  10–17): finding as title, RQ as plain-language question, RQ tracker, numbered findings
  (teal = good news, orange = caveat), "So:" box, encoding footer. Keep that pattern.
- No build step: edit `index.html` / `js/plots.js` / `css/theme.css` and reload the browser.
  It runs from `file://` (data are a `.js` file, not fetched), offline, any modern browser.

## Map

| Path | Role |
|---|---|
| `index.html` | slides: text, speaker notes (`<aside class="notes">`), `data-timing`, fragments, figure slots |
| `js/plots.js` | `FIGS.<name>` = one animated figure: `{w, h, build(svg) -> [step0, step1, ...]}`; `Figure` controller |
| `css/theme.css` | typography, palette, layout classes (`.result`, `.findings`, `.fd`, `.so`, `.foot`, `.rq-track`, `.card`, `.unc-card`) |
| `data/paper_figures.{json,js}` | every plotted value (JSON readable copy; JS = `window.PAPER_DATA`) |
| `assets/mve_box.png`, `assets/training.png` | crops of the paper's Fig. 1 |
| `assets/qr_code.svg` | QR to https://github.com/matteotolloso/mace |
| `assets/paper_figs/*.png` | appendix figures for backup slides, copied from `../papers/iclr27/figures/out/` |
| `lib/` | vendored: reveal.js 5.2.1 (+ notes, zoom, math plugins), D3 7.9.0, KaTeX 0.16.11, Inter + Source Serif 4 fonts |
| `scripts/extract_data.py` | regenerates `data/` (mace env) |
| `scripts/make_assets.py` | regenerates Fig. 1 crops + QR (tool env) |
| `scripts/export_pdf.py` | writes `slides.pdf`, every slide in final state (tool env) |
| `scripts/screenshots.py` | first/final screenshots of every slide + contact sheets (tool env) |
| `slides.pdf` | static fallback for uploads; regenerate after every change |

## Environments

- **mace env** (`conda activate mace`): only for `scripts/extract_data.py`; it imports
  `uq` and the paper figure modules. CPU, ~1 min.
- **tool env**: a separate venv kept outside the repo, with `pymupdf qrcode pillow playwright`
  and Playwright's Chromium. Never install these into the mace env. Create it on a disk
  with space (the root disk of this machine is nearly full; put the venv and the browsers
  on `/raid`):
  ```bash
  python -m venv <tools>/venv && <tools>/venv/bin/pip install pymupdf qrcode pillow playwright
  PLAYWRIGHT_BROWSERS_PATH=<tools>/browsers <tools>/venv/bin/playwright install chromium
  ```
  Always export `PLAYWRIGHT_BROWSERS_PATH=<tools>/browsers` when running the Playwright
  scripts. WebKit cannot run on this machine (missing system libraries, no root), so
  Safari behaviour is untested. There is no Node.js; nothing here needs it.

## Rebuild from scratch

```bash
conda activate mace
python talk/scripts/extract_data.py                       # data/  (needs the evaluation caches, see below)
<tools>/venv/bin/python talk/scripts/make_assets.py       # assets/mve_box.png, training.png, qr_code.svg
cp papers/iclr27/figures/out/{figA_reliability_energy,figA_recalibration,figA_eu_au_ratio,figA_acquisition_id,figA_train_curves}.png talk/assets/paper_figs/
PLAYWRIGHT_BROWSERS_PATH=<tools>/browsers <tools>/venv/bin/python talk/scripts/export_pdf.py
PLAYWRIGHT_BROWSERS_PATH=<tools>/browsers <tools>/venv/bin/python talk/scripts/screenshots.py /tmp/talk-shots
```

Library downloads (only if `lib/` is lost), from jsdelivr:
`reveal.js@5.2.1/dist/{reveal.css,reveal.js,reset.css}` and `plugin/{notes,math,zoom}/*.js`;
`d3@7.9.0/dist/d3.min.js`; `katex@0.16.11/dist/{katex.min.js,katex.min.css,contrib/auto-render.min.js}`
plus every `fonts/*.woff2` referenced in `katex.min.css`;
`@fontsource/inter@5.1.0/files/inter-latin-{300,400,500,600,700,800}-normal.woff2` and
`-400-italic`, `-500-italic`; `@fontsource/source-serif-4@5.1.0/files/source-serif-4-latin-{600,700}-normal.woff2`
(saved as `lib/fonts/inter-<w>.woff2`, `inter-<w>i.woff2`, `source-serif-<w>.woff2`).

## Data provenance (`scripts/extract_data.py`)

- Imports read-only from `../papers/iclr27/figures/`: `common.py` (final metrics, intervals),
  `epoch_data.py` (per-epoch series), `fig5_decomposition.phased_ema`,
  `fig6_acquisition.{gain_rows, selected_quantiles}`. Same statistics as the paper: five
  splits, ten members, training-support filter, no trim, 95% Student-t (df = 4), geometric
  means and log-space intervals for RMSE and ENCE, EMA smoothing (restarted at epoch 300 for
  Fig. 4). Do not reimplement statistics in JS; the browser only draws.
- Inputs: per-split caches under `experiments/<id>/evaluation/split_<s>/` (reliability raw
  CSVs, epoch-quality CSVs, `epoch_predictions_cc_test_ood.npz`) and
  `runs/active-learning/ani1x-energy/` (Fig. 6). If one is missing, the paper figures fail
  the same way; report the gap, never fill it.
- Output keys: `final[test][protocol] = {RMSE: [c, lo, hi], AUSE: {AU|EU|TU: [c, lo, hi]}, ENCE: {...}}`;
  `decomposition[system|energy][train|id|ood][AUSE|ENCE][signal] = {epoch, mean, lo, hi}` (epochs 0–395, fine-tuning ≥ 300);
  `hf_dynamics[kind][protocol][metric][signal]` (HF-stage epochs); `acquisition.gains[protocol][signal] = {mean, lo, hi, splits}`,
  `acquisition.quantiles = {bins, random, <protocol>: {signal: fractions}}`; `meta` (commit, splits, filter).
- Verified on 2026-10-28 against `results/tables/iclr27_main_text_numbers.txt` (ledger entry
  "2026-09-23 ICLR27 main-text numbers"): identical to all printed digits. After re-running,
  spot-check again (e.g. Energy-OOD LF→HF TU ENCE 2.969 [2.079, 4.240]; System-ID HF-only
  RMSE 35.899; LF→HF EU acquisition gain 2.497 [1.889, 3.106]).

## Numbers quoted in slide text and notes

Only these, all from the ledger table above; any new number needs a ledger source.

| Quote | Source line(s) in `iclr27_main_text_numbers.txt` |
|---|---|
| 50k DFT / 5k CC training configurations per split | paper Sec. 3 (data) |
| TU ENCE ID→OOD ≈4× (HF-only), ≈7× (LF→HF); plot labels ×3.7 / ×7.0 computed from `data/` | Energy-ID/OOD TU ENCE: 0.444→1.637, 0.427→2.969 |
| AU ENCE ≈80 / 200 after 100 fine-tuning epochs vs ≈14 / 70 after 300 from scratch | HF-stage dynamics: 81.909, 207.071 vs 13.640, 72.040 |
| AU ~100× worse (error bars far too small) | LF→HF fine-tuning AU ENCE, e.g. id 0.210→67.2, ood 3.4→207 (paper: "two orders of magnitude") |
| Acquisition gain 2.3–2.5 vs 0.8–1.6 meV/atom | AU/EU/TU gains 2.316/2.497/2.347 vs 0.807/1.603/1.140 |
| RMSE before acquisition ≈59 vs ≈75 meV/atom | rmse_before 58.940 vs 75.485 |

Wording deliberately differs from the PhD deck where the data disagree: "Pretraining gives a
larger gain" (not "doubles": EU ratio is 1.6×); "ranking settles during training" (not
"within a few dozen epochs": HF-only levels off later).

## How animation works

- Figure slot: `<div class="d3fig" data-fig="NAME" style="width: ...">`. `Figure.goto(n, anim)`
  rebuilds the SVG and replays steps 0..n instantly, or animates only step n when advancing
  by one. Step 0 plays when the slide opens.
- Step k plays when the k-th `<span class="fragment step" data-fragment-index="i">` of the
  slide becomes visible (count of visible `.step` spans; their index only sets the click).
  Put the step span and the finding it illustrates on the same `data-fragment-index`.
  Fragments that are not `.step` (findings, So box) never trigger plot steps.
- Keys: `S` speaker view (notes + pacing), `R` replay current plot animation, `F` full screen.
- Print/PDF (`?print-pdf`): every figure is drawn in its final state (`Figure.last()`).
- Epoch sweeps clip each panel to the current epoch (`sweepPanel`/`sweeper`) and show a live
  "epoch N" label; focus steps dim other panels (`DIM` = 0.14) or other signals.

### Figure catalogue (`js/plots.js`)

| Name | Paper | viewBox | Steps |
|---|---|---|---|
| `final` | Fig. 2 | 900 x 640 | 0 RMSE · 1 AUSE · 2 EU only (others dimmed) · 3 ENCE (+ dashed box on Energy-OOD TU) |
| `rankcal` | Fig. 3 | 900 x 620 | 0 system ID points · 1 system ID→OOD travel · 2 energy ID + travel · 3 TU only + ×ratio labels |
| `decomposition` | Fig. 4 (3 x 2, paper layout) | 900 x 760 | 0 pretraining sweep · 1 fine-tuning sweep · 2 focus a,b · 3 c,e · 4 d,f · 5 d,f AU · 6 d,f EU · 7 all |
| `hfstage` | Fig. 5 | 900 x 700 | 0 ranking row sweep · 1 calibration row sweep |
| `acquisition` | Fig. 6 | 900 x 620 | 0 HF-only gains · 1 LF→HF gains · 2 histograms + legend |
| `metrics` | schematic | 1460 x 470 | 0 sparsification + AUSE area · 1 reliability diagram |
| `ensemble` | schematic | 920 x 470 | 0 members · 1 EU · 2 AU · 3 TU mixture |
| `mechanism` | schematic | 860 x 620 | 0 train error + AU · 1 unseen error · 2 gap "error bars too small" |

Schematics use synthetic, seeded numbers and must stay labelled as illustration/schematic on
the slide. Plot styling follows `../papers/iclr27/figures/common.py`: protocol colours
HF-only `#009099`, LF→HF `#c0443d`; signals AU circle/dashed, EU square/dotted, TU
diamond/solid; split colours system `#6f4e9c`, energy `#a6761d`; AU/EU faded (0.35) in
calibration panels; RMSE and ENCE on log axes; plot fonts 19–21 px in a 900-unit viewBox.

### Slide catalogue (`index.html`)

| # | Title | Timing s | Figure / clicks |
|---|---|---|---|
| 1 | Title (dark) | 20 | — |
| 2 | Why uncertainty, and why multi-fidelity? | 85 | bars animate on entry; click: open question |
| 3 | Uncertainty has two jobs | 95 | `metrics`; click 1: ENCE |
| 4 | MACE models that predict their own error bars | 150 | `ensemble`; clicks 1 EU card, 2 AU card, 3 TU equation |
| 5 | Experimental setup | 90 | cards + RQ boxes, no clicks |
| 6 | RQ1 Pretraining improves ranking, not always calibration | 130 | `final`; clicks 1–3 findings 2–4 + steps, 4 So |
| 7 | RQ2 Energy shift decouples ranking from calibration | 120 | `rankcal`; clicks 1–3 findings + steps, 4 finding 4, 5 So |
| 8 | RQ3 Calibration is lost during high-fidelity training | 140 | `decomposition`; click 1 fine-tuning, 2–6 findings 1–5 + focus, 7 So |
| 9 | RQ3 Pretraining speeds up both gain and loss | 110 | `hfstage`; clicks 1–2 findings, 3 calibration sweep + finding 3, 4 finding 4, 5 In practice |
| 10 | Why do the error bars shrink? | 60 | `mechanism`; clicks 1–2 findings + steps, 3 So |
| 11 | RQ4 Pretrained uncertainty is a better acquisition function | 110 | `acquisition`; click 1 LF→HF + finding 1, 2–3 findings, 4 histograms + finding 4, 5 So |
| 12 | Take-home messages | 75 | entry animation only |
| 13 | Thanks (dark, QR) | 15 | — |
| B | Backup: reliability (energy), recalibration, EU/AU ratio, acquisition ID, training curves, limitations | — | static PNGs |

Main-slide `data-timing` must keep summing to 1200; speaker notes carry `[m:ss → m:ss]`
cumulative cues that must match the timings.

## Style rules

- Chrome: titles Source Serif 700 (52 px h2), body Inter; ink `#1f2d33`, teal `#13897d`,
  orange `#e05a2b`, card `#e9f1f0`, dark slides `#0f2a33`. Plot colours stay the paper's.
- Result slide skeleton: `h2` finding · `p.sub` "RQn · question" · `.rq-track` (current `on`) ·
  `.result` grid (figure 880 px | findings) · `.fd` items (`.n` teal, `.n.bad` orange) ·
  `.so` box ("So:" / "In practice:") · `.foot` encoding line.
- Keep slides full: no slide should open nearly empty. Figures show axes and first data on
  entry; text appears in sync with clicks.
- After any layout change, run `scripts/screenshots.py` and look at every sheet: no text
  overlaps (legends vs labels, takeaways vs axis labels), nothing cut at the bottom, and the
  final state of each slide fits. Then regenerate `slides.pdf`.

## Known caveats

- The paper's Fig. 1 shows the epistemic term with 1/(M−1); code and slides use 1/M. Only
  the MVE and training crops of Fig. 1 are used, not its equation.
- Safari untested (no WebKit here); recommend presenting from Chrome/Firefox.
- The folder is untracked; commit only when the user asks (it is ~5 MB incl. `lib/`).
