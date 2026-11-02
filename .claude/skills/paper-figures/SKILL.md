---
name: paper-figures
description: Rebuild or change paper figures and quoted numbers in papers/<venue>/figures/ from tracked evaluation outputs (CPU only), verify that numbers did not change, and hand PDFs to the manuscript repository. Use for any figure, caption-number or paper_numbers.py work.
---

# Paper figures

1. Read `papers/<venue>/README.md` (which script makes which figure, panels, caveats)
   and `.claude/rules/paper-writing.md`. Figures read tracked outputs through
   `common.py` (`split_dir`, `evaluation_dir`, `final_metrics`, letters via the registry).
2. Edit the figure script; keep the statistics contract (`docs/concepts.md`) and the
   plotting conventions in `.claude/rules/statistics-and-plots.md`.
3. Rebuild on CPU: `uq figures <venue>` (all) or
   `cd papers/<venue>/figures && python -B <script>.py` (one). Look at the PNG.
4. Check numbers: `cd papers/<venue>/figures && python -B paper_numbers.py > /tmp/numbers.txt`
   and `diff` against `results/tables/iclr27_main_text_numbers.txt`. A label-only change
   must give no numeric difference; a numeric change needs a ledger entry
   (`record-result`) and regenerated table.
5. Do not copy into or commit in the manuscript repository unless asked; when asked,
   copy the PDF into `ICLR27-UQ-MF/figures/` (copy, do not move — the tracked output must
   stay) and report the script, commit and PDF size.
