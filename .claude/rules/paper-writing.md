---
paths:
  - "papers/**"
  - "ICLR27-UQ-MF/**"
---

# Paper writing (Overleaf-synchronized manuscript)

## Where things live

- The manuscript is `ICLR27-UQ-MF/`, an **independent Git repository** synced with
  Overleaf through GitHub and ignored by this repository; commit and push paper
  changes from inside it. Main file `iclr2027_conference.tex` (inputs
  `math_commands.tex`, `iclr2027_conference.{sty,bst,bib}`, `fancyhdr.sty`,
  `natbib.sty`; figures in `figures/`). Keep that layout: Overleaf compiles from its
  root. Anything committed there appears in Overleaf: commit only what the paper needs;
  build products, notes and Claude files are in its `.git/info/exclude` (never pushed).
- Figures are produced in this repository by `papers/iclr27/figures/*.py` into
  `papers/iclr27/figures/out/` (`uq figures iclr27`, CPU only); every quoted number is
  printed by `papers/iclr27/figures/paper_numbers.py` (ledger entry "ICLR27 main-text
  numbers"). Copy a PDF into `ICLR27-UQ-MF/figures/` only to replace a paper figure,
  and record which script and commit produced it. The submitted state is tag
  `iclr27-submission`.
- `papers/iclr27/README.md` maps every figure to its script, panels and caveats.
  Before using any output from `experiments/<id>/evaluation/`, read
  `docs/evaluation-outputs.md` (`../docs/evaluation-outputs.md` from the manuscript
  folder): prefer `-log-scale` epoch-quality variants; do not use `finetune.svg`,
  `energy_ood.svg`, `distribution_*.svg` or energy-split (C, D, F) OOD epoch-quality
  curves (unfiltered or misleading); per-split `split_<s>/` figures are single replicates.

## ICLR27 figure map

| Manuscript | Script (`papers/iclr27/figures/`) |
|---|---|
| Fig. 1 | drawn by hand (`Workflow.pdf`) |
| Fig. 2 | `fig2_final_summary` |
| Fig. 3 | `fig4_rank_vs_calibration` |
| Fig. 4 | `fig5_decomposition` |
| Fig. 5 | `fig3_hf_dynamics` |
| Fig. 6 | `fig6_acquisition` |
| Appendix | `figA_*` (recalibration, EU/AU ratio, acquisition ID, reliability, dynamics, train curves), `figW_*` (water) |

Implementation behind the methods: MVE head and loss `mace/modules/{models,loss}.py`;
AU/EU/TU and metrics `uq/evaluation/{reliability,epoch_quality}.py`; splits
`uq/datasets/ani1x_{system,energy}.py`; hyperparameters `experiments/<id>/config.yaml`.

## Content rules

- Do not rewrite paper text, claims, citations or results unless asked.
- Every number in the paper traces to a ledger entry or to an output such an entry
  lists; flag numbers that cannot be traced instead of repeating them.
- Paths in the manuscript's provenance comments predate 2026-09-26 (`new_figures/...`,
  `experiment_X/...`); translate them with `uq where`.
- Draft sections outside the manuscript (e.g. `papers/iclr27/drafts/results.tex`) are
  not part of it until the user asks to merge them.
- The active-learning study is a proof of concept (paper Fig. 6 and one appendix
  figure only); keep other AL claims out unless the user extends the paper.

## Manuscript consistency checks

Do not silently change code or old results to match the paper. Raise these when
editing methods, equations, captions or reproducibility claims:

1. EU uses the population variance in code (`1/M`); the manuscript equation shows `1/(M-1)`.
2. The code uses five dataset split seeds (`0..4`) and ten member seeds (`0..9`);
   fine-tuning runs 100 epochs, pretraining/direct runs 300.
3. Numbers use five splits, the geometry-only support filter and no trim; figures or
   tables made before 2026-09-22 used split 0 and a total-variance trim.
4. Water: five splits, but the BLYP and CCSD(T) splits are not aligned
   (`docs/known-issues.md`), which favours LF->HF.
5. The old NeurIPS-draft dynamics figure had rotated rows in its ID/OOD columns
   (`docs/known-issues.md`); the ICLR27 figures are rebuilt and not affected.
