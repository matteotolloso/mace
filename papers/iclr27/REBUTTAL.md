# ICLR 2027 rebuttal: record (Submission 18667)

Ratings 1 / 2 / 2 (2026-11-06) -> **2 / 2 / 2 after the discussion**; responses posted on OpenReview
on 2026-11-10 (text: `ICLR27-UQ-MF/rebuttal.tex`, paper commit `f8b4309`); revised PDF with
Appendix M "Additional baselines and ablations". **Withdrawn (decision 2026-11-19) for a journal
version**: what comes next is in [`../strategy/`](../strategy/README.md). Reviews verbatim with
point IDs: [`../strategy/reviews_iclr27.md`](../strategy/reviews_iclr27.md).

Condensed on 2026-11-19 from the folder `papers/iclr27/rebuttal/` (README, PLAN, RUNS, long draft),
which is in Git history: `git show c35487c:papers/iclr27/rebuttal/<file>`. `RUNS.md` there is the
per-job provenance (commits, SLURM ids, systemd units) of the ledger entries of 2026-11-07 to
2026-11-10; the ledger entries themselves carry commit, command and outputs. Citable numbers:
`results/results.md` only. Interpretation of the results: `../strategy/FINDINGS.md`.

## 1. Reviewer asks, answers, and where they are

Type: W writing, A analysis of existing outputs, E new training.

| ID | Ask (short) | Answer given | Type | Where |
|---|---|---|---|---|
| G1 | "HF fine-tuning is not common"; findings not unexpected | DFT -> CC transfer is an established route (ANI-1ccx, Käser & Meuwly, Chen, Bocus); foundation-model fine-tuning; which findings are not trivial | W | reply; Sec. 1 sentence |
| G2 / J1 / Q2 | one model, one dataset | MACE underpins foundation models and spans a design space; ANI-1x has 3114 formulas with paired labels (1246 held out per system split) | W | replies; Sec. 3 sentence; App. M.5 |
| G3 | energies only | CC labels have no forces; DFT forces in pretraining named as a future experiment | W | reply; App. M.5 |
| G4 | compute, learning rates, pretrain-only | optimizer steps per protocol; HF-only at lr 1e-3 / 3e-3; fine-tuning at 1e-4 / 1e-2 | A + E | App. M.2 (ledger 2026-11-07, three entries) |
| G5 | "aleatoric" with deterministic labels | operational definition, representation-relative view; offered to rename | W | reply |
| G6 | white band in Fig. 1 | Safari soft-mask bug fixed (file from the arXiv repository) | W | Fig. 1 |
| G7 | LF-only not reported | LF-only ensembles on CC labels after a per-element reference-energy refit | A | App. M.1 (ledger 2026-11-07) |
| J2 / Q2 / Q5 | fidelity as input, multi-head, joint training | argued (sequential fine-tuning is established and competitive for one target fidelity); alternatives as future work | W | replies; App. M.5 |
| J3 | better arguments for the choices | "Design choices and scope" | W | App. M.5; Sec. 3 |
| Q1 / Q4 | validate the mechanism | variance-head reset, beta-NLL, fine-tuning learning rates, training vs held-out calibration, dynamics on four tests | E | App. M.3, Figs. 21-22 (ledger 2026-11-07, 2026-11-08) |
| Q2 / Q5 | one estimator | MSE-trained ensembles (HF-only and LF->HF, both splits), single mean-variance model, ensemble size | A + E | App. M.4, Figs. 23-25 (ledger 2026-11-02/03/04/08/10) |
| Q3 | novelty; Bilbrey et al. 2025 | read in full; relation stated; sentence in Sec. 1 on how findings (i)-(iv) extend single-fidelity work | W | reply; Sec. 1 |

Figure scripts of Appendix M: `papers/iclr27/figures/figA_{pretrain_only,lr_sensitivity,ablation_dynamics,ablations,estimators,mse_dynamics}.py`
(map in `papers/iclr27/README.md`). Interpretation and caveats of each result:
`papers/strategy/FINDINGS.md`.

## 2. What the replies commit us to

Nothing was promised that is not in the revised PDF. Offers and statements to honour in any later
version (now the journal version):

- ge6X: "happy to rename" the aleatoric term (decision D3 of `papers/strategy/PLAN.md`).
- ge6X: adding DFT forces to the pretraining stage named as a future experiment (P4 there).
- j1p7, Q2LT: fidelity encodings, fidelity-specific heads, joint and alternating training named as
  future work (P6 there covers multi-head replay).

## 3. Decisions taken during the rebuttal

Decided: MSE ensembles as estimator baseline (energy split on the DGX, system split on Leonardo);
3-GPU cap on the DGX only; code to Leonardo from `iclr_rebuttal`; epoch checkpoints pruned after
evaluation (every fifth epoch, each member's best, `.model` kept); multi-head experiments paused
and answered in writing (2026-11-07); no learning-rate sweeps beyond the recorded ones and no
model-size sweep (2026-11-07); held-out-fitted head dropped (2026-11-07); revised PDF with a new
appendix (2026-11-08); AUSE and ENCE only, main-text figure styles (2026-11-08); one reply per
reviewer, no general response (2026-11-09).

Left open, carried to `papers/strategy/PLAN.md`: forces in pretraining (was D3), second dataset or
architecture (was D4), renaming AU / EU (was D5), use of the remaining Leonardo budget (was D10).

