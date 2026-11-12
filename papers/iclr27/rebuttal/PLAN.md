# ICLR 2027 rebuttal: what was asked, what was done (Submission 18667)

Entry point and standing rules: [`README.md`](README.md). Condensed on 2026-11-12 into a record of
the finished rebuttal; the working version with every design, task claim and hour-by-hour log is
`git show ebeb547:papers/iclr27/rebuttal/PLAN.md` (and `git log -p` on this file). Numbers:
`results/results.md` only. What comes after the rebuttal: [`../../strategy/PLAN.md`](../../strategy/PLAN.md).

## 1. Status

- Reviews 2026-11-06: ratings 1 / 2 / 2 (ge6X clear reject; j1p7, Q2LT weak reject); presentation
  rated clear by all. Verbatim with point IDs: [`reviews.md`](reviews.md).
- **Responses posted on OpenReview by the user on 2026-11-10**: one reply per reviewer, text of
  `ICLR27-UQ-MF/rebuttal.tex` at paper commit `f8b4309` (ge6X and Q2LT as two comments each, over
  the 5000-character limit). No general response (user's choice).
- Revised manuscript: new Appendix M "Additional baselines and ablations" (M.1-M.5), additions in
  blue (`\revise{}`); main text still ends on page 9. Paper repository at `1c358d9`, level with
  its GitHub remote (which Overleaf pulls).
- **Until 2026-11-18: replies to reviewers only.** No reviewer follow-up is recorded here as of
  2026-11-12 (append any to `reviews.md` with new point IDs).

## 2. Reviewer asks, answers, and where they are

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

## 3. What the replies commit us to

Nothing was promised that is not in the revised PDF. Offers and statements to honour in any later
version (camera-ready or resubmission):

- ge6X: "happy to rename" the aleatoric term (decision D3 of `papers/strategy/PLAN.md`).
- ge6X: adding DFT forces to the pretraining stage named as a future experiment (P4 there).
- j1p7, Q2LT: fidelity encodings, fidelity-specific heads, joint and alternating training named as
  future work (P6 there covers multi-head replay).
- Each reply ends by asking whether there is anything else to clarify or do: answer follow-ups
  from recorded results only.

## 4. Open items

- Fig. 25 (`figA_mse_dynamics`, App. M.4) was added to the paper repository (`1c358d9`) after the
  posted text was fixed at `f8b4309`, and the replies cite Figs. 21-24 only: **the user should
  confirm that the PDF on OpenReview is the intended one.**
- App. M.4: the ensemble-size result (ledger 2026-11-02) is text only; a five-split figure of the
  M = 1..10 curves is optional.
- Not tested, do not cite: prediction (a) of the mechanism (AU against training MSE per epoch,
  WP3.1; the stored per-epoch means are inconclusive) and the held-out-fitted variance head
  (WP3.4, dropped 2026-11-07; now P2b of `papers/strategy/PLAN.md`).

## 5. Decisions taken during the rebuttal

Decided: MSE ensembles as estimator baseline (energy split on the DGX, system split on Leonardo);
3-GPU cap on the DGX only; code to Leonardo from `iclr_rebuttal`; epoch checkpoints pruned after
evaluation (every fifth epoch, each member's best, `.model` kept); multi-head experiments paused
and answered in writing (2026-11-07); no learning-rate sweeps beyond the recorded ones and no
model-size sweep (2026-11-07); held-out-fitted head dropped (2026-11-07); revised PDF with a new
appendix (2026-11-08); AUSE and ENCE only, main-text figure styles (2026-11-08); one reply per
reviewer, no general response (2026-11-09).

Left open, carried to `papers/strategy/PLAN.md`: forces in pretraining (was D3), second dataset or
architecture (was D4), renaming AU / EU (was D5), use of the remaining Leonardo budget (was D10).

## 6. Log (condensed; full log in Git history)

- 2026-11-06 — reviews received; branch `iclr_rebuttal`; Leonardo access set up.
- 2026-11-07 — ablations and the system-split MSE ensembles launched on Leonardo; pretrain-only
  baseline, gradient steps, HF-only learning-rate sweep and fine-tuning ablations recorded; plan
  condensed around a presentation plan (user: stop adding ablations).
- 2026-11-08 — MSE ensembles after LF->HF recorded on both splits; AU along fine-tuning for every
  ablation, including Energy-OOD; interpretation written to `FINDINGS.md`; Appendix M drafted.
- 2026-11-09 — consistency pass of the draft and Appendix M against the ledger; figures revised
  with the user; `rebuttal.tex` assembled and reworded over several rounds; Fig. 1 fix applied.
- 2026-11-10 — energy-split MSE full evaluation finished and recorded; estimator figures split by
  protocol; Sec. 1 and Sec. 3 sentences added; **responses posted**; `figA_mse_dynamics` built and
  added as Fig. 25.
- 2026-11-12 — rebuttal files condensed; analysis and next-version plan in `papers/strategy/`;
  `FINDINGS.md` moved there, `LEONARDO.md` to `docs/leonardo.md`.
