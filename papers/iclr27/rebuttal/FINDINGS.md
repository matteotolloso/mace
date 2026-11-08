# What the rebuttal results say (interpretation layer, read before writing)

Created 2026-11-08 (mace-leo-3) after discussing the results with the user. This file holds the
*interpretation*: what each new result supports, where it is risky, and how to phrase it. Numbers
stay in `results/results.md`; every statement below names its ledger entry. Update this file when a
result arrives or the user changes the framing.

## 0. User preferences for presenting results (2026-11-08)

- **Metrics: AUSE (ranking) and ENCE (calibration) only**, as in the main text; RMSE for accuracy.
  No Spearman in new figures, appendix text or (when revised) the rebuttal.
- **Plot styles of the main text**: final-model comparisons as Fig. 2 dot plots
  (`papers/iclr27/figures/dotplot.py`), training dynamics as Fig. 5 line plots
  (`fig3_hf_dynamics.py` style). Tables only where a figure cannot work.
- Final models = each member at its lowest validation loss (Gaussian NLL; validation MSE for
  `mace-mse`); say so in captions/text.
- Show the training dynamics, not only final models: the most interesting observations are there.
- The user wants to see the revised appendix PDF (`ICLR27-UQ-MF/iclr2027_conference.pdf`, untracked)
  before the rebuttal text. Waves: A Nov 10, B Nov 12, C (revised PDF) Nov 14, then replies only.

## 1. Results that strengthen the paper

1. **Pretrain-only baseline (G7, G4)** — ledger 2026-11-07 "LF-only (pretrain-only) ensembles on the CC
   labels". Fine-tuning beats the zero-shot DFT ensemble (per-element E0 refit) in ranking on every
   test and in calibration everywhere except Energy-OOD (similar); DFT-learned variances do not
   transfer to CC errors. Side note (pro-pretraining): the zero-shot ensemble ranks in-distribution
   errors as well as or better than HF-only, but is far less calibrated.
2. **Compute (G4)** — ledger 2026-11-07 "Gradient steps per protocol". LF->HF takes ~20x fewer steps
   on CC labels (median 234 vs 4758 / 5304); the extra compute is on DFT labels. Defuses
   "not compute-matched".
3. **Learning rates, accuracy and ranking (G4)** — ledger 2026-11-07 "HF-only learning-rate sweep
   complete" and the FT-lr entries. Against every tuned HF-only, LF->HF stays ~2x more accurate in
   distribution and ranks better in all 8 (split, lr, test) cells. A lower lr helps HF-only only a little.
4. **Estimators (Q2/Q5): the strongest addition** — ledger 2026-11-03, 2026-11-04 (HF-only), 2026-11-08
   "MSE ensembles after LF->HF fine-tuning, ANI-1x system split". MVE TU beats the MSE-ensemble EU on
   every metric and test; one MVE model's AU sparsifies better than ten MSE members. After LF->HF the
   MVE EU, TU and single-model AU all beat the MSE EU, whereas at HF-only the two EUs were similar:
   **pretraining improves the MVE ranking ~5x more than the MSE ranking** (AUSE -0.10/-0.11 vs
   -0.02/-0.03), so the ranking gain of pretraining is tied to the mean-variance members, not to
   starting an ensemble from pretrained weights. The MSE EU is not degraded by pretraining (it has no
   AU to collapse). Energy split pending (DGX `uq-mse-eval3`).
5. **Mechanism (Q1/Q4)** — ledger 2026-11-07 "Fine-tuning ablations of the AU collapse", "Head reset: AU
   calibration along fine-tuning", 2026-11-08 "AU along fine-tuning for all LF->HF ablations".
   - **Held-out effect:** in every variant AU on the *training* set stays calibrated or conservative
     (train ENCE <= 0.91 from FT epoch 10 on, except the unstable system lr 1e-2 epoch 10: 1.16), while held-out AU becomes overconfident: the head follows
     the training residuals, as Appendix "Technical background" says.
   - **Head reset:** the collapse re-emerges with a fresh head, later (ENCE > 1 after 20-25 FT epochs
     instead of 5-10) and ~5x weaker by epoch 95; the inherited head makes it start at epoch 0 —
     this is *why* calibration is lost faster after pretraining (finding iii). The reset head also
     ranks worse at every epoch: part of pretraining's ranking benefit lives in the variance head.
     Validation-NLL selection picks reset members early (epochs 2-19), before the collapse.
   - **beta-NLL** follows the paper's dynamics almost exactly and does not remove the overconfidence.

## 2. Results that need careful phrasing (risks)

1. **The *speed* of the collapse depends strongly on the FT learning rate, non-monotonically.**
   Ledger 2026-11-08 "AU along fine-tuning for all LF->HF ablations". With FT lr 1e-4 *and* 1e-2,
   held-out AU ENCE first exceeds 1 after 25-75 epochs (vs 5-10 at the paper's 1e-3) and ends at
   1.4-4.2 (vs 58-82) at epoch 95. lr 1e-4: the predicted delay (~10x later; it also starts very
   overconfident at epoch 0 because one epoch at the small lr does not absorb the DFT->CC offset,
   training RMSE 31-35 vs 9-14 meV/atom). lr 1e-2: also slow, although its final training error is
   similar (system epoch 10 unstable). **Risk:** a reviewer may read finding (iii) ("pretraining
   accelerates the loss of calibration") as specific to the chosen lr. **Defence:** the collapse
   develops in every variant; at the *selected* checkpoints AU is as overconfident or more with
   1e-4 (AU ENCE x1.15-1.38) and the final-model conclusions hold; the fast start is caused by the
   inherited head (reset result). **Phrase:** "the overconfidence is robust; its speed depends on
   the optimization and is fastest in the main setting". Do not write "the time scale follows the
   optimization speed" without the 1e-2 caveat.
   **Energy-OOD (ledger 2026-11-08 "... on Energy-OOD", added 16:50):** there AU is overconfident from FT
   epoch 0 in every variant with the inherited head, at either lr (lr 1e-4 minimum 5.6, never < 1); only the
   reset head starts calibrated (0.71) and crosses 1 at epoch 5. Paper / beta-NLL end at 207-239, reset and
   lr 1e-2 at ~41, lr 1e-4 at 14. So the "delayed collapse" holds in distribution and on System-OOD only;
   under energy extrapolation no lr makes AU calibrated (App. M.3 text says so).
2. **The calibration contrast between protocols is split-dependent and lr-sensitive.** Ledger
   2026-11-07 "HF-only learning-rate sweep complete": energy split LF->HF less calibrated in all 4
   (split, lr) cells (significant in 2); system split better calibrated in all 4 (significant in 1);
   at HF-only lr 1e-3 the Energy-OOD gap is no longer significant. The abstract's "calibration
   deteriorates more severely under energy extrapolation" holds at the paper's setting but is less
   robust to tuning: state calibration per split, avoid absolute claims.
3. Minor: on the system split the MSE ensemble is more accurate than the MVE ensemble at HF-only
   (RMSE ratio x1.24 OOD, ledger 2026-11-04); equal after LF->HF. The decomposition can cost accuracy.

## 3. Recommended stance

Publish all of it: accuracy, ranking, estimators and the mechanism ablations are the solid core.
Make the two calibration caveats (Section 2.1, 2.2) explicit in the revised appendix (done in App. M.3
for 2.1) and in the Q1 / G4 replies; hiding the lr dependence is riskier because the data are in the
appendix. Open: user review of this framing, D3/D4/D5/D10 (2026-11-09).

## 4. Pending data

- ~~Energy-OOD per-epoch AU for the energy FT ablations~~: done 2026-11-08 16:50 (figure, ledger, App. M.3 caption/text, PDF recompiled).
- Energy MSE LF-only / LF->HF evaluation (`uq-mse-eval3`, RUNS.md handoff).
