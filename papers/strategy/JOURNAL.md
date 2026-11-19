# Journal version: target venues and improvements in priority order

Written 2026-11-19, after ICLR 2027 closed at 2 / 2 / 2 and the authors decided to withdraw. It
turns `PLAN.md` (work packages `A*`, `P*`, `W*`) into one priority list for a Q1 journal, and adds
the shortlist of journals. Where the two differ, this file is newer. Nothing is launched until the
user approves (decisions in `PLAN.md` Section 8; D7 is answered: route B, domain journal).

## 1. Q1 journals that fit

Quartiles depend on the index (Clarivate JCR or Scopus/SCImago) and on the category: **check both
on the official pages at submission time.** Status below as found on 2026-11-19.

| # | Journal | Quartile | Fit | Notes |
|---|---|---|---|---|
| 1 | **Journal of Chemical Theory and Computation** (ACS) | Q1 (JCR and SJR) | very high: ML potentials, UQ, DFT -> CC transfer (the ANI-1ccx line of work) | no length limit; reviewers will expect forces (P4) |
| 2 | **npj Computational Materials** (Nature Portfolio) | Q1 | high: tan2023single (single-model UQ vs ensembles) and other MLIP-UQ work appear there | highest bar of the domain options; open access (APC); molecular data acceptable but materials framing helps |
| 3 | **Digital Discovery** (RSC) | Q1 (JCR, Computer Science Interdisciplinary; first JIF 6.2) | high: ML methodology for chemistry, benchmark-style studies | open access (APC); most receptive to a careful empirical study |
| 4 | **Machine Learning: Science and Technology** (IOP) | Q1 (SJR 2021-2023; check the latest year) | high if the ML framing is kept (uncertainty decomposition as the subject) | open access (APC); closest to the conference version, least rewriting |
| 5 | **Journal of Chemical Information and Modeling** (ACS) | Q1 | medium: UQ and active learning for molecular models | readership more cheminformatics than simulation |
| 6 | **The Journal of Chemical Physics** (AIP) | Q1 in SJR; **Q2 in JCR Chemistry, Physical** (2023) | high topically (ML-potential special topics) | only if the Q1 rule accepts SJR |
| — | Nature Communications, Chemical Science, Nature Computational Science | Q1 | only with forces, a second system and a clear practical consequence (Tier 1 complete plus P5/P11) | ambitious; not the default |
| — | JMLR, IEEE TNNLS, Neural Networks | Q1 | low: general-ML readers raised exactly scope and novelty (two conference panels) | not recommended |

TMLR is not in JCR and has no quartile: it no longer meets the requirement.

**Recommendation:** JCTC first (fit, readership that already accepted the workshop version), with
npj Computational Materials as the ambitious option if P4 and P8a come out clean; Digital Discovery
or MLST as fallback with the same manuscript.

## 2. Improvements in priority order

Priority = how often the concern came back across the nine reviewers (`REVIEWS.md` Section 2) x
what a journal of Section 1 will require, divided by cost. Items 1-6 are the core: without them the
journal reviewers will raise the same points as the conferences.

| # | Improvement | Answers | Package | Cost |
|---|---|---|---|---|
| 1 | **DFT forces in the low-fidelity stage** (energies + forces from `wb97x_tz`, CC energies in the HF stage; same split assignments); report force errors and force disagreement | RE1, G3; the entry ticket for JCTC / npj | P4 | high, Leonardo (pilot split 0, then five splits); start first, it is the long pole and the allocation ends 2027-04-29 |
| 2 | **Fine-tuning geometries unseen in pretraining**: CC train/val sets disjoint from the parents' DFT configurations, plus the matched HF-only | confounded protocol comparison (NM3, NA5, NA12, NMR4; row 4) | P8a | low (~185 GPU-h, DGX) and decisive: if the result changes, the mechanism section changes |
| 3 | **Reframe around one thesis** ("the ordering transfers, the scale does not") and lead with what is not expected; scope claims to atomistic models | novelty (NY1, NY3, NY6, NMR6, RL2, G1, Q3; row 2) | W1 | writing |
| 4 | **Operational names for the two terms** (mean predicted variance, member disagreement) and an **observable calibration target** for the learned variance (each member vs its own squared error) | NM2, NA1, NA2, NA9, NMR1, G5 (row 3); promised to ge6X | W2 + A2 | writing + CPU |
| 5 | **Intervention on the scale**: closed-form rescaling along fine-tuning (P2.0, CPU), then frozen head + rescale, held-out refit, out-of-bag members | turns the mechanism into a test and a recipe (Q1, Q4; NY2 "actionable") | P2 | P2.0 CPU; the rest ~165 GPU-h |
| 6 | **Comparison at matched accuracy** (AUSE / ENCE vs RMSE along training) | "is the benefit explained by the lower RMSE?" (NM6) | A1 | CPU, stored outputs |
| 7 | **Shared-parent ensembles** (ten members fine-tuned from one parent; seed-only, re-initialized head, bootstrap) | realism of the foundation-model use case (ASSESSMENT 4.2) | P1 | ~165 GPU-h; needs D5 |
| 8 | **Second architecture** (ANI-style network or PaiNN) on the six core experiments | one model (5 of 9 reviewers + AC; row 1) | P5 | 1-2 weeks of engineering, little compute |
| 9 | **Checkpoint selection and resolved fine-tuning dynamics** (every few steps in the first epochs; selection by NLL vs RMSE vs budget) | practical use of the dynamics (NY2, NY5) | P3 | ~55 GPU-h |
| 10 | **Second multi-fidelity strategy**: multi-head replay fine-tuning | J2, Q2, Q5 (promised as future work) | P6 | ~100-200 GPU-h; check MVE + multi-head in the fork first |
| 11 | **Broader data**: a second fidelity pair with forces at both levels (`wb97x_dz` -> `wb97x_tz`), or water on aligned splits, or a CC-data-size sweep | dataset part of row 1; NM3 (data size) | P11 / P9 / P12 | one of them, chosen at Gate 1 |
| 12 | **Journal-format writing**: Appendix M and the rebuttal baselines as full sections; related work on fine-tuned and foundation-model UQ (read in full first); ensemble-size figure; backbone-only transfer as part of the ablation section (P8b) | completeness expected without a page limit | W3, W4, P8b | writing; P8b ~55 GPU-h |
| 13 | **Data and code availability**: archive splits, selected models and per-split evaluation outputs with a DOI (e.g. Zenodo), pin the code release | required by every journal of Section 1 | — | low; needs the user's choice of what to archive (sizes) |
| 14 | Multi-round pool-based acquisition | never asked by a reviewer; only if items 1-10 leave budget | P7 | ~480 GPU-h per split design |

Declined on purpose (unchanged from `PLAN.md` Section 5): more learning-rate or model-size
sweeps, more estimators, materials datasets, a ladder of fidelities, other scientific domains.

## 3. Order

1. Now: items 2, 6, 4 (A2), 5 (P2.0) on the DGX / CPU; item 1 dataset variants, force loss, smoke
   test and pilot on Leonardo in parallel; choose the second architecture (item 8, D4).
2. **Gate 1** (`PLAN.md` Section 4): freeze or rewrite the thesis from items 2, 5, 6 and the force
   pilot.
3. Item 1 on five splits, items 7-10.
4. Writing (items 3, 4, 12, 13) on the frozen results; choose the journal of Section 1 by how much
   of items 1, 8, 11 holds.

Sources for the quartiles (2026-11-19): RSC news on the first Digital Discovery JIF
(rsc.org/news/2024/july), the Digital Discovery journal page (pubs.rsc.org/en/journals/journal/dd),
exaly / journalmetrics listings for MLST, SCImago 2024 listing and Peeref for JCP. Not checked
individually: JCTC, npj Computational Materials, JCIM, Nature-family titles (long-standing Q1).
