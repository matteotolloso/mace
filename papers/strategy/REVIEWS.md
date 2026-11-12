# What the reviewers said, across venues

Synthesis of the three sets of reviews. **Rewritten 2026-11-12 (second version)** after the user
supplied the NeurIPS and RPS reviews; the first version of the same day rested on a one-line summary
of them and its diagnosis was wrong in one important respect (Section 3). Verbatim texts with point IDs:

| Venue | File | IDs |
|---|---|---|
| NeurIPS 2026, submission 35263 | [`reviews_neurips26.md`](reviews_neurips26.md) | `NM*`, `NY*`, `NA*`, `NMR*` |
| RPS workshop @ NeurIPS 2026, submission 64 | [`reviews_rps26.md`](reviews_rps26.md) | `RE*`, `RL*`, `RC*`, `RPC` |
| ICLR 2027, submission 18667 | [`../iclr27/rebuttal/reviews.md`](../iclr27/rebuttal/reviews.md) | `G*`, `J*`, `Q*` |

## 1. Venues, versions, outcomes

| Venue | Version reviewed | Scores | Outcome |
|---|---|---|---|
| NeurIPS 2026 (reviews June-July 2026) | first draft: one split seed, total-variance trim, both shifts, no acquisition, no ablations | 2 / 3 / 2 (confidence 4 / 3 / 4) | **withdrawn by the authors 2026-07-28** after the meta-review listed "validity-level problems" |
| RPS workshop (reviews September 2026) | short version, system shift only, still the pre-audit figures | 3 / 4 / 4 of 4 (confidence 2 / 2 / 2) | **accepted, oral**; camera-ready rebuilt on five splits |
| ICLR 2027 (reviews November 2026) | rebuilt paper: five splits, support filter, calibration judged on TU, acquisition experiment | 1 / 2 / 2 (confidence 3 / 4 / 2) | responses posted 2026-11-10 with new baselines and ablations (Appendix M); discussion until 2026-11-18 |

Nine reviewers, one area chair, one set of program chairs.

## 2. Every ask, who raised it, and where it stands today

"Fixed" = the ICLR submission already addressed it; "rebuttal" = addressed in Appendix M of the
revision; "open" = not addressed by any experiment.

| # | Ask | NeurIPS | RPS | ICLR | Raised by | Status |
|---|---|---|---|---|---|---|
| 1 | One dataset, one architecture, one domain | NM1, NMR6 | RE2 | G2, J1, Q2 | 5 of 9 + AC | **open** (argued in the ICLR replies) |
| 2 | Limited novelty; the AU / EU result is "well known" or "expected" | NY1, NY3, NY6, NMR6 | RL2 | G1, Q3 | 4 of 9 + AC | **open**: a matter of what the paper claims (Section 4) |
| 3 | "Aleatoric" / "epistemic" over-interpreted with deterministic labels; component-wise calibration has no observable target | NM2, NA1, NA2, NA9, NMR1 | — | G5 | 3 of 9 + AC | partly fixed (operational definitions, calibration judged on TU); **the names and the AU-ENCE language of Appendix M.3 remain** |
| 4 | HF-only vs LF->HF is confounded: extra data, extra optimization, **geometries of the HF training set already seen in pretraining** | NM3, NA5, NA12, NMR4 | — | G4 (compute, pretrain-only) | 3 of 9 + AC | compute, learning rates and pretrain-only: rebuttal; **reused geometries: open** (acknowledged in Limitations only) |
| 5 | Estimator ablations: ensemble without variance head, single mean-variance model, variance-head reset | NM5, NA12 | RL1 | Q1, Q2, Q4, Q5 | 4 of 9 | rebuttal (asked at all three venues, delivered at the third); "transfer only the backbone" (NA12) open |
| 6 | Forces | — | RE1 | G3 | 2 of 9 | **open** |
| 7 | Statistics: one seed, no intervals | NM4, NA7, NA13, NMR5 | RC6 | — | 3 of 9 + AC | fixed (five splits x ten members, intervals over splits); no ICLR reviewer raised it |
| 8 | Numerical inconsistencies between figures | NA3, NA10, NMR2 | RC4, RC5, RPC | — | 2 of 9 + AC + PC | fixed (support filter instead of the trim; figures rebuilt; rotated rows of the old dynamics figure, `docs/known-issues.md`) |
| 9 | Conclusions contradict the reported numbers (EU "more informative" where AU ranks better; LF->HF "more reliable" where its ENCE is worse) | NA4, NMR3 | RC3 | — | 2 of 9 + AC | fixed (claims rewritten: ranking vs calibration, per test set) |
| 10 | Is the benefit explained by the lower RMSE? Compare at matched RMSE | NM6 | — | — | 1 of 9 | **open**, answerable from stored per-epoch outputs |
| 11 | Other multi-fidelity strategies (fidelity input, multi-head, joint) | — | — | J2, Q2, Q5 | 2 of 9 | **open** (argued) |
| 12 | Mechanism of the AU overconfidence | — | RPC ("limitations of aleatoric variance transfer") | Q1, Q4 | 1 of 9 + PC | rebuttal; direct intervention open |
| 13 | What is the practical use of the training dynamics? | NY2, NY5 | — | — | 1 of 9 | partly fixed (acquisition experiment; "checkpoint selection should monitor calibration"); the selection analysis itself is open |
| 14 | The two shifts come from different training sets: use one factorial design | NA6 | — | — | 1 of 9 | open (acknowledged in Limitations) |
| 15 | Missing reproducibility details | NA8, NMR5 | — | — | 1 of 9 + AC | fixed (appendices on hyperparameters, splits, metrics) |
| 16 | Positioning: general language, single-domain evidence | NM1 | — | — | 1 of 9 | open: a choice of audience (Section 4) |

Positive signals: acquisition is seen as a strength, never as a gap (RC1; ICLR ge6X and Q2LT list
it among the strengths); every venue found the topic important and under-studied and the writing
clear (NeurIPS mM8Y, 3Aye; ICLR all three; RPS all three).

## 3. What the three rounds show

**Correction to the first version of this page.** It assumed that NeurIPS reviewers mainly asked
for actionability (active learning) and ICLR reviewers for forces and fine-tuning strategies, and
explained the difference by reviewer background. The texts say otherwise. At NeurIPS, the use of
the dynamics was one point of one reviewer (NY2); the negative reviews rested on **validity**: no
replication (rows 7), inconsistent figures (8), claims that the tables contradicted (9), an
unidentifiable calibration of the components (3), a confounded comparison (4). Forces were asked by
one workshop reviewer and one ICLR reviewer. Acquisition was asked by nobody.

What actually happened is a progression:

1. **NeurIPS: validity.** The first draft was not sound enough, and the reviewers were right on
   rows 7, 8 and 9. The rebuild of September 2026 (five splits, support filter, rewritten claims)
   fixed them: at ICLR nobody questioned a number, the statistics or the consistency of the
   figures, and the most confident reviewer rated soundness "no substantive concerns".
2. **ICLR: significance.** With validity out of the way, what remained is what the NeurIPS area
   chair had listed last: scope (row 1) and novelty (row 2), plus the two validity residues that
   the rebuild did not close (rows 3 and 4).
3. **RPS: the same content, the right readers.** Reviewers close to the application accepted a
   weaker, earlier version as an oral, and still listed rows 1, 5, 6, 7, 8 as things to do.

So the requests did not diverge between conferences. **Four concerns come back at every round and
are still open**, in order of frequency: one dataset and architecture (row 1); novelty of the
claim (row 2); the aleatoric / epistemic reading and its calibration (row 3); the confounded
protocol comparison (row 4). Everything else was asked once or twice, or is fixed.

## 4. What this implies (input to `PLAN.md`)

1. **Scope (row 1, 5 of 9 reviewers and the area chair)** is the most frequent objection and no argument has moved
   it. A second architecture, and data beyond one sample of ANI-1x, are needed for any conference
   resubmission. Forces (row 6) come with the domain readership rather than from frequency.
2. **Novelty (row 2)** is a problem of the claim, not of the results. The claim reviewers summarize
   back ("aleatoric degrades out of distribution, epistemic is more reliable") is the one they call
   known; a workshop reviewer also finds the non-transfer of the learned variance "expected" (RL2).
   The next version must lead with what is *not* expected (`PLAN.md` Section 2): the ordering of the
   learned variance transfers across fidelities and is carried by the variance head; fine-tuning on
   the correct labels makes its scale worse, also in distribution; the ranking gain of pretraining
   is specific to mean-variance members.
3. **Terminology (row 3)** has now been raised by three reviewers and an area chair at two venues,
   with a wording offered by one of them ("heteroscedastic residual-variance proxy plus ensemble
   disagreement", NA2). Rename. And give the learned variance an observable target wherever its
   calibration is discussed: each member's variance against that member's own squared error.
4. **Confounding (row 4)**: the cheap, decisive control was named twice (NA5, NA12) and never run:
   high-fidelity training geometries that the pretrained model has never seen.
5. **Acquisition is not the missing piece.** It should stay as the consequence that it is; a
   multi-round benchmark is not what any reviewer asked for. The practical use of the dynamics
   (NY2) is checkpoint selection, which is the analysis to add.
6. **Audience (row 16; outcomes).** Two conference panels with general machine-learning readers
   converged on "narrow and not new enough"; reviewers near the application accepted the work
   and asked for forces and more data. This is evidence about where the paper belongs
   (`PLAN.md` Section 6).
7. **ICLR outlook:** unchanged; plan for a resubmission whatever the discussion brings.
