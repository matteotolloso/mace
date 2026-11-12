# Development plan toward publication (after ICLR 2027)

Written 2026-11-12; **revised the same day (version 2) after reading the NeurIPS 2026 and RPS 2026
reviews**: two analyses of stored outputs added (A1, A2), two controls added (P8), the
acquisition benchmark moved to Tier 2, the venue recommendation changed to a domain journal by
default (Section 10 lists the changes). Reasons behind every choice: [`ASSESSMENT.md`](ASSESSMENT.md) (the work against
the literature) and [`REVIEWS.md`](REVIEWS.md) (what reviewers asked and why). This file is the
plan: thesis, work packages, order, compute, venues, decisions. It is a proposal until the user
approves it (Section 8); after that it is the task list, maintained like the rebuttal plan was
(markers `[ ]` open, `[~] (agent, since)` claimed, `[x]` done with ledger entry, `[-]` dropped).

Compute figures are **estimates** from the ledger entry "Training compute from run logs"
(2026-09-25: median GPU-hours per member, energies only: LF pretraining ~15-16, HF-only ~1.1-1.5,
fine-tuning ~0.55) and must be re-measured by a smoke test before a campaign is launched.

## 1. Recommendation in short

1. **Do not add directions; close the four concerns that came back at every venue**
   (`REVIEWS.md` Section 3): scope (one dataset, one architecture; forces), novelty of the claim,
   the aleatoric / epistemic reading and its calibration, the confounded protocol comparison.
   Acquisition was praised and never asked for more: it stays as it is. Everything else is
   declined (Section 5).
2. **Lead with one thesis** instead of four findings: *across a change of fidelity the ordering of
   the uncertainty transfers, its scale does not* (Section 2). The evidence is already in the
   revision appendix; one intervention experiment (P2) turns it from an account into a test.
3. **Start with what is cheap and decisive**: two analyses of stored outputs (A1, A2, CPU) and
   four fine-tuning-scale experiments (P1-P3, P8; a few hundred GPU-hours, DGX) decide the story. Launch the one expensive item (P4, pretraining
   with DFT forces, Leonardo) in parallel because it is the long pole and the allocation ends on
   2027-04-29.
4. **Default venue: a top domain journal** (Section 6), where the closest related work is
   published, where reviewers of that readership already accepted the work as a workshop oral, and
   where forces are the entry ticket. A general conference again only if Gate 2 shows breadth that
   two conference panels found missing.

## 2. Thesis and framing of the next version

**Thesis.** Low-fidelity pretraining teaches a mean-variance model *where* its errors will be
large, and that knowledge survives the change of fidelity; the *magnitude* it attaches to them does
not, and fine-tuning on scarce high-fidelity labels makes it worse on unseen data. Hence ranking
improves and acquisition benefits, calibration is lost during the high-fidelity stage, and the two
separate most under energy extrapolation.

What a reader should *not* be able to call known (reviewers called the old headline "AU degrades
out of distribution, EU is more reliable" well known: NY3, NMR6, RL2, G1): (1) the ordering learned
on low-fidelity residuals is still the right ordering for high-fidelity errors, and it is carried
by the variance head; (2) the scale is not merely inherited wrong, it is damaged by training on the
correct labels, also in distribution; (3) the ranking gain of pretraining belongs to mean-variance
members, not to ensembles in general.

Existing support (ledger 2026-11-07/08/10; `FINDINGS.md` Section 1): pretrain-only ensembles rank
well but are miscalibrated on CC labels; the ranking gain of pretraining is several times larger
for mean-variance than for MSE ensembles; a re-initialized variance head ranks worse at every
epoch and loses calibration later; training-set AU stays calibrated while held-out AU does not.

Consequences for the text (decisions D2, D3 in Section 8):

- The mechanism moves from Appendix M to the main text; the four findings become consequences of
  the thesis; the abstract's calibration claims are stated per split (`FINDINGS.md` Section 2.2).
- Name the two terms by what is computed: *mean predicted variance* and *member disagreement*,
  with the aleatoric / epistemic reading introduced as the interpretation under test. Three
  reviewers and an area chair at two venues objected to the names (NM2, NA1, NA2, NMR1, G5), one
  offering this wording; the paper's own result (the "aleatoric" term behaves as a reducible,
  in-sample quantity) argues for the change.
- Absolute calibration is claimed for TU only; where the learned variance is discussed, its target
  is each member's own squared error (A2), not the error of the ensemble mean.
- Scope the language to atomistic models: general claims on single-domain evidence were penalized
  (NM1).
- Working title, to be improved by the authors: "Ranking transfers, scale does not: predictive
  uncertainty across a change of fidelity in atomistic models".

## 3. Work packages

Tier 1 is the core package; without it a resubmission meets the same reviews. IDs are `P*` to
avoid confusion with the rebuttal's `WP*`. Every package: new experiment ids (nothing existing is
overwritten or retrained), five splits x ten members for anything reported, pilots labelled as
such, ledger entry per result. Every job is registered before launch in the Jobs table of
`PROGRESS.md` (repository root); cluster rules and the sync protocol are in `docs/leonardo.md`.

### Tier 0: analyses of stored outputs (CPU, days)

- [ ] **A1 Matched-accuracy comparison** (NeurIPS NM6: "is the benefit explained away by the lower
  RMSE?"). From the per-epoch tables that every experiment already has (`epoch_quality_*.csv`:
  RMSE, AUSE, ENCE per epoch and split) and the Energy-OOD per-epoch predictions: AUSE and ENCE of
  HF-only and LF->HF as a function of RMSE along training, and the contrast at the checkpoints
  where the two protocols have equal RMSE (System-OOD is such a case at the selected models).
  Uses also the existing learning-rate variants. Output: one figure, one ledger entry.
- [ ] **A2 An observable target for the learned variance** (NeurIPS NA1, NA9, NMR1). From the
  cached per-member predictions of the selected models (`members_*.npz`): calibration of each
  member's variance against that member's own squared error, per protocol and test set, with the
  same statistics as everything else; then restate the "AU overconfidence" results of Appendix M.3
  in these terms. Per-epoch version only where per-member, per-epoch predictions exist or after
  P3. If the member-level picture differs from the AU-vs-ensemble-error picture, that is a result.

### Tier 1a: cheap and decisive (fine-tuning-scale compute, DGX)

- [ ] **P1 Shared-parent ensembles** (validity of the foundation-model motivation; ASSESSMENT 4.2).
  Fine-tune ten members from **one** LF parent per split (member 0), on both split designs.
  Arms: (a) members differ by fine-tuning seed only; (b) each member gets a re-initialized variance
  head; (c) each member fine-tunes on a bootstrap sample of the CC training set (shared with P2c).
  Question: does member disagreement still become a competitive ranking signal, and does TU stay
  as calibrated, when the parents are not independent? Reference: the existing LF->HF ensembles
  (independent parents). Deviates on purpose from the rule "member (s, m) from parent (s, m)", so
  it needs an explicit spec field and the user's approval (D5). ~3 arms x 100 members x 0.55 h
  = ~165 GPU-h + evaluation.
- [ ] **P2 Intervention on the scale** (turns the mechanism into a test; ICLR Q1/Q4; NeurIPS
  "actionable"). In order of cost:
  - P2.0 *no training*: fit one scalar per member on the CC validation set (closed form, mean of
    squared residual over predicted variance) at every stored fine-tuning epoch of the existing
    LF->HF runs and their ablations; report AUSE (unchanged by construction) and ENCE of the
    rescaled AU and TU on the four tests. Separates loss of scale from loss of ordering along
    training. Needs per-configuration, per-epoch predictions on validation and test: check what is
    cached before running inference (`uq.evaluation.epoch_predictions`).
  - P2a *frozen head + scale*: fine-tune with the pretrained variance readout frozen, then the
    scalar of P2.0. The cleanest test of the thesis: inherited ordering, re-estimated scale.
  - P2b *held-out refit*: after fine-tuning, refit only the variance readout on held-out CC
    residuals with the body frozen (the rebuttal's WP3.4, dropped then for lack of time and
    because it resembles the isotonic recalibration of the appendix; here it is evaluated along
    training, not only at the selected checkpoint).
  - P2c *out-of-bag members*: each member fine-tunes on a bootstrap sample and fits its variance
    on its own out-of-bag residuals: no validation labels consumed, and member diversity for P1.
  Success criterion, fixed before running: at any fine-tuning epoch, TU ENCE and the member-level
  calibration of A2, in distribution and on System-OOD, comparable to the best value along the
  reference run, with no loss of AUSE. Expected
  and acceptable negative: Energy-OOD stays miscalibrated (the scale is not constant along the
  extrapolation axis; then the claim is "the scale must be re-estimated where it is used").
  ~3 trained arms x 100 members x 0.55 h = ~165 GPU-h + evaluation.
- [ ] **P3 Resolved fine-tuning dynamics and checkpoint selection** (ASSESSMENT 2.1; this is the
  answer to "what is the practical use of the dynamics?", NeurIPS NY2, NY5: which checkpoint to
  deploy and by which criterion). Re-run the
  fine-tuning of the two main LF->HF experiments as new ids keeping every checkpoint and
  evaluating every few optimizer steps in the first epochs (selected checkpoints sit at a median
  of 234 steps; stored ones are every fifth epoch). Then compare selection by validation NLL,
  validation RMSE and fixed budgets. ~55 GPU-h + evaluation. Not bit-identical to the existing
  runs (GPU nondeterminism): report it as its own experiment.

- [ ] **P8 Controls on what is transferred** (NeurIPS NA5, NA12, NMR4; ASSESSMENT 2.5).
  - P8a *unseen fine-tuning geometries*: a second CC training and validation set per split, drawn
    from the same training pool but **disjoint from the DFT training and validation
    configurations** of the parents (new dataset variant, same test sets). Fine-tune the existing
    LF parents on it, and train HF-only on it as the matched baseline. If the held-out
    overconfidence and the ranking gain are unchanged, the reused-geometry confound is closed with
    one figure; if they change, the mechanism section changes. Both split designs: 100 fine-tuned
    and 100 HF-only members, ~55 + ~130 GPU-h, plus evaluation.
  - P8b *backbone-only transfer*: re-initialize both readouts (energy and variance) before
    fine-tuning; with the existing variance-head reset it separates what the representation
    carries from what the readouts carry. ~55 GPU-h.

### Tier 1b: the setting (new pretraining or new code)

- [ ] **P4 DFT forces in the low-fidelity stage** (ICLR G3; entry ticket for domain journals).
  Protocol of messerly2025multi: LF stage on `wb97x_tz` energies **and forces** (in the raw file
  for all 4.6M TZ configurations; no new labels), HF stage on CC energies as now; HF-only stays
  energy-only because CC forces do not exist. Work: dataset variants that export forces with the
  **same split assignments**; a loss with Gaussian NLL on energies plus a force term (the MVE loss
  is energy-only today, `mace/modules/loss.py`); evaluation of force error and of the force
  disagreement between members against DFT forces. Questions: do the thesis and the AU collapse
  hold with force-trained parents; does the Energy-OOD calibration result change (forces carry the
  slope of the surface, so this is the claim most at risk). Order: smoke test and timing; a
  labelled pilot on split 0 of both designs; then five splits for the design(s) that go in the
  paper. Estimate if forces cost 2-4x per step: 1,600-3,200 GPU-h of pretraining per split
  design, Leonardo.
- [ ] **P5 Second architecture** (all three ICLR reviewers). Criteria: a different model family,
  cheap to train on 50k + 5k energies, a mean-variance head that is easy to add, plugs into the
  `uq` backends (`docs/extending.md`, `uq/registry.py::BACKENDS`). Candidates: an ANI-style
  descriptor network (the architecture of ANI-1ccx itself, minutes per member) or an invariant
  message-passing model (SchNet / PaiNN). Scope: the six core experiments (three protocols x two
  splits), final models and dynamics; not the ablations. Mostly engineering (estimate one to two
  weeks), little compute. Choice of model: D4.
- [ ] **P6 Second multi-fidelity strategy: multi-head replay fine-tuning** (ICLR J2, Q2/Q5;
  tompa2026fine recommends it for broad scope). From the existing LF parents, fine-tune with a CC
  head while replaying DFT data on the LF head (the fork's MACE 0.3.15 has `--heads` /
  `--multiheads_finetuning`; whether the MVE head works with it must be checked first). One arm,
  both splits. Estimate ~1-2 GPU-h per member with 50k replay configurations: ~100-200 GPU-h.
### Tier 2: only if Tier 1 leaves time and budget, or if route A is chosen at Gate 2

- [ ] **P7 Multi-round, pool-based acquisition** (no reviewer asked for it: acquisition is listed
  among the strengths by RC1, ICLR ge6X and Q2LT; it is the consequence a method-oriented venue
  would want enlarged). Pool = CC-labelled
  configurations of the raw file that no split uses (490k CC labels exist, 5k are used for
  training), built per split with the same shift definitions; test sets untouched. Three or more
  rounds; arms: random, TU and EU for both protocols, the MSE-ensemble disagreement (the classical
  query-by-committee baseline, models exist), the highest-LF-energy heuristic, and the
  pretrain-only ensemble choosing the first batch before any CC label. Report **labels needed to
  reach a target error**, not only meV differences. Caveat to state: the CC subset of ANI-1x was
  itself chosen by active learning, so the pool is not a random sample. Estimate ~8 arms x 3
  rounds x 50 members x 0.4 h = ~480 GPU-h per split design.

- [ ] **P12 Data scale** (NeurIPS NM3 "dataset size"): CC training sizes 500 / 5k / 50k at fixed
  LF. Prediction of the mechanism: the held-out overconfidence shrinks with the gap between
  training and held-out residuals. ~1,000 GPU-h for one split design.
- [ ] **P13 One factorial split design** (NeurIPS NA6): train on low-energy configurations of seen
  systems, test the four cells seen / unseen system x low / high energy. Needs a full pretraining
  campaign, so consider it only together with P4 (decision D6).
- [ ] **P9 Water on aligned splits**: rebuild the liquid-water study with one configuration-level
  split shared by both fidelities (`docs/known-issues.md` item 1; new dataset, ~150 small
  members). A second, periodic system whose earlier behaviour (no AU overconfidence when test and
  training distributions coincide) would support the mechanism. Small data: 50 test configurations.
- [ ] **P10 Foundation-model demonstration**: fine-tune a public MACE foundation model for organic
  molecules to CC labels with an added variance head, shared parent by construction (P1 in the
  wild). One figure. Engineering risk: checkpoint compatibility with the fork.
- [ ] **P11 Fidelity pair with forces at both levels**: `wb97x_dz` -> `wb97x_tz` (energies and
  forces for 4.6M configurations in the raw file). The only route to *force* uncertainty across a
  change of fidelity with ground truth and no new labels; the gap is a basis set, not a method.

### Writing (no compute)

- [ ] **W1** Restructure around the thesis: outline in Section 7; mechanism to the main text.
- [ ] **W2** Rename the two variance terms (D3); state calibration contrasts per split.
- [ ] **W3** Related work on fine-tuned and foundation-model UQ from `LITERATURE.md` Section 4,
  after reading each full text (move entries to Section 1 first).
- [ ] **W4** Ensemble-size figure for the existing M = 1..10 result (ledger 2026-11-02), still
  text-only in App. M.4.

## 4. Order and gates

| When | What | Gate at the end |
|---|---|---|
| to 2026-11-18 | ICLR discussion: replies only (`../iclr27/rebuttal/README.md`) | — |
| 2026-11-18 to ~12-15 | CPU: A1, A2. DGX: P2.0, then P8a, P1, P2a-c, P3, P8b. Leonardo: P4 dataset variants, loss, smoke test, pilot. In parallel: choose and wire the second architecture (P5) | **Gate 1**: (i) does disagreement keep its ranking with a shared parent? (ii) does re-estimating the scale restore calibration where predicted? (iii) does the picture hold on fine-tuning geometries the parents never saw, and at matched RMSE? (iv) do force-trained parents change it (pilot)? Freeze the thesis, or rewrite it around what was found |
| ~12-15 to ~2027-01-31 | P4 on five splits, P5, P6 | **Gate 2** (around the ICLR decision): venue and scope (Section 6) |
| 2027-02 to 2027-03 | W1-W4; Tier 2 only if chosen at Gate 2 | submission-ready draft |
| before 2027-04-29 | all Leonardo work finished and synced back (allocation ends) | — |

If a gate result contradicts the thesis, that result becomes the paper's content: do not tune
protocols until the expected outcome appears.

## 5. Declined on purpose

- More learning-rate sweeps, model-size sweeps (ICLR G4 beyond what Appendix M has).
- More estimators (evidential, Bayesian last layer, shallow ensembles, conformal): cite them;
  the MSE ensemble and the single model already isolate the estimator.
- Materials datasets and PBE -> r2SCAN (MatPES and similar): a different paper.
- A ladder of fidelities (HF, MP2 as low fidelity): the energies are in the raw file, but each
  rung costs a full pretraining campaign.
- Other scientific domains (NeurIPS NM1): answered by scoping the claims to atomistic models and
  by the choice of venue, not by experiments.
- Per-request experiments that do not serve one of the four recurring concerns of `REVIEWS.md`
  Section 3.

## 6. Venues and calendar

Facts to verify on the official pages before relying on them: ICML 2027 deadline (a third-party
site projects late January 2027; the official call was not posted on 2026-11-12); NeurIPS 2027
deadline (May in recent years); the ICLR 2027 decision date. A paper under review at ICLR cannot
be submitted elsewhere until the decision or a withdrawal.

| Route | Fit | Needs | Comment |
|---|---|---|---|
| **A. General conference** (NeurIPS 2027, in the physical-sciences area; ICML 2027 is too early) | only with claims scoped to atomistic models and with breadth | everything in Tier 1 including P5, and P7; a second dataset beyond ANI-1x would still be asked (5 of 9 reviewers and the area chair raised scope) | two panels (2 / 3 / 2 and 1 / 2 / 2) gave the same verdict on scope and novelty; a third attempt needs more than the plan's core |
| **B. Domain journal** (npj Computational Materials, J. Chem. Theory Comput., Digital Discovery, Mach. Learn.: Sci. Technol.) | the closest related work is published there (tan2023single, bilbrey2025uncertainty, ko2025data, radova2025fine; Liu 2026; Schäfer 2026 per a listing); reviewers of this readership rated the workshop version 3 / 4 / 4 of 4 | **P4 on five splits** (forces: the one thing they asked, RE1), A1, A2, P1, P2, P8; P5 welcome, not required | no page limit: Appendix M and the rebuttal baselines fit as sections; no calendar |
| **C. TMLR** | judged on correctness and interest to some audience, not on novelty of method; certifications allow presentation at the main conferences | Tier 1a + honest scope | fallback for route A with the same manuscript |

Recommendation (changed in version 2): **route B is the default.** The reviews of three venues
agree on where the work is valued and on what general-conference reviewers find missing, and what
they find missing (more datasets, more domains, a new method) is not what this plan can deliver at
reasonable cost. Build Tier 1 and submit the comprehensive study with forces to a domain journal
when it is ready, without waiting for a conference calendar. Reconsider route A at Gate 2 only if
all of these hold: the intervention (P2) works, the findings survive the shared parent (P1) and
the unseen geometries (P8a), and they replicate on the second architecture (P5); then C is its
fallback with the same manuscript. Do not rush ICML 2027.

## 7. Outline of the next version (for W1)

1. Introduction: the fine-tuning workflow, the thesis, what is new.
2. Setup: data, shifts, protocols (with forces in the LF stage if P4 holds), estimator, statistics.
3. The ordering transfers: pretrain-only ranking, the gain after fine-tuning, mean-variance vs MSE
   ensembles, the re-initialized head.
4. The scale does not: collapse in the high-fidelity stage, training vs held-out and unseen
   geometries (P8a), member-level calibration (A2), the inherited head, energy extrapolation;
   resolved dynamics and checkpoint selection (P3); comparison at matched accuracy (A1).
5. Re-estimating the scale (P2): what restores calibration, where it cannot.
6. Does it hold elsewhere: shared parent (P1), second architecture (P5), multi-head replay (P6).
7. Consequence: acquisition (the existing experiment; several rounds only if P7 is run).
8. Limits. Appendix: the present Appendices A-M.

## 8. Decisions for the user

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Approve this plan and its tiers | approve Tier 1; Tier 2 at Gate 1 |
| D2 | Thesis-led restructure of the paper | yes |
| D3 | Rename AU / EU to operational names | **yes, firmly**: raised at two venues by three reviewers and an area chair (aleatoric / epistemic kept as the interpretation under test) |
| D4 | Second architecture: which model | ANI-style network first (cheapest, historically the DFT -> CC transfer model), PaiNN if a message-passing model is preferred |
| D5 | P1 deviates from "member (s, m) from parent (s, m)": allow it as a new, explicitly named variant | yes |
| D6 | P4: both existing split designs, one of them, or a single factorial design (P13) for the force-trained campaign | pilot both existing designs on split 0 (direct comparison with the energy-only runs), then decide by cost and by whether the conclusions move; the factorial design is cleaner and needs one campaign instead of two, but breaks comparability with the 27 existing experiments |
| D7 | Venue route (A, B or C) | B by default; A only under the conditions of Section 6 |
| D8 | One paper or two (a short method-oriented paper on the thesis + a domain paper with forces and acquisition) | one paper; reconsider only if Gate 2 leaves two self-standing stories |
| D9 | ~~Provide the NeurIPS 2026 and RPS 2026 reviews verbatim~~ | done 2026-11-12 (`reviews_neurips26.md`, `reviews_rps26.md`) |

## 9. Budget

Leonardo `IscrC_UQ-MF`: 80,000 local h, about 73,000 left on 2026-11-08 (~9,100 GPU-h;
`../iclr27/rebuttal/RUNS.md` Section 4), ends 2027-04-29; exceeding the monthly share lowers
priority, it does not stop jobs (`docs/leonardo.md`). DGX: at most 3 GPUs.

| Package | Training (GPU-h, estimate) | Where |
|---|---|---|
| A1 + A2 | none (CPU) | DGX |
| P1 + P2 + P3 | ~400 | DGX |
| P8 (unseen geometries, backbone-only) | ~250 | DGX |
| P4 pilot (split 0, both designs) | ~650-1,300 | Leonardo |
| P4 five splits, per design | ~1,600-3,200 | Leonardo |
| P5 | small (model-dependent) | DGX |
| P6 | ~100-200 | DGX |
| P7, per split design (Tier 2) | ~480 | DGX / Leonardo |
| Evaluation with per-epoch dynamics | ~50 per experiment (about 10 h per split in the rebuttal runs) | where the checkpoints are |

Tier 1 with P4 on both designs fits in the remaining allocation only if the per-step cost of forces
is at the low end; measure it first (D6).

## 10. Log

- 2026-11-12 — plan written with the assessment and the cross-venue review synthesis (this
  folder); nothing launched; waiting for D1-D9.
- 2026-11-12 — **version 2** after the user supplied the NeurIPS and RPS reviews. The first version
  assumed NeurIPS had asked for actionability; the texts show a rejection on validity (fixed by the
  September rebuild) plus the same scope and novelty concerns as ICLR. Changes: A1 (matched RMSE,
  NM6) and A2 (observable target for the learned variance, NA1) added as CPU analyses; P8 added
  (unseen fine-tuning geometries, NA5; backbone-only transfer, NA12); P3 named as the answer on
  the use of the dynamics (NY2); multi-round acquisition (P7) moved to Tier 2 because no reviewer
  asked for it; data scale renumbered P12; factorial split added as P13 / D6 (NA6); D3 (rename)
  made firm; default venue changed from "decide at the gate" to a domain journal; D9 closed.
