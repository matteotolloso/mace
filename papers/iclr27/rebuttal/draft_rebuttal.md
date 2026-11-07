# ICLR 2027 rebuttal: working draft (Submission 18667)

Status: **first draft, 2026-11-06.** Not for posting yet. Plan and task status: `PLAN.md`; point
IDs: `reviews.md`.

Conventions for every agent editing this file:
- Numbers come only from `results/results.md` (entry named in brackets, e.g. `[ledger 2026-11-03]`)
  or from the submitted paper (`[paper Sec. X]`). Unrecorded results are `[PENDING: WPx.y]`.
- Every statement about prior work carries a citation key from Section 7, which records how the
  reference was checked. Do not add a reference you have not checked.
- Tone: polite, concrete, no "the reviewer is wrong". Concede what is true, then show evidence.
- Sentences such as "we added", "we rewrote", "fixed in the revised PDF" describe *planned*
  revisions; before posting, keep each one only if the change exists (WP9/WP10.4 check).
- OpenReview comments have a character limit (fill in from `PLAN.md` Section 1); the per-reviewer
  texts below are deliberately long: trim at assembly time (WP10).

---

## 0. Strategy

1. **Reframe the core objection.** Two reviewers (ge6X G1/G5, Q2LT Q3) read the paper as "known
   facts in a niche setting". The answer is that (i) the setting is the *standard* way coupled-cluster
   MLIPs are built (Section 2.1), and (ii) what is new is not "ranking != calibration" in general but
   *where and when* it happens along the LF->HF transition, and that pretraining changes the
   decomposition (EU becomes a competitive ranking signal) and acquisition (Section 1 table).
2. **Turn the ablation requests into evidence.** Q2LT and ge6X ask for ablations we can run cheaply
   because fine-tuning costs ~0.5 h per member: variance-head reset, beta-NLL, held-out-fitted head,
   learning rates, LF-only baseline, MSE ensembles (HF-only already done), multi-head fine-tuning.
3. **Concede the scope limits honestly** (one architecture, energies) but show they are the realistic
   regime: CC data are energy-only, and ANI-1x is *the* public DFT/CC dataset.
4. **Rename, do not abandon, the aleatoric term** (G5): defend it with the representation-relative
   view of aleatoric uncertainty and with our own ablation showing it is the most useful part of
   the predictive variance.

---

## 1. Novelty table (for Q3, G1; also usable in the revision)

| Finding in the paper | Closest known general result | What is new here (MF-specific) |
|---|---|---|
| Ranking and calibration can disagree | Metrics capture different properties (rasmussen2023uncertain, lind2024uncertainty); calibration vs sharpness (kuleshov2018accurate) | They move in **opposite directions along the energy-extrapolation axis**, and pretraining *amplifies* the split (TU ENCE Energy-OOD 1.637 HF-only vs 2.969 LF->HF while Spearman 0.556 vs 0.640) `[ledger 2026-09-23]`; in-domain isotonic recalibration does not repair it `[paper App. "Post-hoc isotonic recalibration"]` |
| AU unreliable under distribution shift | valdenegro2022deeper | AU also collapses **in domain**, and the collapse is located **in the HF stage** and accelerated by pretraining (OOD AU ENCE, end of pretraining -> end of FT: system 0.993 -> 81.913, energy 10.863 -> 207.071) `[ledger 2026-09-23]`; on the training set AU is never overconfident (train ENCE below 1, mean AU above the mean squared training error: conservative, not "calibrated") `[ledger 2026-09-23, interpretation checks]` |
| NLL variance heads are fragile | nix1994estimating, skafte2019reliable, seitzer2022pitfalls, stirn2023faithful, sluijterman2024optimal | These works study single-task training; we show that the failure emerges during HF optimization in both protocols, is accelerated by pretraining, and quantify it per epoch; whether the target switch itself contributes is what the ablations test `[PENDING: WP3]` |
| Overconfidence of MACE ensemble UQ | bilbrey2025uncertainty: readout ensembles of MACE-MP-0 and fine-tuned models are overconfident by orders of magnitude; quantile readouts are not | Bilbrey et al. retrain only readouts of a foundation model (ensemble and quantile model separately) and do not track AU/EU of a deep ensemble across an LF->HF transition or through training; we do, and find that EU, not AU, carries the improvement after pretraining |
| Pretraining changes the decomposition | - (not found in prior work) | HF-only: EU is the weakest ranking signal on every test set; LF->HF: EU ranks as well as or better than AU on 3 of 4 `[paper Sec. 4.1]` |
| Pretrained UQ is a better acquisition function | QBC with ensemble disagreement (smith2018less), committee AL (schran2020committee) | Despite starting more accurate, the pretrained ensemble benefits more from uncertainty-guided CC acquisition: gain over random 2.316 / 2.497 / 2.347 vs 0.807 / 1.603 / 1.140 meV/atom (AU/EU/TU) `[ledger 2026-09-23]` |

---

## 2. Response to Reviewer ge6X

We thank the reviewer for the careful reading and for finding the paper clear, the data splits
sensible and the acquisition experiment useful. We address each point; new results are in the
revised PDF.

### 2.1 G1: "Using high-fidelity data for fine-tuning is not common"

Low-fidelity pretraining followed by high-fidelity fine-tuning is, to our knowledge, the most common
way to obtain coupled-cluster-quality MLIPs, precisely because CC labels are too expensive to
generate at scale:
- The dataset we use was built for this purpose: ANI-1ccx was obtained by training on ~5M DFT
  energies of ANI-1x and then transfer-learning to ~500k CCSD(T)/CBS energies (smith2019approaching).
- The same strategy is used for small molecules (MP2 -> CCSD(T): kaser2021transfer;
  HF/MP2 -> CCSD(T), sub-microsecond MD: kaser2023transfer), for organic molecules across levels
  (zaverkin2023transfer), for liquid water with <=200 CC energies (chen2023data), for materials
  (kim2024data, ko2025data) and for semi-empirical -> DFT -> CC chains (cui2025multi).
- It is also how atomistic foundation models are adapted: "Adapting MLIP foundation models to
  specialised tasks through fine-tuning is an increasingly important practice" (tompa2026fine);
  see also radova2025fine, and multi-fidelity training studies (gardner2025understanding,
  messerly2025multi).

In all of these, the fine-tuned model is then used in active learning or simulation, where
uncertainty decides which configurations to label next or when to stop trusting the model
(ANI-1x itself was assembled by query-by-committee active learning, smith2018less). How the
uncertainty behaves across the LF->HF transition is therefore a practical question, and to our
knowledge no prior work has studied it (Section 1 table; we added this table to the revision).

On "findings not unexpected": some findings agree with general knowledge (we cite it), but several
are, to our knowledge, not reported before and not obvious a priori: pretraining *improves* ranking
while *worsening* Energy-OOD calibration; the calibration loss occurs in the HF stage of both
protocols and is accelerated by pretraining; pretraining turns EU from the weakest into a
competitive ranking signal; and the more accurate pretrained ensemble benefits *more* from
uncertainty-guided acquisition. See the novelty table (Section 1).

### 2.2 G2: one model (MACE), one dataset

We agree this is a limitation and state it (App. limitations). Our reasons:
- ANI-1x/ANI-1ccx is, to our knowledge, the only large public dataset with CC labels computed on a
  subset of the DFT geometries over many molecular systems, which is what makes the two controlled
  shifts (unseen systems, higher energies) possible [check alternatives: WP7.1].
- MACE is a widely used equivariant MLIP architecture (batatia2022mace, batatia2025design), and it is
  the backbone of MACE-MP-0, the model studied by bilbrey2025uncertainty.
- Within this setting we vary what reviewers asked about: the estimator (ten-member MVE ensemble,
  single MVE model, MSE ensemble without a variance head `[ledger 2026-11-02, 2026-11-03, 2026-11-04]`),
  the objective (beta-NLL, `[PENDING: WP3.3]`), the multi-fidelity strategy (multi-head replay
  fine-tuning `[PENDING: WP6.2]`) and the optimization (`[PENDING: WP4]`).
- [Optional, decision D4: second dataset (liquid water, chen2023data) `[PENDING: WP7.2]`.]

### 2.3 G3: energies only, no forces

The high-fidelity stage is energy-only by necessity, not by choice: the CCSD(T)/CBS labels in ANI-1x
are energies only (the CC fields of the data release contain no forces), and ANI-1ccx was trained on
these CC energies (smith2019approaching). This is the typical situation: "most high-accuracy datasets
are much smaller and often do not contain atomic forces" (messerly2025multi), which is why
multi-fidelity schemes that combine low-level forces with high-level energies are studied
(messerly2025multi). Uncertainty on energies is also what committee-based acquisition uses to build
datasets (smith2018less).

We agree that DFT forces could strengthen pretraining; gardner2025understanding report that the
success of pretraining depends "critically" on force labels. We used energies at both fidelities to
keep the two stages comparable and the uncertainty a single scalar per configuration.
[If D3 = run: we added LF pretraining with DFT forces followed by the same CC fine-tuning
`[PENDING: WP5.2]`.] [Otherwise: we state it as the most important extension and keep the claim
scoped to energy prediction, as in the current conclusion.]

### 2.4 G4: protocol, compute matching, model size, learning rates, pretrain-only

- **Compute matching.** More HF compute does not help HF-only: with validation-based checkpoint
  selection, HF-only members are selected after 30-111 of 300 epochs (median 60 system, 67 energy)
  `[paper App. "Training details and hyperparameters"]`; their validation loss then rises. Matching the step count of LF->HF (300 LF epochs
  on 10x more data) would therefore only add steps after the selected checkpoint. The extra compute of
  LF->HF is spent on cheap LF labels, which is the purpose of multi-fidelity training; the
  comparison is one of data access, as stated in our limitations. Counting optimizer steps up to
  the evaluated checkpoints `[ledger 2026-11-07, results/tables/training_steps.csv]`: on CC labels the
  fine-tuned members have taken a median of 234 steps (2-3 epochs), HF-only members about 20 times more
  (4758 system, 5304 energy, of 23400 available). LF->HF thus uses *less* compute on the scarce labels;
  its extra ~1.1-1.5 x 10^5 steps are on the 10x larger DFT set.
- **Learning rates.** We added sweeps for HF-only (3e-3, 1e-3) and for fine-tuning (1e-4, 1e-2)
  `[PENDING: WP4.2, WP4.3]`. [Write the conclusion once results exist: does the
  ranking-improves/calibration-degrades pattern hold at each protocol's best LR?]
- **Model size.** `[PENDING: WP4.4, optional]`.
- **Pretrain-only (LF-only).** See G7.

### 2.5 G5: "aleatoric" with deterministic labels

We agree that reference energies of a fixed electronic-structure method are deterministic, and the
paper says so (Sec. 2.1, App. "Technical background"). Our use of the term is operational, and we will make this explicit
by renaming the term in the revision [decision D5: e.g. "learned variance (AU)"]. We nevertheless think
the quantity is meaningful and worth reporting, for four reasons:

1. *Aleatoric uncertainty is defined relative to the model's input representation, not to the
   physical process.* hullermeier2021aleatoric stress that the aleatoric/epistemic split is
   context-dependent: it depends on the instance space and hypothesis space, and changing the
   representation can turn aleatoric into epistemic uncertainty and vice versa. A MACE model sees a
   configuration only through local environments within a finite cutoff (6 Å here, App. "Training details"); atomic
   descriptors of finite body order are known to be incomplete, i.e. distinct structures can share
   the same representation (pozdnyakov2020incompleteness). Conditional on the model's
   representation, the target is therefore not a deterministic function, and a non-zero conditional
   variance is well defined.
2. *It is standard practice in atomistic ML.* Mean-variance heads are routinely trained on
   deterministic DFT labels and reported as aleatoric uncertainty (busk2022calibrated on QM9/PC9;
   busk2023graph; tan2023single).
3. *It is the most useful part of the predictive variance in our own data.* Removing the variance
   head (an MSE-trained ten-member deep ensemble whose only signal is member disagreement) gives a
   significantly worse uncertainty than the MVE ensemble's TU on both splits and both test sets,
   e.g. Energy-OOD Spearman difference +0.1179 [0.09477, 0.141] and ENCE ratio x0.1246
   [0.09637, 0.1611] in favour of MVE TU `[ledger 2026-11-03]`; System-OOD +0.04945
   [0.02668, 0.07222] and x0.2649 [0.207, 0.339] `[ledger 2026-11-04]`; most of the gain comes from AU.
   (HF-only protocol; LF->HF `[PENDING: WP3.6]`, energy split in training.)
4. *The reviewer's reading is in fact one of our results.* If labels are deterministic, all error is
   in principle reducible (the excess-risk view of epistemic uncertainty, lahlou2023deup). Our
   training dynamics show exactly this: the learned variance behaves as a reducible estimate of
   in-sample residual difficulty, shrinking with the training error (Sec. 4.2). This is why we
   evaluate calibration on TU only and use AU/EU calibration as a diagnostic (App. "Technical background", the ambiguity
   decomposition, krogh1994neural).

### 2.6 G6: white band in Fig. 1

Thank you: the figure contains nested transparency masks that some viewers (e.g. Safari) render
incorrectly. It is fixed in the revised PDF.

### 2.7 G7: LF-only results

LF-only models predict DFT energies, whose absolute scale differs from CC/CBS energies by large
per-element offsets, so a direct comparison on CC labels would be dominated by that offset. We added
the fair comparison (new appendix table): the converged LF-only ensembles, unchanged except for a
per-member least-squares refit of the per-element reference energies to their residuals on the CC
training set (the analogue of the reference-energy refit with which fine-tuning starts; the most
favourable zero-shot use of the LF model), scored on the CC test sets, with the variance heads
unchanged. Paired over the five splits `[ledger 2026-11-07, results/tables/lf_only_vs_hf.csv]`:

- **LF->HF fine-tuning is better on every uncertainty metric on all four tests**: TU Spearman
  +0.06 to +0.17 and AUSE -0.06 to -0.09 (all intervals exclude 0), TU ENCE x0.20-0.23 except on
  Energy-OOD (interval contains 1). In distribution it also halves the error (RMSE x0.50 Energy-ID,
  x0.67 System-ID); out of distribution accuracy is unchanged or marginally worse (Energy-OOD x1.03,
  System-OOD interval contains 1).
- **Against HF-only**, LF-only is more accurate on Energy-OOD (HF-only RMSE x1.32) and ranks
  in-distribution errors better, but its variances, learned on DFT residuals, are badly calibrated for
  CC errors (TU ENCE 1.5-3.2; HF-only x0.24-0.50 of that), and it ranks Energy-OOD errors worse.

So the pretrained model alone is not a usable CC-level predictor with uncertainty: fine-tuning on the
scarce CC labels is what makes both its errors and its variances fit the target, which is the
transition our paper studies.

---

## 3. Response to Reviewer j1p7

We thank the reviewer for calling the study timely and the validation thorough.

### 3.1 J1: one model, two datasets

We read the "two datasets" as the DFT and the CC label sets (as in J2). They are two fidelities of
one dataset: the CC/CBS energies of ANI-1x are computed on a subset of its DFT geometries
(smith2019approaching), which is what makes a controlled LF->HF study possible (same configurations,
different labels) and the two distribution shifts (unseen molecular systems; higher-energy
configurations) well defined. Why this dataset and this architecture: see our answer to ge6X G2
(Section 2.2), including the estimator and strategy variants we added. [Note (mace-leo): "computed on
a subset of the DFT geometries" is in Section 2.2 and the paper; keep the citation that the authors
use for it.]

### 3.2 J2: is LF pretraining + HF fine-tuning the right approach? fidelity as input

- *Why sequential pretraining/fine-tuning.* It is the protocol of ANI-1ccx (smith2019approaching) and
  of most CC-accuracy MLIPs (Section 2.1). In a systematic comparison of pretraining/fine-tuning and
  multi-headed training, gardner2025understanding find that shared multi-head representations
  "marginally reduce model performance in most cases" relative to pretraining/fine-tuning;
  tompa2026fine find that naive fine-tuning "offers the best convergence for single-system
  applications", whereas multihead replay best preserves out-of-distribution robustness. Since our
  target is a single high fidelity, sequential fine-tuning is the natural baseline.
- *Systematic DFT/CC bias.* Much of it is a per-element offset, which the fine-tuned model absorbs by
  refitting the atomic reference energies on the CC training set (App. "Training details"); the rest is learned when
  the backbone is adapted, which gardner2025understanding show is necessary because pretrained
  representations are method-specific.
- *Fidelity as input.* We agree this is a promising alternative: fidelity encodings are used in
  multi-fidelity graph networks (chen2021learning) and MLIPs (kim2024data, one-hot fidelity), and
  multi-head models with fidelity-specific readouts are their MLIP counterpart
  (gardner2025understanding; MACE multihead replay fine-tuning, tompa2026fine). We added a
  multi-head variant (shared backbone, DFT and CC heads, both mean-variance) fine-tuned from the same
  LF parents, and compare its AU/EU/TU `[PENDING: WP6.2, decision D2]`. [Whether it is "better
  calibrated overall" is the reviewer's hypothesis: report it either way.]

### 3.3 J3: better arguments for model, data and training choices

We rewrote the limitations as "Design choices and scope" (revised App.): MACE (widely used equivariant
MLIP; foundation-model backbone), mean-variance deep ensembles (the estimator that defines AU, EU and
TU through the law of total variance, lakshminarayanan2017simple; strongest across tasks in
tan2023single), ANI-1x (CC labels on DFT geometries, controlled shifts), energies (CC labels have no
forces), sequential fine-tuning (standard; Section 3.2), and what each choice leaves untested.

---

## 4. Response to Reviewer Q2LT

We thank the reviewer for the detailed summary and for the concrete suggestions, which we followed.

### 4.1 Q1 / Q4: validating the mechanism of AU overconfidence

Our explanation predicts four things, and we test each:
(a) on held-out data, AU should track the *training* error rather than the held-out error;
(b) re-initializing the variance head at the start of fine-tuning should *not* prevent the collapse,
    because the new head is fitted to the same shrinking training residuals;
(c) slower fine-tuning should delay the collapse in proportion to the decrease of the training error;
(d) a variance head fitted to *held-out* residuals should not collapse.

[Caution (mace-leo, 2026-11-07): a first per-epoch look with the stored *mean* AU and *mean* squared
errors did not show held-out AU tracking the training MSE in D (means are dominated by heavy tails);
until WP3.1 is redone with robust per-configuration statistics, say "is consistent with (a)" rather
than "supports (a)", or drop the sentence. See PLAN.md WP3.1 note.]
Existing evidence already supports (a): on the training set AU is never overconfident (train ENCE stays
below one and the mean AU exceeds the mean squared training error `[ledger 2026-09-23, interpretation checks]`),
while on held-out data it rises by orders of magnitude during
fine-tuning (App. "Training dynamics"; OOD AU ENCE end of pretraining -> end of FT: system 0.993 -> 81.913, energy
10.863 -> 207.071 `[ledger 2026-09-23]`); the training error keeps falling while held-out error barely
moves (App. "Training dynamics", training-curve figure). New experiments:
- (a) quantitative: AU vs training/held-out MSE per epoch, all protocols `[PENDING: WP3.1]`;
- (b) **variance-head reset** at FT start, 5 splits x 10 members, both shifts `[PENDING: WP3.2]`;
- **alternative objective**: beta-NLL (seitzer2022pitfalls) fine-tuning `[PENDING: WP3.3]`, and
  MSE-trained ensembles (no variance head), already done for HF-only `[ledger 2026-11-03, 2026-11-04]`,
  LF->HF `[PENDING: WP3.6]`;
- (c) fine-tuning at lr 1e-4 `[PENDING: WP3.5]`;
- (d) head fitted on held-out CC data with frozen backbone `[PENDING: WP3.4]`.

[Write the conclusion after results. If (b) shows the collapse re-emerges and (d) removes it, the
mechanism is confirmed; if the reset fixes it, the cause is the inherited head state, and we will say so.]

### 4.2 Q2 / Q5: generality across estimators and multi-fidelity strategies

- *Estimators.* Besides the ten-member MVE ensemble we now report a single MVE model (M = 1) and the
  ensemble-size curve M = 1..10 `[ledger 2026-11-02]`, and MSE deep ensembles `[ledger 2026-11-03,
  2026-11-04]`. Key results: one MVE model's AU already ranks as well as or better than a ten-member
  MSE ensemble on OOD (system: Spearman difference +0.02553 [0.01709, 0.03398]; energy: -0.01237
  [-0.02983, 0.005095], interval contains 0) and is better calibrated (ENCE ratio x0.5149 and
  x0.5312) `[ledger 2026-11-03, 2026-11-04]`; [LF->HF comparison PENDING].
- *Multi-fidelity strategies.* Multi-head replay fine-tuning `[PENDING: WP6.2]`; expected differences
  discussed with gardner2025understanding and tompa2026fine (Section 3.2). Joint training from
  scratch and alternating optimization are discussed but not run [unless decided otherwise].
- *Architecture.* See ge6X G2.

### 4.3 Q3: novelty and Bilbrey et al. (2025)

We added the novelty table (Section 1) to the revision and the related-work paragraph below.
Bilbrey et al. (bilbrey2025uncertainty) quantify the uncertainty of MACE-MP-0 and of versions
fine-tuned to new systems (high-entropy alloys; water in the zeolite H-ZSM-5) with two separate models:
a *readout ensemble* (seven copies whose readout layers alone are retrained with a Huber loss, the
interaction layers frozen) and a *quantile-regression* readout (5th and 95th percentiles). They find that
"the ensemble is overconfident in its predictions and, though ensemble uncertainty tends to increase with
error, the magnitude of uncertainty is lower than the error by orders of magnitude" (coverage of the
90% interval 11-64% for the ensemble vs 86-88% for the quantile model), and interpret the quantile width
as aleatoric and the ensemble spread as mostly epistemic uncertainty. This is consistent with what we
see for disagreement-only uncertainty (our MSE-ensemble ablation, Section 2.5). Our study differs in
three respects: (i) full deep ensembles of mean-variance models, which give AU, EU and TU of one
predictive distribution through the law of total variance (rather than two separately trained
estimators); (ii) a controlled LF->HF transition (DFT -> CC on the same geometries) with an HF-only
control and a pretrain-only baseline, whereas their transfer is to new systems without a
from-scratch comparison; (iii) the evolution of ranking and calibration through training, which
localizes the calibration loss to the HF stage. We added this paragraph to the related work.

---

## 5. General response (to all reviewers and the AC), skeleton

1. Thanks; all reviewers find the paper clear; j1p7 finds no soundness concerns.
2. Motivation: LF->HF fine-tuning is the standard route to CC accuracy (Section 2.1, 6 citations).
3. New experiments (list only those completed): LF-only on CC; variance-head reset; beta-NLL;
   held-out head; LR sweeps; MSE ensembles; single model / ensemble size; multi-head FT; [forces].
4. Revisions: novelty table, renamed AU, design-choices section, Fig. 1 fix, related work.

---

## 6. Theory notes for agents (background, not for posting verbatim)

- **Law of total variance and the ensemble.** TU = AU + EU is the variance of the mixture of the
  members' Gaussians (lakshminarayanan2017simple, depeweg2018decomposition); TU is the only signal
  the observed error can calibrate (paper App. "Technical background").
- **Ambiguity decomposition** (krogh1994neural): e^2 = mean_m e_m^2 - EU per configuration, so
  TU - e^2 = (AU - mean_m e_m^2) + 2 EU; TU can only be overconfident where the heads underestimate
  their own members' errors by more than 2 EU (paper App. "Technical background"). Use this against "AU is meaningless":
  the heads have an observable target, each member's own squared error.
- **Why NLL heads collapse on held-out data.** For fixed mean, the NLL optimum is the conditional
  mean squared residual (nix1994estimating); fitted on training data it inherits training residual
  scale (skafte2019reliable, seitzer2022pitfalls, stirn2023faithful); sluijterman2024optimal recommend
  a warm-up of the mean with fixed variance (an extra ablation option: MSE warm-up then NLL).
- **Multi-fidelity as structured discrepancy.** The classic autoregressive model writes
  f_HF(x) = rho f_LF(x) + delta(x) (kennedy2000predicting); pretraining learns f_LF, fine-tuning
  learns rho and delta from scarce HF data, so the HF-stage uncertainty is about delta, which is
  smaller and smoother than f_HF. This motivates why EU becomes informative after pretraining
  (hypothesis; do not claim as shown).
- **Excess risk.** lahlou2023deup define epistemic uncertainty as excess risk, which includes model
  misspecification; with deterministic labels all error is excess risk. Be careful: this reference
  supports the reviewer's semantic point; we use it to *agree* and to frame our AU finding.

---

## 7. References (key, citation, verification)

`bib` = already in `ICLR27-UQ-MF/iclr2027_conference.bib` (used in the submitted paper).
`web 2026-11-06` = checked against the publisher/arXiv page on that date (title, authors, venue;
quoted phrases copied from the abstract). `memory` = not yet checked: verify before use.

| Key | Reference | Check |
|---|---|---|
| smith2019approaching | Smith et al., Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning, Nat. Commun. 10, 2903 (2019) | bib; web 2026-11-06 (DFT training then transfer learning to CCSD(T)/CBS; ~5M DFT, ~500k CC) |
| smith2020ani | Smith et al., The ANI-1ccx and ANI-1x data sets, Sci. Data (2020) | bib |
| smith2018less | Smith, Nebgen, Lubbers, Isayev, Roitberg, Less is more: sampling chemical space with active learning, J. Chem. Phys. 148, 241733 (2018) | web 2026-11-06 (QBC with ensemble disagreement built ANI-1x) |
| kaser2021transfer | Käser, Boittier, Upadhyay, Meuwly, Transfer learning to CCSD(T): accurate anharmonic frequencies from machine learning models, J. Chem. Theory Comput. 17(6) (2021) | web 2026-11-06 |
| kaser2023transfer | Käser, Meuwly, Transfer-learned potential energy surfaces: towards microsecond-scale molecular dynamics simulations in the gas phase at CCSD(T) quality, J. Chem. Phys. (2023), arXiv:2303.11685 | web 2026-11-06 (volume/pages to add) |
| zaverkin2023transfer | Zaverkin et al., Transfer learning for chemically accurate interatomic neural network potentials, PCCP (2023) | bib |
| chen2023data | Chen et al., Data-efficient machine learning potentials from transfer learning of periodic correlated electronic structure methods: liquid water at AFQMC, CCSD, and CCSD(T) accuracy, JCTC (2023) | bib; web 2026-11-06 (<=200 CC energies) |
| kim2024data | Kim et al., Data-efficient multifidelity training for high-fidelity machine learning interatomic potentials, JACS 147, 1042-1054 (2024/2025) | bib; web 2026-11-06 (SevenNet-MF, one-hot fidelity encoding, simultaneous training) |
| ko2025data | Ko et al., Data-efficient construction of high-fidelity graph deep learning interatomic potentials, npj Comput. Mater. (2025) | bib |
| cui2025multi | Cui, Reuter, Margraf, Multi-fidelity transfer learning for quantum chemical data using a robust density functional tight binding baseline, Mach. Learn.: Sci. Technol. 6, 015071 (2025) | web 2026-11-06 |
| messerly2025multi | Messerly et al., Multi-fidelity learning for interatomic potentials: low-level forces and high-level energies are all you need, Mach. Learn.: Sci. Technol. (2025), arXiv:2505.01590 | web 2026-11-06 (quote "most high-accuracy datasets ... often do not contain atomic forces") |
| gardner2025understanding | Gardner, Schulz, Helie, Sun, Simm, Understanding multi-fidelity training of machine-learned force-fields, arXiv:2506.14963 (v2 2026) | bib; web 2026-11-06 (quotes "critically, the inclusion of force labels"; "marginally reduce model performance in most cases") |
| tompa2026fine | Tompa, Varga-Umbrich, Batatia, Elena, Bernstein, Csányi, Fine-tuning MLIP foundation models: strategies for accuracy and transferability, arXiv:2606.12704 (2026) | web 2026-11-06 (quotes "increasingly important practice"; naive FT "best convergence for single-system applications"; multihead replay preserves OOD robustness) |
| radova2025fine | Radova et al., Fine-tuning foundation models of materials interatomic potentials with frozen transfer learning, npj Comput. Mater. (2025) | bib |
| bilbrey2025uncertainty | Bilbrey, Firoz, Lee, Choudhury, Uncertainty quantification for neural network potential foundation models, npj Comput. Mater. 11, 109 (2025), doi:10.1038/s41524-025-01572-y | full text read 2026-11-07 (mace-leo, publisher PDF): 7-member readout ensemble (interaction layers frozen, Huber loss, different 80k MPtrj subsets; fine-tuned ensembles differ by seed only) + 5th/95th-percentile quantile readouts; MPtrj, HEA25, H-ZSM-5 + water (n = 8, 16 held out); quote "the ensemble is overconfident in its predictions ... lower than the error by orders of magnitude" (Introduction); Table 1 coverage ensemble 11/21/64%, quantile 87/88/86% |
| chen2021learning | Chen, Zuo, Ye, Li, Ong, Learning properties of ordered and disordered materials from multi-fidelity data, Nat. Comput. Sci. 1, 46-53 (2021) | web 2026-11-06 |
| shiota2024taming | Shiota et al., Taming multi-domain, -fidelity data: towards foundation models for atomistic scale simulations, arXiv:2412.13088 | web 2026-11-06 (total energy alignment across fidelities; optional) |
| hullermeier2021aleatoric | Hüllermeier, Waegeman, Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods, Mach. Learn. 110, 457-506 (2021) | bib; web 2026-11-06 (context-dependence: changing the representation turns aleatoric into epistemic and vice versa) |
| pozdnyakov2020incompleteness | Pozdnyakov et al., Incompleteness of atomic structure representations, Phys. Rev. Lett. 125, 166001 (2020) | web 2026-11-06 |
| lahlou2023deup | Lahlou et al., DEUP: Direct epistemic uncertainty prediction, TMLR (2023) | web 2026-11-06 |
| busk2022calibrated | Busk et al., Calibrated uncertainty for molecular property prediction using ensembles of message passing neural networks, Mach. Learn.: Sci. Technol. 3, 015012 (2022) | web 2026-11-06 (aleatoric + epistemic on QM9/PC9; article number from memory: verify) |
| busk2023graph | Busk et al., Graph neural network interatomic potential ensembles with calibrated aleatoric and epistemic uncertainty on energy and forces, PCCP (2023) | bib |
| tan2023single | Tan et al., Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles, npj Comput. Mater. (2023) | bib |
| valdenegro2022deeper | Valdenegro-Toro, Saromo, A deeper look into aleatoric and epistemic uncertainty disentanglement, CVPRW (2022) | bib |
| seitzer2022pitfalls | Seitzer et al., On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks, ICLR 2022 (arXiv:2203.09168) | bib (venue: update bib to ICLR 2022 — memory, verify) |
| sluijterman2024optimal | Sluijterman, Cator, Heskes, Optimal training of mean variance estimation neural networks, Neurocomputing 597, 127929 (2024) | bib; web 2026-11-06 (warm-up with fixed variance; separate regularization) |
| nix1994estimating, skafte2019reliable, stirn2023faithful | NLL variance-head literature | bib |
| lakshminarayanan2017simple, kendall2017uncertainties, depeweg2018decomposition | deep ensembles, decomposition | bib |
| krogh1994neural | ambiguity decomposition | bib |
| rasmussen2023uncertain, lind2024uncertainty, kuleshov2018accurate, levi2022evaluating | UQ metrics, calibration | bib |
| batatia2022mace, batatia2025design | MACE | bib |
| chmiela2018towards | Chmiela, Sauceda, Müller, Tkatchenko, Towards exact molecular dynamics simulations with machine-learned force fields, Nat. Commun. 9, 3887 (2018) | web 2026-11-06 (CCSD(T)-level sGDML; candidate 2nd dataset with CC forces, WP7) |
| kennedy2000predicting | Kennedy, O'Hagan, Predicting the output from a complex computer code when fast approximations are available, Biometrika 87(1), 1-13 (2000) | memory: verify before use |
