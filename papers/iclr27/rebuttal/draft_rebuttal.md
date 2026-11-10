# ICLR 2027 rebuttal: working draft (Submission 18667)

Status: **first draft, 2026-11-06; recorded results filled in 2026-11-08 (all, incl. WP3.6 energy at 17:10, but the open
decisions D3/D4); consistency pass 2026-11-09 (mace-leo-5): AUSE + ENCE only (FINDINGS §0), 4.1 caveats from FINDINGS §2,
claims about the revised PDF checked against `ICLR27-UQ-MF/iclr2027_conference.tex` and marked `[NOT YET IN PDF: ...]`
where the change does not exist yet.** Not for posting yet. Plan and task status: `PLAN.md`; point
IDs: `reviews.md`.

Conventions for every agent editing this file:
- Numbers come only from `results/results.md` (entry named in brackets, e.g. `[ledger 2026-11-03]`)
  or from the submitted paper (`[paper Sec. X]`). Unrecorded results are `[PENDING: WPx.y]`.
- Every statement about prior work carries a citation key from Section 7, which records how the
  reference was checked. Do not add a reference you have not checked.
- Tone: polite, concrete, no "the reviewer is wrong". Concede what is true, then show evidence.
- Sentences such as "we added", "we rewrote", "fixed in the revised PDF" describe *planned*
  revisions; before posting, keep each one only if the change exists (WP9/WP10.4 check).
- OpenReview comments have a per-comment *word* limit and no limit on their number (`PLAN.md` Section 1); the per-reviewer
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
   learning rates, LF-only baseline, MSE ensembles (HF-only already done). [Multi-head fine-tuning paused by the user 2026-11-07: written answer only.]
3. **Concede the scope limits honestly** (one architecture, energies) but show they are the realistic
   regime: CC data are energy-only, and ANI-1x is *the* public DFT/CC dataset.
4. **Rename, do not abandon, the aleatoric term** (G5): defend it with the representation-relative
   view of aleatoric uncertainty and with our own ablation showing it is the most useful part of
   the predictive variance.

---

## 1. Novelty table (for Q3, G1; also usable in the revision)

| Finding in the paper | Closest known general result | What is new here (MF-specific) |
|---|---|---|
| Ranking and calibration can disagree | Metrics capture different properties (rasmussen2023uncertain, lind2024uncertainty); calibration vs sharpness (kuleshov2018accurate) | They move in **opposite directions along the energy-extrapolation axis**, and pretraining *amplifies* the split (Energy-OOD TU ENCE 1.637 HF-only vs 2.969 LF->HF while TU AUSE 0.186 vs 0.143) `[ledger 2026-09-23, results/tables/iclr27_main_text_numbers.txt]`; in-domain isotonic recalibration does not repair it `[paper App. "Post-hoc isotonic recalibration"]` |
| AU unreliable under distribution shift | valdenegro2022deeper | AU also collapses **in domain**, and the collapse is located **in the HF stage** and accelerated by pretraining (OOD AU ENCE, end of pretraining -> end of FT: system 0.993 -> 81.913, energy 10.863 -> 207.071) `[ledger 2026-09-23]`; on the training set AU is never overconfident (train ENCE below 1, mean AU above the mean squared training error: conservative, not "calibrated") `[ledger 2026-09-23, interpretation checks]` |
| NLL variance heads are fragile | nix1994estimating, skafte2019reliable, seitzer2022pitfalls, stirn2023faithful, sluijterman2024optimal | These works study single-task training; we show that the failure emerges during HF optimization in both protocols, is accelerated by pretraining, and quantify it per epoch; our ablations show that held-out AU becomes overconfident while training-set AU stays calibrated or conservative in every variant, that the collapse re-emerges after a variance-head reset (later and about five times weaker), develops more slowly but still develops at a smaller or larger fine-tuning learning rate, and is not removed by beta-NLL `[ledger 2026-11-07, "Fine-tuning ablations of the AU collapse"; "Head reset: AU calibration along fine-tuning"; ledger 2026-11-08, "AU along fine-tuning for all LF->HF ablations"]` |
| Overconfidence of MACE ensemble UQ | bilbrey2025uncertainty: readout ensembles of MACE-MP-0 and fine-tuned models are overconfident by orders of magnitude; quantile readouts are not | Bilbrey et al. retrain only readouts of a foundation model (ensemble and quantile model separately) and do not track AU/EU of a deep ensemble across an LF->HF transition or through training; we do, and find that EU, not AU, carries the improvement after pretraining |
| Pretraining changes the decomposition | - (not found in prior work) | HF-only: EU is the weakest ranking signal on every test set; LF->HF: EU ranks as well as or better than AU on 3 of 4 (AUSE) `[paper Sec. 4.1; ledger 2026-09-23, results/tables/iclr27_main_text_numbers.txt]`; the gain is tied to the mean-variance members: pretraining improves the EU ranking of an MSE-trained ensemble 2-6x less (AUSE) `[ledger 2026-11-08, system and energy splits]` |
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

*How the CC transfer-learning examples optimize (full texts checked 2026-11-09, supervisor's request):*
- kaser2023transfer: PhysNet; LL models (HF/VDZ, B3LYP/aVTZ, MP2/aVTZ) on 71 208 structures (malonaldehyde,
  acetoacetaldehyde, acetylacetone + 49 amons; 9208 = ~13% test), energies + forces + dipoles; TL to CCSD(T)/aVTZ with
  25/50/100/862 malonaldehyde structures (~100 suffice), HL also with forces and dipoles; "all parameters of PhysNet were
  allowed to change in the TL step"; 10 independent TL runs per setting. Optimizer and learning rate not stated.
- chen2023data: Behler-Parrinello NNPs (n2p2), committee of 8; LL HF or BLYP on 531 periodic boxes of 16 waters (energies +
  forces), or the published generation-1 revPBE0-D3 fits of Schran et al. (data size not stated, not checked; the
  main-text CC models start from these); the parameters are the starting point and the committee is "retrained" on <=200
  CCSD / CCSD(T) / AFQMC **energies only** (extended Kalman filter, 90/10 train/validation split, early stopping per
  member). Against random initialisation and Delta-learning on the same 200 energies, only TL gave stable MD.
- smith2019approaching (ANI-1ccx) differs: two of the four hidden layers are frozen during TL (PMC6602931).
- Ours: all MACE weights updated (Adam, lr 1e-3 = 1/10 of pretraining), per-member checkpoint by validation loss. So the
  two new references fine-tune exactly as we do (Chen also: ensemble, CC energies only, per-member early stopping).

*Proposed ge6X **Motivation.** (2026-11-09, given to the user in chat; NOT applied to `ICLR27-UQ-MF/rebuttal.tex`, user
decision).* Recommended (ge6X box 4975/5000 characters):
> **Motivation.** You mention that you may have missed a trend, and we believe there is indeed one. Atomistic modelling is moving towards foundation models that are fine-tuned on a small amount of more accurate data. MACE-MP-0 is meant to reach ab initio accuracy after fine-tuning on a few application-specific data points (Batatia et al., 2023), and fine-tuning these models is becoming common practice (Tompa et al., 2026). Coupled-cluster potentials are built the same way, from ANI-1ccx on the dataset we use (Smith et al., 2019) to small molecules (Käser & Meuwly, 2023) and liquid water (Chen et al., 2023). These models then need uncertainty, for active learning or long simulations. Bilbrey et al. (2025) showed that ensembles of fine-tuned MACE foundation models can be overconfident by orders of magnitude, but to our knowledge how uncertainty changes during the fine-tuning itself had not been studied. Our paper is a controlled version of this workflow.

Variant stating the optimization (box 4998/5000, trim ~20 characters elsewhere if used): "... ANI-1ccx on the dataset we
use (Smith et al., 2019), small molecules (Käser & Meuwly, 2023) and liquid water (Chen et al., 2023), the last two
fine-tuning all weights as we do. These models then need uncertainty for active learning or long simulations. ..."

In all of these, the fine-tuned model is then used in active learning or simulation, where
uncertainty decides which configurations to label next or when to stop trusting the model
(ANI-1x itself was assembled by query-by-committee active learning, smith2018less). How the
uncertainty behaves across the LF->HF transition is therefore a practical question, and to our
knowledge no prior work has studied it (Section 1 table; we added this table to the revision). [NOT YET IN PDF: novelty table (WP9.3).]

On "findings not unexpected": some findings agree with general knowledge (we cite it), but several
are, to our knowledge, not reported before and not obvious a priori: pretraining *improves* ranking
while *worsening* Energy-OOD calibration; the calibration loss occurs in the HF stage of both
protocols and is accelerated by pretraining; pretraining turns EU from the weakest into a
competitive ranking signal; and the more accurate pretrained ensemble benefits *more* from
uncertainty-guided acquisition. See the novelty table (Section 1).

### 2.2 G2: one model (MACE), one dataset

We agree this is a limitation and state it (App. limitations). Our reasons (user directions 2026-11-07):

*Architecture.*
- MACE is among the most accurate and most widely used MLIP architectures, and the backbone of the
  current foundation models (MACE-MP-0, studied by bilbrey2025uncertainty; MACE-OFF) (batatia2022mace).
- MACE is not one point among unrelated models: batatia2025design show that it spans a design space
  that contains, as special cases, the descriptor-based potentials (ACE, and through it the
  fixed-descriptor models built on atom-centred symmetry functions or SOAP) and the equivariant
  message-passing networks (NequIP). So our study covers the representation shared by most current MLIPs.
  [Agent caveat: this unifies the *representations*; it does not prove that the UQ behaviour transfers. Keep
  "in a sense contains", not "implies the same results". Check the exact wording of batatia2025design for
  which models are recovered, before posting.]
- A controlled study multiplies by the number of architectures: per architecture, 2 splits x 3 protocols
  x 5 splits x 10 members = 300 trained models (LF->HF members need their own LF parents). Adding
  architectures with different training recipes would also confound the effect of the fidelity
  transition with architecture-specific tuning. We chose depth on one strong, representative model
  (estimator, objective, learning-rate and protocol ablations) over breadth.

*Dataset.*
- ANI-1x/ANI-1ccx is, to our knowledge, the only large public dataset with CC labels computed on a
  subset of the DFT geometries over many molecular systems, which is what makes the two controlled
  shifts (unseen systems, higher energies) possible [check alternatives: WP7.1].
- ANI-1x is not a single system: it contains ~5M DFT calculations on organic molecules (C, H, N, O)
  drawn from GDB-11, ChEMBL and generated amino acids and dipeptides, sampled by active learning with
  molecular dynamics, normal-mode, dimer and torsion sampling (smith2020ani); ANI-1ccx is ~500k of
  them recomputed at CCSD(T)*/CBS, mostly molecules with fewer than 20 atoms. Our system split
  uses 3114 molecular systems (chemical formulas), of which 1246 are held out entirely as System-OOD
  in each split (`data/ani1x-system/split_<s>/split_summary.json`). So "one dataset" already spans
  thousands of chemically different systems, and the System-OOD test is a generalization test across them.
- Much of the multi-fidelity and UQ literature for MLIPs evaluates on a single system: one molecule
  (rMD17, christensen2020role; 3BPA, kovacs2021linear) or one liquid (water at CC accuracy,
  chen2023data). Our setting is broader than that. [Citations to verify and add to the bib; pick 2-3.]
- Within this setting we vary what reviewers asked about: the estimator (ten-member MVE ensemble,
  single MVE model, MSE ensemble without a variance head `[ledger 2026-11-02, 2026-11-03, 2026-11-04]`),
  the objective (beta-NLL, `[ledger 2026-11-07, "Fine-tuning ablations of the AU collapse"]`) and the
  optimization (HF-only learning rates `[ledger 2026-11-07, "HF-only learning-rate sweep complete"]`, FT learning
  rates `[ledger 2026-11-07, "Fine-tuning ablations of the AU collapse"; "Fine-tuning learning rate on the system
  split"]`).
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
- **Learning rates.** We added sweeps for HF-only (3e-3 and 1e-3, besides the paper's 1e-2) and for fine-tuning
  (1e-4 and 1e-2, besides 1e-3), on both splits, 5 splits x 10 members each (new App. "Learning rates and compute",
  figure). A lower learning rate helps HF-only only a little (TU AUSE lower by at most 0.021, significant in 3 of 8 cells;
  RMSE changes of at most ~6% where significant) `[ledger 2026-11-07, "HF-only learning-rate sweep complete";
  results/tables/ablations_vs_reference.csv]`. Against every tuned HF-only ensemble, LF->HF stays about 2x more
  accurate in distribution and ranks errors better in all 8 (split, learning rate, test) cells (TU AUSE lower by
  0.02-0.09, all intervals exclude 0) `[same entry]`. The calibration contrast is split-dependent: on the energy
  split LF->HF is less calibrated in all 4 cells (significant in 2), on the system split better calibrated in
  all 4 (significant in 1) `[same entry]`; we now state the calibration conclusion per split. For fine-tuning,
  1e-4 and 1e-3 give nearly the same accuracy and ranking, while 1e-2 ranks worse and makes TU underconfident
  in distribution `[ledger 2026-11-07, "Fine-tuning learning rate on the system split" and the energy entry]`.
  So the paper's accuracy and ranking conclusions do not depend on the learning rates we chose.
- **Model size.** [Not run (user 2026-11-07, PLAN WP4.4 dropped): text-only answer still to write; App. M.5 has no model-size paragraph yet. NOT YET IN PDF.]
- **Pretrain-only (LF-only).** See G7.

### 2.5 G5: "aleatoric" with deterministic labels

We agree that reference energies of a fixed electronic-structure method are deterministic, and the
paper says so (Sec. 2.1, App. "Technical background"). Our use of the term is operational, and we will make this explicit
by renaming the term in the revision [decision D5: e.g. "learned variance (AU)". NOT YET IN PDF: depends on D5]. We nevertheless think
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
   e.g. Energy-OOD TU AUSE difference -0.1116 [-0.1291, -0.0942] and ENCE ratio x0.1246
   [0.09637, 0.1611] in favour of MVE TU `[ledger 2026-11-03; results/tables/mse_vs_mve.csv]`; System-OOD -0.05517
   [-0.06456, -0.04578] and x0.2649 [0.207, 0.339] `[ledger 2026-11-04; same file]`; most of the gain comes from AU
   (the two EUs rank alike at HF-only).
   (HF-only protocol.) After LF->HF fine-tuning the gap widens: on the system split MVE TU beats the MSE-ensemble
   EU on every metric and test, e.g. System-OOD AUSE -0.107613 [-0.119144, -0.096082] and ENCE ratio x0.166871
   [0.134028, 0.207762] `[ledger 2026-11-08]`; Energy-OOD AUSE -0.109989 [-0.115209, -0.104769] and ENCE ratio x0.226098
   [0.180046, 0.283930] `[ledger 2026-11-08, energy split]`.
4. *The reviewer's reading is in fact one of our results.* If labels are deterministic, all error is
   in principle reducible (the excess-risk view of epistemic uncertainty, lahlou2023deup). Our
   training dynamics show exactly this: the learned variance behaves as a reducible estimate of
   in-sample residual difficulty, shrinking with the training error (Sec. 4.2). This is why we
   evaluate calibration on TU only and use AU/EU calibration as a diagnostic (App. "Technical background", the ambiguity
   decomposition, krogh1994neural).

### 2.6 G6: white band in Fig. 1

Thank you: the figure contains nested transparency masks that some viewers (e.g. Safari) render
incorrectly. It is fixed in the revised PDF. [NOT YET IN PDF: `ICLR27-UQ-MF/figures/Workflow.pdf` is still the submitted file;
apply the fixed file from the arXiv repository (f342f6b) or rerun `papers/iclr27/figures/fix_workflow_safari.py` (WP9.1).]

### 2.7 G7: LF-only results

[Agent note: why CC labels. G7 asks whether LF-only is worse *than LF+HF or HF-only*, which predict CC energies:
only scoring on the same CC test sets compares the protocols on the same task. LF-only against its own DFT
test labels is a different task (a model scored on the fidelity it was trained on); those results are already
in the paper (App. reliability diagrams, rows "LF-only (DFT labels)"; training-dynamics figures, epochs
0-299) and can be pointed to in one sentence.]

On its own DFT test labels the LF-only ensemble is already shown in the paper (App. reliability diagrams and
training dynamics). To answer the question as asked, against the CC targets of LF->HF and HF-only: 
LF-only models predict DFT energies, whose absolute scale differs from CC/CBS energies by large
per-element offsets, so a direct comparison on CC labels would be dominated by that offset. We added
the fair comparison (new appendix figure, App. M.1): the converged LF-only ensembles, unchanged except for a
per-member least-squares refit of the per-element reference energies to their residuals on the CC
training set (the analogue of the reference-energy refit with which fine-tuning starts; the most
favourable zero-shot use of the LF model), scored on the CC test sets, with the variance heads
unchanged. Paired over the five splits `[ledger 2026-11-07, results/tables/lf_only_vs_hf.csv]`:

- **LF->HF fine-tuning is better on every uncertainty metric on all four tests**: TU AUSE
  -0.06 to -0.09 (all intervals exclude 0), TU ENCE x0.20-0.23 except on
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
(Section 2.2): ANI-1x spans thousands of molecular systems, and we added estimator, objective and learning-rate variants. [Note (mace-leo): "computed on
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
  (gardner2025understanding; MACE multihead replay fine-tuning, tompa2026fine). These, like
  Delta-learning of the HF-LF difference (ramakrishnan2015big) and joint training, are interesting
  multi-fidelity strategies with possibly different uncertainty dynamics; we deliberately focused on
  sequential fine-tuning, the most widely used one, and state the others as future work in the revised
  limitations. [User 2026-11-07: WP6 paused, no multi-head experiment.]

### 3.3 J3: better arguments for model, data and training choices

We added "Design choices and scope" (revised App. M.5): MACE (widely used equivariant
MLIP; foundation-model backbone), mean-variance deep ensembles (the estimator that defines AU, EU and
TU through the law of total variance, lakshminarayanan2017simple; strongest across tasks in
tan2023single), ANI-1x (CC labels on DFT geometries, controlled shifts), energies (CC labels have no
forces), sequential fine-tuning (standard; Section 3.2), and what each choice leaves untested.
[In the PDF since 2026-11-09: estimator paragraph added to M.5; MACE and sequential-FT paragraphs still short,
their citations pending.]

---

## 4. Response to Reviewer Q2LT

We thank the reviewer for the detailed summary and for the concrete suggestions, which we followed.

### 4.1 Q1 / Q4: validating the mechanism of AU overconfidence

[Rewritten 2026-11-07 (mace-leo-2) from the recorded ablations; all four FT ablations x 2 splits x 5 splits x 10
members. Sources: ledger 2026-11-07 "Fine-tuning ablations of the AU collapse", its "Correction ... selected
epochs", "Head reset: AU calibration along fine-tuning", "Fine-tuning learning rate on the system split".]

We thank the reviewer for this suggestion; we ran exactly these ablations. Each fine-tunes the same pretrained
members as in the paper, on both distribution shifts, 5 splits x 10 members, and is compared with the paper's
LF->HF ensembles on the same splits (paired contrasts, 95% intervals over splits; new appendix "Additional
baselines and ablations": per-epoch AU dynamics, App. M.3 Fig. "AU along fine-tuning", and the final models in Fig. 2 style;
the contrasts quoted below are paired over splits). Our explanation makes three testable predictions.

**The overconfidence is a held-out effect.** In every variant below, AU on the *training* set stays calibrated or
conservative from the tenth fine-tuning epoch on (training-set AU ENCE at most 0.909, apart from one unstable epoch at
the largest learning rate), while on held-out data it becomes overconfident `[ledger 2026-11-08, "AU along
fine-tuning for all LF->HF ablations"]`: the variance head follows the training residuals, as our account says.

**(b) Re-initializing the variance head should not prevent the collapse, only restart it.** We re-drew the
variance-specific readout weights at the start of fine-tuning. Along fine-tuning the reset head starts slightly
*under*confident (AU ENCE just below 1), is best calibrated after 10-15 epochs, and then loses AU calibration steadily,
like the inherited head but later (ENCE above 1 after 20-25 epochs instead of 5-10) and about five times less by the
end of fine-tuning (System-ID AU ENCE at FT epoch 95: 12.5 with reset vs 67.2 without)
`[ledger 2026-11-08, "AU along fine-tuning for all LF->HF ablations"]`. So the collapse re-emerges once the head is
fitted to the shrinking training residuals, as predicted. Two further observations refine the paper's account:
(i) with the inherited head the collapse starts from the first fine-tuning epoch, which is the mechanism behind
our finding that calibration is lost faster after pretraining; (ii) the reset costs ranking at every epoch (AU
AUSE higher by 0.04-0.10 at the selected checkpoints, all intervals exclude 0), so part of pretraining's ranking
benefit is carried by the pretrained variance head `[ledger 2026-11-07, "Fine-tuning ablations of the AU collapse";
results/tables/ablations_vs_reference.csv]`. Because
checkpoints are selected by validation NLL, the reset ensembles are selected before their collapse has developed
(epochs 2-19 of 100) and therefore look better calibrated at the end, but rank worse.

**(c) Slower fine-tuning should delay the collapse.** At a 10x smaller FT learning rate (1e-4), held-out AU
ENCE first exceeds 1 after 35-75 fine-tuning epochs instead of 5-10 (System-ID, System-OOD, Energy-ID)
`[ledger 2026-11-08, "AU along fine-tuning for all LF->HF ablations"]`, and the validation-selected epoch moves
from 2 to 25-30 on both splits; accuracy and ranking barely change, and AU at the selected checkpoint is as
overconfident as before or more (AU ENCE x1.15-1.38) `[ledger 2026-11-07, "Fine-tuning learning rate on the system
split" and the energy entry]`. So the delay is as predicted, while the overconfidence of the selected model is not
removed. Two qualifications, which the revised appendix states: (i) a 10x *larger* learning rate (1e-2) also slows
the collapse (ENCE above 1 after 25-40 epochs), although its final training error is similar, so the speed of the
collapse is not a monotone function of the learning rate; it is fastest in the paper's setting and develops in every
variant `[same entry]`. (ii) Under energy extrapolation (Energy-OOD) AU is overconfident from the first fine-tuning
epoch in every variant that keeps the pretrained head, at either learning rate; only the re-initialized head starts
calibrated (ENCE 0.71) and it exceeds 1 after 5 epochs `[ledger 2026-11-08, "AU along fine-tuning for the LF->HF
ablations on Energy-OOD"]`. [Phrase as "the overconfidence is robust; its speed depends on the optimization and is
fastest in the main setting" (FINDINGS §2.1); do not write that the time scale simply follows the optimization speed.]

**Alternative objective.** beta-NLL (seitzer2022pitfalls; beta = 0.5), designed to counter the variance-driven
down-weighting of hard examples in Gaussian NLL, leaves accuracy and ranking unchanged and does not remove the
overconfidence (AU ENCE slightly worse, x1.15-1.26 on all four tests) `[ledger 2026-11-07, "Fine-tuning ablations
of the AU collapse"]`. The overconfidence is thus not an artefact of NLL optimization that a re-weighted
objective fixes. MSE-trained ensembles, which have no variance head, are the complementary control: for HF-only
they are already in the appendix `[ledger 2026-11-03, 2026-11-04]`. We also fine-tuned MSE ensembles from
MSE-pretrained members (both splits): pretraining lowers
the AUSE of the MVE EU by 0.10-0.14 but that of the MSE-ensemble EU by only 0.02-0.04, and does not improve the
MSE-ensemble EU calibration on either split `[ledger 2026-11-08, system and energy splits;
results/tables/mse_vs_mve_pretraining.csv]`. The ranking gain
of pretraining is thus tied to the MVE members; the MSE ensemble, having no AU, cannot show the collapse, and its EU
is not degraded by pretraining.

**(d) A variance scale fitted to held-out residuals should repair calibration where held-out data resemble the
test data.** This is what the isotonic recalibration of the appendix does (fitted on held-out validation data):
it repairs in-distribution calibration but not energy extrapolation (App. recalibration), consistent with a
head that inherits the scale of the training residuals. [Check the exact wording of the recalibration appendix
before posting. WP3.4 (held-out-fitted head) dropped by the user 2026-11-07: do not claim (d) as newly tested.]

[On prediction (a) (held-out AU tracks the training error): the stored per-epoch means were inconclusive
(PLAN WP3.1); do not claim it as tested. If space allows, keep only the existing qualitative evidence: on the
training set AU is never overconfident (train ENCE below one `[ledger 2026-09-23, interpretation checks]`), while
held-out AU ENCE rises by orders of magnitude during fine-tuning (App. "Training dynamics").]

**Revision.** We will revise the mechanism paragraph accordingly: overconfidence arises as the variance head
follows the training residuals during fine-tuning; it starts immediately when the head is inherited from
pretraining and later when it is re-initialized; early stopping on validation NLL therefore determines how much
of it the final model shows. [Draft wording; check against the paper's Discussion before posting. NOT YET IN PDF:
the main-text mechanism paragraph is unchanged; the revised account is only in App. M.3 so far (WP9.5).]

### 4.2 Q2 / Q5: generality across estimators and multi-fidelity strategies

- *Estimators.* Besides the ten-member MVE ensemble we now report a single MVE model (M = 1) and the
  ensemble-size curve M = 1..10 `[ledger 2026-11-02]`, and MSE deep ensembles `[ledger 2026-11-03,
  2026-11-04]`. Key results: one MVE model's AU already ranks errors better than a ten-member
  MSE ensemble on every test (OOD AUSE difference, system -0.02771 [-0.0375, -0.01793]; energy -0.02533
  [-0.03918, -0.01149]) and is better calibrated (OOD ENCE ratio x0.5149 and x0.5312) `[ledger 2026-11-03,
  2026-11-04; results/tables/mse_vs_mve.csv]`. After LF->HF fine-tuning the MVE ensemble ranks better than the MSE
  ensemble on both splits and every test, with EU, TU and even a single member's AU (AUSE difference System-OOD
  -0.095593, -0.107613 and -0.069608; Energy-OOD -0.059281, -0.109989 and -0.020752), and its TU is 4-6x better
  calibrated (ENCE x0.166871 System-OOD, x0.226098 Energy-OOD) `[ledger 2026-11-08, system and energy splits]`.
  Accuracy is equal on the system split (RMSE ratio x1.004406 [0.788443, 1.279523]) and better for the MVE ensemble
  on the energy split (x0.678857 ID, x0.945985 OOD) `[same entries]`. Under energy extrapolation both EUs are
  equally miscalibrated and a single member's AU is worse calibrated than the MSE-ensemble EU (x1.700718): there the
  calibration advantage comes from TU `[ledger 2026-11-08, energy split]`.
- *Multi-fidelity strategies.* We agree that joint, multi-head, alternating and Delta-learning strategies
  (ramakrishnan2015big) are interesting and may show different uncertainty dynamics; we focused on
  sequential fine-tuning because it is the most used route to CC-accurate MLIPs (Section 2.1), and the
  evidence on the alternatives (gardner2025understanding, tompa2026fine; Section 3.2) is discussed in the
  revised limitations. Not run (user decision 2026-11-07).
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
localizes the calibration loss to the HF stage. We added this paragraph to the related work. [NOT YET IN PDF:
related-work paragraph and bib entry `bilbrey2025uncertainty` (WP9.3).]

---

## 5. General response (to all reviewers and the AC), skeleton

1. Thanks; all reviewers find the paper clear; j1p7 finds no soundness concerns.
2. Motivation: LF->HF fine-tuning is the standard route to CC accuracy (Section 2.1, 6 citations).
3. New experiments (list only those completed): LF-only on CC; variance-head reset; beta-NLL; LR sweeps
   (HF-only and fine-tuning); MSE ensembles (HF-only, LF->HF); single model / ensemble size; [forces, if D3].
   (Held-out-fitted head: dropped by the user 2026-11-07, not to be listed.)
4. Revisions: novelty table, renamed AU, design-choices section, Fig. 1 fix, related work. [In the PDF on 2026-11-09:
only the design-choices section (App. M.5) and the new appendix; the other four are NOT YET IN PDF.]

**Table for item 3 (draft, 3.3d, mace-leo-2 2026-11-07; updated mace-leo-3 2026-11-08).** Every row: both distribution
shifts, 5 dataset splits x 10 members, paired contrasts with 95% intervals over splits; details in the new appendix
"Additional baselines and ablations" (items A-D below; in the revised PDF draft of 2026-11-08 this is App. M,
A-E = M.1-M.5). Rows only for recorded results (all rows complete on both splits since 2026-11-08 17:10).

| New experiment (App.) | Asked by | Does the paper's conclusion hold? |
|---|---|---|
| Pretrain-only (LF-only) ensembles on CC labels, per-element E0 refit (A) | ge6X G7, G4 | Yes. Fine-tuning beats the zero-shot DFT ensemble on every uncertainty metric and test; LF-only variances are miscalibrated for CC errors even after the refit `[ledger 2026-11-07, "LF-only (pretrain-only) ensembles on the CC labels"]`. |
| Optimizer steps to the evaluated checkpoints (B, caption) | ge6X G4 | Yes. LF->HF uses ~20x fewer steps on CC labels than HF-only (median 234 vs 4758-5304); its extra compute is on DFT labels `[ledger 2026-11-07, "Gradient steps per protocol"]`. |
| HF-only lr 1e-3 / 3e-3; fine-tuning lr 1e-4 / 1e-2 (B) | ge6X G4 | Yes for accuracy and ranking: against every tuned HF-only, LF->HF stays ~2x more accurate in distribution and ranks better in all 8 cells; the calibration contrast is split-dependent and is now stated per split. FT lr 1e-4 ~ 1e-3; 1e-2 is worse `[ledger 2026-11-07, "HF-only learning-rate sweep complete"; "Fine-tuning learning rate on the system split"]`. |
| Variance-head reset at FT start (C) | Q2LT Q1/Q4 | Yes, and refines it: the collapse re-emerges with a fresh head, more slowly; the inherited head makes it start at once `[ledger 2026-11-07, "Head reset: AU calibration along fine-tuning"]`. |
| beta-NLL fine-tuning (C) | Q2LT Q1/Q4 | Yes: same accuracy and ranking, AU still overconfident `[ledger 2026-11-07, "Fine-tuning ablations of the AU collapse"]`. |
| Fine-tuning lr 1e-4 / 1e-2 (C) | Q2LT Q1/Q4 | Yes: the overconfidence develops later at either learning rate (fastest in the paper's setting) and is not removed at the selected checkpoint; under energy extrapolation it is present from the first epoch `[same entries; ledger 2026-11-08, "AU along fine-tuning ..."]`. |
| Single MVE model, ensemble size M = 1..10, MSE ensembles (HF-only) (D) | Q2LT Q2/Q5 | Yes: the decomposition's TU beats the MSE-ensemble signal on every metric and test; even one MVE model's AU ranks errors better than ten MSE members `[ledger 2026-11-02, 2026-11-03, 2026-11-04]`. |
| MSE ensembles, LF->HF (D) | Q2LT Q1/Q2 | Yes (both splits): pretraining improves the MVE ranking 2-6x more (AUSE) than the MSE-ensemble ranking, so after LF->HF the MVE ensemble ranks better on every test, at equal (system) or better (energy) accuracy `[ledger 2026-11-08, system and energy splits]`. |

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
| smith2019approaching | Smith et al., Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning, Nat. Commun. 10, 2903 (2019) | bib; web 2026-11-06 (DFT training then transfer learning to CCSD(T)/CBS; ~5M DFT, ~500k CC); PMC6602931 2026-11-09: two of four hidden layers frozen in TL |
| smith2020ani | Smith et al., The ANI-1ccx and ANI-1x data sets, Sci. Data (2020) | bib |
| smith2018less | Smith, Nebgen, Lubbers, Isayev, Roitberg, Less is more: sampling chemical space with active learning, J. Chem. Phys. 148, 241733 (2018) | web 2026-11-06 (QBC with ensemble disagreement built ANI-1x) |
| kaser2021transfer | Käser, Boittier, Upadhyay, Meuwly, Transfer learning to CCSD(T): accurate anharmonic frequencies from machine learning models, J. Chem. Theory Comput. 17(6) (2021) | web 2026-11-06 |
| kaser2023transfer | Käser, Meuwly, Transfer-learned potential energy surfaces: towards microsecond-scale molecular dynamics simulations in the gas phase at CCSD(T) quality, J. Chem. Phys. (2023), arXiv:2303.11685 | web 2026-11-06; full text (arXiv v1) read 2026-11-09: PhysNet, all parameters updated in TL, see 2.1; DOI 10.1063/5.0151266 (volume/pages to add); **not in the paper bib** |
| zaverkin2023transfer | Zaverkin et al., Transfer learning for chemically accurate interatomic neural network potentials, PCCP (2023) | bib |
| chen2023data | Chen et al., Data-efficient machine learning potentials from transfer learning of periodic correlated electronic structure methods: liquid water at AFQMC, CCSD, and CCSD(T) accuracy, JCTC (2023) | bib; web 2026-11-06 (<=200 CC energies); full text + SI (arXiv 2211.16619) read 2026-11-09: n2p2 committee of 8, retrained from LL weights on CC energies only, see 2.1 |
| kim2024data | Kim et al., Data-efficient multifidelity training for high-fidelity machine learning interatomic potentials, JACS 147, 1042-1054 (2024/2025) | bib; web 2026-11-06 (SevenNet-MF, one-hot fidelity encoding, simultaneous training) |
| ko2025data | Ko et al., Data-efficient construction of high-fidelity graph deep learning interatomic potentials, npj Comput. Mater. (2025) | bib |
| cui2025multi | Cui, Reuter, Margraf, Multi-fidelity transfer learning for quantum chemical data using a robust density functional tight binding baseline, Mach. Learn.: Sci. Technol. 6, 015071 (2025) | web 2026-11-06 |
| messerly2025multi | Messerly et al., Multi-fidelity learning for interatomic potentials: low-level forces and high-level energies are all you need, Mach. Learn.: Sci. Technol. (2025), arXiv:2505.01590 | web 2026-11-06 (quote "most high-accuracy datasets ... often do not contain atomic forces") |
| gardner2025understanding | Gardner, Schulz, Helie, Sun, Simm, Understanding multi-fidelity training of machine-learned force-fields, arXiv:2506.14963 (v2 2026) | bib; web 2026-11-06 (quotes "critically, the inclusion of force labels"; "marginally reduce model performance in most cases") |
| tompa2026fine | Tompa, Varga-Umbrich, Batatia, Elena, Bernstein, Csányi, Fine-tuning MLIP foundation models: strategies for accuracy and transferability, arXiv:2606.12704 (2026) | web 2026-11-06 (quotes "increasingly important practice"; naive FT "best convergence for single-system applications"; multihead replay preserves OOD robustness) |
| radova2025fine | Radova et al., Fine-tuning foundation models of materials interatomic potentials with frozen transfer learning, npj Comput. Mater. (2025) | bib |
| bilbrey2025uncertainty | (in the paper bib since 2026-11-10, cited in Sec. 1) Bilbrey, Firoz, Lee, Choudhury, Uncertainty quantification for neural network potential foundation models, npj Comput. Mater. 11, 109 (2025), doi:10.1038/s41524-025-01572-y | full text read 2026-11-07 (mace-leo, publisher PDF): 7-member readout ensemble (interaction layers frozen, Huber loss, different 80k MPtrj subsets; fine-tuned ensembles differ by seed only) + 5th/95th-percentile quantile readouts; MPtrj, HEA25, H-ZSM-5 + water (n = 8, 16 held out); quote "the ensemble is overconfident in its predictions ... lower than the error by orders of magnitude" (Introduction); Table 1 coverage ensemble 11/21/64%, quantile 87/88/86% |
| bocus2025operando | Bocus et al., The operando nature of isobutene adsorbed in zeolite H-SSZ-13 unraveled by machine learning potentials beyond DFT accuracy, Angew. Chem. (2025), doi:10.1002/ange.202413637, PubMed 39480685 | web 2026-11-10 (abstract: transfer learning to RI-RPA quality for enhanced-sampling MD, 0.3 meV/atom RMSE, an order of magnitude fewer post-HF single points); full author list not checked; **not in the paper bib** (rebuttal only) |
| chen2021learning | Chen, Zuo, Ye, Li, Ong, Learning properties of ordered and disordered materials from multi-fidelity data, Nat. Comput. Sci. 1, 46-53 (2021) | web 2026-11-06 |
| shiota2024taming | Shiota et al., Taming multi-domain, -fidelity data: towards foundation models for atomistic scale simulations, arXiv:2412.13088 | web 2026-11-06 (total energy alignment across fidelities; optional) |
| hullermeier2021aleatoric | Hüllermeier, Waegeman, Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods, Mach. Learn. 110, 457-506 (2021) | bib; web 2026-11-06 (context-dependence: changing the representation turns aleatoric into epistemic and vice versa) |
| pozdnyakov2020incompleteness | Pozdnyakov et al., Incompleteness of atomic structure representations, Phys. Rev. Lett. 125, 166001 (2020) | web 2026-11-06 |
| lahlou2023deup | Lahlou et al., DEUP: Direct epistemic uncertainty prediction, TMLR (2023) | web 2026-11-06 |
| busk2022calibrated | Busk et al., Calibrated uncertainty for molecular property prediction using ensembles of message passing neural networks, Mach. Learn.: Sci. Technol. 3, 015012 (2022) | web 2026-11-06 (aleatoric + epistemic on QM9/PC9); web 2026-11-09: MLST 3(1), 015012 (2022), doi:10.1088/2632-2153/ac3eb3 |
| busk2023graph | Busk et al., Graph neural network interatomic potential ensembles with calibrated aleatoric and epistemic uncertainty on energy and forces, PCCP (2023) | bib |
| tan2023single | Tan et al., Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles, npj Comput. Mater. (2023) | bib |
| valdenegro2022deeper | Valdenegro-Toro, Saromo, A deeper look into aleatoric and epistemic uncertainty disentanglement, CVPRW (2022) | bib |
| seitzer2022pitfalls | Seitzer et al., On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks, ICLR 2022 (arXiv:2203.09168) | bib; web 2026-11-09: ICLR 2022 (iclr.cc poster 6755, OpenReview aPOpXlnV1T): bib entry updated from arXiv to ICLR 2022 |
| sluijterman2024optimal | Sluijterman, Cator, Heskes, Optimal training of mean variance estimation neural networks, Neurocomputing 597, 127929 (2024) | bib; web 2026-11-06 (warm-up with fixed variance; separate regularization) |
| nix1994estimating, skafte2019reliable, stirn2023faithful | NLL variance-head literature | bib |
| lakshminarayanan2017simple, kendall2017uncertainties, depeweg2018decomposition | deep ensembles, decomposition | bib |
| krogh1994neural | ambiguity decomposition | bib |
| rasmussen2023uncertain, lind2024uncertainty, kuleshov2018accurate, levi2022evaluating | UQ metrics, calibration | bib |
| batatia2022mace, batatia2025design | MACE | bib |
| batatia2025design (content) | ACE and NequIP unified in one design space (ACE recast as one layer of a multi-layer architecture; linearised NequIP as a sparsification of a larger polynomial model), arXiv:2205.06643 | web 2026-11-09 (abstract) |
| batatia2023foundation | Batatia et al., A foundation model for atomistic materials chemistry (MACE-MP-0), arXiv:2401.00096 (2023) | web 2026-11-09 (quote "can be fine-tuned on just a handful of application-specific data points to reach ab initio accuracy"; not in bib) |
| kovacs2025maceoff | Kovács et al., MACE-OFF: short-range transferable machine learning force fields for organic molecules, JACS (2025), doi:10.1021/jacs.4c07099 | web 2026-11-09 (not in bib) |
| chmiela2018towards | Chmiela, Sauceda, Müller, Tkatchenko, Towards exact molecular dynamics simulations with machine-learned force fields, Nat. Commun. 9, 3887 (2018) | web 2026-11-06 (CCSD(T)-level sGDML; candidate 2nd dataset with CC forces, WP7) |
| kennedy2000predicting | Kennedy, O'Hagan, Predicting the output from a complex computer code when fast approximations are available, Biometrika 87(1), 1-13 (2000) | web 2026-11-09 (title, authors, Biometrika 87(1), 2000; GP model combining expensive code runs with cheap approximations; page range from memory) |
| ramakrishnan2015big | Ramakrishnan, Dral, Rupp, von Lilienfeld, Big data meets quantum chemistry approximations: the Delta-machine learning approach, J. Chem. Theory Comput. 11(5), 2087-2096 (2015), doi:10.1021/acs.jctc.5b00099 | web 2026-11-09 (arXiv:1503.04987; ML corrections added to cheap approximate methods) |
| christensen2020role | Christensen, von Lilienfeld, On the role of gradients for machine learning of molecular energies and forces (rMD17), Mach. Learn.: Sci. Technol. (2020), doi:10.1088/2632-2153/abba6f, arXiv:2007.09593 | web 2026-11-09 (title, authors, journal; rMD17; article number 045018 from memory) |
| kovacs2021linear | Kovács et al., Linear atomic cluster expansion force fields for organic molecules: beyond RMSE (3BPA), J. Chem. Theory Comput. 17(12), 7696-7711 (2021), doi:10.1021/acs.jctc.1c00647 | web 2026-11-09 (MD17, ISO17 and the 3BPA molecule) |
| smith2020ani (content) | ANI-1x: ~5M DFT calculations, C/H/N/O, GDB-11 + ChEMBL + generated amino acids and dipeptides, MD/normal-mode/dimer/torsion active-learning sampling; ANI-1ccx ~500k CCSD(T)*/CBS, mostly <20 atoms | web 2026-11-07 (PMC7195467) |
