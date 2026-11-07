# ICLR 2027 rebuttal: plan (Submission 18667)

Entry point and standing rules: [`README.md`](README.md). This file holds *what* we do: reviewer
asks (Section 2), merged work packages (3), the TODO with claims (4), the compute plan (5),
decisions (6) and the log (7). *Where* jobs run: [`RUNS.md`](RUNS.md).

TODO markers: `[ ]` open, `[~] (agent: <name>, since <time>)` claimed, `[x]` done (with output or
ledger entry), `[!]` blocked (with reason). Keep IDs stable; record who did what in the Log.

## 1. Situation (2026-11-06)

- Ratings 1 / 2 / 2 (ge6X clear reject; j1p7 and Q2LT weak reject). All three rate presentation
  "clear". Soundness is not the problem (j1p7: "no substantive concerns"); the attack is on
  **motivation/novelty (G1, Q3)**, **generality (G2, J1, Q2)**, **missing baselines/ablations
  (G4, G7, Q1, J2)** and **framing of "aleatoric" (G5)**.
- Winnable: **Q2LT** asks for concrete ablations (variance-head reset, other objectives,
  other MF strategies) that are cheap for us (fine-tuning runs are ~0.5 h per member).
  **j1p7** (confidence 4, soundness 3) mainly wants justification plus a fidelity-aware
  variant; a multi-head experiment plus arguments could move them. **ge6X** is the hardest
  (informativeness 1); answer every point precisely for the AC, especially G1 (the setting is
  standard: ANI-1ccx itself was built by DFT-pretraining + CC transfer learning), G3 (CC data
  have no forces), G5 and G7 (LF-only).
- **Deadline: author responses by 2026-11-18** (user). Revised-PDF deadline and per-comment
  character limit: [FILL IN from OpenReview, WP10.1]. **Post results as early as possible** to start the
  discussion (user, 2026-11-06): answer in waves, not in one block on the last day.
  - Wave A (target **2026-11-10/11**): text-only answers (G1, G3, G5, G6, J2/J3 arguments, Q3) +
    results that already exist (MSE vs MVE HF-only, single model, ledger) + first cheap results
    (WP2 LF-only, WP3.1 gap analysis, WP3.2 head reset, WP4.1 compute accounting).
  - Wave B (target **2026-11-13/14**): remaining FT ablations (WP3.3-3.5), LR sweeps (WP4.2-4.3),
    multi-head FT (WP6.2), MSE LF->HF (WP3.6).
  - Wave C (target **2026-11-16/17**): LF-scale experiments from Leonardo (forces WP5.2, joint
    fidelity-aware training WP6.3, model size WP4.4, optional 2nd dataset WP7.2) + revised PDF.
  - Keep **2026-11-18** as buffer for replies only; never post a result without a ledger entry.
- **Compute:** local DGX (3 GPUs) + **CINECA Leonardo, ~10,000 GPU-h** (account `IscrC_UQ-MF`).
  LF-scale experiments (~16-19 h per member, ~800 GPU-h per 50-member experiment) go to Leonardo;
  its setup (WP11) is the critical path. Access works since 2026-11-06 (`LEONARDO.md`).
- Already available and citable (ledger): single MVE model vs ensemble size (2026-11-02);
  MSE-ensemble baseline, HF-only, energy (2026-11-03) and system (2026-11-04) splits.
  In training (DGX): MSE-ensemble energy LF-only / LF->HF (complete ~2026-11-08, `RUNS.md`).

## 2. Tasks asked by each reviewer (annotated)

Type: **W** = writing/argument only, **A** = analysis of existing outputs (CPU or GPU
inference, no training), **E** = new training. WP = work package in Section 3.

### ge6X (1, conf 3)

| ID | Ask | Our position | Type | WP |
|---|---|---|---|---|
| G1 | Motivation weak: "HF fine-tuning is not common"; findings expected | Factually wrong: DFT->CC transfer learning is the standard route to CC-accurate MLIPs (ANI-1ccx, Käser & Meuwly, Zaverkin, Chen 2023 water, Kim 2024, Messerly 2025, Cui 2025, Gardner 2025), and fine-tuning foundation models is "an increasingly important practice" (Tompa 2026). Clarify which findings are MF-specific (shared with Q3) | W | WP1 |
| G2 | Only MACE, only ANI-1x | MACE + ANI-1x is the most standard pairing; ANI-1x is the only public dataset with ~0.5M CC labels on the DFT geometries; partially addressed by estimator/strategy ablations; optional 2nd dataset/architecture | W (+E opt.) | WP7 |
| G3 | Energies only, no forces | CC/CBS labels in ANI-1x have no forces (checked: only `ccsd(t)_cbs.energy`); energy-only HF is the realistic setting (Smith 2019, Messerly 2025). DFT forces exist (`wb97x_tz.forces`): optional LF-with-forces pretraining | W (+E opt.) | WP5 |
| G4 | Arbitrary protocol: match compute, model sizes, optimal LRs, pretrain-only | Compute is not the bottleneck (HF-only val-selected at epoch 30-111 of 300); LR sweep for HF-only and FT is cheap; pretrain-only = LF-only (G7); model size optional | W + A + E | WP4, WP2 |
| G5 | "Aleatoric" meaningless for deterministic labels | Agree on semantics, already operational in the paper; defend usefulness (representation-relative noise, standard usage) and **rename in the revision** (e.g. "learned variance (AU)") | W | WP8 |
| G6 | White band in Fig. 1 | Known Safari/soft-mask PDF bug, fixed in the arXiv version | W (fix) | WP9 |
| G7 | LF-only not reported: is it worse? | Evaluate the existing LF-only ensembles (A, C) on CC test labels after refitting atomic reference energies on the CC training set | A | WP2 |

### j1p7 (2, conf 4)

| ID | Ask | Our position | Type | WP |
|---|---|---|---|---|
| J1 | One model, "two datasets" | as G2 (the two "datasets" are probably the two fidelities or the two splits; answer both) | W | WP7 |
| J2 | PT->FT may not be the right approach; systematic DFT/CC bias; fidelity not an input -> missed opportunity, maybe better calibration | PT->FT is the most accurate strategy with two label sources (Gardner 2025; Tompa 2026 for single-target deployment); systematic bias is absorbed by the refit E0s + backbone adaptation; run a **fidelity-aware multi-head variant** and compare UQ | W + E | WP6 |
| J3 | Better arguments for limitations of model/data/training choices | Rewrite limitations as an explicit "design choices and scope" paragraph | W | WP9, WP1 |

### Q2LT (2, conf 2)

| ID | Ask | Our position | Type | WP |
|---|---|---|---|---|
| Q1 / Q4 | Mechanism (variance head follows training residuals) not validated: reset variance head at FT start; alternative objectives | Run the ablations: head reset, beta-NLL, held-out-fitted variance head, slower FT lr; CPU analysis of train/held-out gap vs AU; MSE ensembles = objective without variance head | A + E | WP3 |
| Q2 / Q5 | Generality: one architecture, one dataset, one estimator, sequential strategy only; joint / multi-head / alternating | Estimator: MSE ensembles (done for HF-only) + single MVE model (done); strategy: multi-head replay FT (WP6); architecture/dataset: discuss + optional | W + E | WP3, WP6, WP7 |
| Q3 | Novelty vs known results (ranking != calibration, AU unreliable under shift, NLL pitfalls); Bilbrey et al. 2025 | Table "known in general / what is new in MF" ; Bilbrey 2025 uses readout ensembles + quantile heads of MACE-MP-0, not the MF transition or training dynamics: add to related work | W | WP1 |

## 3. Merged work packages

Similar asks are merged so one piece of work answers several reviewers.

| WP | Theme | Answers | Priority | Cost |
|---|---|---|---|---|
| WP1 | Positioning, motivation, novelty table, related work | G1, Q3, J2, J3 | P0 | writing |
| WP2 | LF-only (pretrain-only) baseline on CC labels | G7, G4 | P0 | GPU inference only |
| WP3 | Mechanism of AU overconfidence + objective/estimator ablations | Q1/Q4, Q2/Q5, G4 | P0-P1 | FT-scale training (cheap) + CPU |
| WP4 | Protocol robustness: compute accounting, LR sweep, (model size) | G4 | P1 | CPU + cheap training |
| WP5 | Forces | G3 | P0 writing, **P1 experiment (Leonardo)** | LF-scale, ~0.9-1.7k GPU-h |
| WP6 | Fidelity-aware / multi-head training | J2, Q2/Q5 | P1 (6.2 local or Leonardo; 6.3 Leonardo) | FT-scale (6.2), LF-scale (6.3) |
| WP7 | Generality: 2nd dataset / architecture | G2, J1, Q2 | P0 writing, P2 experiment (Leonardo, D4) | moderate (water) to large (architecture) |
| WP8 | "Aleatoric" with deterministic labels: theory + renaming | G5 | P0 | writing |
| WP9 | Manuscript revision (figures, limitations, new appendix) | G6, J3, all | P0-P1 | writing + CPU figures |
| WP10 | Rebuttal assembly and logistics | all | P0 | writing |
| WP11 | Leonardo setup and sync (env, data, SLURM launcher, resume, smoke test) | enables WP5, WP6.3, WP7, parts of WP3/4 | **P0, critical path** | engineering + ~10 GPU-h |

LF-scale items (WP3.6 system MSE, WP4.4, WP5.2, WP6.3) need only WP11 and decisions D2-D4, not
compute. FT-scale items run on whichever machine has free GPUs first (DGX GPUs are full until the
MSE energy members finish, ~2026-11-08), so Wave A does not wait.

## 4. TODO (by work package)

Status: `[ ]` open, `[~]` claimed, `[x]` done, `[!]` blocked. Keep IDs stable.

### WP1 Positioning and novelty (G1, Q3, J2, J3)

- [ ] 1.1 Literature table of DFT->CC (or LF->HF) pretrain/fine-tune MLIP works, with year and
      setting; seed list in `draft_rebuttal.md` Section 7 (all verified 2026-11-06). Add 2-3 recent
      foundation-model fine-tuning examples to a higher level of theory.
- [ ] 1.2 Novelty table: for each paper finding, the closest known general result and what is new
      in the MF setting (draft in `draft_rebuttal.md` Section 1). Check that each "known" claim
      cites the right paper (Valdenegro-Toro 2022, Seitzer 2022, Rasmussen 2023, Kuleshov 2018,
      Bilbrey 2025).
- [ ] 1.3 Read Bilbrey et al. 2025 (npj Comput Mater 11, 109) in full; write exactly what they
      observe (readout ensembles + quantile heads of MACE-MP-0, overconfidence) and how it differs
      (no LF/HF transition, no training dynamics, no AU/EU of a deep ensemble). Add to related work.
- [ ] 1.4 Paragraph for the intro making the motivation explicit: CC labels are scarce and
      energy-only, so MF pretraining is how CC-quality MLIPs are built; UQ is what decides where to
      spend the next CC calculation (active learning built ANI-1x itself: Smith 2018).

### WP2 LF-only baseline on CC labels (G7, G4 "pretrain-only")

- [ ] 2.1 Design (write here before coding): predict the converged LF-only ensembles A (system) and
      C (energy) on `cc_train`, `cc_test_id`, `cc_test_ood` of every split; refit atomic reference
      energies by least squares on the CC training residuals (per member, mirroring the E0 refit of
      LF->HF fine-tuning), keep the variance heads unchanged; report RMSE/Spearman/AUSE/ENCE for AU,
      EU, TU next to HF-only and LF->HF. Also report the variant without E0 refit only if
      meaningful (absolute DFT and CC energies differ by large per-element offsets).
- [ ] 2.2 Implement as a `uq evaluate` step (new step, new output files; do not overwrite A/C outputs);
      unit test; `--dry-run`.
- [ ] 2.3 Run (1 GPU, inference only; ~1.5 min per split/test at batch 256 for M=10, cf. ledger
      2026-11-02), aggregate, ledger entry.
- [ ] 2.4 Add an LF-only (E0-refit) row/marker to Fig. 2 or an appendix table (`paper-figures`).

### WP3 Mechanism and objective ablations (Q1/Q4, Q2/Q5)

Hypothesis to test (paper Sec. 4.2, App. "Technical background"): during HF training the variance head fits the
shrinking *training* residuals and transfers that scale to held-out data. Predictions: (a) AU on
held-out data tracks the training MSE, not the held-out MSE; (b) re-initializing the head does not
prevent the collapse (it re-learns the same small residuals); (c) slower fine-tuning delays it in
proportion to the decrease of training error; (d) a head fitted on held-out residuals is calibrated.

- [ ] 3.1 (CPU, P0) From existing `epoch_quality_{train,id,ood}.csv` of B, D, E, F: per epoch, mean AU
      on held-out vs training MSE vs held-out MSE; plot AU-ENCE against the train/held-out MSE gap
      across epochs and protocols. Ledger entry. Checks prediction (a) with zero new compute.
- [ ] 3.2 (E, P0) **Variance-head reset at FT start** (Q1 literal ask): new experiments
      `ani1x-{system,energy}/mace-mve/lf-to-hf-reset-var` (names to confirm with `new-experiment`);
      re-initialize the variance branch of the readout, keep everything else as B/D; 5 x 10 members
      each from the same LF parents. Needs a small fork change (inside `### MVE ###` fences) or a
      checkpoint surgery script. Prediction (b).
- [ ] 3.3 (E, P1) **beta-NLL fine-tuning** (beta = 0.5, Seitzer et al. 2022), same parents. Tests
      "alternative training objective" (Q1).
- [ ] 3.4 (E, P1) **Held-out-fitted variance head**: after the B/D fine-tuning, freeze the backbone and
      mean readout, fit only the variance head on half of the CC validation set, select on the other
      half. Prediction (d); also a practical fix to report.
- [ ] 3.5 (E, P1) **Slower fine-tuning** (lr 1e-4 instead of 1e-3; shared with WP4.3). Prediction (c).
- [~] 3.6 (agent: mace-80) (E/A, P0) **MSE ensembles** (objective without variance head; EU-only
      estimator). HF-only pairs done (ledger 2026-11-03/04). Energy LF-only / LF->HF training on the
      DGX (9 partial members resumed 2026-11-06 20:01); automatic evaluation + pruning at 50/50; then
      `compare_uq` vs D and a ledger entry. System split: on Leonardo since 2026-11-07 01:12
      (mace-leo, 17 jobs, `RUNS.md`); then evaluation there by split (11.7) and sync back (11.9).
- [ ] 3.7 Evaluate 3.2-3.5 (`uq evaluate`, ~10 h per experiment on 1 GPU incl. per-epoch metrics;
      consider restricting to final + epoch quality), `compare_uq`-style paired contrasts vs B/D,
      ledger entries, one appendix figure "ablations of the AU collapse".

### WP4 Protocol robustness (G4)

- [ ] 4.1 (CPU) Count gradient steps and GPU-hours per protocol from run logs (ledger 2026-09-25 has
      hours); write the compute-matching argument: HF-only is validation-selected at epoch 30-111 of
      300 (App. "Training details"), so giving it more steps only increases validation NLL; the extra compute of
      LF->HF buys cheap LF labels, which is the point of MF training.
- [ ] 4.2 (E, P1) HF-only learning-rate sweep (lr 3e-3, 1e-3 vs 1e-2), energy split first, then system.
      ~50 members x ~1.4 h per config = ~8 h wall on 9 slots.
- [ ] 4.3 (E, P1) FT learning-rate sweep (1e-4 = WP3.5, and 1e-2), ~3 h wall per config.
- [ ] 4.4 (E, P2) Model size (64 / 256 channels) for HF-only only (LF->HF would need new LF parents).
- [ ] 4.5 Report in the rebuttal: does the ranking/calibration conclusion survive the best LR of each
      protocol? (paired over splits).

### WP5 Forces (G3)

- [ ] 5.1 (W, P0) Argument: ANI-1x CC/CBS labels are energy-only (verified in `data/raw/ani1x-release.h5`:
      only `ccsd(t)_cbs.energy` at CC level); most HF datasets lack forces (Messerly 2025); ANI-1ccx
      was trained on CC energies (Smith 2019). Energy uncertainty is what QBC acquisition uses (Smith 2018).
- [ ] 5.2 (E, P2, decision D3) LF pretraining with DFT energies + `wb97x_tz.forces` (energy NLL +
      force MSE), then CC energy fine-tuning as B/D. Cost ~ a full LF-only experiment (~16-19 h per
      member, 50 members per split type -> ~4-5 days on 9 slots for one split type). Needs fork change
      (MVE loss + forces).

### WP6 Fidelity-aware / multi-head training (J2, Q2/Q5)

- [ ] 6.1 (W, P0) Argument: Gardner et al. 2025 find PT/FT marginally more accurate than multi-head with
      two label sources; Tompa et al. 2026 find naive FT best for single-target deployment and
      multihead replay best for preserving OOD robustness; fidelity-specific E0s already absorb the
      per-element systematic offset.
- [ ] 6.2 (E, P1, decision D2) **Multi-head replay fine-tuning** from the existing LF-only parents:
      shared backbone, DFT head + CC head (MVE on both), CC fine-tuning with DFT replay. MACE has
      `multiheads_finetuning` upstream (`mace/cli/run_train.py`); check compatibility with the MVE
      readout in the fork first (estimate effort before claiming). Cost ~FT + replay (~1-2 h/member).
- [ ] 6.3 (E, P2) Joint training from scratch with a fidelity input/heads (expensive, LF-scale).

### WP7 Generality: dataset / architecture (G2, J1, Q2)

- [ ] 7.1 (W, P0) Argument: ANI-1x is the only large public dataset with CC labels on DFT geometries
      and a system/energy structure for controlled shifts; we vary split type, protocol, estimator
      (MVE ensemble / single MVE / MSE ensemble) and MF strategy (WP6).
- [ ] 7.2 (E, P2, decision D4) Options, cheapest first: (a) liquid water BLYP->CCSD(T) (Chen 2023) with
      aligned splits (current water splits are misaligned, see memory/known issues; water was removed
      from the paper 2026-11-02); (b) a second architecture (needs MVE readout in another code base).
      Only with explicit user approval.

### WP8 "Aleatoric" with deterministic labels (G5)

- [ ] 8.1 (W, P0) Theory paragraph (draft in `draft_rebuttal.md` Section 2.5): context-dependence of the
      AU/EU split (Hüllermeier & Waegeman 2021); conditional variance given the model's local
      representation is not zero (finite cutoff; incompleteness, Pozdnyakov et al. 2020); standard usage
      on deterministic DFT labels (Busk 2022, 2023; Tan 2023); excess-risk view (Lahlou et al. 2023)
      agrees with the reviewer that it is reducible: this is what our dynamics show.
- [ ] 8.2 (W, decision D5) Renaming in the revision: "learned variance (AU)" / "ensemble disagreement (EU)"
      at first use and in figures, or keep AU/EU with a stronger caveat.

### WP9 Manuscript revision (ICLR27-UQ-MF/)

- [ ] 9.1 Fig. 1: copy the Safari-safe `Workflow.pdf` (arXiv repo commit f342f6b) or run
      `papers/iclr27/figures/fix_workflow_safari.py` (G6). Check in Safari and pdf.js.
- [ ] 9.2 Limitations -> "Design choices and scope" (J3), naming each choice and why.
- [ ] 9.3 Related work: Bilbrey 2025, Messerly 2025, Cui 2025, Tompa 2026, Käser 2021/2023 (bib entries).
- [ ] 9.4 New appendix sections for WP2, WP3, WP4 (and WP6 if run); keep main-text numbers few
      (memory: results qualitative in main text, numbers in figures/appendix).
- [ ] 9.5 Mark all revision changes in `\revise{}` (blue) as ICLR expects; keep a changelog here.

### WP11 Leonardo setup and sync (critical path; target: smoke test passed by 2026-11-08)

- [x] 11.1 Access (2026-11-06 20:16): `ssh leonardo` from the DGX; 12 h certificate renewed by the
      user with `leonardo-login` (`LEONARDO.md` Section 1).
- [x] 11.2 (mace-80, 2026-11-06 21:52; job 63701795, aseMolec dropped from environment.yml) Environment (user, 2026-11-06: separate Miniconda): `$WORK/uq-mf/miniconda3`, env `mace`
      from `environment.yml`; clone `iclr_rebuttal` to `$WORK/uq-mf/mace`;
      `pip install --no-deps --no-build-isolation -e .`; `uq check` and the 58 unit tests pass; torch
      sees a GPU on a Booster node (smoke test). Heavy installs in a job if the login node's 10-min
      CPU limit bites (`LEONARDO.md` Section 3).
- [x] 11.3 (mace-80, 2026-11-06 20:45; 182 files verified on Leonardo) Data: make `data/MANIFEST.sha256` locally (ANI splits + dataset.yaml), rsync
      `data/ani1x-{system,energy}/` (2.3 GB), verify the manifest on Leonardo.
- [ ] 11.4 Run directory and variables: `UQ_RUNS=$WORK/uq-mf/runs` (overflow `$FAST/uq-mf/runs`),
      `PYTHON_BIN`, `WANDB_MODE=offline` with W&B files in `$CINECA_SCRATCH/uq-mf`; at most ~3 unpruned
      LF-scale experiments on `$WORK` (1 TB; ~255 GB each before pruning).
- [x] 11.5 (mace-80, 2026-11-06 21:10; `uq/training/slurm.py`, `uq/tests/test_slurm.py`) `uq train-slurm`
      writes/submits 1-GPU jobs; `uq train-member` gained `--resume`, skip-if-complete and the
      SLURM-assigned GPU. Original spec: SLURM launcher in `uq` (new module, e.g. `uq/training/slurm.py`, + `uq train --slurm`):
      writes sbatch scripts that run exactly `launch.command(job)` per member, optionally k members
      packed per 1-GPU job (`--cpus-per-task=8`, `--mem<=120G` so one GPU is charged as 1/4 node;
      `LEONARDO.md` Section 3), job arrays, `--dependency=afterok` for LF->HF children, `--resume` for
      jobs that hit the 24 h walltime, `--dry-run`; unit tests; local scheduler unchanged.
- [x] 11.6 (mace-80, 2026-11-06) Resume support: fork flag `--resume_exact` (default off; pinned
      commands unchanged) continues at epoch N+1 with the checkpoint's learning rate and scheduler
      state; `uq train --resume` backs up and cuts the results log to epoch N; `uq/tests/test_resume.py`;
      scratch test passed; used for the 9 partial MSE members. Not bit-exact (data-loader RNG restarts).
- [x] (mace-leo, 2026-11-07 01:36; `uq evaluate-slurm`, tested with job 63717684; system MSE jobs queued) 11.7 Evaluation on Leonardo: one job per split (`uq evaluate <id> --splits s --no-aggregate`),
      outputs synced back, `uq aggregate` locally (RUNS.md protocol item 8).
- [x] (mace-leo, 2026-11-07 01:10; jobs 63715545, 63715577, `RUNS.md`) 11.8 Smoke test on `boost_qos_dbg` (30 min): one HF-only member for a few epochs and one FT
      member; compare per-epoch time and first-epoch losses with the DGX run of the same seed (close,
      not bit-identical); check internet from the compute node; charge via `saldo -b`; throughput
      single and packed (2-3 members per GPU) into `RUNS.md`.
- [x] (mace-leo, 2026-11-07 01:36; `uq sync-back`, dry-run tested; first real use when the system MSE evaluation ends) 11.9 Sync-back script (rsync Leonardo -> local, paths from `RUNS.md` protocol), dry-run first.

### WP10 Rebuttal assembly

- [ ] 10.1 Fill revised-PDF deadline and character limits in Section 1 (response deadline: 2026-11-18).
- [ ] 10.2 General response (to all reviewers + AC) summarizing new experiments.
- [ ] 10.3 Per-reviewer responses from `draft_rebuttal.md`; replace every `[PENDING]` or drop the claim.
- [ ] 10.4 Final check: each number -> ledger entry; each prior-work statement -> reference; no author
      names; every "we added / revised" sentence matches a change that exists.
- [ ] 10.5 Post Wave A by 2026-11-10/11, Wave B by 2026-11-13/14, Wave C by 2026-11-16/17 (Section 1).

## 5. Compute plan: local server + Leonardo (timeline to 2026-11-18)

Unit costs (DGX A100 40 GB; ledger 2026-09-25 and MSE runs; Leonardo throughput to be measured in
WP11.8): HF-only member ~1.4 h; LF->HF fine-tuning member ~0.5 h; LF-only member ~15-16 h alone,
~19 h packed 3/GPU; full `uq evaluate` of one experiment ~10-11 GPU-h (parallel by split). One
50-member experiment: FT-scale ~25 GPU-h, HF-scale ~70 GPU-h, LF-scale ~800 GPU-h (+ ~11 GPU-h
evaluation each). Forces in training add an unknown factor (backward through forces): measure.

**Placement rule.** Local server: anything that uses existing local checkpoints and is cheap (FT
ablations, inference, evaluations of local experiments), so Wave A does not depend on Leonardo.
Leonardo: everything LF-scale or from scratch, plus overflow of FT/HF-scale items once WP11 works.
Every placement goes in `RUNS.md` before launch.

### Leonardo budget proposal (~10,000 GPU-h; to approve, D10)

| Item | WP | Machine | Estimate (GPU-h) | Needs |
|---|---|---|---|---|
| Setup, smoke tests, throughput probes | 11 | leonardo | ~20 | - |
| MSE ensemble, system split LF-only + LF->HF (+ eval) | 3.6 | leonardo | ~850 | D1b |
| LF pretraining **with DFT forces** + CC FT, energy split (+ eval) | 5.2 | leonardo | ~900-1,700 | D3 |
| Same, system split | 5.2 | leonardo | ~900-1,700 | D3 |
| Joint fidelity-aware training from scratch (DFT+CC heads), energy split | 6.3 | leonardo | ~900 | D2 |
| HF-only model size 64 / 256 channels, both splits | 4.4 | leonardo | ~300 | - |
| LF->HF model size (new LF parents), energy split, one size | 4.4 | leonardo | ~850 | optional |
| 2nd dataset (water, aligned splits; small) | 7.2 | leonardo | ~200 | D4 |
| Overflow of FT/HF-scale ablations (WP3, 4.2-4.3, 6.2) | 3, 4, 6 | either | ~500 | - |
| **Reserve (reruns, failures, reviewer follow-ups)** | | | **>= 2,000** | |

The full list exceeds the budget only if both forces runs are expensive; start with the energy split
(strongest effects in the paper) and add the system split if throughput allows.

### Timeline (revised 2026-11-06 evening)

| Dates | DGX (3 GPUs) | Leonardo | Writing |
|---|---|---|---|
| Nov 6-7 | GPUs full: MSE energy members (main scheduler 9 + 9 resumed); CPU: WP3.1 gap analysis, code for WP2 / WP3.2-3.4 | WP11.2-11.5 (env, data, layout, SLURM launcher) | WP1, WP8, G6 text; fill deadlines |
| Nov 7-8 | resumed members finish; WP2 inference when a slot frees | WP11.8 smoke test, then FT-scale ablations (WP3.2-3.5) and LF-scale jobs (WP5.2, WP6.3, WP3.6 system if D1b) | Wave A drafts per reviewer |
| Nov 8-10 | MSE energy complete -> automatic evaluation; WP4.2-4.3 sweeps | evaluations by split, sync back | **Post Wave A (Nov 10-11)** |
| Nov 11-13 | remaining evaluations, `compare_uq`, ledger entries | LF->HF children of LF-scale runs, evaluations | **Post Wave B (Nov 13-14)** |
| Nov 14-16 | aggregates and figures for the revision (CPU) | reruns / gaps from the reserve | revised PDF; **Wave C (Nov 16-17)** |
| Nov 17-18 | - | - | replies only; buffer |

What is running now: `RUNS.md` Section 3 (single source).

## 6. Decisions

Decided (user, 2026-11-06):
- **D1** MSE ensembles: resume on the DGX first (done 20:01), cheap experiments on the DGX, heavy ones
  on Leonardo.
- **D7** The 3-GPU cap applies only to the DGX; Leonardo uses its queues and our budget.
- **D8** Code reaches Leonardo by pushing `iclr_rebuttal` to `origin` (pull there, clean checkouts).
- **D9** Leonardo epoch checkpoints: prune after evaluation (every 5th epoch + each member's best +
  `.model`), as for `mace-mse` on the DGX.
- **D11** Leonardo environment: a separate Miniconda in `$WORK/uq-mf/miniconda3`.
- **D12** Anaconda ToS for the `defaults` channels accepted on our Leonardo Miniconda (user, 2026-11-06).
- **D1b** The system-split MSE members (100, LF-scale) run on Leonardo, not on the DGX (done 20:35: the
  local scheduler was replaced by energy-only ones).

Open:

- **D2** Fidelity-aware training: multi-head replay FT from existing parents (WP6.2, cheap) and/or
  joint training from scratch (WP6.3, ~900 GPU-h).
- **D3** Forces in pretraining (WP5.2): energy split only, or both splits.
- **D4** Second dataset/architecture (WP7.2): argue only, or water with aligned splits on Leonardo.
- **D5** Rename AU/EU in the revision (WP8.2).
- **D6** Is a revised PDF planned (ICLR allows revisions during discussion)?
- **D10** Leonardo budget allocation (Section 5 table).

## 7. Log (newest last)

- 2026-11-06 ~14:30 — mace-80: reviews received (1/2/2). On request: work committed on `main`
  (`30afae8`); MSE-ensemble scheduler, post-processing and 9 running members stopped (SIGTERM, ~14:31);
  branch `iclr_rebuttal`; `reviews.md`, this plan and `draft_rebuttal.md` (literature checked online).
- 2026-11-06 14:37 — mace-23 (MSE-study session) misread the stop as a crash and restarted scheduler and
  post-processing as systemd user units; kept, since the user then chose to resume MSE locally (D1).
- 2026-11-06 — user: deadline 2026-11-18, post results early; Leonardo with ~10,000 GPU-h. Added
  `RUNS.md`, WP11, the two-machine compute plan; D7, D8 decided.
- 2026-11-06 15:31 — codex-audit (read-only, on user request): `LOCAL_AUDIT_2026-11-06.md`.
- 2026-11-06 20:01 — mace-80: resume support (WP11.6) tested on a scratch copy, then the 9 partial MSE
  members resumed (units `uq-resume-mse-a/b`); audit findings fixed (missing checkpoint dir; two
  overstated draft claims). Pushed `32c2e64`. User rule: never message other sessions.
- 2026-11-06 20:16 — mace-80: Leonardo access from the DGX works (WP11.1); account, storage and budget
  facts recorded. Pushed `caa794c`.
- 2026-11-06 evening — mace-80: read the CINECA guide (filesystems, SLURM, accounting, linearization):
  `LEONARDO.md` (replaces `LEONARDO_LOGIN.md`). D9, D11 decided. Cleaned all rebuttal Markdown files;
  `README.md` added as entry point; `CLAUDE.md` points to it.
- 2026-11-06 20:35-20:50 — mace-80: D1b done on the DGX (old scheduler replaced by energy-only ones without
  killing members; `RUNS.md`). Leonardo: compute nodes of `lrd_all_serial` have internet; repo cloned,
  Miniconda installed, data copied and verified (WP11.3). Env creation blocked on D12 (Anaconda ToS).
- 2026-11-06 21:10 — mace-80: D12 accepted (user): Anaconda ToS accepted on our Leonardo Miniconda. First env
  job failed to solve (no `__cuda` on the serial node; strict channel priority): fixed with
  `CONDA_OVERRIDE_CUDA=12.4`, `CONDA_CHANNEL_PRIORITY=flexible` (the DGX uses flexible); resubmitted
  (63699144). WP11.5 done (`uq train-slurm`, 58 tests). `git push` from this shell hangs (stale VS Code
  askpass): commits reach Leonardo by `git bundle` (same hashes) until the user pushes.
- 2026-11-06 ~21:30 — mace-80 paused. Next: fix the pip step of the Leonardo env (job 63699144 log), then
  WP11.8 smoke test (`UQ_RUNS=$CINECA_SCRATCH/uq-mf/smoke/runs uq train-slurm ... --qos boost_qos_dbg`),
  then system-split MSE (D1b) and FT ablations. `git push` may need the user (stale VS Code askpass).
- 2026-11-07 01:20 — mace-leo: WP11.8 smoke test passed on Leonardo (losses match the DGX; LF member ~3.5 min/epoch
  with 3 packed per GPU, ~18 h per chain). System-split MSE (WP3.6, D1b) launched: 50 LF->HF chains in 17 1-GPU
  jobs, QOS normal 24 h, all running. Next on Leonardo: per-split evaluation script (11.7) and sync-back (11.9)
  before the jobs end (~2026-11-07 evening); other Leonardo experiments wait for D2-D4/D10 or code (WP3.2-3.5).
- 2026-11-07 01:36 — mace-leo (sole session, owner of all jobs): WP11.7 `uq evaluate-slurm` (per-split GPU jobs after
  training, `afterany` + completeness check, optional prune job) and WP11.9 `uq sync-back` (no overwrite) done; Leonardo
  inference matches the DGX. Queued after the system MSE training: 17 back-up chains (`--resume`), 10 evaluations, prune.
