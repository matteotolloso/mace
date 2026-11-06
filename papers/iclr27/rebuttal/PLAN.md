# ICLR 2027 rebuttal: central plan (Submission 18667)

**This file is the single source of truth for the rebuttal.** Every agent working on the
rebuttal reads it first, claims a task here before starting, and updates it when done.

| File | Role | Who edits |
|---|---|---|
| `papers/iclr27/rebuttal/reviews.md` | verbatim reviews, point IDs `G*` / `J*` / `Q*` | nobody (frozen; append follow-ups only) |
| `papers/iclr27/rebuttal/PLAN.md` (this) | per-reviewer tasks, merged work packages, TODO, decisions, log | every agent |
| `papers/iclr27/rebuttal/draft_rebuttal.md` | draft OpenReview responses with literature | writing agents |
| `papers/iclr27/rebuttal/RUNS.md` | every job on every machine (local, Leonardo), sync protocol, Leonardo budget | every agent launching or checking jobs |
| `results/results.md` | the only source of citable numbers | via the `record-result` skill |
| `PROGRESS.md` (repo root) | machine state: jobs, GPUs, paused MSE study | every agent running jobs |

Branch: `iclr_rebuttal` (created 2026-11-06 from `main` at `30afae8`). Manuscript changes go to
`ICLR27-UQ-MF/` (separate Git repo, synced with Overleaf; do not push there unless asked).

## 0. Coordination rules (read before doing anything)

1. **Claim before working.** Put `[~] (agent: <short name>, since YYYY-MM-DD HH:MM)` on the TODO
   line. Do not touch a line claimed by someone else; leave a note in the Log instead.
   Mark `[x]` when done, with the ledger entry or output path; `[!]` if blocked, with the reason.
2. **Numbers.** The rebuttal states only numbers in `results/results.md` (or in a file an entry
   lists). A new experiment is not done until it has a ledger entry. In `draft_rebuttal.md`,
   unrecorded numbers are written `[PENDING: WPx.y]`, never guessed.
3. **GPUs and machines.** Local server: never more than 3 GPUs for this project, counting every
   running job (`nvidia-smi`; `uq train` enforces it). Leonardo: QOS limits and the 10,000 GPU-h
   budget. **Before launching anything, on either machine, add a row to `RUNS.md`** (experiment,
   machine, commit) and follow its sync protocol; local pids/logs also go to `PROGRESS.md`.
4. **Artifacts.** Never delete, overwrite or retrain existing checkpoints, datasets or evaluation
   outputs. New variants are new experiments (`new-experiment` skill) under new ids.
5. **Five splits x ten members** for anything reported as a result. A reduced pilot is allowed only
   if labelled "pilot" everywhere (ledger, plan, rebuttal) and approved in Section 6.
6. **Anonymity.** No author names, affiliations, absolute paths or the OpenReview forum URL in
   tracked files of this repo (it is mirrored anonymously) or in the rebuttal text.
7. **Literature.** Every factual statement in the rebuttal about prior work cites a reference in
   Section 7 of `draft_rebuttal.md`; add new ones there with how they were verified.
8. **No commits/pushes** unless the user asks.

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
  character limit: [FILL IN from OpenReview]. **Post results as early as possible** to start the
  discussion (user, 2026-11-06): answer in waves, not in one block on the last day.
  - Wave A (target **2026-11-10/11**): text-only answers (G1, G3, G5, G6, J2/J3 arguments, Q3) +
    results that already exist (MSE vs MVE HF-only, single model, ledger) + first cheap results
    (WP2 LF-only, WP3.1 gap analysis, WP3.2 head reset, WP4.1 compute accounting).
  - Wave B (target **2026-11-13/14**): remaining FT ablations (WP3.3-3.5), LR sweeps (WP4.2-4.3),
    multi-head FT (WP6.2), MSE LF->HF (WP3.6).
  - Wave C (target **2026-11-16/17**): LF-scale experiments from Leonardo (forces WP5.2, joint
    fidelity-aware training WP6.3, model size WP4.4, optional 2nd dataset WP7.2) + revised PDF.
  - Keep **2026-11-18** as buffer for replies only; never post a result without a ledger entry.
- **Compute:** local server (3 GPUs) + **CINECA Leonardo, 10,000 GPU-h** (user, 2026-11-06). With
  Leonardo, the LF-scale experiments (~16-19 h per member, ~800 GPU-h per 50-member experiment)
  become affordable; the critical path is setting Leonardo up (WP11) without losing track of what
  runs where (`RUNS.md`).
- Already available and citable (ledger): single MVE model vs ensemble size (2026-11-02);
  MSE-ensemble baseline, HF-only, energy (2026-11-03) and system (2026-11-04) splits.
  Not yet available: MSE-ensemble LF->HF (training paused 2026-11-06, see `PROGRESS.md`).

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

Priorities changed on 2026-11-06 after the user added Leonardo (10,000 GPU-h): the LF-scale items
(WP3.6 system MSE, WP4.4, WP5.2, WP6.3) are no longer blocked by compute, only by WP11 and by
decisions D1-D4. Cheap FT-scale items stay on the local server so that Wave A does not wait for
Leonardo.

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

Hypothesis to test (paper Sec. 4.2, App. B): during HF training the variance head fits the
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
- [~] 3.6 (agent: mace-80) (E/A, P0) **MSE ensembles**, training resumed locally 2026-11-06 (D1; RUNS.md) (objective without variance head; EU-only estimator):
      HF-only pairs done (ledger 2026-11-03/04). LF->HF energy: 27/50 members done, training paused;
      resuming needs a decision (Section 6, D1) and a resume path for 9 partial members.
- [ ] 3.7 Evaluate 3.2-3.5 (`uq evaluate`, ~10 h per experiment on 1 GPU incl. per-epoch metrics;
      consider restricting to final + epoch quality), `compare_uq`-style paired contrasts vs B/D,
      ledger entries, one appendix figure "ablations of the AU collapse".

### WP4 Protocol robustness (G4)

- [ ] 4.1 (CPU) Count gradient steps and GPU-hours per protocol from run logs (ledger 2026-09-25 has
      hours); write the compute-matching argument: HF-only is validation-selected at epoch 30-111 of
      300 (App. D), so giving it more steps only increases validation NLL; the extra compute of
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

- [ ] 11.1 Access (user, 2026-11-06: do this first). Recommended: a CINECA step certificate written
      to files on the local server, so that agents here can `ssh leonardo` for its 12 h validity;
      the user renews it (twice a day, at the RUNS.md sync points). Alternative: a `claude rc`
      session on a Leonardo login node (then it owns the Leonardo rows of RUNS.md and syncs the
      file through git). Commands: final message of session mace-80, 2026-11-06; record the
      choice in `RUNS.md` Section 1.
- [ ] 11.2 Environment on Leonardo: conda env from `environment.yml` (login node has internet; check),
      `pip install --no-deps --no-build-isolation -e .`, torch 2.4 with CUDA working on a Booster node;
      `python -m unittest discover -s uq/tests -t .` (49 tests) and `uq check` pass.
- [ ] 11.3 Data: make `data/MANIFEST.sha256` locally (ANI splits + dataset.yaml), rsync
      `data/ani1x-{system,energy}/` (2.3 GB), verify the manifest on Leonardo.
- [ ] 11.4 Run directory and variables: `UQ_RUNS` on the work filesystem, `PYTHON_BIN`,
      `WANDB_MODE=offline`; check quotas (`cindata` or equivalent) against storage: an LF-scale
      experiment with `save_all_checkpoints` is ~255 GB (17 MB x 300 epochs x 50 members).
- [ ] 11.5 SLURM launcher in `uq` (new module, e.g. `uq/training/slurm.py`, + `uq train --slurm`):
      writes sbatch scripts that run exactly `launch.command(job)` per member, optionally k members
      packed per GPU (3 on 64 GB is safe memory-wise; check CPU threads: 8 cores per GPU), job arrays,
      parent dependencies (`--dependency=afterok`) for LF->HF children, `--dry-run`; unit tests;
      no change to the local scheduler's behaviour.
- [x] 11.6 (agent: mace-80, done 2026-11-06 20:01) Resume support (`uq train --resume`). Fork flag
      `--resume_exact` (default off; pinned commands unchanged) starts at epoch N+1 and keeps the
      checkpoint's learning rate and scheduler state; `launch.prepare_resume` backs up and cuts the
      results log to epoch N; 5 new unit tests (54 total pass); scratch test on a copy of s2_m5
      passed. Not bit-exact (data-loader RNG restarts), as for any resume. Original text:: continue a partial member from its latest epoch
      checkpoint (MACE `--restart_latest`) and keep `results/*_train.txt` free of duplicated epochs.
      Needed for 24 h walltime overruns (forces runs may exceed it; else `boost_qos_lprod`, 4 days,
      max 32 GPUs) and for the 9 paused local MSE members (D1).
- [ ] 11.7 Evaluation on Leonardo: one job per split (`uq evaluate <id> --splits s --no-aggregate`),
      outputs synced back, `uq aggregate` locally (RUNS.md protocol item 8).
- [ ] 11.8 Smoke test on `boost_qos_dbg` (30 min): one HF-only member for a few epochs and one FT
      member; compare per-epoch time and the first epochs' losses with the local run of the same
      seed (close, not bit-identical); measure charge with `saldo -b`; write throughput (single
      and packed) in `RUNS.md`.
- [ ] 11.9 Sync-back script (rsync Leonardo -> local, paths from `RUNS.md` protocol), dry-run first.

### WP10 Rebuttal assembly

- [ ] 10.1 Fill revised-PDF deadline and character limits in Section 1 (response deadline: 2026-11-18).
- [ ] 10.5 Post Wave A by 2026-11-10/11, Wave B by 2026-11-13/14, Wave C by 2026-11-16/17 (Section 1).
- [ ] 10.2 General response (to all reviewers + AC) summarizing new experiments.
- [ ] 10.3 Per-reviewer responses from `draft_rebuttal.md`; replace every `[PENDING]` or drop the claim.
- [ ] 10.4 Final check: each number -> ledger entry; each prior-work statement -> reference; anonymity.

## 5. Compute plan: local server + Leonardo (timeline to 2026-11-18)

Unit costs (local A100 40 GB; `PROGRESS.md`, ledger 2026-09-25; Leonardo throughput to be measured
in WP11.8): HF-only member ~1.4 h; LF->HF fine-tuning member ~0.5 h; LF-only member ~15-16 h alone,
~19 h packed 3/GPU; full `uq evaluate` of one experiment ~10-11 GPU-h (parallel by split). One
50-member experiment: FT-scale ~25 GPU-h, HF-scale ~70 GPU-h, LF-scale ~800 GPU-h (+ ~11 GPU-h
evaluation each). Forces in training add an unknown factor (backward through forces): measure.

**Placement rule.** Local server: anything that uses existing local checkpoints and is cheap (FT
ablations, inference, evaluations of local experiments), so Wave A does not depend on Leonardo.
Leonardo: everything LF-scale or from scratch, plus overflow of FT/HF-scale items once WP11 works.
Every placement goes in `RUNS.md` before launch.

### Leonardo budget proposal (10,000 GPU-h; to approve, D10)

| Item | WP | Machine | Estimate (GPU-h) | Needs |
|---|---|---|---|---|
| Setup, smoke tests, throughput probes | 11 | leonardo | ~20 | - |
| MSE ensemble, system split LF-only + LF->HF (+ eval) | 3.6 | leonardo | ~850 | D1 |
| MSE ensemble, energy split: 14 unstarted LF + children (the 9 partial ones resume locally) | 3.6 | leonardo | ~300 | D1, 11.6 |
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

### Timeline

| Dates | Local server (3 GPUs) | Leonardo | Writing |
|---|---|---|---|
| Nov 6-7 | WP2 inference; WP3.1 (CPU); implement head reset / beta-NLL / held-out head variants (WP3.2-3.4); start WP3.2 head reset (both splits, ~6 h wall on 9 slots) | WP11.1-11.4 (access, env, data, run dir) | WP1, WP8, G6 text; fill deadlines |
| Nov 7-8 | WP3.5 (FT lr 1e-4) and WP4.3; evaluations of WP3.2 by split on free GPUs; resume the 6 nearly finished MSE members if D1 (needs 11.6) | WP11.5-11.8 (SLURM launcher, resume, smoke test); then launch LF-scale jobs (WP5.2, WP3.6, WP6.3, WP4.4) | Wave A drafts per reviewer |
| Nov 9-10 | WP3.3, WP3.4, WP6.2 (or move to Leonardo); evaluations; ledger entries | LF-scale jobs running (~16-19 h, longer with forces) | **Post Wave A (Nov 10-11)** |
| Nov 11-13 | remaining evaluations; WP4.2 HF-only LR sweep | LF->HF children (~0.5 h each), evaluations by split, sync back | **Post Wave B (Nov 13-14)** |
| Nov 14-16 | aggregates, figures for the revision (CPU) | reruns / gaps from the reserve | revised PDF; **Wave C (Nov 16-17)** |
| Nov 17-18 | - | - | replies only; buffer |

### Currently running

See `RUNS.md` Section 3 (single source). Local check 2026-11-06 15:31 CET:
nine fresh MSE members plus the existing scratch resume test on GPUs 0/2/6;
production resume of nine partial members remains unfinished. Details and code-review
flags: [`LOCAL_AUDIT_2026-11-06.md`](LOCAL_AUDIT_2026-11-06.md). Leonardo not checked.

## 6. Decisions needed from the user

- **D1 MSE-ensemble LF->HF: DECIDED (user, 2026-11-06)**: resume the MSE experiments on the local
  server first (the 9 partial members with `uq train --resume`), then cheap experiments here and
  heavy ones on Leonardo. Open sub-question: the system-split MSE members (100, LF-scale) are in the
  local queue; move them to Leonardo to free local GPUs for the cheap ablations? History: training stopped 2026-11-06 on request: energy LF-only 27/50 and energy
  LF->HF 27/50 complete; 9 energy LF-only members stopped mid-run (6 at epoch ~260-269 of 300, 3 at
  ~180), partial outputs kept; system 0/100. It is the most direct answer to Q1/Q2 "other
  objective / estimator" for the *pretrained* protocol. With Leonardo, recommendation: resume the 9
  partial energy members locally (WP11.6), run the 14 unstarted energy members and the whole system
  split on Leonardo (~1,150 GPU-h).
- **D2 Fidelity-aware training**: multi-head replay FT from existing parents (WP6.2, cheap) and/or
  joint training from scratch (WP6.3, ~900 GPU-h).
- **D3 Forces in pretraining (WP5.2)**: energy split only, or both splits.
- **D4 Second dataset/architecture (WP7.2)**: argue only, or water with aligned splits on Leonardo.
- **D5 Rename AU/EU** in the revision (WP8.2).
- **D6** Is a revised PDF planned (ICLR allows revisions during discussion)?
- **D7 GPU cap scope: DECIDED (user, 2026-11-06)**: max 3 GPUs only on the local server; Leonardo
  uses its queues.
- **D8 Code sync: DECIDED (user, 2026-11-06)**: push `iclr_rebuttal` to `origin` once the files are
  updated; Leonardo pulls from it.
- **D9 Storage of Leonardo epoch checkpoints**: keep on Leonardo, prune after evaluation (every 5th
  epoch + best, as approved for `mace-mse` locally), or copy back (~255 GB per LF-scale experiment).
- **D10 Leonardo budget allocation** (Section 5 table).

## 7. Log (newest last)

- 2026-11-06 15:27–15:31 CET — [x] (agent: codex-audit) User requested a local-only,
  read-only handoff audit after the previous agent exhausted its tokens. Checking
  documentation, services, logs, resume changes and CPU checks; only audit notes
  written in `LOCAL_AUDIT_2026-11-06.md` and machine-state pointers. All 49 CPU tests
  and `uq check` pass; production jobs healthy, production resume still pending.
  No GPU ownership transfer, job changes or Leonardo work.

- 2026-11-06 — Reviews received (1/2/2). On user request: work committed on `main` (`30afae8`, not
  pushed); MSE-ensemble scheduler (pid 3432145), post-processing job (pid 2031848) and its 9 running
  members stopped with SIGTERM (no outputs deleted); branch `iclr_rebuttal` created. This plan,
  `reviews.md` and `draft_rebuttal.md` written; literature for the draft verified on the web.
- 2026-11-06 — User: response deadline 2026-11-18, post results early (waves A/B/C in Section 1);
  CINECA Leonardo available with 10,000 GPU-h. Added `RUNS.md` (machine/run registry, sync
  protocol, Leonardo budget ledger), WP11 (Leonardo setup, critical path), the two-machine compute
  plan and timeline (Section 5), decisions D7-D10. Leonardo facts (partitions, walltimes, node
  hardware) from docs.hpc.cineca.it; charging model and compute-node internet still to verify.
- 2026-11-06 — Correction: the MSE stop was at ~14:31 (not ~14:40). At 14:37 session mace-23 restarted
  the MSE scheduler + post-processing as systemd user units (misread the stop as a crash); the user then
  decided to resume MSE locally (D1), so they stay. Ownership of local GPU jobs: mace-80 (`RUNS.md`).
  D7, D8 decided. WP11.6 resume support implemented (fork: `--resume_exact`, default off; uq:
  `uq train --resume`, `launch.prepare_resume`); 49 tests pass; scratch end-to-end test running.
- 2026-11-06 20:05 — mace-80: read `LOCAL_AUDIT_2026-11-06.md` and acted on it: scratch resume test passed;
  `latest_epoch` now tolerates a missing checkpoint directory (audit's static edge case) + unit tests;
  the 9 partial MSE members resumed with two `uq-resume-mse-*` units; draft corrected where the audit
  flagged it (train-set AU "conservative", not "calibrated"; no "triggered by the fidelity change").
  User rule: never message other sessions (cost); coordinate through these files only.
