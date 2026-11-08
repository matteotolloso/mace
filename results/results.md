# Results ledger

The single source of truth for experimental claims. A result may be quoted in the
paper, the README or a chat **only if it has an entry here** (or is read directly
from an output file that an entry lists). Rules: `.claude/rules/reproducibility.md`.

How to use it:

- Add one entry per experiment or analysis, newest last. Never delete or rewrite an
  old entry; if a result is superseded, add a new entry and mark the old one
  `Status: superseded by <entry>`.
- Copy numbers from the output files, with units. Do not round beyond what the
  file stores unless the entry says so.
- Leave a field as `TODO` rather than guessing. An entry with a `TODO` in Commit,
  Command, Seeds or Outputs is not yet citable.
- `results/tables/` and `results/figures/` hold curated copies of tables and
  figures that back a claim; generated outputs otherwise stay where their scripts
  write them (`experiments/<id>/evaluation/`, `papers/<venue>/figures/out/`,
  `runs/active-learning/...`).
- **Paths in entries before 2026-09-26 refer to the old layout** (`experiment_X/...`,
  `dataset/...`, `eval/...`, `new_figures/...`, experiment letters A-F, wA-wC). They are
  kept as written; `uq where <old path>` prints the current location and
  `docs/migration-2026-09.md` has the full map. The old layout is the tag
  `iclr27-submission`.

---

## Template (copy below this line)

### <YYYY-MM-DD> <experiment name>

- **Status:** draft | verified | superseded by <entry>
- **Purpose:** the question this run answers (e.g. which RQ, figure or table).
- **Code commit:** `<git rev-parse HEAD>`; dirty tree: yes/no (list files if yes).
- **Command(s):** exact shell line(s), run from the repository root.
- **Configuration:** config file(s) and every CLI override.
- **Dataset and split:** dataset directory, test file(s), split seeds used, and any
  filter applied (e.g. training-support filter) with the number of rows removed.
- **Random seeds:** dataset split seeds and ensemble member seeds.
- **Compute:** cached data / CPU inference / GPU inference / retraining.
- **Metrics:** name, definition or script, unit, statistic (e.g. five-split mean
  with 95% Student-t interval, df=4) and value.

  | Metric | Protocol | Signal | Value | Source file |
  |---|---|---|---|---|
  | | | | | |

- **Output files:** every file produced or changed, with paths.
- **Interpretation:** what the numbers support, what they do not, and caveats.

---

## Entries

### 2026-09-23 ICLR27 main-text numbers (Figs. 2-6)

- **Status:** verified (recomputed from caches in this task; the numbers match the
  PDFs in `new_figures/out/`, which were built from the same caches on 2026-09-22).
- **Purpose:** every number quoted in the ICLR27 Results section and the captions of
  Figs. 2-6 (`ICLR27-UQ-MF/iclr2027_conference.tex`): RQ1 final models, RQ2
  dynamics, RQ3 shift/recalibration, AL proof of concept.
- **Code commit:** `6e385648e42701355117311f8cae0ffc41dcdbe3`; dirty tree: yes.
  `new_figures/fig5_decomposition.py` and `new_figures/fig6_acquisition.py` differ
  only in docstrings; `new_figures/paper_numbers.py` is new and untracked (reads only).
- **Command(s):**
  `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B paper_numbers.py > ../results/tables/iclr27_main_text_numbers.txt`
  Figures: `bash new_figures/make_all.sh` (not re-run; existing PDFs copied to the paper).
- **Configuration:** `experiment_{B,D,E,F}/config_*.yml` (HF-only E/F: 300 epochs,
  lr 0.01; LF->HF B/D: 100 fine-tuning epochs, lr 0.001, from A/C pretrained 300
  epochs); isotonic calibration from `eval/reliability.py` (per signal, fitted on the
  validation split, `out_of_bounds="clip"`); AL settings in
  `active_learning/ani_energy/runs/aggregate_epochs_50/summary_ci95.json`
  (budget 500, pool fraction 0.5, 50 additional epochs, lr 0.001, warm start).
- **Dataset and split:** `dataset/ani1x_{system,energy}_split_{0..4}/`, `cc_test_id`,
  `cc_test_ood` (5000 each); pretraining curves of Fig. 5 use the DFT files. The
  geometry-only training-support filter (`eval/support_filter.py`) removes 13-16 of
  5000 Energy-OOD and 0-2 ID configurations per split, nothing on the system split.
  AL: support filter of `active_learning/ani_energy/support_filter.py` (6-10 of ~2500
  held-out per split); `C1H5N1:887` was removed post hoc from split 3's held-out test.
- **Random seeds:** dataset split seeds 0-4; ensemble member seeds 0-9 (10 members).
- **Compute:** cached data, CPU only.
- **Metrics:** Spearman and AUSE: five-split arithmetic mean +/- 95% Student-t
  half-width (df=4). ENCE (15 equal-count bins, variance form), RMSE (meV/atom, per
  atom) and EU/AU ratios: geometric mean [95% log-space interval]. Per-epoch values are
  raw five-split means at the named epoch. Selected values (all in the output file):

  | Metric | Protocol | Signal | Value | Source file |
  |---|---|---|---|---|
  | Spearman, Energy-OOD | HF-only / LF->HF | TU | 0.556 +/- 0.018 / 0.640 +/- 0.019 | `results/tables/iclr27_main_text_numbers.txt` |
  | ENCE, Energy-OOD | HF-only / LF->HF | TU | 1.637 [1.318, 2.033] / 2.969 [2.079, 4.240] | same |
  | ENCE, Energy-OOD, isotonic | HF-only / LF->HF | TU | 6.422 / 8.113 | same |
  | RMSE, Energy-OOD (meV/atom) | HF-only / LF->HF | - | 75.145 / 58.579 | same |
  | OOD AU ENCE end pretraining -> end fine-tuning | LF->HF | AU | system 0.993 -> 81.913; energy 10.863 -> 207.071 | same |
  | AL gain over random, held-out Energy-OOD (meV/atom) | LF->HF | AU/EU/TU | 2.316 / 2.497 / 2.347 | `active_learning/ani_energy/runs/aggregate_epochs_50/summary_ci95.json` |
  | AL gain over random, held-out Energy-OOD (meV/atom) | HF-only | AU/EU/TU | 0.807 / 1.603 / 1.140 | same |

- **Output files:** `results/tables/iclr27_main_text_numbers.txt` (new);
  figures `new_figures/out/fig{2..6}_*.pdf` copied unchanged to
  `ICLR27-UQ-MF/figures/`.
- **Interpretation:** LF->HF ranks better (Spearman, AUSE) than HF-only for every
  signal on all four tests; calibration does not follow (AU ENCE lower for HF-only
  on all four; TU ENCE lower for HF-only on Energy-OOD). Energy shift improves
  ranking and degrades calibration; isotonic recalibration worsens AU and TU on
  Energy-OOD. Fine-tuning makes AU strongly overconfident on unseen data while
  training-set calibration stays near 1. The median EU/AU never exceeds 1 in any
  energy bin. In the AL proof of concept all six OOD gains exclude zero; the
  LF->HF-minus-HF-only gain excludes zero for all three signals. Caveats: approximate
  intervals (overlapping splits); AL results are conditional on the support filter
  and on the split-3 deletion; System-ID HF-only RMSE interval is widened by one
  split-0 configuration (4.402 eV/atom, 70% of that split's squared error).

### 2026-09-23 ICLR27 Fig. 3, AU/EU/TU during the high-fidelity stage

- **Status:** verified (recomputed from caches in this task; matches the values reported
  by the experiments session for the rebuilt `new_figures/out/fig3_hf_dynamics.pdf`).
- **Purpose:** Section 4.2 of the ICLR27 manuscript after Fig. 3 was extended from TU to
  AU, EU and TU; also HF-only calibration on its own CC training set.
- **Code commit:** `6e385648e42701355117311f8cae0ffc41dcdbe3`; dirty tree: yes
  (`new_figures/` scripts edited by the experiments session; `new_figures/paper_numbers.py`
  untracked, extended in this task with an all-signal Fig. 3 section, a HF-only train-set
  section and a `max` pick).
- **Command(s):**
  `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B paper_numbers.py > ../results/tables/iclr27_main_text_numbers.txt`
- **Configuration:** as the 2026-09-23 main-text entry; HF-only E/F 300 epochs, LF->HF
  B/D 100 fine-tuning epochs.
- **Dataset and split:** OOD test sets `cc_test_ood` of `dataset/ani1x_{system,energy}_split_{0..4}/`;
  System-OOD from cached per-epoch CSVs (filter removes nothing), Energy-OOD recomputed
  from `new_figures/_cache/epochs/{D,F}_<s>.npz` with the training-support filter;
  HF-only train set from cached `experiment_{E,F}/evaluation/cache/split_<s>/epoch_quality_train.csv`
  (unfiltered; training configurations are inside the support).
- **Random seeds:** split seeds 0-4; member seeds 0-9.
- **Compute:** cached data, CPU only.
- **Metrics:** raw five-split means at the named HF-stage epoch (AUSE arithmetic; ENCE
  geometric), 95% Student-t intervals (df=4). Values quoted in the paper:

  | Metric | Protocol | Signal | Value | Source file |
  |---|---|---|---|---|
  | System-OOD AU ENCE, first -> last HF epoch | HF-only / LF->HF | AU | 0.426 -> 13.640 / 0.374 -> 81.909 | `results/tables/iclr27_main_text_numbers.txt` |
  | Energy-OOD AU ENCE, first -> last HF epoch | HF-only / LF->HF | AU | 3.005 -> 72.040 / 3.405 -> 207.071 | same |
  | Train-set AU ENCE, last epoch | HF-only | AU | system 0.858, energy 0.885 | same |
  | System-OOD AUSE, LF->HF epoch 0 vs HF-only epoch 295 | both | AU/EU/TU | 0.210/0.215/0.201 vs 0.252/0.254/0.242 | same |
  | Energy-OOD AUSE, LF->HF epoch 5 vs HF-only epoch 295 | both | AU/EU/TU | 0.144/0.188/0.139 vs 0.178/0.232/0.165 | same |

- **Output files:** `results/tables/iclr27_main_text_numbers.txt` (regenerated; the values
  of the earlier entry are unchanged); `new_figures/out/fig3_hf_dynamics.pdf` copied to
  `ICLR27-UQ-MF/figures/`.
- **Interpretation:** AU becomes overconfident on unseen data during CC training in both
  protocols, and several times faster and further after LF pretraining, while AU stays
  calibrated on the training set; EU calibration does not deteriorate in the same way.
  At epoch 0 of fine-tuning, LF->HF beats HF-only's epoch-295 AUSE on System-OOD for all
  signals, but on Energy-OOD only ties for AU and EU and is worse for TU; it is better
  for all three within five epochs.

### 2026-09-23 ICLR27 interpretation checks: error scale, per-bin overconfidence, selected epochs, acquisition by signal

- **Status:** verified (computed from caches and training logs in this task; CPU only).
- **Purpose:** back the ICLR27 text that links AU/EU to the irreducible/reducible reading
  (Sections 4.2, 4.4, 4.5) and correct "AU stays calibrated on training data"; record the
  epochs chosen by validation-NLL checkpoint selection.
- **Code commit:** `6e385648e42701355117311f8cae0ffc41dcdbe3`; dirty tree: yes (`new_figures/`
  scripts edited by the experiments session; `new_figures/paper_numbers.py` untracked, extended
  in this task with `error_scale`, `acquisition_signals`, `au_bins`, `selected_epochs`, `system_ratio`).
- **Command(s):**
  `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B paper_numbers.py > ../results/tables/iclr27_main_text_numbers.txt`
- **Configuration:** as the 2026-09-23 main-text entry. Checkpoint selection re-derived with
  `eval/reliability.py::select_best_checkpoints` (selection key `loss`, mode `min`, as in
  `experiment_X/eval_X.sh`) from `experiment_{B,D,E,F}/results_<s>/` and `checkpoints_<s>/`.
- **Dataset and split:** `dataset/ani1x_{system,energy}_split_{0..4}/`. Per-epoch train/ID
  scales from cached `epoch_quality*_{train,id,ood}.csv` (unfiltered; train in support);
  Energy-OOD per-epoch from `new_figures/_cache/epochs/{C,D,F}_<s>.npz` with the support filter;
  converged models from `filtered_rows` (support filter).
- **Random seeds:** split seeds 0-4; member seeds 0-9 (50 members per experiment).
- **Compute:** cached data and log files, CPU only.
- **Metrics:** MSE/AU = mean squared per-atom error over mean per-atom AU on a test set, per split,
  geometric five-split mean [95% log-space interval]; per-bin MSE_b/MV_b in the 15 equal-count
  ENCE bins (bin 1 lowest variance); paired acquisition-gain differences, arithmetic mean +/-
  95% half-width (df=4).

  | Metric | Protocol | Signal | Value | Source file |
  |---|---|---|---|---|
  | Selected HF-stage epoch, median (range) over 50 members | LF->HF B / D | - | 2 (1-3) / 2 (1-3) | `results/tables/iclr27_main_text_numbers.txt` |
  | Selected HF-stage epoch, median (range) | HF-only E / F | - | 60 (30-111) / 67 (34-111) | same |
  | Train MSE/AU, last epoch | LF->HF system / energy | AU | 0.615 [0.431, 0.878] / 0.302 [0.093, 0.985] | same |
  | Train MSE/AU, last epoch | HF-only system / energy | AU | 0.544 [0.268, 1.105] / 0.171 [0.136, 0.214] | same |
  | Energy-OOD MSE_b/MV_b, fine-tuning epoch 95, bins 1/8/15 | LF->HF | AU | 63.935 / 232.221 / 2.390 | same |
  | Energy-OOD MSE_b/MV_b, fine-tuning epoch 95, bins 1/8/15 | LF->HF | TU | 2.413 / 7.502 / 1.726 | same |
  | Acquisition gain EU minus AU, held-out Energy-OOD (meV/atom) | HF-only / LF->HF | - | 0.796 +/- 0.545 / 0.182 +/- 0.226 | same |
  | Acquisition gain EU minus TU (meV/atom) | HF-only / LF->HF | - | 0.463 +/- 0.413 / 0.150 +/- 0.349 | same |
  | Median per-config EU/AU, System-OOD | LF-only / HF-only / LF->HF | - | 0.49 / 0.41 / 0.69 | same |
  | Median per-config EU/AU, System-ID | LF-only / HF-only / LF->HF | - | 0.46 / 0.40 / 0.63 | same |

- **Output files:** `results/tables/iclr27_main_text_numbers.txt` (regenerated; earlier sections
  unchanged, new sections appended at the end).
- **Interpretation:** (1) Validation-NLL selection keeps LF->HF members after 1-3 of 100
  fine-tuning epochs and HF-only members after 30-111 of 300 epochs: the converged LF->HF models
  of Fig. 2 are barely fine-tuned, and validation NLL itself rises as soon as AU starts to
  shrink. (2) On training data AU is not overconfident: train ENCE stays below 1 and the mean
  AU exceeds the mean squared training error in all four cases (point estimates; the HF-only
  system interval reaches 1.1). "Calibrated on training data" overstates this: "never
  overconfident / conservative" is what the numbers support. (3) After fine-tuning, AU
  overconfidence on Energy-OOD is concentrated in the low- and mid-variance bins (x64, x232)
  while the highest-variance bin is near calibrated (x2.4). (4) Without pretraining, EU is the
  best acquisition signal (EU-AU and EU-TU paired differences exclude zero); after pretraining
  the three signals are indistinguishable; AU-guided acquisition still beats random in both
  protocols. (5) Pretraining raises the median EU/AU on the system split as well (ID and OOD),
  above both single-fidelity baselines, with non-overlapping intervals. Caveat: the per-epoch
  train/ID caches are unfiltered (at most two ID configurations per split would be removed).

### 2026-09-23 Periodic water, five-split final-model reliability (wA, wB, wC)

- **Status:** verified (outputs complete: 5 splits x 10 members for each family; all
  three `eval_wX.sh` exited 0). Not part of the current paper.
- **Purpose:** water results with five-split confidence intervals (previously split 0 only).
- **Code commit:** splits 1-4 trained at `6e385648e42701355117311f8cae0ffc41dcdbe3`
  (dirty tree, no changes under `mace/`, `eval/`, `utils/`, `dataset/`, `experiment_w*/`);
  evaluation finished after the user's commit `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`,
  which changed no code under those paths. Split 0 was trained earlier (existing checkpoints).
- **Command(s):**
  ```bash
  PYTHON_BIN=$CONDA_PREFIX/bin/python nohup \
    $CONDA_PREFIX/bin/python experiments/scripts/train_water_splits.py \
    --gpus 6 7 --splits 1 2 3 4 --jobs-per-gpu 3        # 120 members, 11:38-19:19
  bash experiments/scripts/eval_water_after_training.sh <scheduler_pid> 6 7
  # = experiment_wA/eval_wA.sh 6 || experiment_wC/eval_wC.sh 7, then experiment_wB/eval_wB.sh 6
  ```
  Logs: `experiments/runs/water_scheduler.log`, `experiments/runs/water_eval_driver.log`,
  `experiments/runs/water/`.
- **Configuration:** `experiment_wA/config_wA.yml`, `experiment_wB/config_wB.yml`,
  `experiment_wC/config_wC.yml`; data paths overridden per split by the launchers;
  wB member (s, m) fine-tuned from wA member (s, m). 300 epochs (wA, wC), 100 (wB).
- **Dataset and split:** `dataset/water_<s>/blyp/test.xyz` (wA, 133 configurations) and
  `dataset/water_<s>/ccsdt/test.xyz` (wB, wC, 50 configurations), s = 0..4. No support
  filter (the aggregate uses every test configuration); `--trim 0.005` in `eval_wX.sh`
  affects only the per-split diagnostics, not the `*_raw.csv` rows aggregated here.
- **Random seeds:** split seeds 0..4; member seeds 0..9.
- **Compute:** retraining (splits 1-4) + GPU inference (all five splits), at most 2 GPUs (6, 7).
- **Metrics:** `eval/reliability.py` definitions, per-atom; five-split arithmetic mean with
  95% Student-t interval (df=4), as stored in `value_mean`, `value_ci95_low`,
  `value_ci95_high`. RMSE in eV/atom. ENCE with 15 equal-count bins.

  | Metric | Protocol | Signal | Value [95% CI] | Source file |
  |---|---|---|---|---|
  | RMSE (eV/atom) | LF-only wA (BLYP labels) | - | 0.005481279688324983 [0.003130532175280815, 0.007832027201369152] | `experiment_wA/evaluation/reliability_test_nocal_summary.csv` |
  | RMSE (eV/atom) | LF->HF wB | - | 0.008420521960402473 [0.006422621801591316, 0.01041842211921363] | `experiment_wB/evaluation/reliability_test_nocal_summary.csv` |
  | RMSE (eV/atom) | HF-only wC | - | 0.010786383522253978 [0.007743399064477342, 0.013829367980030614] | `experiment_wC/evaluation/reliability_test_nocal_summary.csv` |
  | AUSE | wA | AU / EU / TU | 0.5785812873497076 / 0.54427019600032 / 0.5707899978786812 | wA summary |
  | AUSE | wB | AU / EU / TU | 0.45840684508584084 / 0.34759292379931417 / 0.4118163642017584 | wB summary |
  | AUSE | wC | AU / EU / TU | 0.534796727399185 / 0.38000266012278155 / 0.4874926411801794 | wC summary |
  | ENCE | wA | AU / EU / TU | 0.7360202304179838 / 11.846886822314838 / 0.7071435057238457 | wA summary |
  | ENCE | wB | AU / EU / TU | 0.8026132958826352 / 11.648253810531674 / 0.7250425951359039 | wB summary |
  | ENCE | wC | AU / EU / TU | 1.3944561791464198 / 16.479705533680615 / 1.2580812342400567 | wC summary |
  | Spearman | wA | AU / EU / TU | -0.01690850105780101 / 0.021984398597179065 / -0.013819397546080515 | wA summary |
  | Spearman | wB | AU / EU / TU | 0.03454868641811743 [-0.07811773040118852, 0.14721510323742337] / 0.2339665481972255 [0.11085611508060542, 0.3570769813138456] / 0.0899765493034472 [-0.07175831317183981, 0.25171141177873424] | wB summary |
  | Spearman | wC | AU / EU / TU | -0.025028009994881573 / 0.1830245105718146 [-0.07462839754458608, 0.4406774186882153] / 0.01129409850178405 | wC summary |

  Intervals for the AUSE/ENCE rows and remaining Spearman rows are in the same files.

- **Output files:** `experiment_w{A,B,C}/evaluation/*.{csv,svg}` (five-split aggregates:
  `reliability_{train,test}_{nocal,cal}_*`, `epoch_quality_*`, `epoch_quality_finetune_mixed_*`
  for wB, `distribution_*`, `epoch_raw_train`, `train_curves`) and
  `experiment_w{A,B,C}/evaluation/cache/split_{0..4}/`; checkpoints in
  `experiment_w{A,B,C}/checkpoints_{1..4}/`. Paper-styled figures (added 2026-09-24,
  CPU from these caches): `new_figures/out/figW_{final_summary,reliability,dynamics}.{pdf,svg,png}`
  via `bash new_figures/make_all.sh`.
- **Interpretation:** On CCSDT water, LF->HF has lower RMSE than HF-only (point estimates
  8.4 vs 10.8 meV/atom; intervals overlap) and lower TU ENCE (0.73 vs 1.26). Ranking is weak
  everywhere: AU and TU Spearman intervals include zero for all three families; only EU ranks
  errors above chance, and its interval excludes zero only for wB. EU alone is badly
  miscalibrated (ENCE 10-16) while AU and TU are close to calibrated, i.e. EU is small
  relative to the errors and AU carries TU. Caveats: (1) the BLYP and CCSDT splits are
  shuffled independently, so wB's test geometries may have been seen during wA pretraining,
  biasing wB vs wC in favour of wB (`README.md`, audit notes); (2) 50 test configurations
  with 15 ENCE bins leave ~3 per bin, so ENCE is noisy; (3) wA is scored on BLYP labels and
  is not comparable with wB/wC; (4) no support filter.

### 2026-09-24 Water BLYP/CCSDT split overlap and selected epochs

- **Status:** verified (computed in this task from the dataset files and checkpoint logs).
- **Purpose:** quantify the water split misalignment for all five splits (README gave split 0
  only) and the validation-NLL selected epoch of the water members, for the ICLR27 appendix.
- **Code commit:** `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`; dirty tree: yes (uncommitted
  figure/doc work; `experiments/scripts/water_split_overlap.py` is new).
- **Command(s):**
  `$CONDA_PREFIX/bin/python -B experiments/scripts/water_split_overlap.py | tee results/tables/water_split_overlap.txt`
- **Configuration:** none (reads `dataset/water_<s>/{blyp,ccsdt}/{train,val,test}.xyz`, and
  `experiment_w{A,B,C}/{checkpoints,results}_<s>/` through `eval/reliability.py::select_best_checkpoints`
  with key `loss`, mode `min`, i.e. the validation Gaussian NLL, as in `eval_wX.sh`).
- **Dataset and split:** water splits 0..4 (65/10/25 of 531 BLYP and 200 CCSDT configurations:
  BLYP 345/53/133, CCSDT 130/20/50). Geometries are matched on positions rounded to 1e-4 A;
  every CCSDT geometry is found in the BLYP file.
- **Random seeds:** split seeds 0..4; member seeds 0..9.
- **Compute:** cached data (CPU, file reads only).
- **Metrics:**

  | Metric | Split 0 | 1 | 2 | 3 | 4 | Source file |
  |---|---|---|---|---|---|---|
  | CCSDT test (of 50) in BLYP train / val / test | 37 / 4 / 9 | 31 / 6 / 13 | 37 / 4 / 9 | 37 / 6 / 7 | 35 / 5 / 10 | `results/tables/water_split_overlap.txt` |
  | CCSDT val (of 20) in BLYP train | 12 | 13 | 14 | 10 | 16 | same |
  | CCSDT train (of 130) in BLYP train / val / test | 80 / 19 / 31 | 81 / 10 / 39 | 83 / 16 / 31 | 91 / 11 / 28 | 87 / 15 / 28 | same |

  | Selected epoch (0-based), 50 members | median | range | per-split medians | Source |
  |---|---|---|---|---|
  | wA (BLYP, 300 epochs) | 270.0 | 133-299 | 291.0, 249.5, 253.5, 287.0, 256.0 | same |
  | wB (CCSDT fine-tuning, 100 epochs) | 54.5 | 0-97 | 53.0, 22.5, 12.5, 59.0, 81.5 | same |
  | wC (CCSDT, 300 epochs) | 272.0 | 5-299 | 280.5, 69.5, 291.5, 281.0, 272.5 | same |

- **Output files:** `results/tables/water_split_overlap.txt`, `experiments/scripts/water_split_overlap.py`.
- **Interpretation:** In every split, 31-37 of the 50 CCSDT test geometries were in wA's BLYP
  training set (41-44 in BLYP train or val), so wB saw most of its test geometries at low
  fidelity during pretraining; wC saw none. The wB-vs-wC comparison on water is biased toward
  wB in all five splits, not only split 0. Unlike ANI (LF->HF selected after 1-3 fine-tuning
  epochs), water fine-tuning keeps training: wB's validation NLL is minimised at a median
  fine-tuning epoch of 54.5, with a wide per-split spread.

### 2026-09-24 Appendix reliability diagrams (ANI, A-F)

- **Status:** verified (figures only; no new metric).
- **Purpose:** paper-sized reliability diagrams for the ICLR27 appendix, replacing the
  illegible `experiment_X/evaluation/reliability_*.svg`.
- **Code commit:** `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`; dirty tree: yes (`new_figures/figA_reliability.py` new).
- **Command(s):** `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B figA_reliability.py`
- **Configuration:** none; reads `experiment_{A..F}/evaluation/cache/split_{0..4}/reliability_{id,ood}_{nocal,cal}_raw.csv`.
- **Dataset and split:** ANI system (A, B, E) and energy (C, D, F) splits 0..4, ID and OOD test
  files; training-support filter from `eval/support_filter.py` (same rows as the aggregates).
- **Random seeds:** split seeds 0..4; member seeds 0..9.
- **Compute:** cached data (CPU).
- **Metrics:** the per-panel ENCE annotations are five-split geometric means and equal the
  `value_geometric_mean` column of `experiment_X/evaluation/reliability_{id,ood}_nocal_summary.csv`
  (checked for C, D, F). Top-bin 95% RMSE interval width (hi/lo), printed by the script: system
  HF-only ID 3.5-4.3, system LF->HF OOD ~2.5, system LF-only OOD ~1.9, all energy nocal panels <= 1.35,
  energy HF-only OOD after recalibration 2.1-2.3 (AU, TU).
- **Output files:** `new_figures/out/figA_reliability_{system,energy}{,_cal}.{pdf,svg,png}`.
- **Interpretation:** presentation of already-recorded results; see the ledger entries for Fig. 2.

### 2026-09-24 Figure restyle (arrows, TU emphasis) and appendix dynamics/training-curve figures

- **Status:** verified (presentation only; `new_figures/paper_numbers.py` output is byte-identical
  to `results/tables/iclr27_main_text_numbers.txt` after the rebuild).
- **Purpose:** paper chat request: x-axis lower-is-better arrows point left; TU emphasised and
  AU/EU faded in every calibration panel; paper-sized appendix replacements for the raw
  `epoch_quality*-log-scale.svg` and `train_curves.svg`.
- **Code commit:** `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`; dirty tree: yes. Changed:
  `new_figures/{common,epoch_data,fig2_final_summary,fig3_hf_dynamics,fig4_rank_vs_calibration,
  fig5_decomposition,figA_recalibration,figA_reliability,figW_final_summary,figW_reliability,
  figW_dynamics}.py`, `make_all.sh`; new `figA_dynamics.py`, `figA_train_curves.py`.
- **Command(s):** `cd new_figures && PATH=$CONDA_PREFIX/bin:$PATH MPLCONFIGDIR=/tmp/mpl-cache bash make_all.sh`
- **Configuration:** none.
- **Dataset and split:** ANI A-F and water wA-wC caches, splits 0..4; Energy-OOD per-epoch values
  in `figA_dynamics_energy_*` recomputed from `new_figures/_cache/epochs/{C,D,F}_<s>.npz` with the
  training-support filter. The npz recomputation reproduces the cached unfiltered F RMSE/GNLL
  (split 0, epochs 0/150/295) to float32 precision.
- **Random seeds:** split seeds 0..4; member seeds 0..9.
- **Compute:** cached data (CPU).
- **Metrics:** none new. `figA_dynamics` plots Spearman, AUSE, ENCE, mean AU/EU variance, RMSE and
  mean Gaussian NLL (TU variance, `eval/epoch_quality.py` definition) per epoch; `figA_train_curves`
  plots the aggregated `train_curves.csv` columns.
- **Output files:** all of `new_figures/out/*.{pdf,svg,png}` regenerated; new
  `figA_dynamics_{system,energy}_{hf,lfhf}` and `figA_train_curves`.
- **Interpretation:** presentation change; no number changed.

### 2026-09-25 Training compute from run logs (ICLR27 computational-resources appendix)

- **Status:** verified (read from log timestamps in this task; no computation beyond summing).
- **Purpose:** replace the outdated "approximately 500 GPU hours" in the ICLR27 appendix
  "Computational resources".
- **Code commit:** `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`; dirty tree: yes (unrelated figure work).
- **Command(s):** inline Python (CPU): for every `experiment_{A,B,C,D,E,F,wA,wB,wC}/logs_<s>/mace_run-<m>.log`,
  s = 0..4, m = 0..9, take the difference between the last and the first timestamped log line
  and sum over runs.
- **Configuration / data:** training logs only; each run trains one ensemble member on one GPU.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (450 runs, none missing).
- **Compute:** file reads only.
- **Metrics:** wall-clock hours per run (log span), summed.

  | Experiment | Runs | Total (h) | Median per run (h) | Max (h) |
  |---|---|---|---|---|
  | A (LF-only, system) | 50 | 825.5 | 15.21 | 23.31 |
  | B (LF->HF, system) | 50 | 25.8 | 0.53 | 0.66 |
  | C (LF-only, energy) | 50 | 836.6 | 16.35 | 23.79 |
  | D (LF->HF, energy) | 50 | 29.3 | 0.57 | 0.83 |
  | E (HF-only, system) | 50 | 75.9 | 1.45 | 2.00 |
  | F (HF-only, energy) | 50 | 71.3 | 1.07 | 13.33 |
  | wA / wB / wC (water) | 150 | 32.9 / 4.3 / 12.4 | 0.74 / 0.09 / 0.27 | 0.79 / 0.13 / 0.31 |
  | **All training** | 450 | **1913.9** | | |

- **Output files:** none (numbers above).
- **Interpretation:** about 1,900 GPU hours of training, dominated by the 100 DFT pretraining runs.
  Upper bound on GPU time where several runs shared one GPU (the water scheduler ran up to three
  jobs per GPU). Excludes per-epoch evaluation inference, the acquisition experiment and
  preliminary runs.

### 2026-09-25 Axis-label arrows on Figs. 2-6 and figA_acquisition_id

- **Status:** verified (labels only; `new_figures/paper_numbers.py` output byte-identical to
  `results/tables/iclr27_main_text_numbers.txt`).
- **Purpose:** paper chat request: consistent direction arrows on paper Figs. 2, 4, 5, 6 and the
  Energy-ID acquisition appendix figure.
- **Code commit:** `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`; dirty tree: yes. Label-only edits in
  `new_figures/{fig2_final_summary,fig5_decomposition,fig3_hf_dynamics,fig6_acquisition,figA_acquisition_id}.py`.
- **Command(s):** `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B <script>.py` for each of the five scripts.
- **Configuration / dataset / seeds:** unchanged caches; splits 0..4, members 0..9.
- **Compute:** cached data (CPU).
- **Metrics:** none.
- **Output files:** `new_figures/out/{fig2_final_summary,fig5_decomposition,fig3_hf_dynamics,fig6_acquisition,figA_acquisition_id}.{pdf,svg,png}`.
- **Interpretation:** presentation change only. Fig. 2 titles "AUSE ← (ranking)", "ENCE ← (calibration)";
  rotated y-labels "AUSE ←", "ENCE ←" (render as downward arrows) in Figs. 3 and 5 scripts, and
  "RMSE gain over random (meV/atom) →" (renders as an upward arrow) in fig6 and figA_acquisition_id.

### 2026-09-25 Fig. 5 (fig3_hf_dynamics) y-label brackets

- **Status:** verified (labels only; `new_figures/paper_numbers.py` output byte-identical to
  `results/tables/iclr27_main_text_numbers.txt`).
- **Purpose:** paper chat request: y-labels "AUSE ← (ranking)" and "ENCE ← (calibration)" in paper
  Fig. 5, matching Fig. 2 and fig4 (rotated labels: "←" renders as a downward arrow).
- **Code commit:** `a5c717d3d2dd400229bd40d7761cc9d4247eda9c`; dirty tree: yes (label-only edit in
  `new_figures/fig3_hf_dynamics.py`).
- **Command(s):** `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B fig3_hf_dynamics.py`
- **Configuration / dataset / seeds:** unchanged caches; splits 0..4, members 0..9.
- **Compute:** cached data (CPU).
- **Metrics:** none.
- **Output files:** `new_figures/out/fig3_hf_dynamics.{pdf,svg,png}`.
- **Interpretation:** presentation change only.

### 2026-09-26 Fig. 6 (fig6_acquisition): note removed

- **Status:** verified (label-only; `new_figures/paper_numbers.py` output byte-identical to
  `results/tables/iclr27_main_text_numbers.txt`).
- **Purpose:** supervisor request relayed by the paper chat: remove the italic note
  "favours uncertainty ↑" from panel (a) of paper Fig. 6.
- **Code commit:** `c3bcc5669b5ccf02922e2eb5855b336f32020324`; dirty tree: yes (one `ax_ood.text` call and the then-unused `MUTED` import removed
  from `new_figures/fig6_acquisition.py`).
- **Command(s):** `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache $CONDA_PREFIX/bin/python -B fig6_acquisition.py`
- **Configuration / dataset / seeds:** unchanged caches (`active_learning/ani_energy/runs/aggregate_epochs_50/`).
- **Compute:** cached data (CPU).
- **Metrics:** none.
- **Output files:** `new_figures/out/fig6_acquisition.{pdf,svg,png}`.
- **Interpretation:** presentation change only.

### 2026-09-26 Repository restructure (no result changed)

- **Status:** verified (layout change only; every regenerated output compared with its
  committed version).
- **Purpose:** new layout for long-term work: `uq` package and CLI, experiment specs in
  `experiments/<dataset>/<model>-<uq>/<protocol>/`, datasets in `data/`, heavy artifacts in
  `runs/`, paper material in `papers/iclr27/`, docs in `docs/`.
- **Code commit:** tag `restructure-2026-09` (the branch `restructure`, squashed into one
  commit on 2026-09-26 at the user's request), from tag `iclr27-submission` (`c3bcc56`).
- **Command(s):** `python docs/restructure/migrate_layout.py apply` (moves), then
  `uq status`; `SOURCE_DATE_EPOCH=1767225600 uq aggregate`;
  `SOURCE_DATE_EPOCH=1767225600 uq figures iclr27`;
  `cd papers/iclr27/figures && python -B paper_numbers.py`;
  `python runs/_migration/equivalence.py`.
- **Configuration / dataset / seeds:** unchanged (configs lost only the path fields that
  the launcher always overrode; hyperparameters compared key by key).
- **Compute:** file renames (137,638 files, one filesystem) and CPU regeneration; one GPU
  briefly for an old-vs-new inference check.
- **Metrics:** none new. Verification: `uq status` 450/450 members; aggregate CSVs
  byte-identical, 177/177 aggregate SVGs identical after removing timestamps and random
  element ids; 20 paper figures identical (PNG bytes, SVG/PDF normalized);
  `paper_numbers.py` output identical to `results/tables/iclr27_main_text_numbers.txt`
  except the printed path of the active-learning summary; new launchers issue the old
  commands (training 36/36 + `--wandb_dir`, evaluation 574/574, aggregation 9/9); moved
  inference modules differ from the tag only in imports and docstrings; active-learning
  reports, the `aggregate_epochs_50` inputs (100) and manifest artifacts verify through
  the recorded-path aliases; 30 unit tests pass.
- **Output files:** `runs/_migration/manifest.json` (moves), `runs/_migration/golden/`
  (commands of the old scripts), docs in `docs/`.
- **Interpretation:** every result recorded above is unchanged. GPU inference is not
  bit-reproducible (EU per configuration varies by up to ~5% relative between runs of
  the same code), so re-evaluating on GPU changes per-split files slightly; aggregation
  from existing per-split files is byte-reproducible.

### 2026-09-26 Epoch-checkpoint pruning (no result changed)

- **Status:** verified (only files that no analysis reads were deleted; the pipeline's
  checkpoint and epoch selection was re-checked on every split).
- **Purpose:** reduce disk use (user request of 2026-09-26: delete the experiment epoch
  checkpoints; the active-learning checkpoints, the legacy ANI export and local W&B files
  were kept). MACE saved a checkpoint every epoch; the analyses read epochs divisible by 5
  (every 10th for `epoch_raw`) and each member's best-validation checkpoint.
- **Code commit:** tag `restructure-2026-09` (`uq/training/prune.py`, `uq prune-checkpoints`
  in `uq/cli.py`, `PRUNED.json`-aware `select_best_checkpoints` in
  `uq/evaluation/reliability.py`); run from the uncommitted working tree of branch
  `restructure` at `49e8d80`, whose versions of these files were then committed unchanged.
- **Command(s):**
  `MPLCONFIGDIR=/tmp/mpl-cache uq prune-checkpoints --apply > runs/_logs/prune_checkpoints_2026-09-26.log 2>&1`,
  then the same with `>>`. The first invocation used an earlier `prune.py` (no simulated
  pre-check): it pruned split 0 of `ani1x-energy/mace-mve/lf-only` and stopped at its
  post-check, because the selection code of the time required a checkpoint for every
  logged epoch (traceback in the log). Selection was then made to accept exactly the
  gaps listed in `PRUNED.json`, split 0 was re-checked, and the second invocation pruned
  the remaining 44 splits.
- **Configuration / dataset / seeds:** all nine experiments, splits 0-4, members 0-9.
  Rule: keep `epoch % 5 == 0`, each member's best checkpoint by validation loss (`loss`,
  `min`, as the evaluation pipeline selects), every `.model` and every non-checkpoint file.
- **Compute:** file deletion only (CPU); no inference, no retraining.
- **Metrics:** none. Deleted, from the 45 `runs/<id>/split_<s>/checkpoints/PRUNED.json`
  records (one event each): 83,618 files, 1,337,744,269,588 bytes (1245.9 GiB). Per
  experiment (second invocation; log above): C 9567 (+2391 in the first), D 3950,
  F 11959, A 11958, B 3950, E 11963, wA 11964, wB 3957, wC 11959. Afterwards `du -sh
  --apparent-size runs` reports 653G (experiments 116G + 116G + 94G, active learning 327G).
- **Verification:** before deleting, per split, the sampled epochs (n = 5 and n = 10) and
  the best checkpoints were computed on a simulated pruned folder and compared with the
  real one; after deleting, they were recomputed on the real folder (equal on all 44
  splits of the second invocation). Split 0 of C, pruned by the first invocation, was
  checked afterwards: selection runs on the pruned folder and samples 60 epochs (0-295)
  at n = 5 and 30 at n = 10, with the 10 best checkpoints present. `uq status`:
  450/450 members complete; `uq prune-checkpoints` dry run: 0 files left;
  `python -m uq.datasets.water_overlap` (selected epochs through the pruned folders)
  identical to `results/tables/water_split_overlap.txt`
  (`runs/_migration/water_overlap_after_prune.txt`); `paper_numbers.py` output
  (`runs/_migration/numbers_after_prune.txt`) differs from
  `results/tables/iclr27_main_text_numbers.txt` only in the printed path of the
  active-learning summary; the unit tests pass (46 at the tag; `uq/tests/test_checkpoints.py` covers
  recorded gaps, unrecorded gaps and a pruned best checkpoint).
- **Output files:** `runs/<id>/split_<s>/checkpoints/PRUNED.json` (45), the log above.
- **Interpretation:** every recorded result and tracked output is unchanged and can be
  regenerated from the kept checkpoints (up to GPU nondeterminism). Epoch analyses finer
  than every 5th epoch, and re-selecting checkpoints by another criterion, now need
  retraining; the evaluation code at tag `iclr27-submission` refuses the pruned folders.

### 2026-11-02 Single MVE model vs ensemble size (B, D, E, F; MSE-baseline study, free arm)

- **Status:** verified (computed in this task; the M = 10 rows reproduce the tracked
  `reliability_<test>_nocal_summary.csv` aggregates of the same experiments to the printed
  precision, e.g. D Energy-OOD TU Spearman 0.640 [0.621, 0.659] in both; per configuration,
  ensemble prediction/AU agree to <= 1e-5 relative and EU to a median 0.16% (GPU non-determinism)
  with `reliability_ood_nocal_raw.csv`, D split 0).
- **Purpose:** first arm of the AU/EU-decomposition ablation for the ICLR27 rebuttal / follow-up:
  what a single MVE model (AU only, M = 1) gives compared with the ten-member MVE ensemble
  (AU, EU, TU), and how the metrics grow with ensemble size. The MSE-ensemble arm
  (`*/mace-mse/*`, training since 2026-11-02) will be compared with `python -m uq.evaluation.compare_uq`.
- **Code commit:** `75f349e8167a731ca7a465d81dbda0c7339d0b87`; dirty tree: yes. New
  `uq/evaluation/{member_predictions,ensemble_size,compare_uq}.py`; edited `uq/evaluation/{pipeline,
  aggregate,summary}.py`, `uq/{cli,registry}.py`, `uq/training/*`, specs of B, D, E, F
  (`member_predictions: [id, ood]` appended), docs; unrelated user edits in `talk/`.
- **Command(s):**
  per experiment X in D F B E, split s in 0-4, test t in ood id (GPU 0, one split/test at a time):
  `uq evaluate X --gpu 0 --splits s --steps member_predictions_t ensemble_size_t --no-aggregate`;
  then `uq aggregate D F B E --only ensemble_size_` (CPU; other aggregates untouched);
  table: `for e in E B F D; do for t in id ood; do for m in 1 10; do uq summary $e --test $t --members $m; done; done; done > results/tables/ensemble_size_mve.txt`.
- **Configuration:** `experiments/ani1x-{system,energy}/mace-mve/{lf-to-hf,hf-only}/{experiment.yaml,config.yaml}`;
  final per-member checkpoints selected by validation Gaussian NLL (as `reliability`), batch size 256.
- **Dataset and split:** `data/ani1x-{system,energy}/split_{0..4}/cc_test_{id,ood}.xyz` (5000 each).
  Training-support filter applied: Energy-OOD 13-16, Energy-ID 0-2, System-ID 0-1, System-OOD 0
  configurations removed per split.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (no member or split missing).
- **Compute:** GPU inference (40 runs, ~1.5 min each on a shared A100) + CPU.
- **Metrics:** for ensemble size M, metrics (RMSE of the mean prediction, Spearman and AUSE of each
  signal vs squared error, ENCE 15 equal-count bins) averaged over the ten cyclic windows of M
  consecutive member seeds within a split; M = 1 = each member alone (AU = its own variance, EU
  undefined, TU = AU). Five-split mean [95% Student-t interval, df = 4]; ENCE as geometric mean
  [log-space interval]; RMSE in meV/atom. Uncalibrated. OOD values:

  | Metric (OOD) | Experiment | M = 1 (single model) AU | M = 10 AU / EU / TU | Source file |
  |---|---|---|---|---|
  | Spearman | E (system, HF-only) | 0.387 [0.359, 0.416] | 0.402 [0.358, 0.447] / 0.370 [0.330, 0.409] / 0.411 [0.367, 0.455] | `results/tables/ensemble_size_mve.txt` |
  | Spearman | B (system, LF->HF) | 0.462 [0.451, 0.473] | 0.490 [0.465, 0.514] / 0.501 [0.489, 0.514] / 0.518 [0.498, 0.537] | same |
  | Spearman | F (energy, HF-only) | 0.426 [0.398, 0.453] | 0.533 [0.511, 0.555] / 0.445 [0.430, 0.460] / 0.556 [0.537, 0.574] | same |
  | Spearman | D (energy, LF->HF) | 0.529 [0.521, 0.537] | 0.626 [0.608, 0.644] / 0.589 [0.571, 0.608] / 0.640 [0.621, 0.659] | same |
  | AUSE | E | 0.274 [0.258, 0.291] | 0.252 [0.231, 0.273] / 0.288 [0.265, 0.311] / 0.247 [0.225, 0.269] | same |
  | AUSE | B | 0.201 [0.183, 0.218] | 0.175 [0.156, 0.193] / 0.175 [0.154, 0.195] / 0.163 [0.145, 0.180] | same |
  | AUSE | F | 0.272 [0.254, 0.290] | 0.200 [0.188, 0.212] / 0.298 [0.285, 0.312] / 0.186 [0.178, 0.194] | same |
  | AUSE | D | 0.233 [0.224, 0.241] | 0.148 [0.139, 0.158] / 0.194 [0.179, 0.209] / 0.143 [0.134, 0.153] | same |
  | ENCE (geometric) | E | 0.759 [0.628, 0.919] | 0.332 [0.282, 0.392] / 2.018 [1.588, 2.565] / 0.391 [0.323, 0.472] | same |
  | ENCE (geometric) | B | 2.188 [1.632, 2.933] | 0.618 [0.462, 0.827] / 1.491 [0.971, 2.292] / 0.320 [0.255, 0.401] | same |
  | ENCE (geometric) | F | 6.975 [5.760, 8.446] | 3.246 [2.571, 4.097] / 14.535 [12.474, 16.936] / 1.636 [1.318, 2.031] | same |
  | ENCE (geometric) | D | 22.342 [17.491, 28.539] | 6.698 [5.002, 8.970] / 13.302 [10.513, 16.831] / 2.970 [2.078, 4.246] | same |
  | RMSE (meV/atom, mean) | E / B / F / D | 42.0 / 37.9 / 77.7 / 62.0 (single members) | 34.0 / 33.0 / 75.2 / 58.7 (ensemble) | same |

  ID values and all M = 1..10 rows: same file and `experiments/<id>/evaluation/ensemble_size_{id,ood}.csv`.

- **Output files:** new, tracked: `experiments/ani1x-{system,energy}/mace-mve/{lf-to-hf,hf-only}/evaluation/
  split_{0..4}/{members,ensemble_size}_{id,ood}.{npz,csv}` (14 MB of npz),
  `.../evaluation/ensemble_size_{id,ood}.{csv,svg}`, `results/tables/ensemble_size_mve.txt`.
  No existing output changed (a trial full `uq aggregate` into a scratch directory reproduced every
  existing aggregate CSV byte for byte; SVGs differ only in embedded date and element ids, so they
  were not rewritten).
- **Interpretation:** On all four OOD tests the ten-member TU ranks better than a single MVE model's
  AU (Spearman higher, AUSE lower, non-overlapping intervals on every test except System-OOD HF-only,
  E, where they overlap), and is better calibrated (ENCE lower on every test). Part of the gain is
  averaging AU itself over members (ensemble AU already improves on single-model AU); EU alone ranks
  worse than AU or TU on three of four tests (similar on B). The intervals are over splits for each
  quantity separately, not paired tests; paired contrasts come with `compare_uq` once the MSE
  ensembles are evaluated. Metrics keep improving up to M = 10 on Energy-OOD (curves in
  `ensemble_size_ood.svg`), so ten members is not saturated there.

### 2026-11-03 MSE-ensemble baseline, ANI-1x energy split, HF-only: evaluation and paired comparison with the MVE ensemble (F)

- **Status:** verified (computed in this task; the MVE side reproduces the recorded F values, e.g.
  Energy-OOD TU Spearman 0.556 [0.537, 0.574] and ENCE 1.636 [1.318, 2.031]; the MSE side of the
  comparison table, read from `ensemble_size_*`, matches its own reliability aggregate, e.g. OOD EU
  Spearman 0.438 [0.411, 0.465], AUSE 0.297 [0.277, 0.318]).
- **Purpose:** rebuttal / follow-up ablation: does the AU/EU decomposition (MVE ensemble) beat a
  standard deep ensemble whose only signal is member disagreement (MSE ensemble)? First pair: HF-only
  on the energy split (MVE twin F).
- **Code commit:** `60524e6562fce798f8dae501fa5b0cb6f4d39fe8`; dirty tree: yes. This task:
  `uq/evaluation/{reliability,epoch_quality}.py` (AUSE of a constant signal is NaN),
  `uq/evaluation/aggregate.py` (`mask_constant_signals`, `_defined`), `uq/evaluation/summary.py`
  (n read from the RMSE row); MVE aggregate CSVs of B, D, E, F re-computed into a scratch directory
  are byte-identical to the tracked ones. Other session's uncommitted edits (GPU cap 3, scheduler
  adoption) do not touch evaluation.
- **Command(s):** training `uq train ani1x-energy/mace-mse/hf-only ... --gpus 0 2 --jobs-per-gpu 3 --by-experiment`
  (2026-11-02 12:18 - 2026-11-03 01:26); evaluation `uq evaluate ani1x-energy/mace-mse/hf-only --gpu 0`
  (by `runs/_jobs/mse_postprocess.sh`, 01:43-13:14, log `runs/_jobs/ani1x-energy__mace-mse__hf-only.evaluate.log`);
  then, after the AUSE fix, `uq aggregate ani1x-energy/mace-mse/hf-only` (CPU) and
  `python -m uq.evaluation.compare_uq`.
- **Configuration:** `experiments/ani1x-energy/mace-mse/hf-only/{experiment.yaml,config.yaml}`: as F
  (`experiments/ani1x-energy/mace-mve/hf-only`) except `predict_mve: False`, `loss: "energy_only"`
  (per-atom energy MSE). Per-member checkpoint = lowest validation MSE: selected epochs median 282.5,
  range 244-299 of 300 (F: median 67), i.e. the MSE members use nearly the whole budget.
- **Dataset and split:** `data/ani1x-energy/split_{0..4}/cc_{train,val,test_id,test_ood}.xyz`;
  training-support filter: Energy-OOD 13-16 and Energy-ID 0-2 configurations removed per split.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (50/50 members; no split or member missing).
- **Compute:** retraining (50 members) + GPU inference + CPU.
- **Metrics:** paired per split (same data and member seeds); five-split mean [95% Student-t, df = 4]
  of each value and of the contrast with the MSE-ensemble EU: difference for Spearman/AUSE, geometric
  ratio [log-space interval] for ENCE and RMSE. Signals: `mse_eu` (MSE ensemble, its only signal),
  `mve_eu`, `mve_au`, `mve_tu` (MVE ensemble F), `mve_single_au` (one MVE member, mean over the ten).
  Values copied from `results/tables/mse_vs_mve.csv` (printed to 4 significant digits):

  | Test | Metric | MSE EU | MVE EU - MSE EU | MVE TU - MSE EU | single MVE AU - MSE EU |
  |---|---|---|---|---|---|
  | OOD | Spearman | 0.4379 [0.4112, 0.4646] | 0.006994 [-0.007935, 0.02192] | 0.1179 [0.09477, 0.141] | -0.01237 [-0.02983, 0.005095] |
  | OOD | AUSE | 0.2973 [0.2772, 0.3175] | 0.0009726 [-0.01117, 0.01312] | -0.1116 [-0.1291, -0.0942] | -0.02533 [-0.03918, -0.01149] |
  | OOD | ENCE (ratio) | 13.13 [11.02, 15.64] | x1.107 [0.9703, 1.263] | x0.1246 [0.09637, 0.1611] | x0.5312 [0.4166, 0.6772] |
  | ID | Spearman | 0.2593 [0.2282, 0.2904] | -0.004608 [-0.04112, 0.0319] | 0.05569 [0.02196, 0.08942] | 0.02643 [-0.004547, 0.0574] |
  | ID | AUSE | 0.3965 [0.3762, 0.4168] | -0.0003056 [-0.02465, 0.02404] | -0.05388 [-0.07504, -0.03272] | -0.02904 [-0.05069, -0.007384] |
  | ID | ENCE (ratio) | 1.514 [1.315, 1.741] | x0.8882 [0.7302, 1.08] | x0.2935 [0.2401, 0.3587] | x0.2833 [0.214, 0.3751] |

  RMSE (eV/atom), MSE ensemble: OOD 0.07387 [0.06712, 0.08129], ID 0.0154 [0.01503, 0.01577];
  ratio MVE ensemble / MSE ensemble: OOD 1.017 [0.9695, 1.067], ID 1.013 [0.9938, 1.032]; single MVE
  member / MSE ensemble: OOD 1.051 [1.001, 1.103], ID 1.281 [1.25, 1.313]. Full table (also MVE AU):
  `results/tables/mse_vs_mve.csv`; MSE-ensemble aggregates: `uq summary ani1x-energy/mace-mse/hf-only`.

- **Output files:** new: `experiments/ani1x-energy/mace-mse/hf-only/evaluation/` (per split and
  aggregates; aggregates rebuilt after the AUSE fix), `results/tables/mse_vs_mve.csv` (other pairs
  listed as missing until trained). Caveat: the per-split `epoch_quality_*.csv` of this experiment were
  written before the fix and hold an arbitrary `ause_aleatoric` for the all-zero AU (other AU columns
  NaN); `aggregate.mask_constant_signals` masks it, never read it from the per-split files.
- **Interpretation:** Without pretraining on the energy split: (1) the loss does not change the
  epistemic signal: MSE-ensemble EU and MVE-ensemble EU are indistinguishable on all three metrics on
  both tests (intervals contain 0 / 1); (2) the decomposition's total variance beats the deep-ensemble
  signal on every metric and test: OOD Spearman +0.118, AUSE -0.112, ENCE x0.12 (all intervals exclude
  0 / 1), ID likewise; most of the gain comes from AU (MVE AU alone also beats MSE EU, see the csv);
  (3) even one MVE model's AU ranks as well as the ten-member MSE ensemble on OOD (Spearman difference
  interval contains 0), sparsifies better (AUSE) and is better calibrated (ENCE x0.53), at the cost of a
  less accurate prediction (RMSE x1.05 OOD, x1.28 ID); (4) MVE training costs no ensemble accuracy
  (RMSE ratio intervals contain 1). Caveats: one dataset/protocol so far (system split and LF->HF pending);
  the MSE members were still improving at the end of the shared 300-epoch budget (selected epochs
  244-299), so a longer budget might favour the MSE ensemble's accuracy; ENCE of an uncalibrated EU is
  expected to be poor by construction (no recalibration compared yet).

### 2026-11-03 Epoch-checkpoint pruning: ani1x-energy/mace-mse/hf-only (no result changed)

- **Status:** verified (pruning tool's own pre/post checks; user-approved 2026-11-02 for every
  `mace-mse` experiment after its evaluation).
- **Command(s):** `uq prune-checkpoints ani1x-energy/mace-mse/hf-only --apply` (run by
  `runs/_jobs/mse_postprocess.sh` right after the evaluation, 2026-11-03 13:14; log
  `runs/_jobs/ani1x-energy__mace-mse__hf-only.prune.log`).
- **Effect:** deleted 11958 epoch checkpoints, 189.3 GiB; kept every `epoch % 5 == 0` checkpoint, each
  member's best (validation MSE) and every `.model`; record in
  `runs/ani1x-energy/mace-mse/hf-only/split_<s>/checkpoints/PRUNED.json`. The run folder is now 49 GB.
  Checkpoint re-selection (selected epochs above) works on the pruned folders.

### 2026-11-04 MSE-ensemble baseline, ANI-1x system split, HF-only: evaluation and paired comparison with the MVE ensemble (E)

- **Status:** verified (computed in this task; the MVE side reproduces the recorded E values, e.g.
  System-OOD TU Spearman 0.411 [0.367, 0.455], ENCE 0.391 [0.323, 0.472]; MSE side consistent with
  `uq summary ani1x-system/mace-mse/hf-only`, e.g. OOD EU Spearman 0.362 [0.333, 0.391]).
- **Purpose:** second pair of the MSE-vs-MVE ablation (see the 2026-11-03 energy entry): HF-only on
  the system split (MVE twin E).
- **Code commit:** `60524e6562fce798f8dae501fa5b0cb6f4d39fe8`; dirty tree: yes (the 2026-11-03 AUSE
  fix in `uq/evaluation/{reliability,epoch_quality,aggregate,summary}.py`, active during this
  evaluation; other session's GPU-cap/scheduler edits; ledger, PROGRESS.md).
- **Command(s):** training as in the energy entry (`uq train ... --by-experiment`, GPUs 0, 2 and,
  from 2026-11-03 11:00, 6; members done 2026-11-03 13:52); evaluation
  `uq evaluate ani1x-system/mace-mse/hf-only --gpu 0` (by `runs/_jobs/mse_postprocess.sh`,
  2026-11-03 14:04 - 2026-11-04 02:56); `python -m uq.evaluation.compare_uq`.
- **Configuration:** `experiments/ani1x-system/mace-mse/hf-only/{experiment.yaml,config.yaml}`: as E
  except `predict_mve: False`, `loss: "energy_only"`. Selected epochs (validation MSE): median 273,
  range 118-299 of 300 (E: median 60, range 30-111).
- **Dataset and split:** `data/ani1x-system/split_{0..4}/cc_*.xyz`; training-support filter: System-ID
  0-1, System-OOD 0 configurations removed per split.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (50/50; nothing missing).
- **Compute:** retraining (50 members) + GPU inference + CPU.
- **Metrics:** as in the energy entry (paired per split, df = 4; difference for Spearman/AUSE, geometric
  ratio for ENCE/RMSE). From `results/tables/mse_vs_mve.csv` (4 significant digits):

  | Test | Metric | MSE EU | MVE EU - MSE EU | MVE TU - MSE EU | single MVE AU - MSE EU |
  |---|---|---|---|---|---|
  | OOD | Spearman | 0.3618 [0.333, 0.3905] | 0.007908 [-0.01586, 0.03168] | 0.04945 [0.02668, 0.07222] | 0.02553 [0.01709, 0.03398] |
  | OOD | AUSE | 0.302 [0.2864, 0.3176] | -0.01408 [-0.02996, 0.001793] | -0.05517 [-0.06456, -0.04578] | -0.02771 [-0.0375, -0.01793] |
  | OOD | ENCE (ratio) | 1.475 [1.311, 1.659] | x1.368 [1.019, 1.837] | x0.2649 [0.207, 0.339] | x0.5149 [0.3999, 0.6629] |
  | ID | Spearman | 0.3219 [0.2965, 0.3472] | 0.01624 [-0.003098, 0.03558] | 0.0551 [0.03262, 0.07758] | 0.03307 [0.0131, 0.05305] |
  | ID | AUSE | 0.3511 [0.3344, 0.3677] | -0.02996 [-0.04876, -0.01116] | -0.07823 [-0.0943, -0.06215] | -0.05296 [-0.06751, -0.03842] |
  | ID | ENCE (ratio) | 1.396 [1.064, 1.831] | x1.57 [1.089, 2.263] | x0.2874 [0.1917, 0.4309] | x0.3857 [0.3044, 0.4887] |

  RMSE (eV/atom), MSE ensemble: OOD 0.02745 [0.02384, 0.03161], ID 0.02079 [0.02005, 0.02156];
  ratio MVE ensemble / MSE ensemble: OOD 1.235 [1.072, 1.423], ID 1.727 [1.018, 2.93]; single MVE
  member / MSE ensemble: OOD 1.521 [1.184, 1.955], ID 1.995 [1.123, 3.542]. Per-split ensemble ID RMSE
  (meV/atom, `ensemble_size_id.csv`, M = 10): MVE 74.4 / 27.7 / 31.6 / 29.7 / 30.8, MSE 20.3 / 21.5 /
  20.9 / 20.1 / 21.3 (MVE split 0 carries the known 4.402 eV/atom System-ID configuration, see the
  2026-09-23 main-text entry).

- **Output files:** new: `experiments/ani1x-system/mace-mse/hf-only/evaluation/` (per split and
  aggregates); `results/tables/mse_vs_mve.csv` regenerated (energy HF-only rows unchanged; LF->HF pairs
  still missing).
- **Interpretation:** Consistent with the energy split on uncertainty: MVE TU beats the MSE-ensemble
  EU on every metric and test (OOD Spearman +0.049, AUSE -0.055, ENCE x0.26; intervals exclude 0 / 1),
  and here even a single MVE model's AU ranks better than the ten-member MSE ensemble (OOD Spearman
  +0.026 [0.017, 0.034], AUSE -0.028). The EU itself is similar in ranking (Spearman intervals contain
  0) but the MVE EU is worse calibrated (ENCE x1.37 OOD, x1.57 ID). New relative to the energy split:
  the MSE ensemble is more accurate on the system split (MVE/MSE RMSE x1.24 OOD; ID x1.73, also
  excluding split 0 MVE is ~30 vs MSE ~21 meV/atom), whereas on the energy split accuracy was equal.
  So on unseen molecules the decomposition trades prediction accuracy for much better uncertainty;
  the ranking metrics rank each method's own errors. Caveats: MSE members selected late (median epoch
  273), MVE members early (median 60), under the same 300-epoch budget; LF->HF pairs pending.

### 2026-11-04 Epoch-checkpoint pruning: ani1x-system/mace-mse/hf-only (no result changed)

- **Status:** verified (pruning tool's own checks; user-approved 2026-11-02).
- **Command(s):** `uq prune-checkpoints ani1x-system/mace-mse/hf-only --apply` (by
  `runs/_jobs/mse_postprocess.sh`, 2026-11-04 02:57; log `runs/_jobs/ani1x-system__mace-mse__hf-only.prune.log`).
- **Effect:** deleted 11956 epoch checkpoints, 189.2 GiB; kept every 5th epoch, each member's best
  (validation MSE) and every `.model`; records in `runs/ani1x-system/mace-mse/hf-only/split_<s>/checkpoints/PRUNED.json`.
  Run folder now 49 GB; checkpoint re-selection works on it (selected epochs above).

### 2026-11-07 LF-only (pretrain-only) ensembles on the CC labels, E0 refit (rebuttal WP2; G7, G4)

- **Status:** verified (computed in this task: 5 splits x 10 members for both datasets; the paired table
  reproduces the five-split aggregates exactly, e.g. Energy-OOD LF-only TU Spearman 0.46713 in both files).
- **Purpose:** reviewer ge6X G7 / G4 ("pretrain-only" baseline): is the LF-only ensemble, used as is on
  the CC task, worse than fine-tuning (LF->HF) and than HF-only, in accuracy and in uncertainty?
- **Code commit:** `5b89bac1a3d20b2cf7467401259e2c66a374abfa` (aggregates, table); per-split predictions
  with `uq/evaluation/lf_on_hf.py` as committed in `9f8da8b` (D split 0 ran from the identical
  uncommitted file at 01:54; D s1-4 and B s0-2 on the DGX, B s3-4 on CINECA Leonardo at `c7710ef`, same
  code); dirty tree: no.
- **Command(s):** `python -m uq.evaluation.lf_on_hf --experiment D --split 0`;
  `uq evaluate D --gpu 0 --splits 1-4 --steps lfonly_baseline --no-aggregate`;
  `uq evaluate B --gpu 0 --steps lfonly_baseline --no-aggregate` (splits 0-2; split 3 hit CUDA OOM on the
  shared GPU); on Leonardo (job 63721292) `python -m uq evaluate ani1x-system/mace-mve/lf-to-hf --gpu 0
  --splits 3,4 --steps lfonly_baseline --no-aggregate`, copied back with `uq sync-back B --host leonardo
  --splits 3,4 --apply`; `uq aggregate D B --only lfonly_`; `python -m uq.evaluation.compare_lf_only`.
- **Configuration:** members of A (`ani1x-system/mace-mve/lf-only`) and C (`ani1x-energy/mace-mve/lf-only`),
  each at its best LF validation-NLL checkpoint (the ensembles of their own evaluations), unchanged except
  a per-member, per-element shift of the atomic reference energies fitted by least squares to the CC
  residuals of the split's CC training set (`cc_train.xyz`, 5000 configurations); variance heads unchanged.
  Step `lfonly_baseline`, enabled by `lf_only_baseline: true` in the B and D specs. Shifts and CC-train
  RMSE before/after per member: `lfonly_e0_refit.json` per split.
- **Dataset and split:** `data/ani1x-{system,energy}/split_{0..4}/cc_{train,val,test_id,test_ood}.xyz`
  (key `ccsd(t)_cbs.energy`); training-support filter as in B/D: Energy-OOD 13-16, Energy-ID 0-2,
  System-ID 0-1, System-OOD 0 configurations removed per split.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (nothing missing).
- **Compute:** GPU inference (+ CPU aggregation); no training.
- **Metrics:** uncalibrated (`nocal`), per atom; five-split mean [95% Student-t interval, df = 4];
  RMSE (eV/atom) and ENCE as geometric means. Values from
  `experiments/ani1x-{energy,system}/mace-mve/lf-to-hf/evaluation/lfonly_reliability_<test>_nocal_summary.csv`;
  paired contrasts (same split, same configurations) from `results/tables/lf_only_vs_hf.csv`
  (difference for Spearman/AUSE, ratio for RMSE/ENCE, each vs LF-only).

  | Data | Test | LF-only RMSE | LF-only TU Spearman | LF-only TU AUSE | LF-only TU ENCE | LF->HF vs LF-only: RMSE ratio / Spearman diff / AUSE diff / ENCE ratio | HF-only vs LF-only: RMSE ratio / Spearman diff / AUSE diff / ENCE ratio |
  |---|---|---|---|---|---|---|---|
  | Energy | ID | 0.01567 [0.01517, 0.0162] | 0.3885 [0.3794, 0.3976] | 0.3038 [0.2956, 0.312] | 1.856 [1.651, 2.086] | 0.5011 [0.4829, 0.52] / 0.06806 [0.04939, 0.08673] / -0.05762 [-0.06462, -0.05062] / 0.2298 [0.1947, 0.2714] | 0.995 [0.9527, 1.039] / -0.07355 [-0.1021, -0.045] / 0.03885 [0.02431, 0.0534] / 0.2393 [0.2085, 0.2747] |
  | Energy | OOD | 0.05699 [0.05227, 0.06212] | 0.4671 [0.4504, 0.4838] | 0.2291 [0.2215, 0.2366] | 3.244 [2.753, 3.822] | 1.028 [1.018, 1.038] / 0.1729 [0.1578, 0.1881] / -0.08562 [-0.09737, -0.07387] / 0.9152 [0.7298, 1.148] | 1.319 [1.29, 1.348] / 0.08871 [0.06264, 0.1148] / -0.04337 [-0.05517, -0.03156] / 0.5045 [0.3778, 0.6737] |
  | System | ID | 0.02367 [0.02218, 0.02527] | 0.4147 [0.4004, 0.4291] | 0.2678 [0.2601, 0.2755] | 1.502 [1.42, 1.587] | 0.6703 [0.6034, 0.7446] / 0.08547 [0.07813, 0.09281] / -0.06289 [-0.06662, -0.05916] / 0.2278 [0.1956, 0.2654] | 1.517 [0.9153, 2.513] / -0.03773 [-0.05077, -0.02468] / 0.005032 [-0.008813, 0.01888] / 0.2672 [0.2174, 0.3285] |
  | System | OOD | 0.03572 [0.02568, 0.04967] | 0.4545 [0.4422, 0.4668] | 0.2216 [0.2083, 0.2349] | 1.624 [1.415, 1.863] | 0.8884 [0.7572, 1.042] / 0.06335 [0.03794, 0.08876] / -0.05902 [-0.07292, -0.04511] / 0.1967 [0.1779, 0.2175] | 0.9492 [0.6824, 1.32] / -0.04318 [-0.09234, 0.005987] / 0.02515 [-0.002564, 0.05286] / 0.2407 [0.178, 0.3256] |

  AU of the LF-only ensemble (same files): ENCE Energy-ID 3.294 [2.993, 3.624], Energy-OOD 6.174
  [5.391, 7.071], System-ID 2.643 [2.464, 2.835], System-OOD 2.994 [2.62, 3.421]; AU, EU, AUSE and the
  isotonic (`cal`) variants are in the same summaries and in the table.

- **Output files:** new: `experiments/ani1x-{energy,system}/mace-mve/lf-to-hf/evaluation/split_{0..4}/
  lfonly_reliability_{id,ood}_{nocal,cal}_{raw,bins}.csv` + `.svg`, `lfonly_e0_refit.json`; aggregates
  `experiments/ani1x-{energy,system}/mace-mve/lf-to-hf/evaluation/lfonly_reliability_*`;
  `results/tables/lf_only_vs_hf.csv`. No existing output changed.
- **Interpretation:** Fine-tuning beats the zero-shot LF ensemble on every uncertainty metric and test
  (TU Spearman higher, AUSE lower, all intervals exclude 0; ENCE x0.20-0.23 except Energy-OOD, where the
  ratio interval contains 1), and is more accurate in-distribution (RMSE x0.50 Energy-ID, x0.67 System-ID);
  out of distribution its accuracy is the same as or slightly worse than LF-only (Energy-OOD x1.028, System-OOD
  interval contains 1). Against HF-only, the LF-only ensemble ranks in-distribution errors better (HF-only TU
  Spearman -0.074 Energy-ID, -0.038 System-ID) and is more accurate on Energy-OOD (HF-only RMSE x1.32),
  but ranks Energy-OOD errors worse (HF-only +0.089, AUSE -0.043) and is far less calibrated everywhere
  (HF-only ENCE x0.24-0.50; System-OOD ranking and System accuracy intervals contain 0 / 1). So "pretrain-only" is not a substitute for either
  protocol on the CC task: its variances, learned on DFT residuals, are miscalibrated for CC errors even
  after the per-element refit. Caveats: the refit is the most favourable zero-shot use (fitted on the CC
  training residuals the other protocols train on); other E0 choices (e.g. MACE's `E0s: average`) were not
  run; B s3-4 were computed on a different GPU type (A100 64 GB), within GPU non-determinism.

### 2026-11-07 Gradient steps per protocol to the selected checkpoints (rebuttal WP4.1; G4 compute matching)

- **Status:** verified (computed in this task from logs and result files; selected epochs reproduce the
  paper's "HF-only selected after 30-111 of 300 epochs, median 60 system / 67 energy").
- **Purpose:** reviewer ge6X G4 asks whether the protocols are compute-matched: how many optimizer steps,
  on which labels, does each protocol use up to the checkpoint that is evaluated?
- **Code commit:** `ffe0dca257b84122926625093326cf1e41defc6d`; dirty tree: no.
- **Command(s):** `python -m uq.evaluation.training_steps` (CPU).
- **Configuration:** `experiments/ani1x-{system,energy}/mace-mve/{lf-only,lf-to-hf,hf-only}` (A-F),
  `batch_size: 64`; MACE's training loader drops the last incomplete batch, so steps per epoch =
  floor(training set size / 64); the checkpoint of epoch e is written after e + 1 epochs; checkpoint
  selection as in the evaluation pipeline (`reliability.select_best_checkpoints`, lowest validation NLL).
  LF->HF counts add the parent member's steps to its selected checkpoint (fine-tuning starts there).
- **Dataset and split:** training set sizes from the MACE logs: 50000 DFT configurations (A, C), 5000 CC
  configurations (B, D, E, F), every split and member.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (50 members per experiment, none missing).
- **Compute:** log/file reads only.
- **Metrics:** median [min, max] over the 50 members, from `results/tables/training_steps.csv`.

  | Exp. | Labels | Steps/epoch | Selected epoch | Steps to selected checkpoint (own stage) | Incl. parent | Steps run (own stage) |
  |---|---|---|---|---|---|---|
  | A LF-only, system | DFT | 781 | 137 [78, 213] | 107778 [61699, 167134] | - | 234300 (300 epochs) |
  | B LF->HF, system | CC | 78 | 2 [1, 3] | 234 [156, 312] | 107973 [61855, 167368] | 7800 (100 epochs) |
  | E HF-only, system | CC | 78 | 60 [30, 111] | 4758 [2418, 8736] | - | 23400 (300 epochs) |
  | C LF-only, energy | DFT | 781 | 187.5 [128, 298] | 147218.5 [100749, 233519] | - | 234300 (300 epochs) |
  | D LF->HF, energy | CC | 78 | 2 [1, 3] | 234 [156, 312] | 147491.5 [100983, 233831] | 7800 (100 epochs) |
  | F HF-only, energy | CC | 78 | 67 [34, 111] | 5304 [2730, 8736] | - | 23400 (300 epochs) |

- **Output files:** `results/tables/training_steps.csv` (new).
- **Interpretation:** on CC labels, the evaluated LF->HF members have taken a median of 234 steps
  (2-3 epochs) and HF-only members about 20 times more (4758 / 5304), out of 23400 available; so LF->HF
  does not get more compute on the scarce labels, it gets much less, and HF-only is not short of steps
  (its validation NLL is best long before its budget ends). The extra compute of LF->HF (~1.1-1.5 x 10^5
  steps) is spent on the 10x larger DFT set. Wall-clock hours per protocol: ledger 2026-09-25. Steps are
  counted, not timed; members of LF->HF also differ in their parent's selected epoch.

### 2026-11-07 HF-only at lr 1e-3, energy split (rebuttal WP4.2, first of four LR-sweep experiments; G4)

- **Status:** verified for this experiment (5 splits x 10 members, evaluated and aggregated); WP4.2 as a whole is
  incomplete: `ani1x-energy/mace-mve/hf-only-lr3e-3` and the two system-split twins are trained, evaluated and
  pruned on Leonardo but not yet copied back (the ssh connection ended at ~09:50, certificate expired).
- **Purpose:** ge6X G4 ("optimal learning rates"): does the HF-only baseline improve at a lower learning rate,
  and do the LF->HF vs HF-only conclusions hold against the better-tuned baseline?
- **Code commit:** `1583dbdc42a55d5c43e22f16a2c4f8cf40fa4433` (aggregate, tables); training on Leonardo from the
  clone at `b4d07ac`/`c7710ef` (identical training command; `c7710ef` only adds a default-off fork option),
  evaluation there at `016dbb6`; dirty tree: no.
- **Command(s):** on Leonardo (`UQ_RUNS=/leonardo_scratch/fast/IscrC_UQ-MF/uq-mf/runs`):
  `uq train-slurm ani1x-energy/mace-mve/hf-only-lr1e-3 ani1x-energy/mace-mve/hf-only-lr3e-3 --pack 3 --time 08:00:00 --submit`,
  `uq evaluate-slurm ... --after <training jobs> --prune --submit` (10 evaluation jobs, status 0; prune job
  63720299 freed 189.7 GiB per experiment); on the DGX `uq sync-back ani1x-energy/mace-mve/hf-only-lr1e-3
  --host leonardo --checkpoints --apply`, `uq aggregate ani1x-energy/mace-mve/hf-only-lr1e-3`,
  `python -m uq.evaluation.compare_ablations`; the contrast against D by `compare_ablations.compare(D,
  hf-only-lr1e-3, ...)` (same functions, called inline).
- **Configuration:** `experiments/ani1x-energy/mace-mve/hf-only-lr1e-3/{experiment.yaml,config.yaml}`: F with
  `lr: 0.001` instead of 0.01; three members packed per A100 64 GB GPU.
- **Dataset and split:** `data/ani1x-energy/split_{0..4}/cc_*.xyz`; support filter as for F (Energy-OOD 13-16,
  Energy-ID 0-2 configurations removed per split).
- **Random seeds:** split seeds 0-4; member seeds 0-9 (none missing).
- **Compute:** retraining (50 members, Leonardo) + GPU inference (Leonardo) + CPU aggregation (DGX).
- **Metrics:** uncalibrated, per atom; five-split mean [95% t interval, df = 4]; paired per split; RMSE and ENCE
  geometric (ratios). Values from `experiments/ani1x-energy/mace-mve/hf-only-lr1e-3/evaluation/reliability_*_nocal_summary.csv`
  and `results/tables/ablations_vs_reference.csv` (vs F); the D rows were computed inline as stated.

  | Test | Metric (TU) | HF-only lr 1e-3 | vs F (lr 1e-2) | D (LF->HF) vs HF-only lr 1e-3 |
  |---|---|---|---|---|
  | ID | RMSE (eV/atom) | 0.01555 [0.0153, 0.0158] | x0.997 [0.9638, 1.031] | x0.5052 [0.4831, 0.5282] |
  | ID | Spearman | 0.3341 [0.3145, 0.3536] | +0.01913 [0.01598, 0.02228] | +0.1225 [0.09017, 0.1548] |
  | ID | AUSE | 0.3362 [0.3302, 0.3422] | -0.006515 [-0.012, -0.001025] | -0.08996 [-0.1032, -0.07675] |
  | ID | ENCE | 0.3151 [0.2788, 0.3562] | x0.7095 [0.5939, 0.8475] | x1.354 [1.176, 1.559] |
  | OOD | RMSE (eV/atom) | 0.07082 [0.06703, 0.07483] | x0.9425 [0.8972, 0.9901] | x0.8271 [0.7784, 0.8789] |
  | OOD | Spearman | 0.5867 [0.5702, 0.6032] | +0.03087 [0.01505, 0.04669] | +0.05334 [0.02204, 0.08464] |
  | OOD | AUSE | 0.165 [0.156, 0.174] | -0.02075 [-0.03098, -0.01051] | -0.02151 [-0.0379, -0.005125] |
  | OOD | ENCE | 2.289 [2.024, 2.589] | x1.399 [1.08, 1.811] | x1.297 [0.8442, 1.993] |

- **Output files:** new: `experiments/ani1x-energy/mace-mve/hf-only-lr1e-3/evaluation/` (per split and
  aggregates), `runs/ani1x-energy/mace-mve/hf-only-lr1e-3/` (pruned checkpoints, models, logs, results);
  `results/tables/ablations_vs_reference.csv` (new; only this pair complete so far).
- **Interpretation:** the lower learning rate makes HF-only slightly better at ranking (TU Spearman +0.02 ID,
  +0.03 OOD) and slightly more accurate out of distribution (x0.94), better calibrated in distribution
  (ENCE x0.71) and worse calibrated on Energy-OOD (x1.40). Against this better-tuned baseline, LF->HF (D) is
  still more accurate (x0.51 ID, x0.83 OOD) and ranks better on both tests (intervals exclude 0), and is still
  less calibrated in distribution (ENCE x1.35); on Energy-OOD the calibration gap of the paper (D worse) is no
  longer significant (x1.30, interval contains 1). So the accuracy and ranking conclusions survive this LR;
  the Energy-OOD calibration contrast weakens. Pending: lr 3e-3 and the system split.

### 2026-11-07 HF-only learning-rate sweep complete: lr 1e-3 and 3e-3, energy and system splits (rebuttal WP4.2; G4)

- **Status:** verified (4 experiments x 5 splits x 10 members, evaluated on Leonardo and aggregated here). The
  energy lr 1e-3 numbers repeat the entry "HF-only at lr 1e-3, energy split" above (same files, same values).
- **Purpose:** ge6X G4 ("optimal learning rates"): does HF-only improve at a lower lr than the paper's 1e-2, and do
  the LF->HF vs HF-only conclusions hold against the best-tuned HF-only baseline on both splits?
- **Code commit:** `51341ed617f56cbbc1f34b1c6095c6cdef289575` (sync-back, aggregate, tables); training on
  Leonardo at `b4d07ac` (energy) and `016dbb6` (system), evaluation at `016dbb6`; the training command is the
  same at both; dirty tree: only the new outputs listed below.
- **Command(s):** on Leonardo (`UQ_RUNS=/leonardo_scratch/fast/IscrC_UQ-MF/uq-mf/runs`): `uq train-slurm <exps>
  --pack 3 --time 08:00:00 --submit` (system: `--after-ok 63720299`), `uq evaluate-slurm <exps> --after <training
  jobs> --prune --submit` (energy eval 63720281-63720298, prune 63720299; system eval 63722482-63722492, prune
  63722493; all COMPLETED). On the DGX: `UQ_REMOTE_RUNS=/leonardo_scratch/fast/IscrC_UQ-MF/uq-mf/runs uq
  sync-back <exp> --host leonardo --checkpoints --apply` (`runs/_jobs/sync_morning_2026-11-07.sh`), `uq aggregate
  <exp>` for each, `python -m uq.evaluation.compare_ablations` (vs F / E); LF->HF vs each tuned HF-only by
  `compare_ablations.compare(<lf-to-hf>, <hf-only-lrX>, "", test)` (same functions, inline).
- **Configuration:** `experiments/ani1x-{energy,system}/mace-mve/hf-only-lr{1e-3,3e-3}/{experiment.yaml,config.yaml}`:
  F / E with `lr` 0.001 or 0.003 instead of 0.01; three members packed per A100 64 GB GPU.
- **Dataset and split:** `data/ani1x-{energy,system}/split_{0..4}/cc_*.xyz`; support filter as for F / E.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (none missing).
- **Compute:** retraining (200 members, Leonardo) + GPU inference (Leonardo) + CPU aggregation (DGX).
- **Metrics:** uncalibrated, per atom; TU for Spearman/AUSE/ENCE, ensemble-mean RMSE (eV/atom); five-split mean
  [95% t interval, df = 4], paired per split; RMSE and ENCE geometric (ratios), Spearman and AUSE differences.
  Sources: `experiments/<exp>/evaluation/reliability_*_nocal_summary.csv`,
  `results/tables/ablations_vs_reference.csv` (vs paper lr), LF->HF contrasts printed by the inline call.

  Tuned HF-only vs the paper's HF-only (lr 1e-2):

  | Split | lr | Test | RMSE | Spearman | AUSE | ENCE | RMSE ratio | Spearman diff | AUSE diff | ENCE ratio |
  |---|---|---|---|---|---|---|---|---|---|---|
  | energy | 1e-3 | ID | 0.01555 | 0.3341 | 0.3362 | 0.3151 | 0.997 [0.9638, 1.031] | +0.01913 [0.01598, 0.02228] | -0.006515 [-0.012, -0.001025] | 0.7095 [0.5939, 0.8475] |
  | energy | 1e-3 | OOD | 0.07082 | 0.5867 | 0.165 | 2.289 | 0.9425 [0.8972, 0.9901] | +0.03087 [0.01505, 0.04669] | -0.02075 [-0.03098, -0.01051] | 1.399 [1.08, 1.811] |
  | energy | 3e-3 | ID | 0.01549 | 0.328 | 0.3355 | 0.4093 | 0.9931 [0.9766, 1.01] | +0.01301 [-0.01357, 0.03958] | -0.007189 [-0.02513, 0.01076] | 0.9215 [0.7735, 1.098] |
  | energy | 3e-3 | OOD | 0.07177 | 0.572 | 0.1701 | 1.639 | 0.9551 [0.9077, 1.005] | +0.01621 [0.0006983, 0.03172] | -0.01559 [-0.02447, -0.006698] | 1.001 [0.7896, 1.27] |
  | system | 1e-3 | ID | 0.0302 | 0.4103 | 0.2579 | 0.3524 | 0.8413 [0.501, 1.413] | +0.0333 [0.02653, 0.04007] | -0.0149 [-0.03029, 0.0004951] | 0.8784 [0.6974, 1.106] |
  | system | 1e-3 | OOD | 0.03423 | 0.4467 | 0.233 | 0.3528 | 1.01 [0.9631, 1.058] | +0.03534 [0.004943, 0.06574] | -0.01374 [-0.02852, 0.001035] | 0.9028 [0.7914, 1.03] |
  | system | 3e-3 | ID | 0.02917 | 0.3892 | 0.2724 | 0.3991 | 0.8124 [0.471, 1.401] | +0.01217 [0.002254, 0.02208] | -0.000386 [-0.01802, 0.01725] | 0.9947 [0.9113, 1.086] |
  | system | 3e-3 | OOD | 0.03204 | 0.4265 | 0.2437 | 0.3931 | 0.9452 [0.9009, 0.9916] | +0.01515 [-0.0179, 0.04819] | -0.00304 [-0.02132, 0.01524] | 1.006 [0.8107, 1.248] |

  LF->HF (D energy, B system; paper settings) vs each tuned HF-only, paired (ratio LF->HF / HF-only, difference LF->HF - HF-only):

  | Split | HF-only lr | Test | RMSE ratio | Spearman diff | AUSE diff | ENCE ratio |
  |---|---|---|---|---|---|---|
  | energy | 1e-3 | ID | 0.5052 [0.4831, 0.5282] | +0.1225 [0.09017, 0.1548] | -0.08996 [-0.1032, -0.07675] | 1.354 [1.176, 1.559] |
  | energy | 1e-3 | OOD | 0.8271 [0.7784, 0.8789] | +0.05334 [0.02204, 0.08464] | -0.02151 [-0.0379, -0.005125] | 1.297 [0.8442, 1.993] |
  | energy | 3e-3 | ID | 0.5071 [0.4902, 0.5247] | +0.1286 [0.1011, 0.1561] | -0.08929 [-0.1011, -0.07746] | 1.042 [0.9038, 1.202] |
  | energy | 3e-3 | OOD | 0.8162 [0.766, 0.8696] | +0.068 [0.03903, 0.09697] | -0.02667 [-0.03874, -0.0146] | 1.812 [1.061, 3.093] |
  | system | 1e-3 | ID | 0.5254 [0.4898, 0.5636] | +0.0899 [0.0768, 0.103] | -0.05302 [-0.06339, -0.04266] | 0.9706 [0.7646, 1.232] |
  | system | 1e-3 | OOD | 0.927 [0.6196, 1.387] | +0.07119 [0.04916, 0.09321] | -0.07042 [-0.07861, -0.06222] | 0.905 [0.6438, 1.272] |
  | system | 3e-3 | ID | 0.5441 [0.495, 0.598] | +0.111 [0.09372, 0.1283] | -0.06753 [-0.07955, -0.05552] | 0.8571 [0.7791, 0.9431] |
  | system | 3e-3 | OOD | 0.9903 [0.6964, 1.408] | +0.09138 [0.06498, 0.1178] | -0.08112 [-0.09403, -0.06821] | 0.8123 [0.6488, 1.017] |

- **Output files:** new: `experiments/ani1x-energy/mace-mve/hf-only-lr3e-3/evaluation/`,
  `experiments/ani1x-system/mace-mve/hf-only-lr{1e-3,3e-3}/evaluation/` (per split + aggregates),
  `runs/ani1x-{energy,system}/mace-mve/hf-only-lr*/` (pruned checkpoints, models, logs, results); changed:
  `results/tables/ablations_vs_reference.csv` (now 8 complete (ablation, test) pairs; the others pending sync).
- **Interpretation:** a lower lr helps HF-only only a little: ranking improves by +0.012-0.035 Spearman (6 of 8
  intervals exclude 0), RMSE changes by at most ~6% where the interval excludes 1 (the system-ID RMSE
  intervals are very wide, 0.47-1.41: heavy-tailed split-level errors, read with care), calibration changes are
  mostly not significant (energy lr 1e-3: better ID, worse OOD). Against every tuned HF-only, LF->HF stays about
  2x more accurate in distribution, and ranks better on all 8 (split, lr, test) cells (Spearman and AUSE
  intervals all exclude 0). The calibration contrast depends on the cell (energy: LF->HF ENCE ratio > 1 in all
  4 cells, significant in 2; system: ratio < 1 in all 4, significant in 1, system-ID vs lr 3e-3). So the accuracy and ranking conclusions of the
  paper survive the lr sweep; the calibration statements should be phrased per split/test.

---

### 2026-11-07 Fine-tuning ablations of the AU collapse: variance-head reset, beta-NLL, FT learning rate (rebuttal WP3.2, 3.3, 3.5/4.3; Q1/Q4, G4)

- **Status:** verified for head reset and beta-NLL (both splits) and FT lr 1e-4 / 1e-2 (energy split): each 5
  splits x 10 members, evaluated on Leonardo and aggregated here. **Draft gap:** FT lr on the system split
  (`ani1x-system/mace-mve/lf-to-hf-lr1e-{4,2}`) is still being evaluated on Leonardo; not included.
- **Purpose:** Q2LT Q1/Q4: test the proposed mechanism of AU overconfidence after LF->HF fine-tuning
  (prediction (b): a re-initialized variance head re-learns the small training residuals, so the collapse
  re-appears; (c): slower FT; "alternative objectives": beta-NLL); G4: FT learning rate.
- **Code commit:** `baa2d5ec013b72ba56b58c1f93d5b8f009a9918d` (sync-back, aggregate, table); training on
  Leonardo at `ac60247` (reset-var, fork flag `mve_reset_variance_head`), `b4d07ac` (FT lr), `c7710ef` (beta-NLL,
  fork flag `mve_beta_nll`, beta = 0.5); both flags default off, existing commands unchanged (tests); evaluation
  there from the same clone; dirty tree: only the new outputs.
- **Command(s):** on Leonardo `uq train-slurm <exps> --pack 3 --submit` and `uq evaluate-slurm <exps> --after <training
  jobs> --prune --submit` (eval 63720971-63720980 + prune 63720981 reset-var; 63720982-63720991 + 63720992 FT lr energy;
  63720993-63721009 + 63721011 beta-NLL; all COMPLETED); on the DGX `uq sync-back <exp> --host leonardo --checkpoints
  --apply` (`runs/_jobs/sync_morning_2026-11-07.sh`, retry `sync_morning_retry.sh`), `uq aggregate <exp>`,
  `python -m uq.evaluation.compare_ablations`. Direction table below: ratio of the five-split means of `mse` and
  `mean_variance` per AU bin of `reliability_<test>_nocal_bins.csv` (computed inline in this task).
- **Configuration:** `experiments/ani1x-{system,energy}/mace-mve/lf-to-hf-{reset-var,beta-nll,lr1e-4,lr1e-2}/`:
  each is B (system) / D (energy) with one change: variance-specific readout weights (32 per member) re-drawn
  N(0,1) at FT start (seeded by member seed) / beta-NLL loss with beta = 0.5 (selection still by Gaussian NLL) /
  FT lr 1e-4 or 1e-2 instead of 1e-3. Same LF parents (A, C) and member seeds.
- **Dataset and split:** `data/ani1x-{system,energy}/split_{0..4}/cc_*.xyz`; support filter as for B/D.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (none missing).
- **Compute:** FT retraining (300 members, Leonardo) + GPU inference (Leonardo) + CPU aggregation (DGX).
- **Metrics:** uncalibrated, per atom; five-split mean of the ablation, then in parentheses the paired contrast
  vs its reference (B/D) with 95% t interval (df = 4): ratio (x) for RMSE (eV/atom, ensemble mean) and ENCE,
  difference for Spearman. AL = AU, TO = TU. Source: `results/tables/ablations_vs_reference.csv` (also has AUSE and EU).

| Ablation | Split | Test | RMSE | Spearman AL | Spearman TO | ENCE AL | ENCE TO |
|---|---|---|---|---|---|---|---|
| head reset | system | ID | 0.01584 (x0.9984 [0.9234, 1.079]) | 0.351 (-0.1117 [-0.1349, -0.0884]) | 0.4122 (-0.08799 [-0.114, -0.06194]) | 0.4714 (x1.087 [0.7197, 1.643]) | 0.7108 (x2.078 [1.849, 2.335]) |
| head reset | system | OOD | 0.03413 (x1.076 [0.9571, 1.209]) | 0.3992 (-0.09061 [-0.1171, -0.06409]) | 0.4526 (-0.06531 [-0.09319, -0.03742]) | 0.5398 (x0.8733 [0.6108, 1.248]) | 0.7172 (x2.246 [2, 2.522]) |
| head reset | energy | ID | 0.008822 (x1.123 [1.075, 1.174]) | 0.3024 (-0.1177 [-0.1427, -0.09265]) | 0.3837 (-0.07289 [-0.09951, -0.04627]) | 0.4663 (x1.598 [1.431, 1.785]) | 0.7034 (x1.649 [1.576, 1.726]) |
| head reset | energy | OOD | 0.06199 (x1.058 [1.026, 1.092]) | 0.54 (-0.08607 [-0.1279, -0.04427]) | 0.6009 (-0.03919 [-0.07168, -0.006706]) | 2.676 (x0.3994 [0.2508, 0.6361]) | 1.1 (x0.3704 [0.232, 0.5914]) |
| beta-NLL | system | ID | 0.01529 (x0.9639 [0.9461, 0.982]) | 0.4706 (+0.007953 [0.00232, 0.01359]) | 0.5049 (+0.004669 [0.0009791, 0.00836]) | 0.5003 (x1.154 [1.032, 1.291]) | 0.2972 (x0.8687 [0.8192, 0.9213]) |
| beta-NLL | system | OOD | 0.03158 (x0.9952 [0.981, 1.01]) | 0.492 (+0.002162 [-0.003718, 0.008042]) | 0.5181 (+0.0002636 [-0.007424, 0.007951]) | 0.7475 (x1.209 [1.102, 1.327]) | 0.2872 (x0.8994 [0.7991, 1.012]) |
| beta-NLL | energy | ID | 0.007714 (x0.9822 [0.9578, 1.007]) | 0.4224 (+0.00231 [-0.001422, 0.006041]) | 0.4602 (+0.003633 [-0.0002097, 0.007475]) | 0.3687 (x1.263 [1.118, 1.427]) | 0.3569 (x0.8366 [0.8011, 0.8736]) |
| beta-NLL | energy | OOD | 0.05852 (x0.9991 [0.9943, 1.004]) | 0.6247 (-0.001391 [-0.004889, 0.002107]) | 0.638 (-0.002038 [-0.004707, 0.0006305]) | 8.099 (x1.209 [1.153, 1.267]) | 3.411 (x1.149 [1.096, 1.205]) |
| FT lr 1e-4 | energy | ID | 0.00801 (x1.02 [1.008, 1.032]) | 0.4259 (+0.005872 [0.003851, 0.007893]) | 0.4644 (+0.00787 [0.005917, 0.009824]) | 0.3934 (x1.348 [1.278, 1.423]) | 0.3682 (x0.8632 [0.8473, 0.8793]) |
| FT lr 1e-4 | energy | OOD | 0.05837 (x0.9965 [0.9946, 0.9984]) | 0.6263 (+0.00021 [-0.005402, 0.005822]) | 0.6396 (-0.0004348 [-0.005134, 0.004264]) | 7.726 (x1.153 [1.115, 1.192]) | 3.127 (x1.053 [1.022, 1.085]) |
| FT lr 1e-2 | energy | ID | 0.007622 (x0.9705 [0.9469, 0.9946]) | 0.391 (-0.02906 [-0.0444, -0.01373]) | 0.415 (-0.04154 [-0.05268, -0.0304]) | 0.299 (x1.025 [0.9277, 1.132]) | 0.5862 (x1.374 [1.302, 1.45]) |
| FT lr 1e-2 | energy | OOD | 0.06034 (x1.03 [1.026, 1.034]) | 0.6061 (-0.01999 [-0.03066, -0.009313]) | 0.6182 (-0.02186 [-0.03539, -0.008344]) | 7.376 (x1.101 [0.9457, 1.281]) | 3.475 (x1.171 [1.088, 1.259]) |

  Direction of AU miscalibration (MSE / mean variance per AU bin; > 1 = overconfident):

  | Split | Test | LF->HF: bins with MSE > MV (of 15); MSE/MV range | head reset: same |
  |---|---|---|---|
  | system | ID | 10; 0.5-1.9 | 0; 0.25-0.79 |
  | system | OOD | 11; 0.61-2.8 | 1; 0.29-3.1 |
  | energy | ID | 8; 0.51-1.5 | 0; 0.35-0.68 |
  | energy | OOD | 15; 1.1-19 | 15; 1.7-8.8 |

- **Output files:** new: `experiments/ani1x-{system,energy}/mace-mve/lf-to-hf-{reset-var,beta-nll}/evaluation/`,
  `experiments/ani1x-energy/mace-mve/lf-to-hf-lr1e-{4,2}/evaluation/` (per split + aggregates),
  `runs/<same ids>/` (pruned checkpoints, models, logs, results); changed: `results/tables/ablations_vs_reference.csv`
  (20 of 24 (ablation, test) pairs complete).
- **Interpretation:** (1) **Head reset does not reproduce the collapse, contrary to prediction (b).** With a
  re-initialized head the AU is *under*confident on System-ID, System-OOD and Energy-ID (MSE/MV < 1 in nearly
  every bin) and much less overconfident on Energy-OOD (AU ENCE x0.399 [0.251, 0.636]); TU calibration is worse
  except on Energy-OOD, and ranking is clearly worse everywhere (AU Spearman -0.086 to -0.118, TU -0.039 to -0.088).
  So, under our 100-epoch FT protocol, much of the AU overconfidence of LF->HF comes with the variance head
  *inherited* from LF pretraining (its scale and its ranking ability), not only from re-fitting CC training
  residuals. The mechanism text must be revised (inherited head state; the draft already foresaw this case).
  Caveat: a reset head starts from large variances and FT is short, so underconfidence may partly be
  incomplete convergence; selection is by validation NLL as for B/D. (2) **beta-NLL** leaves accuracy and
  ranking unchanged (Spearman differences within +-0.008) and makes AU calibration slightly worse everywhere
  (AU ENCE x1.15-1.26, all intervals > 1): the overconfidence is not an NLL-optimization artefact that
  beta-NLL removes. (3) **FT lr 1e-4** (energy) changes RMSE by <= 2% and Spearman by <= 0.008 and worsens AU ENCE
  (x1.15-1.35); the final checkpoint does not test "delay" (prediction (c) concerns the dynamics; epoch curves
  in `epoch_quality_*` not analysed here). **FT lr 1e-2** (energy) is worse in ranking (TU Spearman -0.04 / -0.02)
  and TU calibration (x1.37 / x1.17): the paper's lr 1e-3 is a sound choice (G4).

---

### 2026-11-07 Correction to "Fine-tuning ablations of the AU collapse": selected epochs (rebuttal WP3.2-3.5)

- **Status:** verified; corrects one caveat of the entry above (its numbers are unchanged).
- **Purpose:** state at which epoch the evaluated checkpoints were selected (lowest validation loss per member,
  `reliability.select_best_checkpoints`, as for every experiment), since the interpretation depends on it.
- **Code commit:** `a6d50d2` (clean tree). **Command:** `uq.evaluation.training_steps.member_rows(exp)` for each
  experiment (inline; same function as `python -m uq.evaluation.training_steps`), logs and checkpoint names only (CPU).
- **Data / seeds:** as the entry above; 5 splits x 10 members each.
- **Metrics:** selected epoch (0-based) of 100 FT epochs, median [min-max] over the 50 members:

  | Experiment | System | Energy |
  |---|---|---|
  | LF->HF reference (B / D) | 2 [1-3] | 2 [1-3] |
  | head reset | 6 [2-19] | 7 [4-15] |
  | beta-NLL | 2 [1-3] | 3 [2-4] |
  | FT lr 1e-4 | (pending) | 30 [21-37] |
  | FT lr 1e-2 | (pending) | 0 [0-22] |

  All members ran 100 epochs.
- **Output files:** none (printed in the task).
- **Interpretation:** the evaluated ensembles are early-stopped, not converged: validation NLL is lowest after 2-3
  FT epochs for LF->HF. The caveat above ("FT is short, so underconfidence may partly be incomplete convergence")
  is wrong: the reset members select epochs 2-19 of 100, so at their own validation-NLL optimum the re-initialized
  head still predicts large variances; later epochs shrink AU but raise validation NLL. FT lr 1e-4 moves the
  selected epoch ~10x later (2 -> 30), consistent with prediction (c) on the time scale, while the final metrics
  barely change.

---

### 2026-11-07 Head reset: AU calibration along fine-tuning (rebuttal WP3.2; Q1 prediction (b))

- **Status:** verified (five-split aggregates already on disk; read in this task). Complements the two entries above.
- **Purpose:** does a re-initialized variance head also lose AU calibration during FT (prediction (b)), only later?
- **Code commit:** `196bc36` (clean). **Command:** pandas read of `experiments/<id>/evaluation/epoch_quality_<test>.csv`
  (columns `ence_aleatoric_geometric_mean`, `spearman_aleatoric_mean`) for B, D and their `-reset-var` twins.
  Energy-split OOD curves not used (not support-filtered; `.claude/rules/paper-writing.md`).
- **Data / seeds:** as above; epochs available: those kept by pruning (every 5th + best), here 0, 5, 10, 20, 40, 60.
- **Metrics:** AU ENCE (five-split geometric mean) / AU Spearman (mean) per FT epoch:

  | Test | Experiment | ep 0 | ep 5 | ep 10 | ep 20 | ep 40 | ep 60 |
  |---|---|---|---|---|---|---|---|
  | System-ID | B (LF->HF) | 0.21 / 0.42 | 1.2 / 0.45 | 3.8 / 0.43 | 11 / 0.42 | 24 / 0.41 | 45 / 0.41 |
  | System-ID | head reset | 0.95 / 0.17 | 0.49 / 0.34 | 0.24 / 0.37 | 1 / 0.39 | 3.6 / 0.35 | 7.4 / 0.33 |
  | System-OOD | B (LF->HF) | 0.37 / 0.43 | 1.6 / 0.49 | 4.8 / 0.48 | 13 / 0.47 | 30 / 0.46 | 54 / 0.45 |
  | System-OOD | head reset | 0.91 / 0.23 | 0.57 / 0.39 | 0.49 / 0.41 | 1.6 / 0.43 | 4.7 / 0.39 | 9.1 / 0.38 |
  | Energy-ID | D (LF->HF) | 0.12 / 0.37 | 0.7 / 0.42 | 2.4 / 0.41 | 7.8 / 0.40 | 20 / 0.39 | 40 / 0.38 |
  | Energy-ID | head reset | 0.97 / 0.10 | 0.55 / 0.29 | 0.28 / 0.31 | 0.56 / 0.31 | 2.5 / 0.27 | 4.9 / 0.25 |

  (values printed with 2 significant digits from the CSV columns named above)
- **Output files:** none new.
- **Interpretation:** the re-initialized head starts miscalibrated (ENCE ~0.9-1.0, large variances), becomes best
  calibrated around epoch 10, and then loses calibration steadily (ENCE 1 -> 7.4 on System-ID by epoch 60), the
  same monotone degradation as the inherited head, only ~5-8x slower at each epoch. So prediction (b) holds for
  the dynamics: the collapse re-emerges after a reset; the inherited head makes it start from the first epoch,
  which matches the paper's finding (iii) that calibration is lost faster after pretraining. The reset ensembles'
  better final AU calibration (entry above) arises because their validation-NLL optimum (epochs 2-19) falls
  before the collapse has developed. The reset also costs ranking at every epoch (AU Spearman 0.10-0.43 vs
  0.37-0.49), i.e. part of pretraining's ranking benefit (finding (i)) is carried by the variance head.

---

### 2026-11-07 Fine-tuning learning rate on the system split; completes "Fine-tuning ablations of the AU collapse" (rebuttal WP3.5, WP4.3; Q1 (c), G4)

- **Status:** verified; closes the draft gap of the entry "Fine-tuning ablations of the AU collapse" (FT lr, system
  split) and the two "(pending)" cells of its correction. Earlier entries unchanged.
- **Purpose:** as in that entry: prediction (c) (slower FT delays the collapse) and G4 (is the paper's FT lr 1e-3 a
  sound choice?), now on both splits.
- **Code commit:** `03e6b946e6f053d730a49bf792a3b403e7ea5525` (sync-back, aggregate, table; clean tree apart from
  `PLAN.md`/`RUNS.md`); training and evaluation on Leonardo from a clean clone at `016dbb6`.
- **Command(s):** on Leonardo `uq train-slurm ... --pack 3 --submit` and `uq evaluate-slurm ... --prune --submit`
  (eval 63722646-63722655, prune 63722656; all COMPLETED); on the DGX `runs/_jobs/sync_ftlr_system_2026-11-07.sh`
  (log next to it): `uq sync-back <exp> --host leonardo --checkpoints --apply`, `uq aggregate <exp>`,
  `python -m uq.evaluation.compare_ablations` (24 complete pairs, 0 missing; the 20 earlier pairs byte-identical in
  `git diff`). Selected epochs: `uq.evaluation.training_steps.member_rows(registry.experiment(exp))` (inline).
  Direction: ratio of five-split means of `mse` and `mean_variance` per AU / TU bin of
  `reliability_<test>_nocal_bins.csv` (inline; reproduces the B row of the entry above).
- **Configuration:** `experiments/ani1x-system/mace-mve/lf-to-hf-lr1e-{4,2}/`: B with FT lr 1e-4 / 1e-2 instead of
  1e-3; same LF parents (A) and member seeds.
- **Dataset and split:** `data/ani1x-system/split_{0..4}/cc_*.xyz`; support filter as for B.
- **Random seeds:** split seeds 0-4; member seeds 0-9 (50/50 members per experiment, 5/5 splits evaluated).
- **Compute:** FT retraining (100 members, Leonardo) + GPU inference (Leonardo) + CPU aggregation (DGX).
- **Metrics:** as in the entry above (uncalibrated, per atom; ablation five-split mean, then paired contrast vs B with
  95% t interval, df = 4). Source: `results/tables/ablations_vs_reference.csv`.

| Ablation | Split | Test | RMSE | Spearman AL | Spearman TO | ENCE AL | ENCE TO |
|---|---|---|---|---|---|---|---|
| FT lr 1e-4 | system | ID | 0.016 (x1.008 [0.9903, 1.027]) | 0.4646 (+0.001904 [-1.203e-05, 0.003821]) | 0.5043 (+0.004067 [0.001235, 0.006898]) | 0.5984 (x1.38 [1.262, 1.51]) | 0.2808 (x0.821 [0.7526, 0.8957]) |
| FT lr 1e-4 | system | OOD | 0.03199 (x1.008 [1.001, 1.015]) | 0.492 (+0.002214 [-0.003282, 0.007711]) | 0.5218 (+0.003926 [-0.0004411, 0.008293]) | 0.8424 (x1.363 [1.315, 1.412]) | 0.2692 (x0.8431 [0.795, 0.8941]) |
| FT lr 1e-2 | system | ID | 0.0153 (x0.9645 [0.9179, 1.013]) | 0.436 (-0.0267 [-0.05513, 0.001735]) | 0.4558 (-0.04442 [-0.07299, -0.01586]) | 0.2603 (x0.6004 [0.5285, 0.682]) | 0.5333 (x1.559 [1.522, 1.596]) |
| FT lr 1e-2 | system | OOD | 0.02982 (x0.9398 [0.8872, 0.9955]) | 0.4645 (-0.02535 [-0.03809, -0.0126]) | 0.4821 (-0.03578 [-0.05416, -0.0174]) | 0.3386 (x0.5477 [0.4742, 0.6326]) | 0.4831 (x1.513 [1.283, 1.785]) |

  (values printed with 4 significant digits from the CSV columns `ablation_mean`, `contrast_mean`, `contrast_ci95_*`)

  Selected FT epoch (0-based, of 100; all members ran 100), median [min-max] over 50 members: system lr 1e-4
  25 [18-40], lr 1e-2 1 [0-24] (B: 2 [1-3]); energy lr 1e-4 29.5 [21-37], lr 1e-2 0 [0-22] (the correction above
  printed the energy lr 1e-4 median as 30).

  Direction (bins with MSE > MV, of 15; MSE/MV range): AU, system: B ID 10, 0.5-1.9 / OOD 11, 0.61-2.8; lr 1e-4
  ID 11, 0.59-2.2 / OOD 13, 0.76-3.2; lr 1e-2 ID 5, 0.26-1.2 / OOD 7, 0.4-1.9. TU, lr 1e-2 vs reference: system ID
  0, 0.21-0.62 (B 0, 0.36-0.89), OOD 0, 0.27-0.96 (B 1, 0.42-1.3); energy ID 0, 0.13-0.56 (D 0, 0.35-0.78), OOD 13,
  0.49-9.2 (D 12, 0.66-9.9).
- **Output files:** new: `experiments/ani1x-system/mace-mve/lf-to-hf-lr1e-{4,2}/evaluation/` (per split +
  aggregates), `runs/<same ids>/` (pruned checkpoints, models, logs, results); changed:
  `results/tables/ablations_vs_reference.csv` (24 of 24 pairs). Figures reading it (not new numbers):
  `papers/iclr27/figures/out/figA_{ablations,lr_sensitivity}.*`.
- **Interpretation:** the system split reproduces the energy-split pattern. (1) **FT lr 1e-4** changes RMSE by
  <= 0.8% and Spearman by <= 0.004, makes AU more overconfident (AU ENCE x1.38 / x1.36, MSE > MV in 11 / 13 of 15
  bins) and TU slightly better calibrated (x0.82 / x0.84), and moves the selected epoch from 2 to 25: slower FT
  shifts the time scale (prediction (c)) without removing the AU overconfidence at the selected checkpoint.
  (2) **FT lr 1e-2** lowers AU ENCE (x0.60 / x0.55; AU no longer mostly overconfident) but ranks worse (TU Spearman
  -0.044 / -0.036, AU -0.027 with an interval touching 0 on ID) and makes TU *under*confident in-distribution (TU
  ENCE x1.56 / x1.51; MSE/MV of TU <= 0.96 in every bin), with members selected at epoch 0-1 (median), i.e.
  essentially the first FT epoch. RMSE x0.96 ID (interval contains 1), x0.94 OOD. On both splits lr 1e-4 and the
  paper's 1e-3 give nearly the same accuracy and ranking (1e-4: AU more overconfident, TU slightly better calibrated
  on three of four tests), while 1e-2 costs ranking and TU calibration: the paper's conclusions do not depend on
  the FT lr in 1e-4 to 1e-3, and 1e-2 is worse (G4). With the energy entry, the FT-lr rows are now complete for the
  appendix (items B and C).

---

### 2026-11-08 MSE ensembles after LF->HF fine-tuning, ANI-1x system split (rebuttal WP3.6; Q2LT Q1 "no variance head", Q2)

- **Status:** verified for the system split (computed in this task from complete per-split files: 5 splits x 10
  members for `ani1x-system/mace-mse/{lf-only,lf-to-hf}`; the HF-only rows of `results/tables/mse_vs_mve.csv` are
  byte-identical to the 2026-11-04 entry). Energy split pending (evaluating on the DGX), separate entry.
- **Purpose:** does LF pretraining change the EU of a plain MSE ensemble (no variance head, so no AU to collapse) the
  way it changes the MVE ensemble's EU/TU? And at LF->HF, how does the MSE-ensemble EU compare with its MVE twin B?
- **Code commit:** training on CINECA Leonardo at `7e1b998` (LF-only) / `0323ada` (LF->HF); evaluation on Leonardo at
  `016dbb6` (jobs 63718301-310, printed in each `.out`); sync-back, aggregate and tables on the DGX at
  `5577d20eaad3d3b2d5111b0f5211bb391e4a5094`, dirty tree: `papers/iclr27/rebuttal/RUNS.md` (notes) and the new
  `uq/evaluation/compare_uq_protocols.py` (this entry).
- **Command(s):** on Leonardo `uq train-slurm ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf --pack 3
  --submit` (train 63716417-441) and `uq evaluate-slurm ... --prune --submit` (eval 63718301-310, prune 63718311,
  all COMPLETED 2026-11-08 07:57); on the DGX `runs/_jobs/sync_mse_system_2026-11-08.sh` (`uq sync-back <exp> --host
  leonardo --checkpoints --apply`, then `uq aggregate <exp>`, log `runs/_jobs/sync_mse_system_2026-11-08.log`);
  `python -m uq.evaluation.compare_uq`; `python -m uq.evaluation.compare_uq_protocols`.
- **Configuration:** `experiments/ani1x-system/mace-mse/{lf-only,lf-to-hf}/{experiment.yaml,config.yaml}`: as A / B
  except `predict_mve: False`, `loss: "energy_only"`; LF->HF member (s, m) fine-tuned from MSE LF-only member (s, m).
  Selected epochs (validation MSE, `uq.evaluation.training_steps.member_rows`): MSE LF-only median 285, range
  211-299 of 300; MSE LF->HF median 9.5, range 1-99 of 100 (MVE B: median 2, range 1-3).
- **Dataset and split:** `data/ani1x-system/split_{0..4}/`; LF-only on `dft_*`, LF->HF on `cc_*.xyz`; support filter
  as for B/E (the ensemble-size tables are built from `members_<test>.npz`, as in the 2026-11-03/04 entries).
- **Random seeds:** split seeds 0-4; member seeds 0-9 (50/50 members per experiment; nothing missing).
- **Compute:** retraining (100 members, Leonardo) + GPU inference (Leonardo) + CPU (DGX).
- **Metrics:** uncalibrated, per atom; M = 10 unless "single"; five-split mean [95% Student-t, df = 4]; contrasts
  paired per split: difference for Spearman/AUSE, geometric ratio [log-space interval] for RMSE/ENCE. (1) MSE vs MVE
  at LF->HF, from `results/tables/mse_vs_mve.csv` (rows `ani1x-system,lf-to-hf`):

  | Test | Metric | MSE EU | MVE EU - MSE EU | MVE TU - MSE EU | single MVE AU - MSE EU |
  |---|---|---|---|---|---|
  | OOD | Spearman | 0.383849 [0.364706, 0.402992] | 0.117636 [0.101460, 0.133811] | 0.133991 [0.115563, 0.152418] | 0.078157 [0.060745, 0.095569] |
  | OOD | AUSE | 0.270191 [0.247562, 0.292821] | -0.095593 [-0.110085, -0.081101] | -0.107613 [-0.119144, -0.096082] | -0.069608 [-0.083307, -0.055910] |
  | OOD | ENCE (ratio) | 1.917238 [1.313071, 2.799391] | x0.777860 [0.488524, 1.238560] | x0.166871 [0.134028, 0.207762] | x1.141149 [0.685608, 1.899367] |
  | ID | Spearman | 0.351971 [0.329968, 0.373973] | 0.139945 [0.118109, 0.161782] | 0.148200 [0.125263, 0.171136] | 0.075478 [0.049414, 0.101543] |
  | ID | AUSE | 0.332668 [0.313315, 0.352021] | -0.115140 [-0.139948, -0.090333] | -0.127812 [-0.153246, -0.102377] | -0.084282 [-0.110850, -0.057714] |
  | ID | ENCE (ratio) | 1.622512 [1.084572, 2.427265] | x0.737723 [0.583246, 0.933114] | x0.211060 [0.128868, 0.345675] | x0.835977 [0.581798, 1.201202] |

  RMSE (eV/atom), MSE ensemble: OOD 0.031592 [0.021249, 0.046971], ID 0.014243 [0.013763, 0.014741]; ratio MVE
  ensemble / MSE ensemble: OOD x1.004406 [0.788443, 1.279523], ID x1.113989 [0.930848, 1.333162].

  (2) Effect of pretraining, LF->HF minus HF-only on the same split, from `results/tables/mse_vs_mve_pretraining.csv`
  (rows `ani1x-system`):

  | Test | Metric | MSE EU | MVE EU | MVE TU |
  |---|---|---|---|---|
  | OOD | Spearman | 0.022088 [-0.007000, 0.051176] | 0.131816 [0.100692, 0.162939] | 0.106627 [0.077427, 0.135827] |
  | OOD | AUSE | -0.031811 [-0.044979, -0.018642] | -0.113318 [-0.141212, -0.085425] | -0.084252 [-0.104822, -0.063682] |
  | OOD | ENCE (ratio) | x1.299972 [0.928895, 1.819287] | x0.738941 [0.454851, 1.200468] | x0.818863 [0.555936, 1.206140] |
  | ID | Spearman | 0.030117 [0.017197, 0.043037] | 0.153823 [0.138290, 0.169357] | 0.123222 [0.113173, 0.133271] |
  | ID | AUSE | -0.018393 [-0.030092, -0.006694] | -0.103572 [-0.131385, -0.075759] | -0.067976 [-0.084179, -0.051772] |
  | ID | ENCE (ratio) | x1.162340 [0.811730, 1.664391] | x0.546147 [0.420290, 0.709693] | x0.853624 [0.806909, 0.903043] |

  RMSE ratio LF->HF / HF-only, MSE ensemble: OOD x1.150773 [0.840819, 1.574986], ID x0.685133 [0.649926, 0.722247];
  MVE ensemble: OOD x0.936004 [0.635113, 1.379443], ID x0.441992 [0.256304, 0.762211]. MVE AU after pretraining
  (same file): ENCE OOD x1.860774 [1.260638, 2.746611], ID x2.005605 [1.627671, 2.471292] (the AU collapse).
  MSE LF-only on its DFT System-OOD test (`experiments/ani1x-system/mace-mse/lf-only/evaluation/
  reliability_ood_nocal_summary.csv`, `uq summary ... --test ood`): EU Spearman 0.408 [0.395, 0.421].
- **Output files:** new `experiments/ani1x-system/mace-mse/{lf-only,lf-to-hf}/evaluation/` (per split + aggregates,
  synced from Leonardo); `results/tables/mse_vs_mve.csv` regenerated (system LF->HF rows filled; energy LF->HF still
  missing; all other rows unchanged); new `results/tables/mse_vs_mve_pretraining.csv` and
  `uq/evaluation/compare_uq_protocols.py`; checkpoints in `runs/ani1x-system/mace-mse/{lf-only,lf-to-hf}/` (pruned on
  Leonardo by 63718311 before the copy).
- **Interpretation:** (1) Pretraining helps the MVE ensemble's uncertainty much more than the MSE ensemble's: MVE EU
  ranking improves by +0.13 to +0.15 Spearman (AUSE -0.10 to -0.11), the MSE-ensemble EU by +0.02 to +0.03 (OOD
  interval contains 0), and the MSE-ensemble EU calibration does not improve (ENCE ratio intervals contain 1). So the
  large gain in EU ranking after LF->HF is specific to the MVE members, not a generic ensemble effect of starting
  from a pretrained model. (2) Consequently, at LF->HF the MVE ensemble beats the MSE ensemble on every ranking metric
  and test (EU, TU, and even a single member's AU), whereas at HF-only the two EUs ranked alike (2026-11-04 entry);
  accuracy is now equal (RMSE ratio intervals contain 1; at HF-only the MSE ensemble was more accurate). (3) The MSE
  ensemble has no AU, so it cannot show the collapse; its EU is not hurt by pretraining, while the MVE AU's
  calibration worsens ~2x (ENCE) as recorded for B. Caveats: MSE LF->HF members are selected later (median epoch 9.5)
  than MVE (median 2), so their EU reflects longer FT; one split only until the energy entry is recorded; the
  ensemble-size tables follow the same filtering as the 2026-11-03/04 MSE entries.

---

### 2026-11-08 AU along fine-tuning for all LF->HF ablations (rebuttal WP3.2, 3.3, 3.5; Q1 (b), (c)); appendix figure

- **Status:** verified (computed in this task from complete five-split aggregates; extends the entry "Head reset: AU
  calibration along fine-tuning" to beta-NLL and both FT learning rates, and to every 5th epoch).
- **Purpose:** the per-epoch AU dynamics behind the final-model ablations: does each variant change *when* the AU
  overconfidence develops, and how far it goes by the end of fine-tuning?
- **Code commit:** `dcd2a087e5f3dcafb8db7820cbaca89033eacd4d`; dirty tree: yes (the new figure scripts of this entry and
  the AUSE-for-Spearman change of `figA_ablations.py` / `figA_lr_sensitivity.py`, which change no number).
- **Command(s):** pandas read of `experiments/ani1x-{system,energy}/mace-mve/<variant>/evaluation/epoch_quality_<test>.csv`
  (columns `ence_aleatoric_geometric_mean` / `_n`, `ause_aleatoric_mean`), variants `lf-to-hf` (B / D),
  `lf-to-hf-reset-var`, `lf-to-hf-beta-nll`, `lf-to-hf-lr1e-4`, `lf-to-hf-lr1e-2`; figure
  `python papers/iclr27/figures/figA_ablation_dynamics.py` (from the per-split files, EMA-smoothed).
- **Configuration / data / seeds:** as in "Fine-tuning ablations of the AU collapse"; tests System-ID, System-OOD, Energy-ID
  (cached curves; Energy-OOD per-epoch curves are not support-filtered and are not used); epochs 0-95 every 5; five splits
  x ten members for every variant and epoch (`_n` = 5 everywhere).
- **Compute:** CPU (cached per-epoch outputs).
- **Metrics:** AU ENCE, five-split geometric mean, uncalibrated, per atom: value at FT epoch 0, minimum (epoch), first epoch
  after the minimum where it exceeds 1, value at epochs 50 and 95 (3 significant digits):

  | Test | Variant | ep 0 | min (ep) | first > 1 | ep 50 | ep 95 |
  |---|---|---|---|---|---|---|
  | System-ID | LF->HF (B) | 0.21 | 0.21 (0) | 5 | 31 | 67.2 |
  | System-ID | head reset | 0.946 | 0.24 (10) | 20 | 4.88 | 12.5 |
  | System-ID | beta-NLL | 0.442 | 0.442 (0) | 10 | 26.7 | 67 |
  | System-ID | FT lr 1e-4 | 25.4 | 0.455 (10) | 50 | 1.07 | 2.28 |
  | System-ID | FT lr 1e-2 | 0.373 | 0.373 (0) | 40 | 1.43 | 3.35 |
  | System-OOD | LF->HF (B) | 0.374 | 0.374 (0) | 5 | 37.7 | 81.9 |
  | System-OOD | head reset | 0.911 | 0.495 (10) | 20 | 6.1 | 14.9 |
  | System-OOD | beta-NLL | 0.71 | 0.71 (0) | 5 | 31.3 | 77.7 |
  | System-OOD | FT lr 1e-4 | 28.9 | 0.674 (15) | 35 | 1.44 | 2.96 |
  | System-OOD | FT lr 1e-2 | 0.33 | 0.33 (0) | 25 | 1.87 | 4.17 |
  | Energy-ID | LF->HF (D) | 0.116 | 0.116 (0) | 10 | 26.6 | 60.5 |
  | Energy-ID | head reset | 0.969 | 0.192 (15) | 25 | 3.55 | 9.41 |
  | Energy-ID | beta-NLL | 0.435 | 0.435 (0) | 10 | 21.7 | 57.6 |
  | Energy-ID | FT lr 1e-4 | 25.7 | 0.294 (15) | 75 | 0.633 | 1.42 |
  | Energy-ID | FT lr 1e-2 | 0.382 | 0.36 (5) | 35 | 1.23 | 3.02 |

  On the training set (`epoch_quality_train.csv`, AU ENCE geometric mean, epochs 10-95) every variant stays at or below
  0.909 on both splits (0.755-0.909 at epoch 95), except system lr 1e-2 at its unstable epoch 10 (1.16): AU is not
  overconfident on the data it is fitted to. At FT epoch 0,
  lr 1e-4 has a training RMSE of 35.4 (system) / 30.7 (energy) meV/atom (mean over splits) against 14.3 / 9.0 for the
  paper's lr: after one epoch at the small lr the DFT-to-CC offset is not yet absorbed, hence the overconfident start.
  The system lr 1e-2 runs show an unstable epoch 10 (mean training RMSE 771.9 meV/atom), recovered by epoch 50 (10.0).
- **Output files:** new `papers/iclr27/figures/figA_ablation_dynamics.py` -> `papers/iclr27/figures/out/figA_ablation_dynamics.{pdf,svg,png}`.
- **Interpretation:** (1) With the inherited head (paper, beta-NLL) held-out AU ENCE exceeds 1 within 5-10 FT epochs and
  reaches 58-82 by epoch 95; beta-NLL follows the paper's curve almost exactly. (2) A re-initialized head starts near 1,
  is best calibrated at epochs 10-15 and then degrades like the inherited head but ~5x less by epoch 95 (9-15).
  (3) Both FT learning rates away from the paper's 1e-3 slow the collapse strongly: ENCE first exceeds 1 at epochs 35-75
  (1e-4) and 25-40 (1e-2) instead of 5-10, and ends at 1.4-4.2 instead of 58-82. For 1e-4 this is the predicted
  delay (prediction (c), ~10x later); for 1e-2 it is not a slower fit of the training set (training RMSE at epoch 95 is
  similar: 9.6 vs 9.0 meV/atom system), so the extent of the collapse is not a monotone function of the learning rate.
  So the statement "the time scale follows the optimization speed" holds for the smaller lr only; the general claim is
  that the overconfidence is fastest under the paper's setting and develops in every variant, at different rates. In all
  variants AU stays calibrated or conservative on the training set while it becomes overconfident on held-out data.

---
