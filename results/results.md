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
