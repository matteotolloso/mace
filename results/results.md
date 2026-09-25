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
  write them (`experiment_X/evaluation/`, `new_figures/out/`,
  `active_learning/ani_energy/runs/`).

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
