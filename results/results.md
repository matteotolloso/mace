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
  `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache /raid/m.tolloso/miniconda3/envs/mace/bin/python -B paper_numbers.py > ../results/tables/iclr27_main_text_numbers.txt`
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
  `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache /raid/m.tolloso/miniconda3/envs/mace/bin/python -B paper_numbers.py > ../results/tables/iclr27_main_text_numbers.txt`
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
  `cd new_figures && MPLCONFIGDIR=/tmp/mpl-cache /raid/m.tolloso/miniconda3/envs/mace/bin/python -B paper_numbers.py > ../results/tables/iclr27_main_text_numbers.txt`
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
