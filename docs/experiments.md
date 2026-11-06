# Experiments: specs, training, completeness

An experiment is one folder `experiments/<dataset>/<model>-<uq>/<protocol>/` with
`experiment.yaml` (what to run) and `config.yaml` (MACE hyperparameters). Its id is
that path; the nine experiments of the ICLR27 paper also answer to their old letters.

```bash
uq list                  # all experiments: id, legacy letter, protocol, parent
uq show B                # spec + resolved data, runs and evaluation paths
uq check                 # validate every spec
uq status [B ...] [-v]   # trained members per split, evaluated splits
```

## `experiment.yaml`

```yaml
legacy_id: B                      # optional alias (A-F, wA-wC for the ICLR27 experiments)
dataset: ani1x-system             # data/<dataset>/dataset.yaml
model: mace
uq: mve-ensemble                  # AU/EU/TU from an ensemble of MVE members; or mse-ensemble (concepts.md)
protocol: lf-to-hf                # lf-only | lf-to-hf | hf-only
target: hf                        # fidelity of the training labels
parent: ani1x-system/mace-mve/lf-only   # lf-to-hf only: member (s, m) starts from parent member (s, m)
splits: [0, 1, 2, 3, 4]
members: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training:
  config: config.yaml
  test_role: id                   # file passed as MACE --test_file (logged once, never used for selection)
evaluation:                       # which steps run; docs/evaluation.md
  batch_size: 256
  trim: 0.0                       # keep 0.0 (concepts.md, support filter)
  reliability: [id, ood]          # sets scored by the final ensemble (cal + nocal)
  sets: [id, ood, train]          # epoch quality, LF->HF trajectory, distributions
  finetune_comparison: true       # optional: parent vs fine-tuned ensemble
  energy_quantiles: true          # optional: AU/EU by within-system energy quantile
  reliability_log_axes: ["1e-4", "1e-1"]   # optional: log-log per-split reliability plots
  train_curves_batch_size: 16     # optional (legacy water setting)
```

Rules checked by `uq check`: the protocol is known; `lf-to-hf` needs a parent that
shares dataset, splits and members, other protocols have none; the target fidelity
and every evaluated role exist in the dataset; `config.yaml` exists; legacy ids are unique.

## The ICLR27 experiments

| Experiment | Legacy | Trained on | Parent |
|---|---|---|---|
| `ani1x-system/mace-mve/lf-only` | A | DFT | — |
| `ani1x-system/mace-mve/lf-to-hf` | B | CC (fine-tuned) | A |
| `ani1x-system/mace-mve/hf-only` | E | CC | — |
| `ani1x-energy/mace-mve/lf-only` | C | DFT | — |
| `ani1x-energy/mace-mve/lf-to-hf` | D | CC (fine-tuned) | C |
| `ani1x-energy/mace-mve/hf-only` | F | CC | — |
| `water-random/mace-mve/lf-only` | wA | BLYP | — |
| `water-random/mace-mve/lf-to-hf` | wB | CCSD(T) (fine-tuned) | wA |
| `water-random/mace-mve/hf-only` | wC | CCSD(T) | — |

All 450 members (9 x 5 x 10) are trained. Pretraining and direct runs: 300 epochs;
fine-tuning: 100 epochs at 1/10 of the learning rate (see each `config.yaml`).

## The MSE-ensemble baseline (2026-11, rebuttal / follow-up)

Each ANI-1x MVE experiment has an `mace-mse` twin: same data, splits, member seeds,
architecture and hyperparameters; only `predict_mve: False` and `loss: "energy_only"`
(per-atom energy MSE) differ. The ensemble spread is the only uncertainty (EU; AU is
written as 0, TU = EU), so it answers whether the AU/EU decomposition adds anything over
a standard deep ensemble. LF->HF members are fine-tuned from the MSE parents, never from
MVE parents. Checkpoints are selected per member by validation MSE (the member's own
loss; same `--selection-key loss` code path). Water is not included (not in the paper).

| Experiment | Trained on | Parent | MVE counterpart |
|---|---|---|---|
| `ani1x-system/mace-mse/lf-only` | DFT | — | A |
| `ani1x-system/mace-mse/lf-to-hf` | CC (fine-tuned) | `ani1x-system/mace-mse/lf-only` | B |
| `ani1x-system/mace-mse/hf-only` | CC | — | E |
| `ani1x-energy/mace-mse/lf-only` | DFT | — | C |
| `ani1x-energy/mace-mse/lf-to-hf` | CC (fine-tuned) | `ani1x-energy/mace-mse/lf-only` | D |
| `ani1x-energy/mace-mse/hf-only` | CC | — | F |

## Training

```bash
uq train-member B 2 7 --gpu 4 --dry-run    # one member: split 2, member seed 7
uq train B --gpus 6 7 --dry-run            # every missing member of B (<= 3 GPUs)
uq train wA wB wC --gpus 6 7 --jobs-per-gpu 3
```

`train-member` runs `mace/cli/run_train.py --config experiments/<id>/config.yaml`
with the split's train/val/test files, the run directories
`runs/<id>/split_<s>/{models,logs,checkpoints,results}`, `--seed <member>`,
`--wandb_name split_<s>_mace_seed_<m>`, `--wandb_dir runs/wandb` and, for `lf-to-hf`,
`--foundation_model runs/<parent>/split_<s>/checkpoints/mace_run-<m>.model`. It
refuses to start if an input is missing. `uq train` skips complete members, trains
parents first, waits for the matching parent member before a fine-tuning job, and
never uses more than three GPUs; logs go to `runs/_logs/`. Fine-tuning members start as
soon as their parent member is complete; `--by-experiment` finishes experiments in the
order given instead of interleaving members. Each member is capped at
`--threads-per-job` CPU threads (default 8) and packed jobs use PyTorch's expandable
CUDA segments. The scheduler refuses a member that left partial outputs from an aborted
run (MACE appends to its result log): remove those files, then rerun. Measured on ANI-1x
LF (2026-11-02, A100): one member ~93 s/epoch alone, ~184 s/epoch with three per GPU
(1.5x throughput); GPU-bound beyond that. `$PYTHON_BIN` overrides the
interpreter. Configs set `wandb: True` (entity `uq-mf`), so training needs W&B.

A member is **complete** when both `checkpoints/mace_run-<m>.model` and
`results/mace_run-<m>_train.txt` exist; epoch checkpoints alone count as partial.

## Pruning epoch checkpoints

MACE writes a checkpoint every epoch (`save_all_checkpoints: True`), but the analyses
read only epochs divisible by 5 (epoch quality, train curves, mixed LF->HF
trajectories; every 10th for `epoch_raw`) and each member's best-validation checkpoint.
Once an experiment is trained **and evaluated**:

```bash
uq prune-checkpoints <exp>            # dry run: files and GiB that would go
uq prune-checkpoints <exp> --apply    # irreversible: only when the user asks for it
```

It keeps every `epoch % 5 == 0` checkpoint, the best checkpoint of each member (the
pipeline's own selection: validation loss) and every `.model`; refuses splits
with incomplete members; checks on a simulated folder that the pipeline would select
exactly the same epochs and checkpoints; and writes `checkpoints/PRUNED.json` (rule,
date, every deleted file). Checkpoint selection accepts exactly the gaps listed there
and fails on any other missing epoch. The nine ICLR27 experiments were pruned on
2026-09-26 (ledger entry of that date): finer epoch analyses or re-selection by
another criterion now need retraining, and code older than that (tag
`iclr27-submission`) refuses the pruned folders.

## What LF->HF fine-tuning inherits

From the parent member (`mace/tools/finetuning_utils.py`,
`mace/tools/model_script_utils.py`): the architecture including `predict_mve`; the
embedding, interaction, product and readout weights; the scale/shift buffers.
Because `multiheads_finetuning: False`, the atomic reference energies are re-fitted
on the HF training set (`E0s: "average"`), absorbing the DFT-to-CC offset. Members
of one split differ only by weight initialization and batch order (`--seed`).

## Adding an experiment

See [extending.md](extending.md#a-new-experiment) (short version: copy the closest
experiment folder, edit `experiment.yaml`/`config.yaml`, `uq check`, `uq train --dry-run`).
