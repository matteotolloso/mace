# Code changes

## Before editing

- Read the files involved first. Project code is spread across the upstream
  MACE package (`mace/`) and project directories (`eval/`, `dataset/`, `utils/`,
  `active_learning/`, `new_figures/`, `experiment_X/`).
- Changes to `mace/` alter training for every experiment. Keep them minimal,
  marked with the existing `### MVE ###` ... `### /MVE ###` fences, and never
  change the behaviour of an already-trained configuration silently.
- `model: "MACE"` in the experiment configs builds `ScaleShiftMACE`
  (`mace/tools/model_script_utils.py::_build_model`); MVE changes must keep that
  class and the plain `MACE` class consistent.

## Paths are part of the contract

- Launchers resolve the repository root from their own location
  (`experiment_X/train_X.sh`, `experiment_X/eval_X.sh`, `utils/train_split_member.sh`),
  and evaluation caches, aggregates and paper figures use fixed relative paths.
  Do not move or rename `experiment_X/`, `eval/`, `dataset/`, `utils/`,
  `active_learning/` or `new_figures/` without updating every caller.
- Run commands from the repository root.

## Keep changes focused and verified

- Match the surrounding style (the repo uses black, isort and pylint through
  `.pre-commit-config.yaml`).
- Do not hand-edit generated caches or aggregate CSVs; regenerate them with the
  script that owns them.
- Verify with the narrowest check that covers the change, and report the result:

```bash
# statistics / plotting / aggregation
cd eval && MPLCONFIGDIR=/tmp/mpl-cache \
  $CONDA_PREFIX/bin/python -B -m unittest -v test_plotting
# shell launchers
bash -n eval.sh eval_water.sh check_experiments.sh experiment_*/train_*.sh experiment_*/eval_*.sh
# upstream MACE package (slow; CI runs it on push)
$CONDA_PREFIX/bin/python -m pytest tests --ignore=tests/test_cueq_oeq.py
```

- A change that alters numbers must be followed by regenerating the affected
  artifacts (normally `bash eval.sh --plots-only`, CPU-only) and updating the
  matching `results/results.md` entry.

## Code map

- MVE model/readout: `mace/modules/models.py`; Gaussian NLL: `mace/modules/loss.py`
- MVE CLI/config wiring: `mace/tools/arg_parser.py`, `mace/tools/scripts_utils.py`,
  `mace/tools/model_script_utils.py`, `mace/cli/run_train.py`
- Fine-tuning weight transfer: `mace/tools/finetuning_utils.py`
- Ensemble prediction: `eval/reliability.py` and `mace/cli/eval_configs.py`
- ANI splitters: `dataset/ani1x_system_splitter.py`, `dataset/ani1x_energy_splitter.py`;
  orchestration: `dataset/make_dataset.sh`
- Water conversion/splits: `dataset/convert_water_n2p2_to_extxyz.py`,
  `dataset/make_dataset_w.sh`
- Shared training launcher: `utils/train_split_member.sh`
- Epoch metrics: `eval/epoch_quality.py`; LF->HF mixed trajectories:
  `eval/epoch_quality_finetune_mixed_dataset.py`
- Five-split statistics and plots: `eval/replicate_statistics.py`,
  `eval/aggregate_replicates.py`; training-support filter: `eval/support_filter.py`
- Top-level evaluators: `eval.sh`, `eval_water.sh`
- Paper figures: `new_figures/` (shared helpers in `common.py`, `epoch_data.py`)
- Active learning: `active_learning/ani_energy/`
