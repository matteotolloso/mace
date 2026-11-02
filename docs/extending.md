# Extending: new datasets, experiments, models, UQ methods, studies, papers

The design rule: **specs say what, `uq` modules say how, and every prediction backend
writes the same per-configuration table.** Downstream code (aggregation, statistics,
`uq summary`, paper figures) only reads that table, so new models or UQ methods plug
in without touching it.

## The interface: per-configuration rows

`reliability_<set>_<cal|nocal>_raw.csv`, one row per test configuration in file order:

```text
config_index,num_atoms,ref_energy,pred_energy,error,sq_error,
aleatoric_var,epistemic_var,total_var,aleatoric_var_raw,epistemic_var_raw,total_var_raw
```

Per-atom energies (eV/atom) and variances ((eV/atom)^2); `*_raw` are the
uncalibrated variances (equal to the others in `nocal` files). `reliability_*_bins.csv`
(15 equal-count bins per signal) and `epoch_quality_<set>.csv` (one row per epoch:
`spearman_*`, `ause_*`, `ence_*`, `magnitude_*`, `rmse_e_atom`, `nll_energy`) complete
the contract. Produce these and `uq aggregate`, `uq summary` and the figure helpers
work unchanged. A method without an aleatoric part writes `aleatoric_var = 0`
(then TU = EU); one without an epistemic part writes `epistemic_var = 0`.

## A new dataset

1. Builder: `uq/datasets/<name>.py` with an argparse `main()` that writes
   `train/val/<test roles>` extxyz files for one split seed into `--outdir`
   (never overwriting). Keep raw sources under `data/raw/`.
2. Spec: `data/<dataset>/dataset.yaml` (copy `ani1x-energy/dataset.yaml`): `splits`,
   `tests`, `support_filter`, `fidelities.{lf,hf}` (`name`, `method`, `energy_key`,
   `files` per role), `build.{module,source,args}` with `{source}`, `{split_dir}`,
   `{split}` placeholders.
3. `uq data build <dataset> --dry-run`, then without `--dry-run`; commit
   `dataset.yaml`, split metadata and a note in [datasets.md](datasets.md). Large
   files stay ignored (`data/.gitignore`).
4. Support filter: set `support_filter: true` only if the extxyz `info` has
   `system` and `conf_idx` (the geometry cache stores them; ANI-specific today), then
   `uq data support --build`.
5. Record the dataset generation in `results/results.md` if results will depend on it.

## A new experiment

1. Copy the closest folder, e.g.
   `cp -r experiments/ani1x-energy/mace-mve/hf-only experiments/<dataset>/mace-mve/hf-only`,
   and remove its `evaluation/`.
2. Edit `experiment.yaml` (drop `legacy_id`; set `dataset`, `protocol`, `target`,
   `parent`, `evaluation` sets) and `config.yaml` (hyperparameters only; set a new
   `wandb_project`).
3. `uq check`, `uq show <id>`, `uq train <id> --gpus G --dry-run`, then train,
   `uq status <id>`, `uq evaluate <id> --gpu G`, `uq summary <id>`.
4. Regenerate the index in `experiments/README.md` from `uq list --markdown` (a unit
   test checks it).
5. Record every citable result in the ledger (`record-result` skill) and freeze the
   experiment's commands: `python -m uq.tests.test_commands --update` (the diff must
   only add the new experiment).

Naming: `experiments/<dataset>/<model>-<uq>/<protocol>`; a variant of a protocol gets
a descriptive suffix (`hf-only-lr1e-3`), not a new letter.

## A new model

`model` in `experiment.yaml` selects the training/inference backend. Implemented
`(model, uq)` pairs are listed in `BACKENDS` (`uq/registry.py`); `uq check` rejects
specs naming any other pair and `uq train`/`uq evaluate` refuse them, so add the pair
in the same change as its code. Today only `mace` exists: training is `uq/training/launch.py::command` (builds the MACE command)
and inference loads MACE checkpoints in `uq/evaluation/reliability.py`
(`discover_checkpoints`, `select_best_checkpoints`, `load_models`, `evaluate_split`)
and `epoch_quality.py`. For another model: add a launcher that builds its training
command from the spec (dispatch on `exp.model` in `launch.command`), keep outputs under
`runs/<id>/split_<s>/`, and add an evaluation backend that writes the tables above
(dispatch on `exp.model` in `uq/evaluation/pipeline.py::steps`). Keep MACE-specific
code out of shared modules.

## A new UQ method

`uq` in `experiment.yaml` names it (`mve-ensemble`: ensemble of MVE members,
AU = mean member variance, EU = population variance of member means; `mse-ensemble`:
ensemble of MSE members without variance head, AU = 0, EU as above). A new method
(single MVE model, MC dropout, evidential, Laplace, ...) needs: how members/samples
are trained (spec `members`, config), and a prediction function returning per
configuration `pred_energy`, `aleatoric_var`, `epistemic_var`. Implement it next to
`evaluate_split` (or in a new `uq/evaluation/<method>.py`), dispatch on `exp.uq` in the
pipeline, write the same tables, and document the decomposition in
[concepts.md](concepts.md). The experiment folder is then
`experiments/<dataset>/<model>-<method>/<protocol>`.

## A new active-learning study

The current workflow (`uq/active_learning/`) is specialized to ANI-1x Energy-OOD:
`common.py` fixes `DATASET`, `ENERGY_KEY`, `BUDGET`, `REGIMES` (parent experiment
ids), and `RUNS`. For a new study, parametrize those (or add a `study.yaml` under
`experiments/active-learning/<study>/`) and write runs to
`runs/active-learning/<study>/`. Keep its provenance checks.

## A new paper

`papers/<venue>/` with `README.md` (which figure comes from which script and ledger
entry), `figures/` (scripts + `make_all.sh`, outputs in `figures/out/`), `drafts/`.
Figure scripts resolve experiments with `uq.registry` and paths with `uq.paths`
(see `papers/iclr27/figures/common.py`). The manuscript itself is a separate
Overleaf-synced repository; copy PDFs into it and record which script and commit
produced them. `uq figures <venue>` runs `figures/make_all.sh`.
