---
name: new-experiment
description: Create a new experiment folder (experiment.yaml + config.yaml) under experiments/<dataset>/<model>-<uq>/<protocol>, validate it, and prepare training. Use when the user wants a new experiment, a variant of an existing one (hyperparameters, protocol, dataset), or a new UQ method/model setting.
---

# Create an experiment

1. Decide the id `<dataset>/<model>-<uq>/<protocol>` (`docs/layout.md`): the dataset must
   exist in `data/` (else use `add-dataset` first); a variant of an existing protocol gets
   a descriptive suffix (e.g. `hf-only-lr1e-3`), never a letter.
2. Copy the closest experiment without its outputs:
   `mkdir -p experiments/<id> && cp experiments/<src>/{experiment.yaml,config.yaml} experiments/<id>/`.
3. Edit `experiment.yaml` (`docs/experiments.md`): drop `legacy_id`; set `dataset`,
   `protocol`, `target`, `parent` (only for `lf-to-hf`: same dataset/splits/members),
   `evaluation` (`reliability` roles, `sets`, `batch_size`, `trim: 0.0`, optional steps).
   Edit `config.yaml`: hyperparameters only (no paths), a new `wandb_project`, and
   comments explaining what differs from the source experiment.
4. `uq check`, `uq show <id>`, `uq train <id> --gpus G --dry-run`,
   `uq evaluate <id> --gpu G --dry-run`: the resolved files and commands must be what you
   intend.
5. Regenerate the index: rewrite the table in `experiments/README.md` from
   `uq list --markdown` (a unit test fails otherwise).
6. A new model or UQ method needs code first (`docs/extending.md`) and its
   `(model, uq)` pair in `registry.BACKENDS`; keep the per-configuration output schema
   so aggregation and figures work unchanged.
7. Train with `train-experiment`, evaluate with `evaluate-experiment`, record with
   `record-result`, then freeze its commands:
   `python -m uq.tests.test_commands --update` (the diff must only add this experiment).
