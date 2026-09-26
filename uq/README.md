# `uq` — research package

Everything specific to this project; the forked MACE package is `../mace/`. Installed
in editable mode (`pip install --no-deps --no-build-isolation -e .`), which also
provides the `uq` command (`uq --help`).

| Module | Role |
|---|---|
| `paths.py` | the directory layout (the only module that hard-codes paths) |
| `registry.py` | load and validate `experiments/**/experiment.yaml` and `data/*/dataset.yaml` |
| `cli.py` | the `uq` command; thin wrappers around the modules below |
| `legacy.py` | experiment letters A-F/wA-wC and pre-2026-09 paths -> current ones |
| `datasets/` | builders: `ani1x_system`, `ani1x_energy`, `water_n2p2`; `build` (`uq data build`) |
| `training/` | `launch` (one member), `scheduler` (`uq train`, <= 2 GPUs), `status`, `prune` (`uq prune-checkpoints`) |
| `evaluation/` | one module per evaluation step, `pipeline` (spec -> steps), `aggregate`, `replicate_statistics`, `support_filter`, `summary`, `epoch_predictions`; `extra/` = exploratory plots |
| `active_learning/` | ANI-1x Energy-OOD active-learning workflow (`uq al ...`) |
| `tests/` | `python -m unittest discover -s uq/tests -t .`; `snapshots/commands.json` pins every recorded experiment's commands |

Evaluation modules are runnable: `python -m uq.evaluation.<module> --help`.
