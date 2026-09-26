---
paths:
  - "experiments/**"
  - "data/**"
  - "uq/training/**"
  - "uq/datasets/**"
  - "uq/registry.py"
---

# Experiments and datasets

Schema and procedures: `docs/experiments.md`, `docs/datasets.md` (skills
`new-experiment`, `add-dataset`, `train-experiment`).

- An experiment is exactly `experiments/<dataset>/<model>-<uq>/<protocol>/` with
  `experiment.yaml` + `config.yaml`; its id is that path. Keep the naming; variants get a
  descriptive protocol suffix, never a new letter. Only the nine ICLR27 experiments have
  `legacy_id`s.
- `config.yaml` holds hyperparameters only; data files, run directories, seeds,
  `--wandb_dir` and the foundation model are added by `uq train-member`. Do not change
  the config of an experiment that has trained members (it is their provenance); create
  a new experiment instead.
- Every experiment has five splits and ten members; `lf-to-hf` needs a `parent` with the
  same dataset, splits and members. Run `uq check` after editing any spec.
- Existing dataset splits are never regenerated (`uq data build` refuses); a changed
  split design is a new dataset folder. Keep large files out of Git (`data/.gitignore`).
- Training: dry-run first (`uq train <exp> --gpus G --dry-run`), respect the two-GPU
  limit, check `uq status` afterwards. Evaluation outputs of the ICLR27 experiments are
  tracked and cited: do not regenerate them unless asked.
