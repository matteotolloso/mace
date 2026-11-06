---
paths:
  - "uq/**"
  - "mace/**"
  - "papers/**/*.py"
  - "setup.cfg"
---

# Code changes

- Read the files involved first; match the surrounding style (black, isort, pylint via
  `.pre-commit-config.yaml`; lines up to ~110 characters in `uq/`).
- Paths: only `uq/paths.py` knows the directory layout, only `uq/registry.py` reads
  specs. Resolve experiments with `registry.experiment(...)` (ids or legacy letters) and
  build paths with `paths.*`; never hard-code `runs/...`, `data/...` or letters.
- New code goes in the `uq` package (importable as `uq.<...>`, run as
  `python -m uq.<module>`); no `sys.path` hacks. A new top-level package needs
  `pip install --no-deps --no-build-isolation -e .` to become importable.
- Keep the evaluation interface stable: per-configuration rows
  (`reliability_*_raw.csv` columns), bins and epoch-quality columns are consumed by
  aggregation and figures (`docs/extending.md`). Output file names are part of it.
- `mace/` is the upstream fork: changes there alter training for every experiment.
  Keep them minimal, inside `### MVE ###` ... `### /MVE ###` fences, and keep
  `ScaleShiftMACE` (what `model: "MACE"` builds) and the plain `MACE` class consistent.
  Never change the behaviour of an already-trained configuration silently.
- Do not hand-edit generated outputs (`experiments/*/evaluation/**`,
  `papers/*/figures/out/**`, anything in `runs/`); regenerate them with their command.
- A change that alters numbers: rebuild the affected outputs (`uq aggregate`, figures)
  and add a ledger entry.
- `uq/tests/snapshots/commands.json` pins the training/evaluation commands of every
  recorded experiment. If a change alters them on purpose, run
  `python -m uq.tests.test_commands --update`, review the diff and report which
  experiments changed; otherwise the change is a regression.
- A new model or UQ method: add its `(model, uq)` pair to `registry.BACKENDS` in the same
  change as its launcher and evaluation dispatch (`docs/extending.md`); `uq check`,
  `uq train` and `uq evaluate` refuse pairs without a backend.

Verify with the narrowest check:

```bash
uq check                                             # specs
python -m unittest discover -s uq/tests -t .         # 54 CPU tests
uq evaluate <exp> --gpu 0 --dry-run                  # pipeline commands
python -m pytest tests --ignore=tests/test_cueq_oeq.py   # upstream MACE (slow, CI; needs pytest)
```
