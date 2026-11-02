---
name: add-dataset
description: Add a new dataset or split design under data/<dataset>/ with a builder module in uq/datasets/, a dataset.yaml spec and generated splits via `uq data build`. Use when the user wants experiments on new data, a new split design (e.g. aligned water splits), or new fidelities.
---

# Add a dataset

1. Name it `<source>-<split design>` (e.g. `water-aligned`); an existing dataset's splits
   are never regenerated or edited, so a changed split design is always a new dataset.
2. Put raw inputs in `data/raw/` (large files are git-ignored; record where they came from
   in `docs/datasets.md`).
3. Builder: `uq/datasets/<name>.py` with an argparse `main()` that writes one split seed's
   files (`<fidelity files per role>`, extxyz with the energy key in `info`) into
   `--outdir`/`--root`, deterministic in `--seed`, refusing to overwrite. Reuse
   `ani1x_energy.py`/`water_n2p2.py` patterns (stratified quotas, unit conversion).
4. Spec `data/<dataset>/dataset.yaml` (copy an existing one): `description`, `splits`,
   `tests`, `support_filter`, `fidelities.{lf,hf}.{name,method,energy_key,files}`,
   `build.{module,source,args}` with `{source}`, `{split_dir}`, `{split}` placeholders.
5. `uq data build <dataset> --dry-run`, then build; spot-check sizes and disjointness of
   roles (IDs must not repeat across train/val/test; CC subsets of DFT if applicable).
6. Support filter: only if `info` carries `system` and `conf_idx`; set
   `support_filter: true` and run `uq data support --build`.
7. Document it in `docs/datasets.md` (table + section); commit the spec, builder and
   metadata only (never `*.xyz`/`*.h5`). Record the generation in the ledger if results
   will cite it.
