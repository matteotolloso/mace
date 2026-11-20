# Data

`data/<dataset>/dataset.yaml` declares a dataset (split seeds, test roles, fidelities
with energy keys and per-role files, builder arguments); `split_<s>/` holds the files.
Large files (`*.xyz`, `*.h5`) are git-ignored; split metadata, the support-filter
geometry caches (`split_<s>/geometry/`) and the raw water sources are tracked.

| Folder | Content |
|---|---|
| `ani1x-system/` | ANI-1x split by molecular system (System-ID / System-OOD) |
| `ani1x-energy/` | ANI-1x split by within-system energy quantile (Energy-ID / Energy-OOD) |
| `ani1x-system-forces/`, `ani1x-energy-forces/` | the same two splits (same builder arguments and seeds, verified frame by frame), with wB97x/def2-TZVPP forces in the DFT files, for pretraining with forces (2026-11-20) |
| `water-random/` | liquid water, random split per level of theory (fidelities not aligned) |
| `raw/` | `ani1x-release.h5` (ignored), `water-n2p2/<theory>/input.data` |
| `legacy/` | unused legacy exports (ignored) |

`uq data build <dataset>` generates missing splits and never overwrites one; details
in [docs/datasets.md](../docs/datasets.md).
