# Datasets

Each dataset is declared in `data/<dataset>/dataset.yaml`: split seeds, test roles,
fidelities (short name, level of theory, energy key, file per role) and the exact
builder arguments. Experiments refer to files by role (`train`, `val`, `id`,
`ood`, `test`), never by path. `uq show <experiment>` prints the resolved files.

| Dataset | Split design | Fidelities (energy key) | Test roles | Support filter |
|---|---|---|---|---|
| `ani1x-system` | by molecular system | LF DFT `wb97x_tz.energy`, HF CC `ccsd(t)_cbs.energy` | `id`, `ood` | yes |
| `ani1x-energy` | by within-system energy quantile | same | `id`, `ood` | yes |
| `water-random` | random 65/10/25 per level of theory | LF BLYP, HF CCSD(T) (`REF_energy`) | `test` | no |

## Building

```bash
uq data build ani1x-system --splits 0-4 --dry-run   # print the builder commands
uq data build water-random --splits 5               # a new split (existing ones are never rebuilt)
uq data support              # support-filter bound and removals per split
uq data support --build      # build missing geometry caches (CPU, ~5 min)
```

A split that already has any of its files is never regenerated: completed
experiments depend on the exact split files. ANI-1x needs
`data/raw/ani1x-release.h5`; water reads `data/raw/water-n2p2/<theory>/input.data`.

## ANI-1x

Builder settings (`build.args` in the dataset specs; not the splitters' defaults):

- **System split** (`uq/datasets/ani1x_system.py`): 60% of molecular systems form
  the seen domain, whose configurations are partitioned 60/20/20 into candidate
  train/val/ID pools; the remaining systems supply OOD. This tests generalization
  to unseen systems, not a high-energy threshold.
- **Energy split** (`uq/datasets/ani1x_energy.py`): configurations are ranked by DFT
  energy *within each system*; ranks `<= 0.5` form the low-energy domain
  (partitioned 60/20/20 into train/val/ID), ranks `>= 0.55` the OOD domain, the band
  between is discarded. CC labels are never used for ranking. Systems with fewer than
  four valid energies are dropped; with exactly four, the highest goes to OOD and the
  others one each to train/val/ID, which is why a few seen-domain quantiles reach 0.667.
- Files are sampled with system-stratified quotas and `max_per_system=64` as a soft
  weighting cap. Sizes: DFT train/val/ID/OOD 50000/10000/50000/50000, CC
  5000/1000/5000/5000. Each CC file is an exact subset of the DFT file with the same role.
- Forces are not exported; training is energy-only.
- `split_summary.json` and `system_assignments.csv` record each split (tracked).

Verified on 2026-09-22 (seeds 0 and 3 exhaustively, by `system:conf_idx`): no
configuration shared between train, val, ID and OOD; CC files are exact subsets of
the DFT files; system-split OOD systems are disjoint from seen systems (0 of 1460);
energy-split OOD configurations lie above every seen-domain configuration of the same
system (0 violations in 2148 systems).

## Liquid water

16 H2O (48 atoms) in a 7.831 A cubic cell, periodic. Source: n2p2 `input.data` per
level of theory (`afqmc`, `blyp`, `ccsd`, `ccsdt`, `hf`, `revpbe-d3`; 531 BLYP and 200
CCSD(T) configurations), converted to extended XYZ (Hartree/Bohr -> eV/A) by
`uq/datasets/water_n2p2.py`; energy key `REF_energy`. Keep the periodic cells.

**Caveat — the fidelities are not aligned.** The converter shuffles each theory with
its own derived seed (`seed + sum(ord(c) for c in theory)`), so a CCSD(T) test geometry
may sit in the BLYP training set: 31-37 of the 50 CCSD(T) test geometries per split are
in BLYP train (`results/tables/water_split_overlap.txt`, `python -m uq.datasets.water_overlap`).
LF->HF water results are therefore biased in favour of pretraining. Fixing it needs one
configuration-level split shared by all theories (a new dataset, e.g. `water-aligned`)
and retraining.

## Adding a dataset

See [extending.md](extending.md#a-new-dataset).
