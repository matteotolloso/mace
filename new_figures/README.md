# Paper figures (ICLR27)

Main-text Figs. 2–6 and three appendix figures for the ICLR27 manuscript, rebuilt
from the repository caches following `notes/new_figures.md`. Fig. 1 (pipeline
drawing) is edited by hand and is not here. The paper chat copies the PDFs from
`out/` into `ICLR27-UQ-MF/figures/`; nothing here writes into the paper repository.

```bash
conda activate mace
python -B eval/support_filter.py --build               # once, CPU, ~5 min
python -B new_figures/run_epoch_predictions.py 0 2     # once, GPU 0, ~3.5 h (Figs. 3, 5 only)
bash new_figures/make_all.sh                           # all figures, CPU, ~1 min
```

Every figure writes `out/<name>.{pdf,svg,png}`. The PDFs are vector with embedded
TrueType fonts; scale them to `\linewidth` in LaTeX.

| Figure | Script / output name | Panels |
|---|---|---|
| Main Fig. 2 | `fig2_final_summary` | (a) RMSE, (b) AUSE, (c) ENCE of the final models |
| Main Fig. 3 | `fig3_hf_dynamics` | AUSE (a, b) and ENCE (c, d) for AU, EU and TU during the HF stage; System-OOD (a, c), Energy-OOD (b, d) |
| Main Fig. 4 | `fig4_rank_vs_calibration` | ENCE–AUSE plane: (a) system shift, (b) energy shift |
| Main Fig. 5 | `fig5_decomposition` | LF→HF over training: (a, b) train, (c, d) ID, (e, f) OOD; AUSE left, ENCE right |
| Main Fig. 6 | `fig6_acquisition` | (a) held-out Energy-OOD gain over random, (b) selected energy quantiles |
| Appendix | `figA_recalibration` | ENCE before/after isotonic recalibration: (a) System-OOD, (b) Energy-OOD |
| Appendix | `figA_eu_au_ratio` | (a) OOD EU/AU during LF→HF training, (b) final-model median EU/AU by energy quantile |
| Appendix | `figA_acquisition_id` | Energy-ID gain over random (single panel) |

The appendix figures hold exactly the panels removed from the main text on
2026-09-23 (old Fig. 4b, Figs. 5g–h, Fig. 6b); the numbers did not change. Fig. 2
no longer draws Spearman; the values are still computed by `paper_numbers.py`.

## Where the current paper figures come from

| Paper figure | Repo source |
|---|---|
| 2 (system reliability + tables) | `experiment_{A,E,B}/evaluation/cache/split_0/reliability_{id,ood}_nocal*` |
| 3 (energy reliability + tables) | `experiment_{C,F,D}/evaluation/cache/split_0/reliability_{id,ood}_nocal*` |
| 4 (LF→HF dynamics, system) | `experiment_B/.../epoch_quality_finetune_mixed_{train,id,ood}` |
| 5 (LF→HF dynamics, energy) | `experiment_D/.../epoch_quality_finetune_mixed_{train,id,ood}` |
| 6 (HF-only vs fine-tuning, system) | `experiment_E/.../epoch_quality_{id,ood}`, `experiment_B/.../epoch_quality_{id,ood}` |
| App. 9 (AU/EU vs energy quantile) | `experiment_{C,D,F}/.../energy_ood_bins.csv` |
| App. 16/17 (isotonic) | `experiment_*/.../reliability_{id,ood}_cal*` |

The side tables of the current Figs. 2 and 3 are **dataset split 0 only**, after
the 0.5% total-variance trim of `eval/reliability.py --trim 0.005` (reproduced to
all printed digits). The new figures use all five splits.

## Evaluation protocol of the new figures

- **Five dataset splits** everywhere. Each metric is computed within a split, then
  summarized as mean ± 95% Student-t interval (df = 4). Log-scaled quantities
  (ENCE, RMSE, ratios) use geometric means with log-space intervals.
- **Training-support filter instead of the total-variance trim.** Test
  configurations whose smallest interatomic distance is below the smallest one in
  that split's training data (`cc_train`, `cc_val`, `dft_train`, `dft_val`) are
  excluded, for every model and signal alike. This reads geometry only. The trim
  removed the highest-*uncertainty* configurations, i.e. it selected on the
  quantity being evaluated, and it did not always catch the blow-ups (appendix
  Fig. 17: calibrated LF→HF Energy-OOD RMSE 1.70 eV/atom, ENCE ≈ 4700).
  - Energy split: removes 13–16 of 5000 Energy-OOD and 0–2 Energy-ID configurations.
  - System split: removes nothing (its DFT training data covers compressed geometries).
  - The rule lives in `eval/support_filter.py` and is shared with
    `eval/aggregate_replicates.py`, so the repo's aggregate tables and these
    figures now agree to all printed digits (checked on experiment D, Energy-OOD).
    `active_learning/ani_energy/support_filter.py` applies the same rule to the
    AL runs (Fig. 6).
- Because of both changes, **numbers differ from the current paper text**, mostly
  in the system split (e.g. System-ID LF→HF RMSE 16.0 instead of 10.0 meV/atom,
  since the trim also removed legitimate hard configurations there).

## Figure-by-figure notes and deviations from `notes/new_figures.md`

**Fig. 2, final-model summary.** Panels (a) RMSE, (b) AUSE, (c) ENCE (accuracy first); the
Spearman column was removed for space. LF-only is dropped (evaluated on DFT
labels, not comparable). The data do *not* support "LF→HF is better on every
metric": it is better on Spearman and AUSE everywhere and on RMSE except
System-OOD (33.0 vs 34.0 meV/atom, a tie), but HF-only has lower
AU ENCE on all four tests and lower TU ENCE on Energy-ID and Energy-OOD
(Energy-OOD TU ENCE: HF-only 1.66, LF→HF 3.07). HF-only System-ID RMSE has a wide
interval because of one configuration in split 0 (C2H6, 0.652 Å, error −4.4 eV/atom).
It lies *inside* the training support and TU ranks it the most uncertain of 5000,
so it is kept.

**Fig. 3, HF-stage dynamics.** AU (dashed), EU (dotted) and TU (solid) for both
protocols; colour is the protocol, as in Fig. 5 (the first version drew TU only).
Adding AU shows that AU ENCE grows during CC training in *both* protocols, not only
after LF pretraining; it grows fastest for LF→HF. System-OOD comes from the existing
per-epoch caches, which the filter does not change. Energy-OOD is recomputed
from new per-configuration, per-epoch predictions (`run_epoch_predictions.py`),
because the cached AUSE is outlier-dominated (LF→HF final TU AUSE 0.044 raw vs
0.143 filtered). Shaded bands are the 95% interval over the five splits. Result:
the ranking half of the hypothesis holds (LF→HF TU AUSE is low from the first CC
epoch and flat; HF-only never catches up), the calibration half does not: TU ENCE
worsens with CC training for both protocols (LF→HF Energy-OOD 1.5 → 5.7, HF-only
minimum 0.86 at epoch ~15, then 8.4). The last-epoch values are above Fig. 2's
because final models are best-validation checkpoints.

**Fig. 4, ranking vs. calibration.** Only the ENCE–AUSE plane, with (a) system
shift and (b) energy shift. AU, EU and TU share one encoding: hollow marker = ID,
filled marker = OOD, joined by a thin line (TU used to be an arrow with a filled
ID marker, which broke the hollow/filled convention). **Appendix `figA_recalibration`** (formerly Fig. 4b) is a dumbbell (hollow = uncalibrated,
filled = isotonic) rather than bars, since bars on a log axis have no meaningful
baseline. It shows AU and EU as well as TU because they change the reading:
under system shift recalibration fixes EU (HF-only 2.05 → 0.48) but worsens
LF→HF TU (0.32 → 0.52); under energy shift it worsens AU and TU for both
protocols. Calibrated values come from `reliability_ood_cal_raw.csv` with the
support filter.

**Fig. 5, decomposition.** Six panels in a 3×2 grid: ranking (AUSE, left) and
calibration (ENCE, right) on (a, b) the training data
(`epoch_quality_finetune_mixed_train.csv`, unfiltered, since training
configurations are inside the support), (c, d) the ID test set, and (e, f) the OOD
test set. **Appendix `figA_eu_au_ratio`** (formerly Fig. 5g–h) has (a) the OOD
EU/AU ratio over training and (b) final models by energy quantile. EU does not grow during
pretraining (Energy-OOD EU/AU 0.42 → 0.24 → 0.35); it rises only during fine-tuning
(→ 0.51). During fine-tuning AU becomes badly overconfident on OOD data (AU ENCE
system → 82, energy → 207) while TU ENCE stays far lower (0.83, 5.7): EU partly
compensates. `figA_eu_au_ratio` (a) uses log₁₀(mean EU / mean AU), not the median of
per-configuration log-ratios: the system-split caches only store per-epoch means,
and getting medians would need another ~70 GPU-minutes (experiments A, B). With
the filter the means are no longer outlier-driven. For Energy-OOD the pretraining
stage is scored on the same `cc_test_ood` geometries as fine-tuning (against DFT
labels). The cached curves use the 50k `dft_test_ood` file instead; both are
Energy-OOD samples. `figA_eu_au_ratio` (b) uses the median per-configuration EU/AU per bin: the
system-balanced *mean* of the current Fig. 9 is set by a few configurations
(LF→HF top bin: mean EU 80.9 eV², median EU/AU < 1). With a robust estimator
EU/AU stays below 1 in every bin, rises only mildly from ID into Energy-OOD
(LF→HF ≈ 0.6 → 0.9) and falls again in the top bin. So "EU grows with depth into
the energy tail" is not supported. What *is* supported is that EU becomes
extreme on the few out-of-support geometries, which TU ranks first.

**Fig. 6, active learning.** (a) held-out Energy-OOD gains, (b) selected energy
quantiles. **Appendix `figA_acquisition_id`** (formerly Fig. 6b) shows the Energy-ID
gains. Both are drawn without the highest-DFT-energy
baseline: that needs new training runs (2 protocols × 10 members × 5 splits).
Data: the support-filtered `active_learning/ani_energy/runs/aggregate_epochs_50/`.
Caption values, random-500 RMSE (meV/atom): Energy-OOD HF-only 67.4, LF→HF 53.1;
Energy-ID HF-only 15.06, LF→HF 7.42.

## Files

| File | Role |
|---|---|
| `common.py` | paths, metric recomputation, statistics, palette, style (filter from `eval/support_filter.py`) |
| `epoch_data.py` | per-epoch metrics (cached CSVs for system, new predictions for energy) |
| `build_geometry_cache.py` | wrapper for `eval/support_filter.py --build` (cache in `eval/cache/geometry/`) |
| `epoch_predictions.py`, `run_epoch_predictions.py` | per-configuration per-epoch inference (GPU) |
| `fig2_*.py` … `fig6_*.py` | one script per main-text figure |
| `figA_*.py` | one script per appendix figure; they import the drawing helpers of Figs. 5 and 6 |
| `paper_numbers.py` | prints every number quoted in the paper (added by the paper chat) |
| `make_all.sh` | regenerates all eight figures from the caches (CPU only) |

Palettes were checked with the dataviz validator (CVD separation, normal-vision
floor, chroma, contrast): protocol teal `#009099` / red `#c0443d`; AU `#2f6fc0`,
EU `#e8952f`, TU `#1d6b3a`; System-OOD `#6f4e9c`, Energy-OOD `#a6761d`.
