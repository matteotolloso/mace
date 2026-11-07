# Rebuttal runs: what runs where (DGX + CINECA Leonardo)

Every training or evaluation job of the rebuttal gets a row here **before** launch, on either machine,
and is updated when it ends. *What* and *why*: `PLAN.md`; Leonardo rules and layout: `LEONARDO.md`;
provenance of recorded numbers: the ledger entries. Condensed 2026-11-07; the full per-job notes of
2026-11-06/07 (smoke tests, storage plans, reconciliation log) are in Git history.

## 1. Machines and ownership

| Machine | Resources | Limits | Access |
|---|---|---|---|
| **DGX** (shared, 8x A100 40 GB) | GPUs 0, 2, 6 | **max 3 GPUs for this project at any time**; no extra inference next to packed training (OOM) | shell; jobs as systemd user units `uq-*` (`KillMode=control-group`: stopping a unit kills its members) |
| **Leonardo** (Booster, 4x A100 64 GB per node) | `IscrC_UQ-MF`, 80,000 local h | SLURM QOS; `$WORK` and `$FAST` 1 TB each | `ssh leonardo` while the user's 12 h certificate (`leonardo-login`) is valid |

**Owner of all jobs:** session `mace-leo` (since 2026-11-07 01:30). A later session takes over by writing its name here.

## 2. Sync protocol (short)

1. One machine per experiment id, written here before launch.
2. Leonardo runs clean checkouts of `iclr_rebuttal` commits (pull, or `git bundle` with identical hashes); no edits there.
3. Data are copied from the DGX and checked against `data/MANIFEST.sha256`; never `uq data build` on Leonardo.
4. Members run exactly `uq.training.launch.command()` through `uq train-slurm`; LF->HF children need their parent's `.model`.
5. Outputs flow Leonardo -> DGX only: evaluate there per split (`uq evaluate-slurm ... --prune`), prune (D9), then on the
   DGX `uq sync-back` (never overwrites), `uq aggregate`, `compare_*`, ledger entry.
6. Refresh Sections 3-4 at least morning and evening (`squeue --me`, `sacct`, `saldo -b`; DGX `systemctl --user list-units 'uq-*'`, `nvidia-smi`); without access write "unknown since <time>".

Sync-back (DGX, repo root):
```bash
export UQ_REMOTE_ROOT=/leonardo_work/IscrC_UQ-MF/uq-mf/mace
export UQ_REMOTE_RUNS=/leonardo_work/IscrC_UQ-MF/uq-mf/runs     # $FAST experiments: /leonardo_scratch/fast/IscrC_UQ-MF/uq-mf/runs
uq sync-back <exp> --host leonardo --checkpoints --apply         # rerun after an interruption: copies only what is missing
uq aggregate <exp>; python -m uq.evaluation.compare_ablations    # -> results/tables/ablations_vs_reference.csv
```
`$WORK`: system MSE, `*/lf-to-hf-reset-var`, `*/lf-to-hf-lr1e-{4,2}`. `$FAST`: `*/hf-only-lr*`, `*/lf-to-hf-beta-nll`.
A failed evaluation cancels its prune job (`afterok`): read the `.out`, `uq evaluate-slurm <exp> --splits <s> --submit`.
A training job at walltime: `uq train-slurm <exps> --resume --pack 3 --submit`.

## 3. Runs

Status: `queued` -> `running` -> `evaluated` (on Leonardo) -> `synced` -> `aggregated` -> `recorded` (ledger), or `failed`.
Last verified **2026-11-07 12:30** (mace-leo), both machines.

| Experiment | WP | Machine | Commit | Job ids / unit | Status | Notes |
|---|---|---|---|---|---|---|
| `ani1x-energy/mace-mse/{lf-only,lf-to-hf}` | 3.6 | DGX GPUs 0 2 6 | 30afae8 / c7fa140 | `uq-mse-energy-x/y`, `uq-resume-mse-a/b`, `uq-mse-postprocess` | running, 70/100 | 9 LF members running; at 50/50 `uq-mse-postprocess` evaluates (GPU 0) and prunes; markers in `runs/_jobs/` keep it off the system split |
| `ani1x-system/mace-mse/{lf-only,lf-to-hf}` | 3.6 | Leonardo `$WORK` | 7e1b998 / 0323ada | train 63716417-441 (17), back-up 63718283-300, eval 63718301-310, prune 63718311 | running | LF parents ~18:30-19:00, children in the same jobs, then evaluation ~10 h per split |
| `ani1x-{energy,system}/mace-mve/hf-only-lr{1e-3,3e-3}` | 4.2 | Leonardo `$FAST` | b4d07ac / 016dbb6 | prune 63720299, 63722493 | **recorded** (ledger 2026-11-07, `5c16649`) | |
| `ani1x-energy/mace-mve/lf-to-hf-reset-var` | 3.2 | Leonardo `$WORK` | ac60247 | `runs/_slurm/reset-var*/`; prune 63720981 | aggregated | ledger entry pending |
| `ani1x-system/mace-mve/lf-to-hf-reset-var` | 3.2 | Leonardo `$WORK` | ac60247 | same | evaluated; sync interrupted 11:36, retry queued (`runs/_jobs/sync_morning_retry.sh`) | |
| `ani1x-energy/mace-mve/lf-to-hf-lr1e-{4,2}` | 3.5, 4.3 | Leonardo `$WORK` | b4d07ac | `runs/_slurm/ft-lr*/`; prune 63720992 | aggregated | ledger entry pending |
| `ani1x-system/mace-mve/lf-to-hf-lr1e-{4,2}` | 3.5, 4.3 | Leonardo `$WORK` | 016dbb6 | eval 63722646-655, prune 63722656 | evaluating (~7 h per split, from 10:30) | |
| `ani1x-energy/mace-mve/lf-to-hf-beta-nll` | 3.3 | Leonardo `$FAST` | c7710ef | `$FAST/.../_slurm/beta-nll*/`; prune 63721011 | aggregated | ledger entry pending |
| `ani1x-system/mace-mve/lf-to-hf-beta-nll` | 3.3 | Leonardo `$FAST` | c7710ef | same | evaluated; syncing | |
| WP2 LF-only on CC (B, D; step `lfonly_baseline`) | 2 | DGX GPU 0 + Leonardo 63721292 | 9f8da8b / c7710ef | | **recorded** | caused one MSE OOM on the DGX (retried, complete) |
| Leonardo setup and smoke tests | 11 | Leonardo | adc12bd-c7710ef | 63701795; 63715545, 63715577, 63718842, 63720530 | done | smoke tests cancelled on purpose; 63698806, 63699144 = failed env attempts (fixed) |

## 4. Leonardo budget (local h; 1 GPU-h = 8 local h)

| Date | Consumed | Remaining of 80,000 | Note |
|---|---|---|---|
| 2026-11-07 12:00 | `saldo -b`: 0 (daily update lag); **sacct: ~594 GPU-h = ~4,750** (287 GPU jobs) | ~75,250 | ~2,500 more committed by running/queued jobs (system MSE, system FT-lr evaluation) |

## 5. Log (newest last; earlier entries in Git history)

- 2026-11-07 10:40 — mace-leo: certificate renewed by the user. Leonardo: 263 COMPLETED, 27 RUNNING, 29 PENDING; the
  only FAILED/CANCELLED jobs are the known env attempts and smoke tests. DGX: 3 GPUs, no new FAILED, `/raid` 81%.
- 2026-11-07 10:48-12:00 — mace-leo: `runs/_jobs/sync_morning_2026-11-07.sh` copied back 8 of 9 finished experiments
  (log `runs/_jobs/sync_morning_2026-11-07.log`); system reset-var interrupted by a closed connection (rsync leaves no
  partial files), retried after the loop.
