# Rebuttal runs: what runs where (DGX + CINECA Leonardo)

**Every training or evaluation job of the rebuttal is listed here before it is launched**, on
either machine, and updated when it ends. `PLAN.md` says *what* to run and why; this file says
*where it runs, from which commit, and whether its outputs are back*. Leonardo rules and layout:
`LEONARDO.md`.

## 1. Machines and ownership

| Machine | Resources for us | Limits | Access |
|---|---|---|---|
| **DGX** (shared, 8x A100 40 GB) | GPUs 0, 2, 6 (3 jobs per GPU normally) | **max 3 GPUs for this project at any time** (`uq train` enforces it) | shell; jobs as systemd user units `uq-*` |
| **Leonardo** (Booster, 4x A100 64 GB per node) | account `IscrC_UQ-MF`, ~10,000 GPU-h | SLURM QOS (`LEONARDO.md` Section 3); `$WORK` and `$FAST` 1 TB each | `ssh leonardo` while the user's 12 h certificate is valid |

**Ownership of DGX GPU jobs:** session **`mace-80`** (since 2026-11-06 15:00). Other sessions report
anomalies to the user and never start, stop or restart jobs. The `uq-*` units use
`KillMode=control-group`: `systemctl --user stop <unit>` also kills its training members, so a
stop is announced here first and done only by the owner.

## 2. Sync protocol

1. **One owner machine per experiment**, written in Section 3 *before* launch; never the same
   experiment id on both machines (a split-by-split division gets one row per split).
2. **Code:** Leonardo runs clean checkouts of commits pushed to `origin/iclr_rebuttal` (D8); no edits
   on Leonardo. Record the commit per row.
3. **Data:** never `uq data build` on Leonardo. rsync `data/ani1x-{system,energy}/` (2.3 GB) into the
   Leonardo clone and check it against `data/MANIFEST.sha256` made on the DGX (WP11.3).
4. **Parents:** LF->HF children need their parent's `.model` (~7 MB each); copy explicitly, list in the row.
5. **Run directory:** on Leonardo `UQ_RUNS=$WORK/uq-mf/runs` (overflow `$FAST/uq-mf/runs`), same
   layout `<id>/split_<s>/{checkpoints,results,logs,models}`.
6. **Same command:** members run exactly `uq.training.launch.command()` (pinned by the tests),
   wrapped in SLURM (WP11.5); no hand-written `run_train.py` lines.
7. **W&B:** offline on compute nodes (`WANDB_MODE=offline`, files in `$CINECA_SCRATCH/uq-mf`), synced
   later from a login node or not at all.
8. **Outputs flow Leonardo -> DGX only.** After training: evaluate on Leonardo (one job per split,
   `uq evaluate <id> --splits <s> --no-aggregate`), prune epoch checkpoints (D9), rsync back
   `runs/<id>/split_*/{results,logs,models}`, kept checkpoints and `experiments/<id>/evaluation/split_*`;
   `uq aggregate <id>` and the ledger entry happen **on the DGX only**.
9. **Status refresh:** whoever can reach Leonardo runs `squeue --me`, `saldo -b`, `cindata` and
   updates Sections 3-4 with a time stamp; without access write "unknown since <time>", never guess.
10. **Sync points:** at least morning and evening, reconcile this file with `squeue --me` (Leonardo)
    and `systemctl --user list-units 'uq-*'` + `nvidia-smi` (DGX); one line in Section 5.

## 3. Runs

Status: `planned` -> `queued` -> `running` -> `trained` -> `evaluated` -> `synced` -> `recorded`
(ledger entry), or `failed` / `cancelled`. Last verified: **2026-11-06 20:27** (mace-80).

| Experiment | WP | Machine | Scope | Commit | Unit / job ids | Status | Started | ETA | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `*/mace-mse/*`, main scheduler | 3.6 | DGX, GPUs 0 2 6 | energy LF-only s1-s4 m7, s0-s4 m8 running; queue: s0-s4 m9, LF->HF children of m7-m9, then the **system split** (100 members) | 30afae8 | `uq-train-mse` (pid 2144073); log `runs/_logs/scheduler_mace-mse_2026-11-02.log` | running | 2026-11-06 14:37 | energy ~2026-11-08; system ~2026-11-13 | started by mace-23; system split: D1b (move to Leonardo?) |
| MSE post-processing | 3.6 | DGX, GPU 0 | at 50/50: `uq evaluate <exp> --gpu 0`, then `uq prune-checkpoints <exp> --apply` | 30afae8 | `uq-mse-postprocess` (pid 2144075); log `runs/_jobs/mse_postprocess.log` | running (waiting) | 2026-11-06 14:37 | first: energy ~2026-11-08 | prune approved 2026-11-02 |
| energy MSE, 8 resumed parents s2-s4 m5, s0-s4 m6 + 8 LF->HF children | 3.6 | DGX, GPUs 0 2 6 | 8 + 8 | 30afae8 + resume code (committed in 32c2e64) | `uq-resume-mse-a`; log `runs/_logs/scheduler_mace-mse-resume_2026-11-06.log` | running | 2026-11-06 20:01 | m5, s0-s2 m6 ~23:00; s3-s4 m6 ~08:00-09:00 Nov 7; children +0.5 h | resumed after epochs 269 266 266 266 262 260 184 180; ~6 min/epoch with 6-7 jobs per GPU |
| energy MSE, resumed parent s0 m7 + child | 3.6 | DGX, GPU 6 | 1 + 1 | same | `uq-resume-mse-b`; same log | running | 2026-11-06 20:01 | ~09:00 Nov 7 + 0.5 h | resumed after epoch 179 |
| Leonardo smoke test | 11.8 | Leonardo | 1 HF + 1 FT member, `boost_qos_dbg` | | | planned | | | after WP11.2-11.5 |

## 4. Leonardo budget (local h; 1 GPU-h = 8 local h)

| Date | Item | Consumed (`saldo -b`) | Remaining of 80,000 | Month quota used |
|---|---|---|---|---|
| 2026-11-06 | nothing run yet | 0 | 80,000 | 0 of 8,759 |

## 5. Reconciliation log (newest last)

- 2026-11-06 ~14:31 — DGX: MSE study stopped (user request). 14:37: mace-23 restarted it as `uq-*`
  units; kept (D1).
- 2026-11-06 15:31 — DGX: read-only audit (`LOCAL_AUDIT_2026-11-06.md`): cap respected, jobs healthy.
- 2026-11-06 20:01 — DGX: 9 partial MSE members resumed (`uq-resume-mse-a/b`) after a scratch test.
- 2026-11-06 20:16 — Leonardo: first login from the DGX (certificate valid until 2026-11-07 08:14);
  read-only checks; nothing installed or submitted.
- 2026-11-06 20:27 — DGX: 4 `uq-*` units active, GPUs 0 2 6 only; Leonardo: no jobs.
