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
2. **Code:** Leonardo runs clean checkouts of commits of `iclr_rebuttal` (D8): normally pulled from
   `origin`; when a push from the DGX is not possible, the same commits are carried by `git bundle` over
   ssh and fast-forwarded (identical hashes). No edits on Leonardo. Record the commit per row.
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
(ledger entry), or `failed` / `cancelled`. Last verified: **2026-11-06 20:50** (mace-80).

| Experiment | WP | Machine | Scope | Commit | Unit / job ids | Status | Started | ETA | Notes |
|---|---|---|---|---|---|---|---|---|---|
| energy MSE, main members s1-s4 m7, s0-s4 m8 (started by the old scheduler `uq-train-mse`) | 3.6 | DGX, GPUs 0 2 6 | 9 running | 30afae8 | moved to scope `uq-mse-adopted` (2026-11-06 20:35), adopted by `uq-mse-energy-x/y` | running | 2026-11-06 14:37 | ~2026-11-07 afternoon | old scheduler stopped 20:35 (D1b) without killing them |
| energy MSE, remaining s1-s4 m7-m9 + their LF->HF children | 3.6 | DGX, GPUs 0 2 6 | 12 + 12 (8 adopted) | c7fa140 | `uq-mse-energy-x` (`--splits 1-4 --members 7,8,9`); log `runs/_logs/scheduler_mace-mse-energy_2026-11-06.log` | running | 2026-11-06 20:36 | ~2026-11-08 | s1 m9 started 20:36 (GPU 2) |
| energy MSE, s0 m8-m9 + children | 3.6 | DGX, GPU 2 | 2 + 2 (1 adopted) | c7fa140 | `uq-mse-energy-y` (`--splits 0 --members 8,9`, 2 jobs on GPU 2); same log | running | 2026-11-06 20:36 | ~2026-11-08 | s0 m9 started 20:36 |
| system MSE, `ani1x-system/mace-mse/{lf-only,lf-to-hf}` | 3.6 | **Leonardo** (D1b) | 100 | | | planned | | | after the smoke test; nothing of it runs on the DGX |
| MSE post-processing | 3.6 | DGX, GPU 0 | at 50/50: `uq evaluate <exp> --gpu 0`, then `uq prune-checkpoints <exp> --apply` | 30afae8 | `uq-mse-postprocess` (pid 2144075); log `runs/_jobs/mse_postprocess.log` | running (waiting) | 2026-11-06 14:37 | energy ~2026-11-08 | prune approved 2026-11-02; **owner stops it after the two energy evaluations** (the system split is evaluated on Leonardo) |
| energy MSE, 8 resumed parents s2-s4 m5, s0-s4 m6 + 8 LF->HF children | 3.6 | DGX, GPUs 0 2 6 | 8 + 8 | 30afae8 + resume code (committed in 32c2e64) | `uq-resume-mse-a`; log `runs/_logs/scheduler_mace-mse-resume_2026-11-06.log` | running | 2026-11-06 20:01 | m5, s0-s2 m6 ~23:00; s3-s4 m6 ~08:00-09:00 Nov 7; children +0.5 h | resumed after epochs 269 266 266 266 262 260 184 180; ~6 min/epoch with 6-7 jobs per GPU |
| energy MSE, resumed parent s0 m7 + child | 3.6 | DGX, GPU 6 | 1 + 1 | same | `uq-resume-mse-b`; same log | running | 2026-11-06 20:01 | ~09:00 Nov 7 + 0.5 h | resumed after epoch 179 |
| Leonardo env setup | 11.2 | Leonardo, `lrd_all_serial` (budget-free) | conda env + `pip -e` + checks | 1f92419 (via bundle) | 63698806 failed (solve); 63699144 **failed in the pip step** of `conda env create` | stopped (no job queued) | 2026-11-06 21:12 | | conda part solved; partial env in `$WORK/uq-mf/miniconda3/envs/mace`: next agent reads `uq-setup-env_63699144.out`, fixes the pip step, deletes the partial env (ours) and resubmits | ToS accepted (D12); data copied + manifest OK |
| Leonardo smoke test | 11.8 | Leonardo | 1 HF + 1 FT member, `boost_qos_dbg` | | | planned | | | after WP11.2-11.5 |

## 4. Leonardo budget (local h; 1 GPU-h = 8 local h)

| Date | Item | Consumed (`saldo -b`) | Remaining of 80,000 | Month quota used |
|---|---|---|---|---|
| 2026-11-06 | network test job 63697024 (`lrd_all_serial`, budget-free) | 0 | 80,000 | 0 of 8,759 |

## 5. Reconciliation log (newest last)

- 2026-11-06 ~14:31 — DGX: MSE study stopped (user request). 14:37: mace-23 restarted it as `uq-*`
  units; kept (D1).
- 2026-11-06 15:31 — DGX: read-only audit (`LOCAL_AUDIT_2026-11-06.md`): cap respected, jobs healthy.
- 2026-11-06 20:01 — DGX: 9 partial MSE members resumed (`uq-resume-mse-a/b`) after a scratch test.
- 2026-11-06 20:16 — Leonardo: first login from the DGX (certificate valid until 2026-11-07 08:14);
  read-only checks; nothing installed or submitted.
- 2026-11-06 20:27 — DGX: 4 `uq-*` units active, GPUs 0 2 6 only; Leonardo: no jobs.
- 2026-11-06 20:35 — DGX (D1b): the 27 processes of `uq-train-mse` (9 members + W&B helpers) moved to the
  delegated scope `uq-mse-adopted` (tested first on a dummy unit), then `uq-train-mse` stopped (only its
  main process died); `uq-mse-energy-x/y` adopted the 9 members and train the rest of the energy split.
  GPU memory after the two new m9 members: 35.8 / 35.6 / 31.9 GB of 41 on GPUs 0 / 2 / 6; no OOM.
- 2026-11-06 20:40-20:50 — Leonardo: compute (serial) nodes have internet (job 63697024: PyPI, conda,
  GitHub reachable); repo cloned to `$WORK/uq-mf/mace` (c7fa140); Miniconda in `$WORK/uq-mf/miniconda3`;
  ANI-1x splits rsynced (2.3 GB, 26 s), `data/MANIFEST.sha256` verified (182 files). Env creation waits
  for the ToS decision. Untouched: `$WORK/envs`, `$WORK/mace` (Aug 2026, user's earlier setup).
- 2026-11-06 ~21:30 — mace-80 stopped (user's token budget). Leonardo: **no job queued or running**; env not
  ready (pip step failed). DGX: units `uq-mse-energy-x/y`, `uq-resume-mse-a/b`, `uq-mse-postprocess`, scope
  `uq-mse-adopted` keep training the energy MSE split. Smoke-test inputs staged in
  `$CINECA_SCRATCH/uq-mf/smoke/runs` (parent C s0 m0 model + log). DGX reference losses for the smoke test:
  F s0 m0 epochs 0-2: 0.0134 0.0100 0.0098; D s0 m0: -0.0036 -0.0056 -0.0062 (~4 s compute per epoch).
