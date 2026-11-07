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

**Ownership of all jobs (DGX and Leonardo):** session **`mace-leo`** since 2026-11-07 01:30 (the user: it is the
only session operating; earlier owner `mace-80` has ended). A later session takes over by writing its name here.
The `uq-*` units use `KillMode=control-group`: `systemctl --user stop <unit>` also kills its training members, so a
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
(ledger entry), or `failed` / `cancelled`. Last verified: **2026-11-06 20:50** (mace-80) for the DGX; Leonardo **2026-11-07 01:20** (mace-leo).

| Experiment | WP | Machine | Scope | Commit | Unit / job ids | Status | Started | ETA | Notes |
|---|---|---|---|---|---|---|---|---|---|
| energy MSE, main members s1-s4 m7, s0-s4 m8 (started by the old scheduler `uq-train-mse`) | 3.6 | DGX, GPUs 0 2 6 | 9 running | 30afae8 | moved to scope `uq-mse-adopted` (2026-11-06 20:35), adopted by `uq-mse-energy-x/y` | running | 2026-11-06 14:37 | ~2026-11-07 afternoon | old scheduler stopped 20:35 (D1b) without killing them |
| energy MSE, remaining s1-s4 m7-m9 + their LF->HF children | 3.6 | DGX, GPUs 0 2 6 | 12 + 12 (8 adopted) | c7fa140 | `uq-mse-energy-x` (`--splits 1-4 --members 7,8,9`); log `runs/_logs/scheduler_mace-mse-energy_2026-11-06.log` | running | 2026-11-06 20:36 | ~2026-11-08 | s1 m9 started 20:36 (GPU 2) |
| energy MSE, s0 m8-m9 + children | 3.6 | DGX, GPU 2 | 2 + 2 (1 adopted) | c7fa140 | `uq-mse-energy-y` (`--splits 0 --members 8,9`, 2 jobs on GPU 2); same log | running | 2026-11-06 20:36 | ~2026-11-08 | s0 m9 started 20:36 |
| MSE post-processing | 3.6 | DGX, GPU 0 | at 50/50: `uq evaluate <exp> --gpu 0`, then `uq prune-checkpoints <exp> --apply` | 30afae8 | `uq-mse-postprocess` (pid 2144075); log `runs/_jobs/mse_postprocess.log` | running (waiting) | 2026-11-06 14:37 | energy ~2026-11-08 | prune approved 2026-11-02; **owner stops it after the two energy evaluations** (the system split is evaluated on Leonardo) |
| energy MSE, 8 resumed parents s2-s4 m5, s0-s4 m6 + 8 LF->HF children | 3.6 | DGX, GPUs 0 2 6 | 8 + 8 | 30afae8 + resume code (committed in 32c2e64) | `uq-resume-mse-a`; log `runs/_logs/scheduler_mace-mse-resume_2026-11-06.log` | running | 2026-11-06 20:01 | m5, s0-s2 m6 ~23:00; s3-s4 m6 ~08:00-09:00 Nov 7; children +0.5 h | resumed after epochs 269 266 266 266 262 260 184 180; ~6 min/epoch with 6-7 jobs per GPU |
| energy MSE, resumed parent s0 m7 + child | 3.6 | DGX, GPU 6 | 1 + 1 | same | `uq-resume-mse-b`; same log | running | 2026-11-06 20:01 | ~09:00 Nov 7 + 0.5 h | resumed after epoch 179 |
| Leonardo env setup | 11.2 | Leonardo, `lrd_all_serial` (budget-free) | conda env + `pip -e` + checks | adc12bd (via bundle) | 63698806, 63699144 failed; **63701795 done** (6 min) | **done**: env `$WORK/uq-mf/miniconda3/envs/mace`, `uq check` OK, 58 tests OK | 2026-11-06 21:12 | | conda part solved; partial env in `$WORK/uq-mf/miniconda3/envs/mace`: next agent reads `uq-setup-env_63699144.out`, fixes the pip step (cause: `asemolec==1.0.0` in environment.yml is not on PyPI; drop or install it from its source), deletes the partial env (ours) and resubmits | ToS accepted (D12); data copied + manifest OK |
| Leonardo smoke test (a) | 11.8 | Leonardo, `boost_qos_dbg` | F s0 m0 + D s0 m0 packed on 1 GPU (`--pack 2`), `UQ_RUNS=$CINECA_SCRATCH/uq-mf/smoke/runs` | 7e1b998 | 63715545 | **done** (cancelled after 12 min, enough data) | 2026-11-07 00:57 | | first val losses F 0.01341 0.01001 0.00868, D -0.00356 -0.00560 -0.00616 (DGX: 0.0134 0.0100 0.0098; -0.0036 -0.0056 -0.0062); ~11 s/epoch with 2 packed; checkpoints + results written; W&B offline works |
| Leonardo smoke test (b) | 11.8 | Leonardo, `boost_qos_dbg` | `ani1x-system/mace-mse/lf-only` s0 m0-m2 packed 3 on 1 GPU, scratch `UQ_RUNS` | 7e1b998 | 63715577 | **done** (cancelled after 12 min) | 2026-11-07 00:57 | | LF epoch ~3.5 min per member with 3 packed (epoch 0 incl. warm-up 4.3 min) -> ~17.5 h per 300-epoch member; no OOM; scratch outputs, discard |
| system MSE `ani1x-system/mace-mse/lf-only` + `lf-to-hf` | 3.6 | **Leonardo**, `boost_usr_prod`, QOS `normal`, 24 h | 50 chains (LF member, then its LF->HF child in the same job), `--pack 3` -> 17 1-GPU jobs; `UQ_RUNS=$WORK/uq-mf/runs` (W&B offline: `runs/wandb` -> `$CINECA_SCRATCH/uq-mf/wandb-runs`) | 7e1b998 | 63716417-63716422, 63716431-63716441 (list: `$WORK/uq-mf/runs/_slurm/system-mse/jobs.tsv`) | **running** (all 17 started by 01:18; epoch 0 OK, no errors) | 2026-11-07 01:12 | ~2026-11-07 20:00 (18-19 h per job) | estimate ~17 x 19 h x 8 = ~2,600 local h; if a job hits 24 h: `uq train-slurm ... --resume` |
| system MSE back-up chains | 3.6 | Leonardo, QOS normal, 24 h | same 50 chains with `--resume`, `--after` the 17 jobs (`afterany`): complete members skip in seconds; a child that its chain did not start (parent exited non-zero after saving, as s0 m6 on the DGX) or a member stopped by the walltime continues | 0323ada | 63718283-63718300 (`runs/_slurm/system-mse-backup/jobs.tsv`) | queued (dependency) | 2026-11-07 01:36 | after the 17 jobs (+ minutes if all complete) | `uq train-slurm ... --pack 3 --resume --after <17 ids> --submit` |
| system MSE evaluation + prune | 3.6, 11.7 | Leonardo, 10 x 1 GPU (QOS normal, 24 h) + 1 CPU job (`lrd_all_serial`) | `uq evaluate <exp> --splits s --no-aggregate` per experiment and split, after the 34 jobs above (`afterany` + completeness check of the split and of the parent's split); then `uq prune-checkpoints lf-only lf-to-hf --apply` (D9) only if all 10 succeeded (`afterok`) | 0323ada | eval 63718301-63718310, prune 63718311 (`runs/_slurm/system-mse-eval/jobs.tsv`) | queued (dependency) | 2026-11-07 01:36 | eval ~2026-11-08 morning | `uq evaluate-slurm ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf --after <34 ids> --prune --submit`; inference checked first (job 63717684: `members_id.npz` of energy MSE HF-only s0 equal to the DGX within float32, max 0.0098 eV on ~28,000 eV); then sync back with `uq sync-back <exp> --host leonardo --checkpoints --apply` and `uq aggregate` + ledger on the DGX |
| WP2 LF-only baseline on CC labels (E0 refit), B and D | 2 | DGX, GPU 0 (inference only, next to the training jobs) | `uq evaluate D --splits 1-4 --steps lfonly_baseline --no-aggregate`, then B all splits (D s0 run by hand at 01:54, same command via `python -m uq.evaluation.lf_on_hf`) | commit of this row | background shell of mace-leo; log `runs/_jobs/wp2_lfonly_baseline_2026-11-07.log` | running | 2026-11-07 01:58 | ~02:20 | new files `lfonly_*` in `experiments/ani1x-{energy,system}/mace-mve/lf-to-hf/evaluation/split_*`; then `uq aggregate D B --only lfonly_` (CPU) + ledger |
| Leonardo smoke test (c), WP3.2 reset | 3.2 | Leonardo, `boost_qos_dbg` | `ani1x-energy/mace-mve/lf-to-hf-reset-var` s0 m0, scratch `UQ_RUNS` | ac60247 | 63718842 | **done** (cancelled after epoch 7) | 2026-11-07 01:45 | | log shows `Variance head reset: 32 weights`; val NLL 0.00906 0.00151 -0.00073 (D: -0.0036 -0.0056 -0.0062); ~6 s/epoch alone |
| WP3.2 variance-head reset, `ani1x-{system,energy}/mace-mve/lf-to-hf-reset-var` | 3.2 | Leonardo, QOS normal, 4 h | 100 members (50 + 50), `--pack 3` -> 34 1-GPU jobs; parents A and C copied (`.model` + results log, 200 files, md5 identical to the DGX) | ac60247 | `runs/_slurm/reset-var/jobs.tsv` | queued | 2026-11-07 01:50 | ~1 h after start | ~400 local h; evaluation: needs parent epoch checkpoints (epoch_quality_mixed, finetune_comparison) -> copy pruned A/C checkpoints (~2 x 50 GB) or evaluate on the DGX |
| WP3.5/WP4.3 FT lr sweep, energy: `ani1x-energy/mace-mve/lf-to-hf-lr1e-4`, `-lr1e-2` | 3.5, 4.3 | Leonardo, QOS normal, 4 h, `UQ_RUNS=$WORK/uq-mf/runs` | 100 members, `--pack 3` -> 34 jobs; parents C (copied) | b4d07ac | `runs/_slurm/ft-lr/jobs.tsv` | queued | 2026-11-07 02:10 | ~1 h after start | system split later (storage: ~85 GB per unpruned FT experiment) |
| WP4.2 HF-only lr sweep, energy: `ani1x-energy/mace-mve/hf-only-lr1e-3`, `-lr3e-3` | 4.2 | Leonardo, QOS normal, 8 h, **`UQ_RUNS=$FAST/uq-mf/runs`** (`/leonardo_scratch/fast/IscrC_UQ-MF/uq-mf/runs`) | 100 members, `--pack 3` -> 34 jobs | b4d07ac | `$FAST/uq-mf/runs/_slurm/hf-lr/jobs.tsv` | queued | 2026-11-07 02:10 | ~1.5 h after start | on `$FAST` because each unpruned 300-epoch experiment holds ~255 GB; system split after pruning |

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
- 2026-11-07 00:57-01:10 — mace-leo: Leonardo clone fast-forwarded to 7e1b998; smoke tests (a) 63715545 and (b)
  63715577 on `boost_qos_dbg` passed (losses match the DGX; LF ~3.5 min/epoch with 3 packed), cancelled after
  12 min. WP11.8 done.
- 2026-11-07 01:12-01:20 — mace-leo: system MSE production submitted (`uq train-slurm ani1x-system/mace-mse/lf-only
  ani1x-system/mace-mse/lf-to-hf --pack 3 --qos normal --time 24:00:00 --out $UQ_RUNS/_slurm/system-mse --submit`,
  `UQ_RUNS=$WORK/uq-mf/runs`); all 17 jobs running at 01:18. `saldo -b`: 0 consumed so far (accounting lags).
  DGX not checked by mace-leo (owner mace-80).
- 2026-11-07 01:25 — mace-leo (now owner of all jobs): DGX units healthy (energy MSE 60/100); `FAILED
  ani1x-energy__mace-mse__lf-only_s0_m6` at 01:05 is harmless (300 epochs done and model saved; MACE's closing
  metrics on the training set hit CUDA OOM with 6-7 jobs on GPU 2; the scheduler started its child). Markers
  `runs/_jobs/ani1x-system__mace-mse__{lf-only,lf-to-hf}.eval_failed` make `uq-mse-postprocess` skip the system
  split for good (it is evaluated on Leonardo; without them the DGX loop would re-evaluate and prune it after sync).
- 2026-11-07 01:36 — mace-leo: Leonardo clone at 0323ada (via bundle; children started after this run it; the
  training command is unchanged, pinned by the tests). Back-up chains, evaluation and prune jobs queued (Section 3).
