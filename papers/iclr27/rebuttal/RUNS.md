# Rebuttal runs: what runs where (local server + CINECA Leonardo)

**Every training or evaluation job of the rebuttal is listed here before it is launched**, on
either machine, and updated when it ends. `PLAN.md` says *what* to run and why; this file says
*where it runs, from which commit, and whether its outputs are back*. `PROGRESS.md` keeps the
local machine's detailed state (pids, logs) as before.

No absolute paths in this file (tracked, anonymously mirrored): write `<repo>` for the local
checkout and `$UQ_HOME` / `$UQ_RUNS` for the Leonardo checkout and run directory.

## 1. Machines

| Machine | GPUs | Limits | Budget | Access |
|---|---|---|---|---|
| **local** (shared server) | 8x A100 40GB, shared with other users | **max 3 GPUs for this project at any time** (CLAUDE.md; `uq train` enforces it); ~3 jobs per GPU packed | free, ~3 x 24 = 72 GPU-h/day | agents have shell access |
| **leonardo** (CINECA, Booster) | 4x A100 64GB per node, 32 cores, 512 GB RAM per node | `boost_usr_prod` 24 h walltime; `boost_qos_lprod` 4 days, max 32 GPUs; `boost_qos_dbg` 30 min, max 2 jobs (docs.hpc.cineca.it, checked 2026-11-06); booster `$TMPDIR` is 10 GB in RAM | **10,000 GPU-h** (user, 2026-11-06); how a 1-GPU job is charged (per GPU or per node, packed jobs): **check with `saldo -b` after the smoke test** | 2FA (CINECA step certificate); agents on the local server have **no** SSH configuration for it yet (WP11.1) |

**Confirmed by the user (2026-11-06, D7):** the 3-GPU cap applies only to the shared local server;
Leonardo is managed by its SLURM queues and our budget.

**Ownership:** all GPU jobs on the local server are owned by one agent session at a time, named
here; other sessions report anomalies to the user and to the owner and never start, stop or
restart jobs. Current owner: **session `mace-80`** (since 2026-11-06 15:00; session `mace-23`, which
started the systemd units below, agreed). Local jobs run as **systemd user units**
(`systemctl --user list-units 'uq-*'`), which survive the end of a login session; stopping a job
means `systemctl --user stop <unit>`, and is announced here first.

## 2. Sync protocol (both machines)

1. **One owner machine per experiment.** Each experiment id (`<dataset>/<model>-<uq>/<protocol>`)
   is trained on exactly one machine, written in the table below *before* launch. Never launch the
   same id on both. If an experiment is split across machines (e.g. by split), list each split as
   its own row.
2. **Code.** Leonardo runs only clean checkouts of commits that exist on the local branch
   `iclr_rebuttal` (no edits on Leonardo; fix locally, commit, update). Record the commit per row.
   How the commit reaches Leonardo (push the branch to `origin`, or `git bundle` + `scp`) is
   decision D8 in `PLAN.md`.
3. **Data.** Never run `uq data build` on Leonardo. Copy `data/ani1x-{system,energy}/` (dataset.yaml +
   `split_*`, 2.3 GB) with `rsync`, and check it against a sha256 manifest made locally
   (`<repo>/data/MANIFEST.sha256`, WP11.3) on both sides before the first launch.
4. **Parents.** LF->HF children need their parent's `.model` files (~7 MB each, 350 MB per parent
   experiment). Copy parents to the child's machine explicitly and list them in the row.
5. **Run directory.** On Leonardo set `UQ_RUNS` (supported by `uq/paths.py`) to a directory on the
   work filesystem, and `PYTHON_BIN` to the env's python. The runs keep the same relative layout
   `<id>/split_<s>/{checkpoints,results,logs,models}`.
6. **Same command.** Members are launched with exactly `uq.training.launch.command()` (what the unit
   tests pin), wrapped in SLURM (WP11.5). No hand-written `run_train.py` lines.
7. **W&B.** Compute nodes are assumed offline (verify): `WANDB_MODE=offline`, sync later from a
   login node, or disable.
8. **Outputs flow one way: Leonardo -> local.** After a job ends: evaluate on Leonardo (one job
   per split, `uq evaluate <id> --splits <s> --no-aggregate`), then `rsync` back
   `runs/<id>/split_*/{results,logs,models}`, the final `.model` and best-epoch checkpoints, and
   `experiments/<id>/evaluation/split_*`; run `uq aggregate <id>` and write the ledger entry
   **locally only**. Epoch checkpoints stay on Leonardo until the user decides (D9).
9. **Status refresh.** Whoever has a Leonardo session runs `squeue --me` and `saldo -b` and updates
   the tables here (time-stamped). Agents without access must not guess Leonardo state: they write
   "unknown since <time>".
10. **Sync points:** at least twice a day (morning / evening) reconcile this file with `squeue --me`
    on Leonardo and `nvidia-smi` + `ps` locally; log the reconciliation in Section 5.

## 3. Runs

**Latest local verification: 2026-11-06 20:05 CET** (session mace-80). Four units active:
`uq-train-mse` (main scheduler), `uq-mse-postprocess`, `uq-resume-mse-a`, `uq-resume-mse-b`. The
9 partial members were resumed at 20:01 with `uq train --resume` (scratch test passed first: epochs
270-271 appended, checkpoint 269 untouched, scheduler state continuous, no duplicate validation
epochs). Earlier audit: [`LOCAL_AUDIT_2026-11-06.md`](LOCAL_AUDIT_2026-11-06.md).

**Do not stop any `uq-*` unit with a plain `systemctl --user stop`**: the units use
`KillMode=control-group`, so stopping the unit kills its member processes too (audit, item 4).

Status: `planned` -> `queued` -> `running` -> `trained` -> `evaluated` -> `synced` -> `recorded`
(ledger entry written), or `failed` / `cancelled`.

| Experiment id | WP | Machine | Splits x members | Commit | Job ids / pids | Status | Started | ETA | Outputs back? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| all six `*/mace-mse/*` (scheduler) | 3.6 | local | energy lf-only 27/50 + 9 running (s1-s4 m7, s0-s4 m8); then s0-s4 m9 and the children of m7-m9 | 30afae8 + uncommitted (scheduler adoption) | unit `uq-train-mse.service`, pid 2144073 | running | 2026-11-06 14:37 | energy members ~2026-11-07 10:00 | yes (local) | restarted by session mace-23; queue continues with the remaining energy members, then their LF->HF children, then the system split (see D1 note below) |
| `uq-mse-postprocess.service` | 3.6 | local | evaluates + prunes each MSE experiment at 50/50 | same | pid 2144075 | running (idle) | 2026-11-06 14:37 | | | prune approved 2026-11-02 |
| ani1x-energy/mace-mse/{lf-only,lf-to-hf}, 8 partial parents s2-s4 m5, s0-s4 m6 + their 8 children | 3.6 | local, GPUs 0 2 6 (3 per GPU, on top of the main scheduler's 3) | 8 + 8 | 30afae8 + uncommitted resume code (to be committed as the resume commit) | unit `uq-resume-mse-a.service`; pids 2671737 2671757 2671793 (GPU 0), 2671830 2671853 2671889 (GPU 2), 2671912 2671928 (GPU 6) | running | 2026-11-06 20:01 | m5 / s0-s2 m6: ~2-3 h; s3-s4 m6: ~8 h; children +0.5 h | yes | `uq train ani1x-energy/mace-mse/lf-only ani1x-energy/mace-mse/lf-to-hf --gpus 0 2 6 --jobs-per-gpu 3 --by-experiment --members 5,6 --resume`; log `runs/_logs/scheduler_mace-mse-resume_2026-11-06.log`; resumed after epochs 269 266 266 266 262 260 184 180 |
| ani1x-energy/mace-mse/{lf-only,lf-to-hf} s0 m7 (partial parent + child) | 3.6 | local, GPU 6 | 1 + 1 | same | unit `uq-resume-mse-b.service`; pid 2671727 | running | 2026-11-06 20:01 | ~8 h + 0.5 h | yes | `uq train ... --gpus 6 --jobs-per-gpu 1 --splits 0 --members 7 --resume`; same log; resumed after epoch 179 |
| ani1x-system/mace-mse/{lf-only,lf-to-hf} | 3.6 | local (in the scheduler queue) | 0/100 | | | queued | | ~2026-11-12/13 at 9 slots | | **move to Leonardo?** the local queue reaches them after the energy split (~Nov 8); moving them frees the local GPUs for the cheap rebuttal ablations (decision pending) |
| (smoke test, WP11.8) | 11 | leonardo | 1 HF + 1 FT member | | | planned | | | | |

## 4. Leonardo budget ledger (GPU-h)

| Date | Item | Charged (from `saldo -b`) | Cumulative | Remaining of 10,000 |
|---|---|---|---|---|
| 2026-11-06 | (nothing run yet) | 0 | 0 | 10,000 |

## 5. Reconciliation log (newest last)

- 2026-11-06 ~14:31 — local: MSE study stopped by session mace-80 (user request). Leonardo: not set up.
- 2026-11-06 14:37 — local: session mace-23 restarted the MSE scheduler and post-processing as systemd
  user units (misread the stop as a crash). User then chose to resume MSE locally: units kept.
- 2026-11-06 15:20 — local: 9 MSE members on GPUs 0, 2, 6 (3 per GPU) + a 2-epoch resume test on
  GPU 6 (scratch dir, not a result). Ownership of local GPU jobs: mace-80.
- 2026-11-06 15:31 CET — codex-audit: local read-only reconciliation; details in
  `LOCAL_AUDIT_2026-11-06.md`. Three-GPU cap respected, services active, no new
  production errors observed; resume still pending. No launch/stop/restart.
- 2026-11-06 20:01-20:05 — mace-80: scratch resume test passed (16:31, `Done`); 2 resume units launched for
  the 9 partial members (+ their children). Local GPUs: 0, 2, 6 only (cap respected); 6-7 of our jobs
  per GPU until the resumed members finish. Energy MSE ensembles (50/50) expected when the main
  scheduler's m9 members finish (~2026-11-08); the post-processing unit then evaluates and prunes.
