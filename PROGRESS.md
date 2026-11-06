# Progress: ongoing work

Living status of work in progress, so that any new session can continue. Update the
**State** and **Next** sections in place and append to the **Log** (newest last).
Numbers are not stated here: citable values live only in `results/results.md`.

## Current study: MSE-ensemble baseline (rebuttal / follow-up)

**Question.** Does splitting uncertainty into AU and EU (MVE ensembles) beat a standard
deep ensemble whose only signal is member disagreement? Three arms, all on ANI-1x
(water is out of the paper for now):

| Arm | What | Where |
|---|---|---|
| MVE ensemble | AU, EU, TU of the paper's models | `experiments/ani1x-*/mace-mve/*` (done) |
| Single MVE model | AU of one member (M = 1), ensemble size M = 1..10 | `ensemble_size_*` of B, D, E, F (done, ledger 2026-11-02) |
| MSE ensemble | EU of ten MSE members (no variance head) | `experiments/ani1x-*/mace-mse/*` (training) |

**Decisions (user, 2026-11-02).** Checkpoint selection per member by its own validation
loss (MSE for `mace-mse`); LF->HF MSE members fine-tuned from MSE parents (full MSE
pretraining); same hyperparameters and 300/100-epoch budgets as the MVE twins; up to
3 jobs per GPU on GPUs 0 and 2 (never more than 2 GPUs; raised to 3 GPUs, adding GPU 6, on 2026-11-03); commit/push only when asked.

**How to check.**
```bash
uq status ani1x-energy/mace-mse/hf-only ani1x-system/mace-mse/hf-only \
          ani1x-energy/mace-mse/lf-only ani1x-energy/mace-mse/lf-to-hf \
          ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf
tail runs/_logs/scheduler_mace-mse_2026-11-02.log      # started/done/FAILED per member
pgrep -af "uq train ani1x-energy/mace-mse"            # the scheduler (one process)
nvidia-smi                                            # only GPUs 0, 2 and 6 may be ours
```
Training command (restartable; complete members are skipped, partial ones refused):
```bash
uq train ani1x-energy/mace-mse/hf-only ani1x-system/mace-mse/hf-only \
  ani1x-energy/mace-mse/lf-only ani1x-energy/mace-mse/lf-to-hf \
  ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf \
  --gpus 0 2 6 --jobs-per-gpu 3 --by-experiment
```
(run detached with `setsid nohup ... >> runs/_logs/scheduler_mace-mse_2026-11-02.log`. A restarted
scheduler adopts members still running from the previous one, so stopping only the
scheduler process and relaunching is safe.)

## State (2026-11-03 11:05)

- **GPU cap raised to 3 (user, 2026-11-03).** Scheduler restarted at 11:00 on GPUs 0, 2, 6
  (3 jobs each, 9 total). Order unchanged: energy HF-only (done), system HF-only (in
  progress, 7/10 per split at the restart), energy LF-only (+ LF->HF as parents finish),
  system LF-only (+ LF->HF). The old scheduler (pid 1442604, started 2026-11-02 12:18) was
  stopped alone with SIGTERM; its 6 running members kept running as orphans (ppid 1) and
  the new scheduler (pid 3432145) **adopted** them (log lines `adopted ...` at 11:00:43):
  `system/hf-only` s4_m6 (GPU 0, pid 3328512), s0_m7 s1_m7 s2_m7 (GPU 2, pids 3339401
  3359187 3359188), s3_m7 s4_m7 (GPU 0, pids 3366250 3381566). New on GPU 6: s0_m8 s1_m8
  s2_m8 (pids 3432153-5). At 11:04 no FAILED/REFUSED/SKIPPED lines in the log.
- **Code used by the restart (uncommitted):** `uq/training/scheduler.py`: `MAX_GPUS = 3`;
  `running_members()` finds live processes whose `/proc/<pid>/cmdline` equals
  `launch.command(job)` exactly (GPU from their `CUDA_VISIBLE_DEVICES`) and puts them in
  that GPU's slots (`_Adopted`, polled via `/proc/<pid>`; no exit code, so "done" means
  `member_complete`); before each launch a member already complete is skipped (`skipped
  ... completed meanwhile`). `uq/cli.py`: `MAX_GPUS = 3`. Docs/skills/CLAUDE.md say 3.
  49 unit tests pass; adoption tested only on this live run.
- **What could have gone wrong in the transition — check this first in a new session:**
  1. The 6 adopted members (pids above) finish: each must give a `done ... (GPU 0|2)` line,
     not `FAILED`, and `uq status` must count them. An adopted member that ended badly
     shows as `FAILED` (no exit code is available, only completeness). If one fails, look at
     `runs/_logs/ani1x-system__mace-mse__hf-only_s<S>_m<M>.log` (the orphan kept writing to
     it); its partial outputs will make a rerun `REFUSED` until removed (ask the user).
  2. No member is trained twice: no `started` line for a job that is already
     `adopted`/running: after the restart line, `sed -n '/scheduler restarted on GPUs 0 2 6/,$p'
     runs/_logs/scheduler_mace-mse_2026-11-02.log | grep -E "started|adopted" | awk '{print $4}'
     | sort | uniq -d` must print nothing (over the whole log, the 6 adopted members and the
     energy m0/m1 starts of 2026-11-02's restarts appear twice legitimately), and at
     most 3 of our run_train processes per GPU (`ps -eo pid,ppid,cmd | grep run_train`;
     `CUDA_VISIBLE_DEVICES` in `/proc/<pid>/environ`).
  3. Only GPUs 0, 2, 6 are ours (plus the evaluation of the post-processing job on GPU 0).
  4. GPU 6 throughput: if its members are much slower than those on 0/2, or GPU 6 gets
     taken by another user (OOM in member logs), tell the user.
  5. One scheduler only: `pgrep -af "python -m uq train ani1x"` / `ps -eo pid,ppid,cmd | grep "uq train"`
     must show pid 3432145 (or a later restart), never two.
  If the scheduler died: relaunch the command above with `--gpus 0 2 6` (it adopts live
  members); never start a second one next to a live one.
- Post-processing job running detached since 18:03: `runs/_jobs/mse_postprocess.sh`
  (log `runs/_jobs/mse_postprocess.log`). For each `mace-mse` experiment, once 50/50 members
  are complete: `uq evaluate <exp> --gpu 0`, then (only if that succeeded)
  `uq prune-checkpoints <exp> --apply`. Markers in `runs/_jobs/`: `<exp>.pruned`,
  `<exp>.eval_failed` (needs a human; delete the marker to retry), `<exp>.prune_failed`.
  Restart it the same way if it is not running (`pgrep -af mse_postprocess`):
  `setsid nohup runs/_jobs/mse_postprocess.sh >> runs/_jobs/mse_postprocess.log 2>&1 < /dev/null &`.
- Not automated (needs judgment): `python -m uq.evaluation.compare_uq`, checking the
  outputs, ledger entries (evaluation and pruning), paper/rebuttal text.
- Post-processing is independent of the scheduler (it only polls `uq status`); at 11:04 it
  was evaluating `ani1x-energy/mace-mse/hf-only` (split 5/5) on GPU 0, then it prunes it.
- Estimate (2026-11-03 11:45, 9 slots): HF member (300 ep) median 1.4 h wall-clock
  (87 done); LF-only member assumed ~16-17 h (DFT train/val files ~12x the CC ones,
  consistent with the 2026-11-02 6-slot estimate), LF->HF ~0.5 h. System HF-only training
  done ~15:00 today; energy LF-only ~2026-11-07; system LF-only and both LF->HF training
  ~2026-11-11/12; last automatic evaluations (~10 h each, sequential on GPU 0) ~2026-11-13.
  Adopted members confirmed working: s0_m7, s1_m7, s2_m7 gave `done` at 11:18-11:29.
- Evaluation pipeline verified on MSE models (smoke test on 3 complete members, scratch):
  every step runs; AU = 0, TU = EU, AU metrics NaN as designed. Aggregate plots fixed to
  skip a signal undefined at every epoch (otherwise log panels fell back to linear);
  no MVE plot affected (2660 series checked).
- Observation (one member, not a result): validation MSE still decreasing at epoch 299,
  so MSE members use the whole 300-epoch budget (MVE twins stop at ~30-110).

## Open questions for the user

- (none)

**Approved 2026-11-02:** prune each `mace-mse` experiment right after its evaluation:
`uq prune-checkpoints <exp>` (dry run, check), then `--apply` (keeps every 5th epoch + each
member's best + `.model`; ~1/5 of 17 MB x 300 epochs x 50 members per experiment). Record
each pruning in the ledger like the 2026-09-26 entry. An `lf-only` parent may be pruned once
it is evaluated (its LF->HF children only need its `.model` and every-5th epochs).

## Next

1. HF-only pairs (energy, system): DONE (ledger 2026-11-03, 2026-11-04).
2. Next automatic evaluations: `*/mace-mse/lf-only` (evaluation only, parents) and `*/mace-mse/lf-to-hf` (compare with B, D).
3. Possible extra arm: compare after the same recalibration for both methods (isotonic outputs
   `*_cal` exist for both; a scalar variance scaling is not implemented yet).
4. Rebuttal/paper text from the ledger only. First pair (energy, HF-only) recorded 2026-11-03.

## Log

- 2026-11-02 — Opinion given; user approved the MSE-ensemble ablation. Created the six
  `mace-mse` experiments (docs, CLAUDE.md, README). Throughput probe: 3 jobs/GPU = 1.5x.
  Scheduler improvements (children-first, `--by-experiment`, 8 CPU threads/job after
  load hit 428 on 256 cores, partial-output guard). Training started 12:18.
- 2026-11-02 — Single-model / ensemble-size arm for B, D, E, F evaluated and recorded
  (ledger "Single MVE model vs ensemble size"); `uq summary --members`, `uq aggregate --only`,
  `python -m uq.evaluation.compare_uq` added.
- 2026-11-02 — Disk: project 744 GB apparent size (runs/active-learning 327, runs/ani1x-energy
  197 incl. 82 of MSE so far, runs/ani1x-system 116, runs/water-random 94, data 8, experiments 1).
  User approved pruning each MSE experiment after its evaluation.
- 2026-11-02 — At the user's request, `restructure` merged into `main` (7027577), tags
  pushed, other branches deleted; work committed and pushed as 60524e6.
- 2026-11-02 — Evaluation smoke test on MSE models passed after one plotting fix
  (`aggregate._defined`); detached evaluate-then-prune job started (`runs/_jobs/`).
- 2026-11-03 01:26 — `ani1x-energy/mace-mse/hf-only` complete (50/50, no failures, faster than
  estimated); system HF-only training started; automatic evaluation of energy HF-only queued.
- 2026-11-03 11:00 — User raised the GPU limit to 3 (another user was on 4 GPUs, 6-7 idle).
  `MAX_GPUS = 3` (cli, scheduler) and docs/skills/CLAUDE.md updated; scheduler now adopts
  live members of a previous scheduler and skips members completed meanwhile. Old scheduler
  (pid 1442604) stopped alone, its 6 jobs kept running; new one (pid 3432145) on GPUs 0 2 6,
  9 jobs. System HF-only at 7/10 per split; energy HF-only evaluation on split 5/5. ETA not
  recomputed (expect roughly x1.5 throughput).
- 2026-11-03 13:14 — Energy HF-only MSE evaluated and pruned automatically (189.3 GiB freed).
  Then (session of 2026-11-02): outputs checked; found and fixed AUSE of a constant signal
  (MSE AU = 0) being reported as a number (now NaN in `reliability`/`epoch_quality`; aggregate masks
  old per-split files; MVE aggregates byte-identical); re-aggregated; `compare_uq` run; ledger
  entries "MSE-ensemble baseline, ... energy split, HF-only" and its pruning. DONE.
- 2026-11-03 13:52 — `ani1x-system/mace-mse/hf-only` complete (50/50, no failures); its automatic
  evaluation started 14:04 (~11 h). Energy LF-only (pretraining) training now.
- 2026-11-04 02:57 — System HF-only MSE evaluated and pruned automatically (189.2 GiB freed).
  Then (session of 2026-11-02): checked, `compare_uq` run, ledger entries "MSE-ensemble baseline,
  ... system split, HF-only" and its pruning. DONE. New: MSE ensemble more accurate on the system
  split (not on energy); uncertainty conclusions as on energy.
- 2026-11-04 03:00 — ETA update: energy LF-only members run ~4-5 min/epoch with 9 packed jobs on
  3 GPUs (+ evaluations), i.e. ~20 h per member; first 9 members (started 2026-11-03 ~11:00-14:00) at
  epoch ~170. Energy LF-only (+ its LF->HF fine-tunes) ~2026-11-08/09; system LF-only + LF->HF
  ~2026-11-13/14. /raid: 3.1 TB free.
- 2026-11-04 12:05 — Status check: 14/200 LF members complete (energy LF-only 8/50, energy LF->HF 6/50,
  system 0/100), no FAILED/REFUSED in the scheduler log; one scheduler (pid 3432145), post-processing
  job alive (idle, waiting for 50/50); ours: 3 run_train per GPU on 0, 2, 6 only. Second wave of energy
  LF-only members at epoch ~60 after ~3.7 h (~3.7 min/epoch, ~19 h/member): ETA above unchanged.
