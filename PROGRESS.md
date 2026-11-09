# Progress: ongoing work

Living status of work in progress, so that any new session can continue. Update the sections in
place and append to the **Log** (newest last). Numbers are not stated here: citable values live
only in `results/results.md`.

## Current focus: ICLR 2027 rebuttal (since 2026-11-06)

**Start at [`papers/iclr27/rebuttal/README.md`](papers/iclr27/rebuttal/README.md)** (standing rules
and file map), then `PLAN.md` (tasks, decisions) and `RUNS.md` (every job on the DGX and on CINECA
Leonardo, with owner, commit and status) in the same folder. Branch `iclr_rebuttal`. Responses are
due 2026-11-18. Jobs and machine state are tracked in `RUNS.md`, not here.

## Study in progress: MSE-ensemble baseline (part of the rebuttal, WP3.6)

**Question.** Does splitting uncertainty into AU and EU (MVE ensembles) beat a standard deep
ensemble whose only signal is member disagreement? ANI-1x only (water is out of the paper).

| Arm | What | Where | Status |
|---|---|---|---|
| MVE ensemble | AU, EU, TU of the paper's models | `experiments/ani1x-*/mace-mve/*` | done |
| Single MVE model | AU of one member, ensemble size M = 1..10 | `ensemble_size_*` of B, D, E, F | done (ledger 2026-11-02) |
| MSE ensemble | EU of ten MSE members (no variance head) | `experiments/ani1x-*/mace-mse/*` | HF-only done (ledger 2026-11-03/04); energy LF-only/LF->HF training on the DGX; system split training on Leonardo since 2026-11-07 01:12 (17 jobs, ETA evening Nov 7) |

**Decisions (user, 2026-11-02).** Checkpoint per member by its own validation loss (MSE for
`mace-mse`); LF->HF MSE members fine-tuned from MSE parents; same hyperparameters and 300/100-epoch
budgets as the MVE twins; prune each `mace-mse` experiment right after its evaluation (every 5th
epoch + each member's best + `.model`; record the pruning in the ledger).

**How to check (DGX).**
```bash
uq status ani1x-energy/mace-mse/lf-only ani1x-energy/mace-mse/lf-to-hf \
          ani1x-system/mace-mse/lf-only ani1x-system/mace-mse/lf-to-hf
systemctl --user list-units 'uq-*'                       # scheduler, resume units, post-processing
tail runs/_logs/scheduler_mace-mse-energy_2026-11-06.log   # energy schedulers uq-mse-energy-x/y
tail runs/_logs/scheduler_mace-mse-resume_2026-11-06.log   # the 9 resumed members and their children
tail runs/_jobs/mse_postprocess.log                        # automatic evaluate + prune at 50/50
nvidia-smi                                                 # only GPUs 0, 2 and 6 may be ours
```
History: the original scheduler (`uq-train-mse`, log `scheduler_mace-mse_2026-11-02.log`, with harmless
`REFUSED`/`SKIPPED` lines of 14:37 for the 9 partial members) was stopped on 2026-11-06 20:35 without
killing its members; the system split moved to Leonardo (D1b).
Post-processing markers in `runs/_jobs/`: `<exp>.pruned`, `<exp>.eval_failed` (needs a human; delete
the marker to retry), `<exp>.prune_failed`.

**After each experiment is evaluated (not automated):** check the outputs, run
`python -m uq.evaluation.compare_uq`, write the ledger entries (evaluation and pruning).

**Notes.** AU of an MSE ensemble is identically 0 (TU = EU; AU metrics NaN by design; never report
it). MSE members use nearly the whole 300-epoch budget (selected epochs ~244-299), MVE twins stop
at ~30-111. Possible extra arm: compare both methods after the same recalibration.

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
- 2026-11-06 — ICLR reviews out. User asked to commit (`30afae8` on `main`, not pushed), stop the
  MSE-ensemble work and switch to branch `iclr_rebuttal`. Scheduler, post-processing job and 9
  running energy LF-only members stopped (SIGTERM, ~14:31; list in "Current study"); GPUs 0, 2, 6
  freed. Rebuttal plan in `papers/iclr27/rebuttal/PLAN.md`.
- 2026-11-06 14:37 — (session of 2026-11-02) Restarted the MSE scheduler and post-processing as
  systemd user units `uq-train-mse` (GPUs 0 2 6, 9 jobs) and `uq-mse-postprocess`, wrongly believing
  the 14:31 stop was a crash (it was the user's deliberate stop, see the entry above). The user then
  decided to resume the MSE runs, so both units stay; ownership of all GPU jobs is session mace-80
  (`RUNS.md`). This session only reports anomalies from now on and does not start/stop jobs.
- 2026-11-06 15:31 CET — Local-only check requested by user after token exhaustion:
  live production jobs healthy, resume unfinished; `uq check` and all 49 CPU tests
  pass. Persistent audit linked above; no Leonardo work or operational changes.
- 2026-11-06 20:01 — (session mace-80) The 9 partial energy LF-only members resumed after their latest
  epoch checkpoint with the new `uq train --resume` (scratch test passed first), as units
  `uq-resume-mse-a` (`--members 5,6`, GPUs 0 2 6) and `uq-resume-mse-b` (`--splits 0 --members 7`,
  GPU 6), log `runs/_logs/scheduler_mace-mse-resume_2026-11-06.log`; their LF->HF children follow in the
  same units. Each resumed results log was backed up as `<log>.pre-resume-<time>` and cut to the
  checkpointed epoch. Full job table: `papers/iclr27/rebuttal/RUNS.md`.
- 2026-11-06 20:35 — (mace-80) D1b: `uq-train-mse` replaced by `uq-mse-energy-x/y` (energy split only); its 9
  running members moved to scope `uq-mse-adopted` first and adopted, none killed. System split -> Leonardo.
- 2026-11-07 01:12 — (mace-leo) System-split MSE (`ani1x-system/mace-mse/lf-only` + `lf-to-hf`, 100 members as 50
  parent->child chains) running on Leonardo after a passed smoke test; 17 1-GPU jobs, 3 chains each; jobs in
  `papers/iclr27/rebuttal/RUNS.md`. Evaluation on Leonardo by split, then sync back and aggregate on the DGX.
- 2026-11-07 01:00-02:45 — (mace-leo, sole session; owner of all jobs) Leonardo: smoke tests passed; queued there, each
  with automatic per-split evaluation and pruning: system-split MSE (this study, D1b), and the rebuttal ablations
  WP3.2 (variance-head reset), WP3.3 (beta-NLL), WP3.5/4.3 (FT lr 1e-4, 1e-2), WP4.2 (HF-only lr 1e-3, 3e-3).
  New tools: `uq evaluate-slurm`, `uq sync-back`, `uq train-slurm --after/--after-ok`. DGX: energy MSE continues;
  the child `lf-to-hf` s2 m6 died of OOM next to an inference run of mine and was retried (unit
  `uq-mse-retry-s2m6`); markers keep `uq-mse-postprocess` off the system split. Next: sync back and aggregate
  (`papers/iclr27/rebuttal/RUNS.md` Section 2) once the user renews the Leonardo certificate.
- 2026-11-07 10:00 — (mace-leo) System MSE on Leonardo: LF parents at epoch ~145/300 at 09:40 (~3.5 min/epoch), so
  parents ~18:30, children ~+30 min, then evaluation (~10 h per split) and pruning, all queued. Leonardo is not
  reachable until the user renews the certificate. Energy MSE on the DGX: 70/100, no failures since 01:55.

- 2026-11-07 12:30 — (mace-leo) Certificate renewed; Leonardo and DGX healthy. Rebuttal ablations synced back and
  aggregated (HF-only lr sweep recorded; energy reset-var / FT lr / beta-NLL aggregated, entries pending). User:
  no further LR ablations, multi-head paused; `PLAN.md` condensed with a presentation plan (Section 3).
- 2026-11-08 16:50 — (mace-leo-4) Rebuttal: Energy-OOD per-epoch AU of the energy FT ablations done (ledger "AU along
  fine-tuning for the LF->HF ablations on Energy-OOD"); App. M.3 figure/caption updated. Energy MSE evaluation
  (`uq-mse-eval3`, 3 DGX GPUs) still running, slower than planned: results expected 2026-11-09.
- 2026-11-08 17:15 — (mace-leo-4) Energy-split MSE LF->HF final models evaluated first (GPU 4, user-approved) and
  recorded (ledger "MSE ensembles after LF->HF fine-tuning, ANI-1x energy split"): confirms the system split.
  WP3.6 numbers complete; full curves/aggregates of the energy MSE experiments still running (RUNS.md).
- 2026-11-09 12:05 — (mace-leo-5) Energy MSE full evaluation: `uq-mse-eval3` failed overnight on 9/10 splits (CUDA OOM, other users' jobs on the
  shared GPUs); missing steps relaunched as `uq-mse-eval4` (GPUs 7 2 0, per-step retries); expected end ~2026-11-10 evening
  (RUNS.md). Rebuttal draft and App. M made consistent (AUSE only, caveats, Fig. 22 raw values); first rebuttal post planned tonight.
- 2026-11-09 23:12 — Energy MSE full evaluation: LF-only complete (all 5 splits); every remaining LF->HF step had OOM'd twice
  (GPU 0 filled by other users) and was on its last attempt. Remaining 27 steps moved to `uq-mse-eval5`
  (`runs/_jobs/mse_energy_eval5.sh`: GPU with most free memory per step, max 3 GPUs incl. eval4, attempts reset);
  eval4 finishes its 3 running steps, then eval5 runs aggregate + prune. Expected end ~2026-11-10 evening (RUNS.md).
