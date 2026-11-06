# Local handoff audit — 2026-11-06, 15:23–15:31 CET

Requested scope: check the interrupted agent's work and local execution; write a
persistent handoff, but do not implement, launch, stop, restart or repair anything.
Leonardo settings and access were excluded. GPU owner remains `mace-80`.

**Conclusion:** the existing production jobs are making progress, but the approved
resume of nine interrupted members has not been launched. Leaving the current
scheduler alone will not produce complete energy ensembles. The resume code is
uncommitted and its scratch test has finished its two training epochs, with final
dataset evaluation still running at the last observation. No production failure
was found during this audit; this is a snapshot, not a guarantee of future completion.

## Verified local state

- Branch `iclr_rebuttal`, HEAD `30afae80b7695d9878fad39bd985e94d62e629d4`.
  Existing modifications: `mace/cli/run_train.py`, `mace/tools/{arg_parser,train}.py`,
  `uq/cli.py`, `uq/training/{launch,scheduler}.py`, `PROGRESS.md`; the rebuttal folder
  is untracked. No code/config changes were made by this audit. Manuscript worktree
  `ICLR27-UQ-MF/` was clean. Nothing was committed or pushed.
- `uq-train-mse.service`: active/running, PID **2144073**, started 14:37.
  Exactly one production scheduler; its command has `--gpus 0 2 6
  --jobs-per-gpu 3 --by-experiment`, **without `--resume`**.
- `uq-mse-postprocess.service`: active/running, PID **2144075**, started 14:37.
  Waiting for complete ensembles; no `.eval_failed` or `.prune_failed` markers.
  The existing HF-only `.pruned` handling prevents their reevaluation.
- Nine production LF-only members are live, with recent logs and finite validation
  losses. Snapshot at 15:27:

  | GPU | Members | PIDs | Last logged epochs |
  |---|---|---|---|
  | 0 | s1_m7, s2_m7, s3_m7 | 2144084, 2144085, 2144086 | 11, 12, 12 |
  | 2 | s4_m7, s0_m8, s1_m8 | 2144088, 2144089, 2144090 | 11, 12, 11 |
  | 6 | s2_m8, s3_m8, s4_m8 | 2144091, 2144092, 2144093 | 11, 11, 11 |

- A tenth process, scratch resume test PID **2214645**, also uses GPU 6. Thus the
  project uses exactly **three GPUs**, with four jobs temporarily sharing GPU 6.
  GPU 0 also has another user's process; do not treat all GPU-0 use as ours.
  No OOM/traceback/runtime-error lines found in the inspected current member logs.
- Free disk at 15:26: approximately **2.8 TiB** on the project filesystem and
  **138 GiB** on the root filesystem (scratch test lives there). No immediate
  disk-full condition observed; these are not estimates of future storage needs.

`python -m uq status` (mace environment):

| MSE experiment | Complete members by split | Evaluation |
|---|---|---|
| energy HF-only | 10, 10, 10, 10, 10 | 5/5 + aggregate |
| system HF-only | 10, 10, 10, 10, 10 | 5/5 + aggregate |
| energy LF-only | 6, 6, 5, 5, 5 | 0/5 |
| energy LF->HF | 6, 6, 5, 5, 5 | 0/5 |
| system LF-only | 0, 0, 0, 0, 0 | 0/5 |
| system LF->HF | 0, 0, 0, 0, 0 | 0/5 |

## Unfinished work that matters

1. **Production resume is still pending (WP11.6 / D1 / WP3.6).** At 14:37 the
   scheduler refused the nine partial LF-only members and removed their nine
   LF->HF children from its pending queue (`REFUSED` / `SKIPPED`). These are
   expected historical refusals, not nine newly crashed jobs. Editing the Python
   files does not update this running scheduler. Even finishing the nine parents
   elsewhere later will not put their skipped children back in this queue.
   Without intervention, energy LF-only and LF->HF can each reach at most 41/50;
   their automatic evaluation will keep waiting while the scheduler proceeds to
   the queued system experiments. The remaining five unstarted energy parents
   are s0–s4 m9 (27 complete + 9 partial + 9 live + 5 unstarted = 50).
2. **Scratch resume status:** copied s2_m5 checkpoint 269, using `--restart_latest
   --resume_exact --max_num_epochs 272`. Log records resume at lr 0.01, epoch 270
   at 15:25:42 and epoch 271 at 15:30:23, then `Training complete`, model export
   and final train/validation/test evaluation. Its results file contains exactly
   one validation row per epoch 0–271 (272 rows, no duplicate validation epochs).
   Full process success/exit code was **not yet established**. This scratch model
   is not a production result and must not be copied over the production member.
   Locate its files from `/proc/2214645/cmdline` while live, or the existing
   `scratchpad/resume_test/` under the previous agent's temporary directory;
   key files are `run.log`, `cmd.txt` and the scratch `runs/` tree.
3. **The nine production partials are intact.** Latest checkpoint epochs:
   s2_m5=269, s3_m5=266, s4_m5=266, s0_m6=266, s1_m6=262,
   s2_m6=260, s3_m6=184, s4_m6=180, s0_m7=179. No final `.model`,
   malformed JSON lines or duplicate validation epochs in these nine logs.
   Their latest validation epochs equal their latest checkpoints. Do not delete
   these outputs or interpret the scratch model as completion of s2_m5.
4. **Restart coordination needs care:** the production unit has
   `KillMode=control-group`, `Restart=no`. A plain `systemctl --user stop` or
   `restart` would kill its running member processes too. The owner must choose
   a deliberate transition after validating resume; do not blindly use the old
   “stop only scheduler and adopt children” advice with a systemd unit. This
   audit did not change units or recommend an immediate restart.
5. **Existing scratch waiter is suspect:** PID 2215201 is a shell polling
   `pgrep -f "resume_test/runs.*resume_exact"`. Its own shell command contains
   that pattern, so it can match itself and wait indefinitely. Check the actual
   training PID and log for completion rather than trusting that waiter. This
   affects observation, not the training process; left untouched.

## Resume code review and checks

- `python -m uq check`: **15 experiments, 3 datasets: OK**.
- `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest
  discover -s uq/tests -t .`: **49 tests passed**, 15.879 seconds, exit 0.
  Used the existing mace environment. Tests wrote only their temporary fixtures.
- `git diff --check`: passed. No new GPU job or inference was started by this audit.
- The changes add next-epoch resume, preserve loaded optimizer LR, step the LR
  scheduler at the resumed epoch, trim rows beyond the saved checkpoint with a
  backup, and recognize resumed processes for adoption. Default command snapshots
  still pass. The existing suite has **no dedicated coverage of this new resume
  path**; passing 49 tests does not certify it.
- `resume_exact` is not a bit-exact continuation: checkpoint serialization holds
  model/optimizer/LR-scheduler state, but not RNG/data-loader state or the training
  loop's `lowest_loss`/`patience_counter`; those counters reset. EMA/SWA continuation
  also needs separate validation before broad reuse. The current MSE config has
  SWA disabled. No evidence of harm to the running fresh jobs was found.
- Additional static edge case for the owner: `launch.latest_epoch()` calls
  `iterdir()` without checking whether the checkpoint directory exists, whereas
  the scheduler catches only `SystemExit` from `prepare_resume()`. A partial run
  with a log but no checkpoint directory can therefore abort the scheduler with
  `FileNotFoundError`. The nine actual partials all have checkpoint directories,
  so this is not their current blocker. No code repair was made.

## Documentation and plan reconciliation

Read the repository Markdown documentation, rules/procedures, full results ledger,
the manuscript's local guide, and all four rebuttal Markdown files. Prior literature
citations were not reverified externally, and no scientific outputs were regenerated.

- `PLAN.md` “Currently running: nothing” and WP3.6 “paused / decision needed” are
  stale; D1 and `RUNS.md` correctly describe the authorized local restart.
  `PROGRESS.md` retains the older paused-study state below its current-study notes.
  This audit adds a fresh timestamped state rather than rewriting historical logs.
- The local timetable is not executing all advertised Nov 6–7 tasks: currently
  only MSE production and the resume scratch test are running. Registry validation
  still lists the original 15 experiments; head reset, beta-NLL, held-out head,
  LR sweeps and multi-head variants are not registered. WP2 inference and WP3.1
  CPU analysis remain open. All nine production slots are occupied; the queued
  system LF work also competes with the cheap local ablations. No revised ETA
  was established here, and the documented dates should not be treated as guarantees.
- Draft remains explicitly **not for posting**. It contains future-completion
  prose (“we added”, “we rewrote”, “fixed in the revised PDF”), pending results
  and references marked for verification. In particular, “AU stays calibrated on
  training data” conflicts with the ledger's 2026-09-23 interpretation correction
  (conservative / not overconfident is supported); “triggered by the fidelity
  change” is stronger than the still-unrun mechanism ablations establish.
  These are handoff flags only; draft/reviews/manuscript were not edited.
- Other existing documentation drift: two-GPU descriptions in `uq/README.md`
  and `docs/layout.md` versus current three-GPU rule; old figure numbering in
  parts of the paper README; the known manuscript EU-denominator discrepancy.
  Follow the current canonical rules and ledger rather than those old summaries.

## Next session, before changing jobs

Recheck PIDs/services and scratch completion; finish validation of WP11.6; arrange
one coordinated scheduler transition that accounts for the nine live members,
nine partial parents and nine skipped children; verify complete logs/checkpoints
and parent-child matching afterwards. Keep the system-queue placement decision
explicit before allocating local slots to other rebuttal experiments. These are
handoff actions for the owning agent, not actions executed by this audit.

Audit changes only: this file and timestamped pointers/state in `PLAN.md`, `RUNS.md`
and `PROGRESS.md`. No training, data, checkpoint, evaluation, configuration,
manuscript, service or code file was changed.
