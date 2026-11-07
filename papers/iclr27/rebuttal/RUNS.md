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

**Owner of all jobs:** session `mace-leo-2` (since 2026-11-07 18:15; took over from `mace-leo`). A later session takes over by writing its name here.

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
Last verified **2026-11-07 18:15** (mace-leo-2), both machines.

| Experiment | WP | Machine | Commit | Job ids / unit | Status | Notes |
|---|---|---|---|---|---|---|
| `ani1x-energy/mace-mse/{lf-only,lf-to-hf}` | 3.6 | DGX GPUs 0 2 6 | 30afae8 / c7fa140 | `uq-mse-energy-x/y`, `uq-resume-mse-a/b`, `uq-mse-postprocess` | running, 84/100 (18:15) | 9 LF members running; at 50/50 `uq-mse-postprocess` evaluates (GPU 0) and prunes; markers in `runs/_jobs/` keep it off the system split |
| `ani1x-system/mace-mse/{lf-only,lf-to-hf}` | 3.6 | Leonardo `$WORK` | 7e1b998 / 0323ada | train 63716417-441 (17), back-up 63718283-300, eval 63718301-310, prune 63718311 | running | LF parents ~18:30-19:00, children in the same jobs, then evaluation ~10 h per split |
| `ani1x-{energy,system}/mace-mve/hf-only-lr{1e-3,3e-3}` | 4.2 | Leonardo `$FAST` | b4d07ac / 016dbb6 | prune 63720299, 63722493 | **recorded** (ledger 2026-11-07, `5c16649`) | |
| `ani1x-energy/mace-mve/lf-to-hf-reset-var` | 3.2 | Leonardo `$WORK` | ac60247 | `runs/_slurm/reset-var*/`; prune 63720981 | **recorded** (ledger 2026-11-07 "Fine-tuning ablations of the AU collapse") | |
| `ani1x-system/mace-mve/lf-to-hf-reset-var` | 3.2 | Leonardo `$WORK` | ac60247 | same | **recorded** (ledger 2026-11-07 "Fine-tuning ablations of the AU collapse") | sync interrupted 11:36, completed by the retry 12:06 |
| `ani1x-energy/mace-mve/lf-to-hf-lr1e-{4,2}` | 3.5, 4.3 | Leonardo `$WORK` | b4d07ac | `runs/_slurm/ft-lr*/`; prune 63720992 | **recorded** (ledger 2026-11-07 "Fine-tuning ablations of the AU collapse") | |
| `ani1x-system/mace-mve/lf-to-hf-lr1e-{4,2}` | 3.5, 4.3 | Leonardo `$WORK` | 016dbb6 | eval 63722646-655, prune 63722656 | **recorded** (ledger 2026-11-07 "Fine-tuning learning rate on the system split") | synced 18:17-18:31 (`runs/_jobs/sync_ftlr_system_2026-11-07.sh`) |
| `ani1x-energy/mace-mve/lf-to-hf-beta-nll` | 3.3 | Leonardo `$FAST` | c7710ef | `$FAST/.../_slurm/beta-nll*/`; prune 63721011 | **recorded** (ledger 2026-11-07 "Fine-tuning ablations of the AU collapse") | |
| `ani1x-system/mace-mve/lf-to-hf-beta-nll` | 3.3 | Leonardo `$FAST` | c7710ef | same | **recorded** (ledger 2026-11-07 "Fine-tuning ablations of the AU collapse") | |
| WP2 LF-only on CC (B, D; step `lfonly_baseline`) | 2 | DGX GPU 0 + Leonardo 63721292 | 9f8da8b / c7710ef | | **recorded** | caused one MSE OOM on the DGX (retried, complete) |
| Leonardo setup and smoke tests | 11 | Leonardo | adc12bd-c7710ef | 63701795; 63715545, 63715577, 63718842, 63720530 | done | smoke tests cancelled on purpose; 63698806, 63699144 = failed env attempts (fixed) |

## 4. Leonardo budget (local h; 1 GPU-h = 8 local h)

| Date | Consumed | Remaining of 80,000 | Note |
|---|---|---|---|
| 2026-11-07 12:00 | `saldo -b`: 0 (daily update lag); **sacct: ~594 GPU-h = ~4,750** (287 GPU jobs) | ~75,250 | ~2,500 more committed by running/queued jobs (system MSE, system FT-lr evaluation) |
| 2026-11-07 18:55 | `saldo -b`: 0 (still lagging; monthTotal 8,759); **sacct since 11-06: ~773 GPU-h = ~6,180** (297 jobs, running ones counted to now) | **~73,800** | still committed: the rest of the system MSE chain (training ~7 h to walltime, then evaluation ~10 h per split), roughly 1,500-2,500 more |

## 5. Log (newest last; earlier entries in Git history)

- 2026-11-07 10:40 — mace-leo: certificate renewed by the user. Leonardo: 263 COMPLETED, 27 RUNNING, 29 PENDING; the
  only FAILED/CANCELLED jobs are the known env attempts and smoke tests. DGX: 3 GPUs, no new FAILED, `/raid` 81%.
- 2026-11-07 10:48-12:00 — mace-leo: `runs/_jobs/sync_morning_2026-11-07.sh` copied back 8 of 9 finished experiments
  (log `runs/_jobs/sync_morning_2026-11-07.log`); system reset-var interrupted by a closed connection (rsync leaves no
  partial files), retried after the loop.
- 2026-11-07 12:06 — mace-leo: retry completed; all 9 experiments synced and aggregated; `compare_ablations`: 20 complete
  (ablation, test) pairs, 4 missing (system FT lr, still evaluating on Leonardo).
- 2026-11-07 ~13:30 — mace-leo, **handoff for the next session** (this one ends here):
  1. Leonardo certificate valid until ~22:40 today; the user renews it (`leonardo-login`; tunnel from the Mac
     `ssh -N -L 10000:127.0.0.1:10000 m.tolloso@<DGX IP, see memory>`; run the script in the background and give the user
     the printed URL).
  2. After ~18:00: `sacct -j 63722656` (prune of the system FT-lr sweep) COMPLETED -> sync back
     `ani1x-system/mace-mve/lf-to-hf-lr1e-{4,2}` (`$WORK`, Section 2), `uq aggregate`, `python -m
     uq.evaluation.compare_ablations` (then 24/24 pairs), and a ledger entry completing "Fine-tuning ablations of the
     AU collapse" (system FT lr; also the selected epochs, `uq.evaluation.training_steps.member_rows`).
  3. Tomorrow morning: system MSE prune 63718311 COMPLETED -> sync back `ani1x-system/mace-mse/{lf-only,lf-to-hf}`,
     aggregate, `compare_uq`-style contrast vs B and a ledger entry (WP3.6). Energy MSE on the DGX: check
     `runs/_jobs/mse_postprocess.log` (evaluates automatically at 50/50).
  4. Then the CPU work of `PLAN.md` Section 3.3 (appendix figures/tables) and the Q1 rewrite (draft note in 4.1).
  No reminders or scheduled wake-ups (user rule).
- 2026-11-07 18:15 — mace-leo-2 takes over (certificate valid, until ~22:40). Leonardo: system FT-lr evaluations COMPLETED
  (s1 of lr 1e-2 last, 18:14), prune 63722656 COMPLETED (125.7 GiB); system MSE: 15 LF/FT training jobs running ~17 h,
  back-up/eval/prune pending on dependencies. DGX: energy MSE 42/50 + 42/50 members, `uq-mse-*` units active.
- 2026-11-07 18:17-18:31 — mace-leo-2: system FT lr synced back, aggregated, `compare_ablations` 24/24; recorded (ledger
  "Fine-tuning learning rate on the system split"); appendix figures B, C and table A (`03e6b94`).
- 2026-11-07 ~18:50 — mace-leo-2, **handoff for the session of 2026-11-08 morning** (steps 2 and 4 above done; the
  next session takes over by writing its name in Section 1). In order:
  1. **Certificate:** it expires ~22:40 tonight, so tomorrow ask the user to run `leonardo-login` first (tunnel from the
     Mac `ssh -N -L 10000:127.0.0.1:10000 m.tolloso@<DGX IP, see memory>`; run the script in the background, give the
     user the printed URL). Never log in yourself.
  2. **State check (Sections 3-4):** `ssh leonardo 'squeue --me; sacct -X -S 2026-11-07 --state=FAILED,CANCELLED,TIMEOUT,OUT_OF_MEMORY; saldo -b'`;
     DGX `systemctl --user list-units 'uq-*'`, `nvidia-smi`, `uq status ani1x-energy/mace-mse/lf-only ani1x-energy/mace-mse/lf-to-hf`.
     System MSE chain: training 63716417-441 (started 01:10, 24 h walltime: if any hit TIMEOUT, back-up jobs
     63718283-300 resume them), eval 63718301-310 (~10 h per split), prune 63718311. Expect the evaluations still
     running in the morning if the children finished late; then do 4-5 first and come back.
  3. **System MSE, when prune 63718311 is COMPLETED:** with `UQ_REMOTE_RUNS` = `$WORK` (Section 2):
     `uq sync-back ani1x-system/mace-mse/{lf-only,lf-to-hf} --host leonardo --checkpoints --apply` (one at a time;
     rerun after an interruption, copies only what is missing; template `runs/_jobs/sync_ftlr_system_2026-11-07.sh`,
     run it with nohup and wait on its log), `uq aggregate` each, `uq status` must show 5/5 + aggregate, check that
     `evaluation/split_*/ensemble_size_*.csv` exist (needed by `compare_uq`), then `python -m uq.evaluation.compare_uq`
     (-> `results/tables/mse_vs_mve.csv`, LF->HF pairs vs B become complete; HF-only rows must stay byte-identical,
     check with `git diff`).
  4. **Energy MSE (DGX):** `uq-mse-postprocess` evaluates each experiment on GPU 0 when it reaches 50/50 and then prunes;
     read `runs/_jobs/mse_postprocess.log` (`EVALUATION FAILED` -> marker `runs/_jobs/<tag>.eval_failed`, needs a
     human: tell the user). Do not start other DGX GPU work next to it (OOM rule). When both energy experiments are
     evaluated, rerun `compare_uq` (energy LF->HF rows vs D).
  5. **Ledger (WP3.6):** one entry "MSE ensembles after LF->HF fine-tuning" per completed split, format of the
     2026-11-03/04 MSE entries (`record-result` skill): MSE LF-only and LF->HF vs their MVE twins (A/B, C/D), the
     question being whether pretraining changes the MSE-ensemble EU the way it changes MVE EU (Q2LT Q1 "no variance
     head" control, Q2). Five splits x ten members only; if a member or split is missing, record the gap.
  6. **Then CPU:** add the MSE LF->HF row (EU-only: RMSE, EU Spearman, EU/TU ENCE vs B/D) to
     `papers/iclr27/figures/figA_ablations.py` (read `mse_vs_mve.csv`, not `ablations_vs_reference.csv`, which has no
     MSE pairs) and to the general-response table in `draft_rebuttal.md` Section 5; replace `[PENDING: WP3.6]` in 4.1/4.2.
     Update PLAN WP3.6, Sections 3-4 here, commit and push to `iclr_rebuttal`.
  7. **Revised paper (user, 2026-11-07): start writing the new appendix** "Additional baselines and ablations"
     (PLAN Section 3.1, items A-E; WP9.4) in `ICLR27-UQ-MF/iclr2027_conference.tex` after the existing appendix
     sections (it has `\appendix` at l. ~347; macros `\revise{}` (blue) and `\LFHF` exist). Content: A table from
     `papers/iclr27/figures/out/tabA_lf_only.tex`, B `figA_lr_sensitivity.pdf` + one sentence on gradient steps, C
     `figA_ablations.pdf` + the mechanism text (from draft 4.1), D existing estimator results (ledger 2026-11-02/03/04,
     + MSE LF->HF if recorded), E design choices and scope (rewritten limitations). Copy PDFs into
     `ICLR27-UQ-MF/figures/` (copy, never move; note script + commit in a LaTeX comment), mark everything new with
     `\revise{}`, one pointer sentence in Results. Rules: `.claude/rules/paper-writing.md`, memory "paper prose, few
     numbers" (numbers in figures/appendix only, keep the authors' prose). That repo is synced with Overleaf: compile
     locally, show the user, and commit/push there only when the user says so.
  8. **Refine the rebuttal** (WP10.2-10.5): per-reviewer replies in `draft_rebuttal.md` (G4 learning-rate paragraph
     in 2.4 still says "[PENDING: WP4.2, WP4.3]": now both recorded), replace or drop every `[PENDING]`, trim to the
     OpenReview word limits, check numbers -> ledger and prior work -> Section 7, no author names.
  9. **Waiting on the user:** review of the Q1 rewrite (`draft_rebuttal.md` 4.1, including the proposed mechanism
     sentence for the revision); open decisions D3/D4/D5/D10 in PLAN Section 5. Wave A is planned for Nov 10-11.
  No reminders or scheduled wake-ups (user rule); never message other sessions.
