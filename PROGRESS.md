# Progress: where the project is

Living status, so that any new session can continue: read this first, update it as you work.
Numbers are never stated here: citable values live only in `results/results.md`. Rewritten on
2026-11-12; the earlier version, with the day-by-day log of the MSE-ensemble study and of the
rebuttal, is `git show ebeb547:PROGRESS.md`.

## Current focus

**Journal version of the paper.** Running: **pretraining with DFT forces** on one DGX GPU
(Jobs table; measured on split 0: ~9-10 min per epoch with three packed, so ~2 days per LF member
of 300 epochs and ~2 weeks for the 20 LF members of split 0 alone; split 0 complete with
fine-tuning, evaluation and the fw1 arm around late December; the whole campaign on one GPU well
beyond 3 months. User, 2026-11-20: keep one GPU for now, may add GPUs later). Then: ICLR 2027 ended at 2 / 2 / 2 after the rebuttal and is being
withdrawn (decision 2026-11-19); the target is now a Q1 journal. Start at
`papers/strategy/JOURNAL.md` (journal shortlist, improvements in priority order), then
`papers/strategy/PLAN.md` (work packages, gates, budget). **Nothing is launched: it waits for the
user's approval of the priority list and decisions D1-D6, D8** (`PLAN.md` Section 8).

Other workspaces: `talk/` (oral at the RPS workshop, NeurIPS 2026; own `CLAUDE.md`), `RPS-UQ-MF/`
(workshop camera-ready), `UQ-MF-Arxiv-/` (arXiv version), `ICLR27-UQ-MF/` (latest full manuscript
with Appendix M, posted rebuttal). Record of the ICLR rebuttal: `papers/iclr27/REBUTTAL.md`.

## Jobs

Register every training or evaluation job here **before** launching it (experiment id, machine,
GPUs or SLURM job ids, commit, owner session, status), and update the row when it ends. One owner
session for the DGX GPUs at a time. Rules: `CLAUDE.md` (3 GPUs on the DGX), `docs/leonardo.md`
(Leonardo, sync protocol).

| Experiment | Package | Machine | Commit | Job / unit | Owner | Status |
|---|---|---|---|---|---|---|
| PILOT forces_weight 1/10/100 (`ani1x-system-forces` split 0, member 0, 30 epochs; not reportable) | DFT-forces pretraining | DGX GPU 6 (user: one GPU, 2026-11-20) | a6d4565 + uncommitted spec | `uq-forces-pilot` (`runs/_pilot/forces-weight/run.sh`) | this session | running |
| `ani1x-{system,energy}-forces/mace-mve/{lf-only,lf-to-hf}`, splits 0-4 x members 0-9 | DFT-forces pretraining (JOURNAL item 1, PLAN P4) | DGX GPU 6 only, 3 members packed (user: one free GPU, no Leonardo, 2026-11-20) | commit of the forces_weight pick (made by the campaign) | `uq-forces-campaign` (`runs/_jobs/forces/campaign.sh`, log `runs/_jobs/forces/campaign.log`, markers `campaign.<step>.{ok,failed}`; restart = `systemd-run --user --unit=uq-forces-campaign -p KillMode=control-group --working-directory=$PWD -E GPU=6 bash runs/_jobs/forces/campaign.sh` (done steps are skipped; `uq train` skips complete members); user units stop if every login session closes (Linger=no) | this session | **PAUSED 2026-11-24 by the user** (GPU 6 needed for another project; resume only when the user says so, with the restart command). Split 0: LF members 0-2 complete in both designs; interrupted system m3 (epoch 14), m4 (epoch 10), energy m3 (epoch 10); `campaign.sh` train steps now pass `--resume` |

The per-job registry of the rebuttal (2026-11-06 to 11-10) is in Git history
(`git show c35487c:papers/iclr27/rebuttal/RUNS.md`); its results are all in the ledger.

## Finished studies (results in the ledger)

| Study | Where | Ledger entries |
|---|---|---|
| Main experiments A-F (MVE ensembles, three protocols, two shifts) | `experiments/ani1x-*/mace-mve/{lf-only,lf-to-hf,hf-only}` | 2026-09-23 onwards |
| Acquisition proof of concept (Energy-OOD, one round) | `runs/active-learning/ani1x-energy/` | 2026-09-23 |
| Single MVE model and ensemble size | `ensemble_size_*` of B, D, E, F | 2026-11-02 |
| MSE-ensemble baseline, all three protocols, both splits | `experiments/ani1x-*/mace-mse/*` | 2026-11-03, 11-04, 11-08, 11-10 |
| Rebuttal baselines and ablations (pretrain-only on CC labels, steps, learning rates, head reset, beta-NLL) | `experiments/ani1x-*/mace-mve/{hf-only-lr*,lf-to-hf-*}` | 2026-11-07, 11-08 |
| Water (not in the paper; fidelity splits not aligned) | `experiments/water-random/*` | 2026-09-23, 09-24 |

Notes that outlive the studies: the AU of an MSE ensemble is identically zero (TU = EU; AU metrics
are NaN by design, never report them); MSE members are selected late (most of the 300-epoch
budget), MVE members early; each `mace-mse` experiment and every experiment trained on Leonardo
was pruned after evaluation (every fifth epoch, each member's best, `.model`).

## Log (newest last)

- 2026-11-10 — rebuttal posted; energy-split MSE evaluation recorded; DGX idle for this project.
- 2026-11-12 — whole-project review: `papers/strategy/` created (assessment against the
  literature, cross-venue review synthesis, development plan, literature file); rebuttal files
  condensed; `FINDINGS.md` and `LEONARDO.md` moved (`papers/strategy/`, `docs/leonardo.md`).
- 2026-11-12 — NeurIPS 2026 and RPS 2026 reviews received from the user and stored verbatim
  (`papers/strategy/reviews_{neurips26,rps26}.md`); review synthesis rewritten, assessment and plan
  revised (version 2: analyses A1, A2 and controls P8 added, acquisition benchmark to Tier 2,
  default venue a domain journal).
- 2026-11-19 — ICLR 2027 closed at 2 / 2 / 2; withdraw decided. Repository cleaned (rebuttal folder
  condensed into `papers/iclr27/REBUTTAL.md`, ICLR reviews to `papers/strategy/reviews_iclr27.md`,
  Leonardo setup job to `docs/leonardo-setup-env.sbatch`, budget to `docs/leonardo.md`); `iclr_rebuttal` merged
  into `main`, which is the working branch from now on. `papers/strategy/JOURNAL.md` written.
- 2026-11-20 — DFT-forces pretraining started (user: one GPU, no Leonardo, no hurry): datasets
  `ani1x-{system,energy}-forces` built and verified against the original splits; four experiments
  added; force-weight pilot (1 / 10 / 100, 30 epochs, split 0 member 0) on GPU 6; campaign unit
  `uq-forces-campaign` waits for it, picks the weight by the rule in `runs/_jobs/forces/pick_weight.py`
  (lowest validation energy RMSE per atom at epochs 25-29, largest weight within 5%), commits it,
  then trains / evaluates / prunes split by split. Force-error and force-disagreement evaluation
  is not written yet (to do while training runs; checkpoints are kept: every fifth epoch + best).
- 2026-11-20 — force weight: user chose option 1 (weight picked on validation by the pre-set rule)
  plus a **sensitivity arm**: `lf-only-fw<alt>` / `lf-to-hf-fw<alt>` on both designs, created by the
  campaign at the pick (alt = 1 if the pick is 10, else 10; specs list five splits, split 0 is
  trained right after the main split 0, before splits 1-4, so the first look at the results can
  check that the conclusions do not depend on the weight). Early pilot epochs (labelled, not
  reportable): weight 100 stops energy learning (variance absorbs the energy error).
- 2026-11-20 — pick done (forces_weight 10, fw1 arm; `1367d1d`); split 0 LF training started
  06:34. Epochs take ~9-10 min with three packed (pilot estimate was optimistic): ~2 days per member.
  User keeps one GPU for now, may add GPUs later.
- 2026-11-24 — forces campaign paused by the user (GPU 6 for another project): unit stopped, GPU 6 free.
  Interrupted members resume from their latest epoch checkpoint (`--resume` added to `campaign.sh`).
- 2026-11-24 — A1 poster figures: `papers/iclr27/figures/poster_figures.py` (CPU, reuses the paper
  scripts' data/statistics functions; only size/layout/typography change) -> `poster_figures/poster_fig{2..6}.{pdf,png}`
  (paper Figs. 2-6; Fig. 4 = test panels c-f only). Fig. 1 not produced (Figma export, no source).
  Round 2 (same day): all five at 4.4 in height (10.8, 8.7, 5.7, 6.0, 6.5 in wide).
- 2026-11-25 — poster figures round 3: top row Fig. 2 (8.8 x 4.4) + Fig. 4 with all six panels
  (10.75 x 4.4, 2 x 3); bottom row Figs. 5, 3, 6 at 4.1 in (5.6, 6.6, 6.0 wide). Sizes: `SIZE` in the script.
