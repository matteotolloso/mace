# Progress: where the project is

Living status, so that any new session can continue: read this first, update it as you work.
Numbers are never stated here: citable values live only in `results/results.md`. Rewritten on
2026-11-12; the earlier version, with the day-by-day log of the MSE-ensemble study and of the
rebuttal, is `git show ebeb547:PROGRESS.md`.

## Current focus

**Journal version of the paper.** Running: **pretraining with DFT forces** on one DGX GPU
(Jobs table; estimate from the pilot's first epoch: ~16 h per LF member with three packed, so
~2.5-3 months for both designs, five splits, fine-tuning and evaluation; split 0 of both designs
is complete after ~3-4 weeks and is the first checkpoint for a look at the results). Then: ICLR 2027 ended at 2 / 2 / 2 after the rebuttal and is being
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
| `ani1x-{system,energy}-forces/mace-mve/{lf-only,lf-to-hf}`, splits 0-4 x members 0-9 | DFT-forces pretraining (JOURNAL item 1, PLAN P4) | DGX GPU 6 only, 3 members packed (user: one free GPU, no Leonardo, 2026-11-20) | commit of the forces_weight pick (made by the campaign) | `uq-forces-campaign` (`runs/_jobs/forces/campaign.sh`, log `runs/_jobs/forces/campaign.log`, markers `campaign.<step>.{ok,failed}`; restart = `systemd-run --user --unit=uq-forces-campaign -p KillMode=control-group --working-directory=$PWD -E GPU=6 bash runs/_jobs/forces/campaign.sh` (done steps are skipped; `uq train` skips complete members); user units stop if every login session closes (Linger=no) | this session | running since 2026-11-20 01:01, waiting for the pilot (~05:00); then per split: train -> evaluate -> prune |

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
