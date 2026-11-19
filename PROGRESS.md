# Progress: where the project is

Living status, so that any new session can continue: read this first, update it as you work.
Numbers are never stated here: citable values live only in `results/results.md`. Rewritten on
2026-11-12; the earlier version, with the day-by-day log of the MSE-ensemble study and of the
rebuttal, is `git show ebeb547:PROGRESS.md`.

## Current focus

**Journal version of the paper.** ICLR 2027 ended at 2 / 2 / 2 after the rebuttal and is being
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
| — | — | — | — | — | — | nothing running or queued for this project (DGX checked 2026-11-10; Leonardo last checked 2026-11-08) |

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
