# ICLR 2027 rebuttal — start here

Rebuttal of ICLR 2027 submission 18667 (reviews 1 / 2 / 2, received 2026-11-06). **The responses
were posted on 2026-11-10; until 2026-11-18 the only work here is replying to reviewer
follow-ups.** Branch `iclr_rebuttal`; manuscript and posted text in `ICLR27-UQ-MF/` (own Git
repository, synced with Overleaf). What comes next for the paper: `papers/strategy/`.

## Files

| File | What | State |
|---|---|---|
| `README.md` (this) | entry point and standing rules | current |
| [`PLAN.md`](PLAN.md) | reviewer asks -> answers -> where they are in the revised PDF; commitments; open items; decisions | record, condensed 2026-11-12 |
| [`reviews.md`](reviews.md) | reviews verbatim with point IDs `G*` `J*` `Q*` | frozen; append follow-ups with new IDs |
| [`draft_rebuttal.md`](draft_rebuttal.md) | long draft of the replies | superseded by `ICLR27-UQ-MF/rebuttal.tex`; kept for its arguments |
| [`RUNS.md`](RUNS.md) | every job of the rebuttal on the DGX and on Leonardo, Leonardo budget snapshots | closed registry (provenance) |
| `leonardo/` | scripts used on Leonardo | as used |

Moved on 2026-11-12: `FINDINGS.md` -> `papers/strategy/FINDINGS.md` (interpretation of the
results); `LEONARDO.md` -> `docs/leonardo.md` (cluster access, rules and the sync protocol);
reference table, novelty table and theory notes of the draft -> `papers/strategy/LITERATURE.md`.
Citable numbers: `results/results.md` only.

## Replying to a reviewer follow-up

1. Append the follow-up verbatim to `reviews.md` with new point IDs (never renumber).
2. Answer from recorded results: ledger entries, Appendix M, `papers/strategy/FINDINGS.md` (which
   also says how to phrase the two calibration caveats). A reply never states an unrecorded number.
3. Every statement about prior work cites a reference checked in `papers/strategy/LITERATURE.md`
   Section 1.
4. No new experiment without the user's decision: nothing new was to be posted after 2026-11-14.
5. The user posts; text goes into `ICLR27-UQ-MF/rebuttal.tex`. Posted text never names the authors.

## Standing rules (in addition to `CLAUDE.md`)

1. **Never message other agent sessions**: it reloads their whole context. Coordinate through
   files; if another session's action must change, tell the user.
2. **One owner for GPU jobs on the local DGX** at a time, named in the job registry; others report
   anomalies to the user and never start, stop or restart jobs. Local jobs run as systemd user
   units `uq-*` with `KillMode=control-group`: stopping a unit kills its training members too.
3. **Register before launching** any job, on either machine. Local cap: 3 GPUs (DGX only);
   Leonardo: queues and budget (`docs/leonardo.md`).
4. **Five splits x ten members** for anything reported; a pilot is labelled and approved.
5. **Never delete, overwrite or retrain** existing datasets, checkpoints or evaluation outputs
   unless the user asks for that exact action; variants are new experiment ids. Approved
   exception: pruning epoch checkpoints after evaluation (every fifth epoch, each member's best
   and the `.model` are kept).
6. **Commits:** commit and push finished work to `iclr_rebuttal`, never to `main`. The anonymous
   mirror is built from `main`: before this branch is merged, files that name the authors or the
   reviewers' forum must be checked (`docs/leonardo.md` holds a user name and an e-mail address).
7. **Leonardo access** is the user's: never try to log in or ask for passwords or OTP; when the
   certificate has expired, ask the user to run `leonardo-login` (`docs/leonardo.md`).
