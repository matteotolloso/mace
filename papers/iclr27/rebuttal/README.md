# ICLR 2027 rebuttal — start here

Since 2026-11-06 the project's active work is the rebuttal of ICLR 2027 submission 18667
(reviews 1 / 2 / 2). **Author responses are due 2026-11-18**; results are posted in waves, as
early as possible, to start the discussion. Work happens on branch **`iclr_rebuttal`**; the
manuscript is `ICLR27-UQ-MF/` (own Git repo, synced with Overleaf).

## Files (read in this order)

| File | What | Edited by |
|---|---|---|
| `README.md` (this) | entry point and standing rules | update only when a rule changes |
| [`PLAN.md`](PLAN.md) | reviewer asks -> status, **presentation plan** (appendix items, posts, figures), TODO with claims, decisions, short log | every agent |
| [`RUNS.md`](RUNS.md) | **every job on every machine** (local DGX, CINECA Leonardo), sync protocol, Leonardo budget | every agent that launches or checks jobs |
| [`LEONARDO.md`](LEONARDO.md) | Leonardo login (2FA) and cluster rules: filesystems, SLURM, accounting, our layout there | when a fact about Leonardo changes |
| [`draft_rebuttal.md`](draft_rebuttal.md) | draft OpenReview responses + verified reference table | writing agents |
| [`reviews.md`](reviews.md) | reviews verbatim with point IDs `G*` `J*` `Q*` | frozen (append follow-ups only) |

Elsewhere: `results/results.md` (the only source of citable numbers), `PROGRESS.md` at the repo
root (short machine-state notes and the history of the MSE-ensemble study).

## Standing rules (in addition to `CLAUDE.md`)

1. **Claim, then work.** Mark the TODO line in `PLAN.md` `[~] (agent: <name>, since <time>)`
   before starting, `[x]` with the output/ledger entry when done; append a line to its Log.
2. **Never message other agent sessions** (no `SendMessage` to peers): it reloads their whole
   context (one message cost ~500k tokens on 2026-11-06). Coordinate only through these files;
   if another session's action must change, tell the user.
3. **One owner for GPU jobs on the local DGX**, named in `RUNS.md`; others report anomalies to the
   user and never start, stop or restart jobs. Local jobs run as systemd user units `uq-*` with
   `KillMode=control-group`: `systemctl --user stop` kills their training members too.
4. **Register before launching**: a row in `RUNS.md` (experiment, machine, commit) for every job,
   on either machine. Local cap: **3 GPUs** (only on the DGX); Leonardo: queues + budget
   (`LEONARDO.md`).
5. **Numbers** only from the ledger; unrecorded results are `[PENDING: WPx.y]` in the draft.
   Five splits x ten members for anything reported; a "pilot" must be labelled and approved.
6. **Never delete, overwrite or retrain** existing datasets, checkpoints or evaluation outputs
   unless the user asks for that exact action; new variants are new experiment ids. Approved
   exceptions: pruning epoch checkpoints after evaluation (every 5th epoch + each member's best
   + `.model` kept) for `mace-mse` locally and for every experiment on Leonardo.
7. **Commits:** commit and push to `iclr_rebuttal` when a coherent block of work is done (user,
   2026-11-06); never to `main`. The anonymous mirror is built from `main` only, so this branch
   needs no anonymization; posted rebuttal text must still not name the authors.
8. **Literature:** every statement about prior work cites a reference checked and listed in
   `draft_rebuttal.md` Section 7.
9. **Leonardo access** is the user's: never try to log in or ask for passwords/OTP; when the
   certificate has expired, ask the user to run `leonardo-login` (`LEONARDO.md`).
