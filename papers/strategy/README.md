# Strategy: where the paper stands and where it goes

Venue-independent analysis and planning, written 2026-11-12 after the ICLR 2027 rebuttal was
posted, and revised the same day once the NeurIPS and RPS reviews were available. Read in this order; each file is self-contained.

| File | What | Maintained by |
|---|---|---|
| [`JOURNAL.md`](JOURNAL.md) | **start here**: Q1 journal shortlist and the improvements for the journal version in priority order (2026-11-19) | revise when a gate result or the user's choice of venue changes it |
| [`PLAN.md`](PLAN.md) | **the development plan**: thesis, work packages `P*` with tiers, gates, venues, budget, decisions `D*` for the user | every agent working on the next version (claim tasks there) |
| [`ASSESSMENT.md`](ASSESSMENT.md) | validity and position of the work against the literature of November 2026; strengths, gaps, verdict | revise when a gate result or a new related paper changes it |
| [`REVIEWS.md`](REVIEWS.md) | every ask of the nine reviewers at NeurIPS 2026, RPS 2026 and ICLR 2027, who raised it, what is fixed and what is open | revise when new reviews arrive |
| [`reviews_neurips26.md`](reviews_neurips26.md), [`reviews_rps26.md`](reviews_rps26.md), [`reviews_iclr27.md`](reviews_iclr27.md) | verbatim reviews with point IDs | frozen |
| [`FINDINGS.md`](FINDINGS.md) | interpretation of the recorded results (what they support, what is risky, how to phrase it) and the user's presentation preferences | whoever records or interprets a result |
| [`LITERATURE.md`](LITERATURE.md) | verified reference table, novelty map, theory notes, and the unverified landscape scan of 2026-11-12 | whoever checks a reference |

## One-screen summary

- **State.** NeurIPS 2026: withdrawn after reviews 2 / 3 / 2 (first draft). RPS workshop @ NeurIPS
  2026: accepted, oral (`RPS-UQ-MF/`, talk in `talk/`). arXiv version: `UQ-MF-Arxiv-/`. ICLR 2027: ratings 1 / 2 / 2, 2 / 2 / 2 after the rebuttal,
  **withdrawn 2026-11-19** (`../iclr27/REBUTTAL.md`); next: a Q1 journal (`JOURNAL.md`).
- **Assessment.** Sound and replicated since the September rebuild; the niche (uncertainty
  decomposition across a change of fidelity, along training) is still unoccupied. The limits are
  in the setting (energies only, an independently pretrained parent per member, one architecture,
  one dataset) and in two open controls (fine-tuning geometries already seen in pretraining; no
  observable target for the calibration of the learned variance).
- **What reviewers asked.** NeurIPS reviewers faulted the first draft on validity, which the rebuild fixed;
  ICLR then judged significance. Four concerns recur at every venue and are open: scope, novelty
  of the claim, the aleatoric / epistemic reading, the confounded protocol comparison. Acquisition
  was praised, never requested.
- **Plan.** One thesis ("the ordering of the uncertainty transfers across fidelities, its scale
  does not"), tested by intervention; first two CPU analyses and the cheap decisive experiments
  (unseen fine-tuning geometries, shared parent, scale re-estimation, resolved dynamics), DFT
  forces in pretraining in parallel, then a second architecture and multi-head replay. Venue: a Q1
  domain journal (shortlist and priorities in `JOURNAL.md`).
- **Waiting for the user:** approval of the priority list (`JOURNAL.md` Section 2) and decisions
  D1-D6, D8 in `PLAN.md` Section 8.

Numbers are never stated here without their ledger entry (`results/results.md`).
