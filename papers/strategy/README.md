# Strategy: where the paper stands and where it goes

Venue-independent analysis and planning, written 2026-11-12 after the ICLR 2027 rebuttal was
posted. Read in this order; each file is self-contained.

| File | What | Maintained by |
|---|---|---|
| [`PLAN.md`](PLAN.md) | **the development plan**: thesis, work packages `P*` with tiers, gates, venues, budget, decisions `D*` for the user | every agent working on the next version (claim tasks there) |
| [`ASSESSMENT.md`](ASSESSMENT.md) | validity and position of the work against the literature of November 2026; strengths, gaps, verdict | revise when a gate result or a new related paper changes it |
| [`REVIEWS.md`](REVIEWS.md) | what reviewers asked at NeurIPS 2026, RPS 2026 and ICLR 2027, and the common cause | revise when the missing verbatim reviews are added |
| [`FINDINGS.md`](FINDINGS.md) | interpretation of the recorded results (what they support, what is risky, how to phrase it) and the user's presentation preferences | whoever records or interprets a result |
| [`LITERATURE.md`](LITERATURE.md) | verified reference table, novelty map, theory notes, and the unverified landscape scan of 2026-11-12 | whoever checks a reference |

## One-screen summary

- **State.** RPS workshop @ NeurIPS 2026: accepted, oral (`RPS-UQ-MF/`, talk in `talk/`). arXiv
  version: `UQ-MF-Arxiv-/`. ICLR 2027: ratings 1 / 2 / 2, responses posted 2026-11-10, discussion
  until 2026-11-18 (`../iclr27/rebuttal/`).
- **Assessment.** Sound and replicated; the niche (uncertainty decomposition across a change of
  fidelity, along training) is still unoccupied. The limits are in the setting: energies only,
  an independently pretrained parent per ensemble member, one architecture, one dataset.
- **Why reviewers diverged.** They did not: an observational paper needs two of {tested mechanism,
  breadth, demonstrated consequence}; NeurIPS asked for the third, ICLR for the first two.
- **Plan.** One thesis ("the ordering of the uncertainty transfers across fidelities, its scale
  does not"), tested by intervention; cheap decisive experiments first (shared parent, scale
  re-estimation, resolved dynamics), DFT forces in pretraining in parallel, then a second
  architecture, multi-head replay and multi-round acquisition. Venue chosen at a gate.
- **Waiting for the user:** decisions D1-D9 in `PLAN.md` Section 8, among them the verbatim
  NeurIPS and RPS reviews.

Numbers are never stated here without their ledger entry (`results/results.md`).
