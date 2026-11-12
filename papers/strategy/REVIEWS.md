# What the reviewers said, across venues

Synthesis written 2026-11-12. Verbatim reviews: ICLR 2027 in
[`../iclr27/rebuttal/reviews.md`](../iclr27/rebuttal/reviews.md) (point IDs `G*`, `J*`, `Q*`).
**The NeurIPS 2026 and RPS 2026 reviews are not in this repository**: their rows below rest on the
user's summary of 2026-11-12 and are marked so. Add the verbatim texts as `reviews_neurips26.md`
and `reviews_rps26.md` in this folder (no author names, no forum URLs) and then revise this page.

## 1. Venues and outcomes

| Venue | Version | Outcome | Source of what follows |
|---|---|---|---|
| NeurIPS 2026 (main track) | "NeurIPS draft" (split 0 only for the tables, trim instead of the support filter; `../iclr27/README.md`) | not accepted | user's summary: reviewers asked about the **actionability of the signal**, i.e. active learning |
| RPS workshop @ NeurIPS 2026 | 5-page version, system shift only (`RPS-UQ-MF/`) | **accepted, oral**; reviews "very good" | user's summary |
| ICLR 2027, submission 18667 | full paper, five splits, both shifts, acquisition proof of concept (`ICLR27-UQ-MF/`) | ratings 1 / 2 / 2; responses posted 2026-11-10; discussion until 2026-11-18 | verbatim reviews |

## 2. ICLR 2027: the asks, ranked by how many reviewers raised them

| Ask | Who | Answered in the rebuttal by | Still open after the rebuttal |
|---|---|---|---|
| One architecture (MACE) | ge6X, j1p7, Q2LT (3/3) | argument only (design space, foundation models) | **yes: no second architecture** |
| One dataset | ge6X, j1p7, Q2LT (3/3) | argument only (3114 formulas, paired labels) | yes: no second dataset (water was removed from the paper) |
| Other multi-fidelity strategies (fidelity as input, multi-head, joint) | j1p7, Q2LT (2/3) | argument + future work | **yes: no experiment** |
| Mechanism of the AU overconfidence not validated | Q2LT | new ablations (head reset, beta-NLL, fine-tuning lr, train vs held-out) | partly: the direct test (variance fitted on held-out data) was not run |
| Other estimators | Q2LT | new baselines (MSE ensemble, single MVE model, ensemble size) | no |
| Protocol: compute, learning rates, pretrain-only | ge6X | new (steps, lr sweeps, LF-only on CC labels) | model size not varied |
| Energies only, no forces | ge6X | argument (CC labels have no forces) + future work | **yes: no experiment** |
| Motivation ("is HF fine-tuning common?"), "findings not unexpected" | ge6X; Q2LT (novelty vs Bilbrey 2025) | citations, novelty statement in Sec. 1 | a matter of framing, see Section 4 |
| "Aleatoric" with deterministic labels | ge6X | operational definition; offered to rename | rename not done |

Every reviewer rated the presentation "clear"; j1p7 (the most confident, 4) rated soundness 3 and
called the study timely and the validation thorough. No reviewer disputed the correctness of a result.
The low scores come from *informativeness* (ge6X 1, the others 2), i.e. from "what do I learn and
does it hold beyond this setting", not from flaws.

## 3. Why the two conferences asked for different things

They did not really ask for different things. Both sets of reviewers pointed at the same gap from
two sides:

| | NeurIPS 2026 (user's summary) | ICLR 2027 (verbatim) |
|---|---|---|
| Surface request | show the signal is *actionable*: active learning | forces, other fine-tuning strategies, other models and data, mechanism |
| Underlying question | "what can I **do** with these observations?" | "do these observations hold **where people actually work**?" |
| What was missing | a decision that improves because of the finding | external validity of the setting, and an explanation that predicts |

A paper that reports observations, without a new method, is accepted at these venues when it has at
least two of three properties: **(a) an explanation that is tested by intervention** and predicts new
cases, **(b) breadth** (the observation holds across models, data, protocols that matter in
practice), **(c) a demonstrated consequence** (a procedure that works better because of it). The
NeurIPS draft had none in full; reviewers asked for (c). The ICLR version added a small (c)
(single-round acquisition), so reviewers moved on to (b) and (a). Each reviewer fills the gap with
the property closest to their own background: uncertainty people ask for decisions, interatomic-
potential people ask for forces and foundation-model fine-tuning. Chasing each request one by one
is therefore endless; closing two of the three properties convincingly is not.

The workshop accepted the paper as an oral because its audience is exactly the intersection of the
two communities and its bar is "an interesting, well-supported observation", which the paper clears
easily. That is evidence that the *content* is valued by the right readers, and that the problem at
the conferences is *positioning and completeness*, not quality.

## 4. What this implies (input to `PLAN.md`)

1. **Recurring across venues, so non-negotiable for the next version:** a tested mechanism in the
   main text (a), one more architecture and forces in the low-fidelity stage (b), and an acquisition
   study that is more than one round from a test-set pool (c).
2. **Raised once, cheap, worth doing:** a second multi-fidelity strategy (multi-head replay
   fine-tuning from the existing parents), renaming the two variance terms.
3. **Raised once, expensive, not worth doing in this paper:** model-size sweeps, more learning-rate
   sweeps, a zoo of estimators, materials datasets.
4. **Framing:** "findings are not unexpected" (ge6X) and "learned something interesting" (j1p7,
   Q2LT) are reactions to the same results. The difference is whether the reader is given one sharp
   claim to test against their expectations. The current abstract lists four findings of equal
   weight; the next version should lead with one thesis (`PLAN.md` Section 2).
5. **ICLR outlook:** with 1 / 2 / 2 and no reviewer reply recorded by 2026-11-12, plan for a
   resubmission whatever happens in the discussion; a raise to borderline by Q2LT or j1p7 would not
   change the plan, only the venue calendar.
