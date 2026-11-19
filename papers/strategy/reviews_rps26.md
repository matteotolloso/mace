# RPS workshop @ NeurIPS 2026 reviews: Submission 64 (verbatim)

Source: OpenReview forum page printed 2026-11-12 (PDF provided by the user). Text copied from the
PDF (ligatures and line breaks normalized). Author names and the forum URL are left out on
purpose: this tree is mirrored anonymously.

Submission: *Multi-Fidelity Transfer Learning Reshapes Uncertainty Quantification in Atomistic
Models*, workshop "Representations for the Physical Sciences" (RPS) at NeurIPS 2026: the short
version, system-level shift only, reviewed in September 2026 **before** the five-split rebuild of
the figures (the camera-ready in `RPS-UQ-MF/` uses the rebuilt ones). **Decision 2026-09-28:
Accept (Oral).**

**Point IDs** (`RE*` reviewer EZy7, `RL*` LPEk, `RC*` ecuE, `RPC` program chairs) are used by
`REVIEWS.md`. Never renumber them.

| Reviewer | Rating | Confidence |
|---|---|---|
| EZy7 | 3: Weak Accept (minor revisions; no new experiments needed) | 2: medium |
| LPEk | 4: Strong Accept (accept as is) | 2: medium |
| ecuE | 4: Strong Accept (accept as is) | 2: medium |

---

## Decision (Program Chairs): Accept (Oral)

**RPC.** The paper presents a timely and valuable study on how multi-fidelity transfer learning
affects uncertainty quantification and decomposition in atomistic potentials. Reviewers appreciated
the clear motivation, empirical analysis of training dynamics, and insights into the shift between
aleatoric and epistemic uncertainty. For the camera-ready version, please carefully address
reviewer comments regarding figure consistency, clarifying the apparent discrepancies between
Figures 1 and 2, and explicitly discussing the limitations of aleatoric variance transfer across
fidelity levels.

---

## Reviewer EZy7 (Rating 3, Confidence 2): "Nice paper! Recommend training with forces in the future."

**Summary.** Paper tries to answer how different uncertainty decompositions behave under
multi-fidelity transfer learning in atomistic models. Decides epistemic uncertainty is more
reliable than aleatoric under distribution shift.

**Review.**
- Good figures and good design methodology.
- **RE1 (forces).** Should try training with forces as well.
- **RE2 (datasets).** Try more datasets.

---

## Reviewer LPEk (Rating 4, Confidence 2): "Good work, but needs discussion of limitations"

**Summary.** The work studies how uncertainty estimates and its decomposition change during
transfer across fidelity levels. High-fidelity fine-tuning produces more reliable uncertainty
estimates than only training on limited high-fidelity data. The epistemic uncertainty estimates
improve and become a useful indicator of generalization. Aleatoric uncertainty on the other hand
loses calibration and is useful only in-domain.

**Review.** The work addresses a practically important problem and is clearly written and
explained, even for non-experts. It is suitable for presentation as such, but I would encourage the
authors to also discuss some limitations. **RL1 (other estimators).** Would this result hold for
other uncertainty prediction methods? **RL2 (is it expected?).** Isn't the non-transfer of
aleatoric component expected because of the inherent difference in target qualities?

---

## Reviewer ecuE (Rating 4, Confidence 2): "Interesting, impactful and well written paper with a few missing details"

**Summary.** This work investigates the effects of how low fidelity to high fidelity fine tuning of
atomistic models changes uncertainty quantifications. The authors break uncertain quantification
into aleatoric uncertainty and epistemic uncertainty and examine how their behaviors differ during
transfer learning compared to normal training. They show that fine tuning not only improves
uncertainty quantification but alters the tradeoff of aleatoric uncertainty and epistemic
uncertainty in out of domain vs in domain settings.

**Strengths.**
- The paper is overall well written, with clear motivation and explores the novel area of
  uncertainty quantification in low to high fidelity transfer learning for machine learned
  potentials.
- The breakdown of the types of uncertainty in different training regimes is clear and provides
  better understanding of how accurate uncertainty quantification can be.
- **RC1 (acquisition).** This work has the potential to affect recent works atomistic models +
  active learning, where more work on acquisition functions are needed.

**Weaknesses.**
- **RC2.** The last two paragraphs of section 3 are written in a confusing way. Although the
  analysis is nice, it is unclear what conclusions are meant to be drawn from this section.
- **RC3 (claim vs figure).** The conclusion states "after fine-tuning EU became the more
  informative signal on unseen data [than AU]", yet in figure 1j,l. AU has higher correlation and
  lower ENCE. Please clarify.
- **RC4.** Multiple panels 1k, 2e, 2k, 2l all have a line at the top of the border. Please ensure
  that this is because the value ends there, rather than the data going out of bounds.
- **RC5 (figure consistency).** Figures 1 and 2 disagree with each other on certain datapoints. For
  instance, figure 1 shows that in the finetuned setting AUSE for aleatoric uncertainty jumps to
  ~0.5 in the OOD setting. However, in figure 2 for transfer learning on OOD the AUSE for AU is
  only ~0.2. Please clarify how the two settings are different.
- **RC6 (statistics).** Some of the differences between methods are small in magnitude. Since the
  differences are small, multiple seeds and statistical error bars would strengthen the work.

Overall, this work is impactful, and aims to address a relevant question. However, there are a few
easily correctable inconsistencies that are undercutting the value of the work.
