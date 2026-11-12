# NeurIPS 2026 reviews: Submission 35263 (verbatim)

Source: OpenReview forum page printed 2026-11-12 (PDF provided by the user). Text copied from the
PDF (ligatures and line breaks normalized, obvious extraction typos fixed). Author names and the
forum URL are left out on purpose: this tree is mirrored anonymously.

Submission: *Multi-Fidelity Training Reshapes Uncertainty Decomposition in Atomistic Models*
(submitted 2026-05-05; primary area "Generalization and multi-task learning", secondary "AI/ML for
physical sciences"). This is the "NeurIPS draft": one split seed, total-variance trim, no
acquisition experiment (`../iclr27/README.md`). **Withdrawn by the authors on 2026-07-28**, after
the reviews and the meta-review, with a comment that the concerns needed experiments and revisions
that did not fit the rebuttal period.

**Point IDs** (`NM*` reviewer mM8Y, `NY*` YZpQ, `NA*` 3Aye, `NMR*` meta-review) are used by
`REVIEWS.md` and `PLAN.md`. Never renumber them.

| Reviewer | Rating | Confidence | Quality | Clarity | Significance | Originality |
|---|---|---|---|---|---|---|
| mM8Y | 2: Reject | 4 | 2: not good | 2: not good | 2: not good | 2: not good |
| YZpQ | 3: Borderline reject | 3 | 3: good | 3: good | 3: good | 2: not good |
| 3Aye | 2: Reject | 4 | 1: poor | 2: not good | 2: not good | 3: good |

---

## Meta-review (Area Chair vpHw, 2026-07-24)

The paper is an empirical study of how aleatoric vs. epistemic uncertainty behaves under
multi-fidelity training of atomistic models. Reviewers find the topic important and under-studied
and the paper well organized, but raise validity-level problems:
- **NMR1** component-wise calibration is not identifiable — the observed residual targets total
  variance, so per-component ENCE against it is only a proxy, undercutting the absolute calibration
  claims;
- **NMR2** numerical inconsistencies: LF-to-HF Energy-OOD RMSE is 0.0539 in Fig 3 but 1.7002 in
  Fig 15 after mean-preserving calibration, and AUSE shifts under a monotonic map;
- **NMR3** conclusions conflict with the numbers — in Energy-OOD the aleatoric proxy actually
  outranks epistemic and LF-to-HF ENCE is worse than HF-only;
- **NMR4** the LF-to-HF comparison is confounded by extra DFT data and reused geometries;
- **NMR5** single seed/ensemble, missing reproducibility details;
- **NMR6** limited novelty (classic Kendall & Gal machinery; a known OOD result) on a single
  dataset/architecture.

Authors should rebut any misreading, though several of these may not be repairable within a rebuttal.

---

## Reviewer mM8Y (Rating 2, Confidence 4)

**Summary.** The paper analyzes uncertainty decomposition in multi-fidelity training settings as it
relates to atomistic machine learning. The work builds on the MACE architecture to produce a mean
and heteroscedastic variance, train deep ensembles, and decompose total predictive variance into
average predicted variance. Experiments use ANI-1ccx with DFT as low fidelity and CCSD(T)/CBS as
high fidelity, comparing different settings for pretraining and fine-tuning. Follow-up analysis
evaluates improved model performance under LF-to-HF training and investigates the roles of
aleatoric vs epistemic components by training phase and shift type.

**Strengths.**
- The paper's stated motivations are clear and attempt to provide new insight in an under-studied
  and somewhat niche area. The in-depth focus on a specific atomistic case study enables highly
  detailed observations to be made, in particular for the MACE architecture and on the selected low
  and high-fidelity datasets.
- The positioning of the work is well-motivated: the core gaps identified in the introduction are
  key limitations of many existing methods, and addressing them and/or providing insightful,
  domain-specific analysis serves as a foundation for novelty.
- The paper is well-organized, includes a comprehensive literature review, provides high quality
  figures/diagrams, and I appreciated the extensive discussion-style analysis across the entirety
  of Section 4.

**Weaknesses.**
- **NM1 (positioning, scope).** I find the core positioning of the paper to be somewhat difficult
  to pin down. There is a clear focus on atomistic ML as the primary domain of interest, but the
  proposed formulation and broader methodological language is very general. The scope and impact of
  the work is therefore hard to evaluate, provided all analysis and conclusions are drawn from a
  single dataset, architecture, and domain. To fully demonstrate the generality of the scheme, more
  (scientific) domains would need to be evaluated to verify the approach transfers well to other
  settings and the stated conclusions signify broad trends across architectures and datasets.
  Note that this isn't to take away from the relevant conclusions drawn from this particular case
  study, but the scope of the paper's core claims is ultimately bound by the extent of its singular
  task focus.
- **NM2 (aleatoric / epistemic interpretation).** The interpretation of epistemic and aleatoric
  uncertainty is not particularly well supported. The targets in this case are deterministic
  quantum-chemistry energies, and the learned MVE variance is trained to explain residuals under
  many factors, e.g., model misspecification, optimization, data imbalance, fidelity mismatch,
  label error, etc. The terminology used in the paper sometimes suggests a stronger interpretation
  of aleatoric and epistemic uncertainty than the experiments can reliably support. That is, the
  learned variance head is not clearly identifiable with irreducible aleatoric noise, and ensemble
  disagreement is only one operational proxy for epistemic uncertainty. Without more empirical
  evidence or strong, isolated ablations that suggest a particularly clear relationship here, I
  generally find the claims are misleading and should be more cautiously framed, or at the very
  least do more to highlight the challenges of using empirical scores as proxies for true
  aleatoric/epistemic uncertainties.
- **NM3 (comparison not controlled for information).** The comparison between HF-only and LF -> HF
  appears meaningful as a practical multi-fidelity data-access comparison, but it doesn't serve as
  a controlled comparison of training dynamics or uncertainty decomposition under fixed information
  content. Given the LF->HF scheme is exposed to a much larger LF dataset before fine-tuning,
  improvements in RMSE, ranking, etc cannot be attributed uniquely to the fidelity transition or to
  multi-fidelity uncertainty effects. Additional ablations that attempt to limit the information
  exposure or some other meaningful factor (e.g., dataset size or optimization budget) seem
  necessary to allow more direct metric comparisons across these two settings.

**Questions.**
- **NM4 (seeds).** How stable are results across random seeds for split construction and ensemble
  initialization? The paper mentions seed 0 is used for sampling; the empirical results would
  benefit from reporting uncertainty over splits/seeds and could extend extra credence to claims
  under just the single dataset.
- **NM5 (estimator ablations).** In line with the ablation remark above: what happens with an
  ensemble-only baseline without the variance head, and an MVE single-model baseline without
  ensemble disagreement? This would help clarify whether total uncertainty is directly benefiting
  from decomposition, or if primarily from ensemble disagreement.
- **NM6 (matched RMSE).** How much of the LF -> HF benefit comes from multi-fidelity transfer rather
  than simply better RMSE? For example, if HF-only and LF -> HF checkpoints were matched by RMSE, do
  other differences in other metrics (e.g., AUSE/ENCE) still remain, or are they "explained away"
  by the headline RMSE?

---

## Reviewer YZpQ (Rating 3, Confidence 3)

**Summary.** The paper provides an empirical evaluation of the behavior of aleatoric and epistemic
uncertainty quantification within multi-fidelity atomistic machine learning models. The authors
extend the existing MACE graph neural network architecture using a standard mean-variance
predictions and construct deep ensembles to estimate how uncertainty metrics change across
low-fidelity pretraining and high-fidelity fine-tuning phases. Evaluated on system-level and
energy-based distribution shifts using the ANI-1x dataset, their work demonstrates that
transitioning from low-fidelity pretraining to high-fidelity fine-tuning shifts the informativeness
of uncertainty estimations, with epistemic uncertainty being a more robust sign for prediction
errors on out-of-distribution samples.

**Strengths.**
- The paper is well written and easy to follow, with a clear structure and intuitive explanations.
- The method is conceptually simple and easy to apply across a wide range of architectures and models.
- The authors provide an anonymized repository link containing their standardized dataset splits,
  training protocols, and custom evaluation metrics.

**Weaknesses.**
- **NY1 (novelty of the estimator).** The core uncertainty estimation approaches introduces limited
  novelty. The authors rely significantly on the classic heteroscedastic mean-variance estimator
  and deep ensemble decomposition described in Kendall & Gal (2017).
- **NY2 (actionability of the dynamics).** The practical utility of the approach is poorly
  justified. Analyzing the change of uncertainty estimations across intermediate training epochs
  lacks actionable value. Because uncertainty metrics are utilized in post-convergence settings
  (e.g., active learning data selection or risk-aware deployment), tracking how these components
  shift mid-training offers limited benefit.
- **NY3 (known result).** One of the paper's empirical conclusions -- that aleatoric uncertainty
  degrades and loses its reliability under out-of-distribution (OOD) shifts while epistemic
  uncertainty proves to be a more informative marker for generalization -- is a well known
  observation.

**Questions.**
- **NY4.** Given that your uncertainty framework relies heavily on the classic heteroscedastic
  mean-variance estimator and deep ensemble decomposition from Kendall & Gal (2017), what do you
  consider to be the primary methodological novelty?
- **NY5.** Can you clarify the explicit practical utility of tracking the trajectory of uncertainty
  components across intermediate training epochs?
- **NY6.** Since the degradation of aleatoric uncertainty and the corresponding robustness of
  epistemic uncertainty under out-of-distribution (OOD) shifts is a well-known observation in
  machine learning, what specific novel insights does your multi-fidelity framework uncover?

---

## Reviewer 3Aye (Rating 2, Confidence 4)

**Summary.** This paper studies how predictive uncertainty evolves in atomistic neural networks
across low-fidelity pretraining, high-fidelity training, and low-to-high-fidelity fine-tuning. The
authors extend MACE with heteroscedastic Gaussian outputs and use deep ensembles to interpret the
average predicted variance as an aleatoric component and ensemble disagreement as an epistemic
component. Using ANI-1x/ANI-1ccx, the paper evaluates LF-only, HF-only, and LF-to-HF protocols
under system-level and energy-based distribution shifts.

The paper's main empirical message is that the relative usefulness of aleatoric and epistemic
uncertainty changes across fidelity regimes, training stages, and distribution shifts, while total
uncertainty is often presented as the most robust signal.

**Strengths and weaknesses.** The topic is important and relevant. Reliable uncertainty estimates
are crucial for atomistic models because these models are often deployed outside their training
distribution and may guide expensive quantum-chemical calculations or active-learning pipelines.
The paper is also useful in trying to separate uncertainty ranking from numerical calibration, and
in examining training dynamics rather than only final checkpoints.

However, I do not think the current evidence supports the central claims.

- **NA1 (component calibration not identifiable).** The paper evaluates the calibration of
  aleatoric uncertainty and epistemic uncertainty separately using the same observed squared
  residual. The observed residual is a target for the total predictive variance, not for either
  unobserved component individually. A decomposition can be correct in total while each component
  is not individually equal to the full residual. Therefore, component-wise ENCE should be treated
  as a proxy and is not identifiable without component-level targets, repeated observations,
  controlled synthetic noise, or a suitable hierarchical model. Component-wise AU and EU can be
  evaluated as ranking signals, but their absolute calibration against the full residual is not
  established.
- **NA2 (terminology).** The interpretation of the MVE variance as aleatoric uncertainty is also
  delicate in this setting. For a fixed molecular geometry and fixed electronic-structure protocol,
  the target energy is essentially deterministic. The paper does acknowledge a generalized
  operational interpretation, but the terminology can still be misleading. The predicted variance
  may capture input-dependent residual difficulty, model bias, limited data, optimization
  artifacts, fidelity mismatch, or model misspecification. More cautious terminology, such as
  heteroscedastic residual-variance proxy plus ensemble disagreement, would better match what is
  actually observable.
- **NA3 (numerical inconsistencies).** There are serious numerical inconsistencies in the
  calibration appendix. In Figure 3, LF-to-HF Energy-OOD RMSE is reported as 0.0539, while Figure
  15 reports 1.7002 after post-hoc isotonic calibration. Calibration should change uncertainty
  estimates, not the mean predictions, so RMSE should remain unchanged. AUSE also changes
  substantially despite isotonic calibration being monotonic and therefore expected to preserve
  ranking except for possible ties. This suggests that different predictions, checkpoints, splits,
  or result files may have been mixed. The calibration conclusions are not reliable until these
  figures are audited and regenerated.
- **NA4 (claims vs numbers).** Several conclusions conflict with the reported final metrics. In the
  Energy-OOD results, the aleatoric proxy has higher Spearman correlation and lower AUSE than the
  epistemic component for all three training protocols. This conflicts with the broad claim that
  epistemic uncertainty becomes the more informative signal for high-energy configurations. Also,
  LF-to-HF does not consistently improve calibration: in Energy-OOD, its total ENCE is worse than
  HF-only.
- **NA5 (confounding: extra data, reused geometries).** The LF-to-HF comparison is confounded.
  LF-to-HF uses 50,000 DFT labels in addition to 5,000 CC labels, while HF-only uses only 5,000 CC
  labels. Thus, improved performance may come from extra data, extra optimization, or
  representation pretraining rather than the uncertainty method being isolated by the current
  design. Moreover, the paper states that CC configurations are selected from the DFT subset; for
  HF training configurations, LF-to-HF has already seen the same geometries during low-fidelity
  pretraining. A stronger experiment would exclude all high-fidelity training geometries from the
  low-fidelity pretraining set.
- **NA6 (shifts not in one design).** The comparison between system shift and energy shift is not
  fully controlled. These shifts use differently constructed training distributions, so the
  observed differences may reflect different training sets rather than the type of distribution
  shift. A cleaner design would evaluate seen/unseen systems and low/high energy configurations in
  a common factorial setup.
- **NA7 (statistics).** The statistical analysis is weak. The paper reports one split seed and no
  repeated complete ensemble experiments or system-level bootstrap confidence intervals. The
  variation among members of one ensemble is the object being studied; it is not a substitute for
  uncertainty over method-level metrics across data splits, training repeats, or molecular-system
  samples.
- **NA8 (reproducibility details).** Many reproducibility details are missing or ambiguous,
  including optimizer, learning-rate schedule, variance positivity parameterization, variance
  floor, exact per-atom uncertainty normalization, fine-tuning initialization, optimizer-state
  handling, energy-split thresholds, and the exact definition of ensemble GNLL.

**Questions.**
1. **NA9 (= NA1).** What observable target justifies separately calibrating aleatoric uncertainty
   and epistemic uncertainty against the full prediction residual? Please either provide a setting
   where the two components are identifiable, or restrict absolute calibration claims to total
   predictive uncertainty.
2. **NA10 (= NA3).** Can the authors audit and regenerate Figures 14 and 15? In particular, why
   does LF-to-HF Energy-OOD RMSE change from 0.0539 to 1.7002 after uncertainty calibration, and
   why do AUSE values change substantially under a monotonic mapping?
3. **NA11 (same configurations across the transition).** Are all checkpoints in Figures 4 and 5
   evaluated on identical configurations and identical reference labels throughout both training
   stages? If the target changes from DFT to CC at the transition, the jump cannot be interpreted
   solely as a change in uncertainty decomposition.
4. **NA12 (ablations).** Can the authors add ablations to isolate the multi-fidelity effect?
   Important comparisons include an MSE ensemble without an MVE head, a single MVE model, resetting
   the variance head at the fidelity transition, transferring only the representation backbone, and
   LF pretraining that excludes all geometries later used for HF training.
5. **NA13 (= NA7).** Can the authors report uncertainty estimates over the metrics? At minimum,
   please provide repeated complete ensembles or molecular-system-level bootstrap intervals, rather
   than relying on one split and one ensemble.

**Limitations.** The paper contains a limitations section, but it omits several consequential
limitations of the analysis: the lack of component-level calibration ground truth, the proxy nature
of component-wise ENCE, confounding from additional LF data and repeated HF-training geometries,
and the absence of repeated complete ensembles or split-level uncertainty estimates.

**Paper formatting concerns.** Figure 15 has serious presentation issues: the axes clip much of the
reported Energy-OOD behavior, including a panel whose stated RMSE is far outside the displayed
range. Several appendix figures are also too small to inspect reliably.
