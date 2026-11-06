# ICLR 2027 reviews: Submission 18667 (verbatim)

Source: OpenReview author console, printed 2026-11-06 (PDF provided by the user). Text
copied verbatim from the PDF (ligatures and line breaks normalized). Author names and the
forum URL are left out on purpose: this tree is mirrored anonymously.

**Point IDs** (`G*`, `J*`, `Q*`) are used by `PLAN.md` and `draft_rebuttal.md`. Never renumber
them; add new IDs at the end if a reviewer posts follow-ups.

| Reviewer | Rating | Confidence | Presentation | Informativeness | Soundness |
|---|---|---|---|---|---|
| ge6X | 1: Clear rejection | 3: familiar with topic and literature | 3: Clear | 1: learned little or nothing new | 2: minor or fixable concerns |
| j1p7 | 2: Weak rejection | 4: substantial direct research experience | 3: Clear | 2: learned something interesting | 3: no substantive concerns |
| Q2LT | 2: Weak rejection | 2: broad familiarity, little direct experience | 3: Clear | 2: learned something interesting | 2: minor or fixable concerns |

All three rate presentation 3 (clear); no ethics flags.

---

## Reviewer ge6X (Rating 1, Confidence 3)

**Summary.** The paper examines how pretraining atomistic energy models on low-fidelity data
affects uncertainty estimates through ensembling. In particular, they train an ensemble of MACE
models with variance heads on a dataset containing DFT energy estimates and, for a smaller subset,
more accurate coupled-cluster (CC) estimates. Then they compare using CC data alone vs. pretraining
using DFT and fine-tuning using CC on OOD systems and OOD configurations of ID systems. They report
that pretraining generally improves uncertainty ranking but not calibration. They also monitor these
metrics during training and compare them for uncertainty-based active acquisition.

**Strengths.**
- The write-up is readable and avoids the clarity issues typical of heavily LLM-assisted writing.
- The results are clearly presented, and it makes good use of figures.
- The choice of dataset for the multi-fidelity study seems reasonable, and the data splits make sense.
- The experiments using uncertainty in active learning are a nice addition.

**Weaknesses.**
- **G1 (motivation, novelty).** The study's motivation is weak, and its scope is quite narrow. I may
  have missed a trend in this area, but I don't think using high-fidelity data for fine-tuning is
  common right now, so the study of ensemble uncertainty in this setting is not well motivated, and
  on top of this, the reported findings are not unexpected.
- **G2 (one model, one dataset).** The scope of experiments is limited to a specific model (MACE) on
  a specific dataset.
- **G3 (energies only).** All experiments use only energies, and they ignore forces, which are a
  stronger training signal and are widely used to train MLIPs. So the conclusions are further limited
  to this less common training setting.
- **G4 (protocol, compute matching).** The models being compared (DFT+CC vs CC) are contrasted under
  a specific rather arbitrary training protocol: a model is pretrained for a certain number of epochs,
  with optional fine-tuning at a lower learning rate on a smaller set of higher-fidelity data. A proper
  study could match these settings in total compute budget (training steps), consider various model
  sizes given data availability, use optimal learning rates, and also report pretrain-only results.
- **G5 (aleatoric with deterministic labels).** Given that this is a deterministic task, reporting
  aleatoric uncertainty metrics doesn't make much sense. The paper correctly notes that this variance
  component could model approximation or other errors, yet it still reports it as aleatoric
  uncertainty throughout.
- **G6 (minor, figure).** The champion figure has a white band in two of my PDF viewers but I'm not
  sure if this is a problem on my side.

**Questions.**
- **G7 (LF-only).** LF-only results are currently not reported. Is LF-only significantly worse than
  LF+HF or HF-only?

---

## Reviewer j1p7 (Rating 2, Confidence 4)

**Summary.** This paper investigates uncertain assessment for models trained on low and high fidelity
data in the surrogate molecular energy modeling. The paper is not proposing any new methodology or
datasets. Instead it is addressing an important question about the combination of different levels of
theory data and how that affects uncertainty assessment and calibration. The validation is thorough
although as always more and different could have been done.

**Strengths.** A timely investigation and thorough of the combination of different quality datasets.

**Weaknesses.**
- **J1 (one model, two datasets).** Only one model (MACE) and only two datasets used.
- **J2 (training strategy, fidelity as input).** Not clear that the training setup: low fidelity
  pretraining and high fidelity finetuning is the right approach. The two datasets come from different
  distributions with systematic biases and if I understand correctly, the fidelity label is not given
  as input to the model. This seems to be a missed opportunity because it would allow the model to
  learn more and also potentially be better calibrated overall.

**Questions.**
- **J3 (justify choices).** Better arguments for the limitations in the model, data and training
  strategy choices.

---

## Reviewer Q2LT (Rating 2, Confidence 2)

**Summary.** This paper investigates how multi-fidelity training affects predictive uncertainty in
atomistic machine learning models. Using mean-variance MACE deep ensembles, the authors compare
high-fidelity training from scratch with low-fidelity pretraining followed by high-fidelity
fine-tuning. They evaluate prediction accuracy, uncertainty ranking, and calibration under both
in-distribution settings and two types of distribution shift: unseen molecular systems and
higher-energy configurations.

The main findings suggest that low-fidelity pretraining improves predictive accuracy and
uncertainty-based error ranking, but does not consistently improve uncertainty calibration,
particularly under energy extrapolation. By analyzing training dynamics, the authors show that
calibration degradation occurs primarily during high-fidelity optimization and is associated with
increasingly overconfident aleatoric uncertainty estimates. The paper further demonstrates that
uncertainty estimates from pretrained models can provide more effective guidance for
uncertainty-based data acquisition.

**Strengths.**
1. Systematic empirical investigation of uncertainty in multi-fidelity learning. The paper addresses
   an interesting and practically relevant question that has received relatively limited attention.
   Rather than evaluating uncertainty only at convergence, the authors examine its evolution throughout
   pretraining and fine-tuning. The separate analysis of uncertainty ranking, calibration, and
   aleatoric and epistemic components provides a more detailed understanding of model behavior. The
   evaluation across two types of distribution shift and five independent data splits further
   strengthens the empirical analysis.
2. Interesting findings with practical implications. The observation that improved prediction accuracy
   and uncertainty ranking do not necessarily lead to better calibration is important for reliable
   scientific machine learning. The paper also identifies distinct behaviors of aleatoric and epistemic
   uncertainty during high-fidelity optimization. Furthermore, the uncertainty-guided acquisition
   experiment demonstrates the practical usefulness of uncertainty ranking, even when the predicted
   uncertainty magnitudes are not well calibrated.

**Weaknesses.**
- **Q1 (mechanism not validated).** Limited experimental validation of the proposed explanation for
  uncertainty degradation. Although the paper provides a detailed empirical characterization of
  uncertainty evolution, the underlying mechanism is not sufficiently established. The authors suggest
  that the learned aleatoric variance becomes overconfident because the variance head follows
  decreasing training residuals while errors on held-out data do not decrease accordingly. While this
  explanation is plausible and supported by the observed training dynamics, the paper lacks targeted
  ablation experiments to verify it. For example, resetting the variance head during fine-tuning or
  investigating alternative training objectives could help distinguish this explanation from other
  effects related to optimization and overfitting. Consequently, the work provides valuable empirical
  observations, but the mechanistic understanding remains somewhat limited.
- **Q2 (generality: model, estimator, MF strategy).** Limited generalizability across models,
  uncertainty estimators, and multi-fidelity training strategies. The experiments are primarily
  conducted using a single model architecture (MACE), one dataset (ANI-1x), and one uncertainty
  estimation approach (mean-variance deep ensembles). In addition, the multi-fidelity experiments
  focus on a sequential pretraining-fine-tuning strategy. It remains unclear whether the observed
  uncertainty behavior is a general property of multi-fidelity learning or is specific to this
  combination of model architecture, uncertainty estimator, and training procedure. Other
  multi-fidelity strategies, such as joint training, multi-head learning, or alternating optimization
  across fidelity levels, might exhibit different uncertainty dynamics. Investigating or discussing
  these alternatives would help clarify the scope of the conclusions.
- **Q3 (novelty clarification, Bilbrey et al. 2025).** Several aspects of the reported findings are
  closely related to prior observations on the distinction between uncertainty ranking and
  calibration, the unreliability of aleatoric uncertainty under distribution shift, and the
  optimization difficulties of Gaussian NLL-based variance estimation. While the analysis of
  uncertainty evolution during low-to-high-fidelity training appears to offer additional empirical
  insights, the manuscript would benefit from a clearer explanation of which findings are specific to
  the multi-fidelity setting and how they extend existing knowledge. In particular, Bilbrey et al.
  (2025) reported related overconfidence behavior in ensemble-based uncertainty estimates for MACE
  foundation models and appears relevant to this discussion.

**Questions.**
- **Q4 (= Q1).** Regarding the mechanism of calibration degradation: Could the authors provide
  additional ablation experiments to better validate the proposed explanation for the overconfidence
  of aleatoric uncertainty? For example, would resetting the variance prediction head at the
  beginning of high-fidelity fine-tuning, or using a different training objective, change the
  observed calibration dynamics?
- **Q5 (= Q2).** Regarding the generality of the findings: To what extent do the observed uncertainty
  behaviors depend on the sequential low-fidelity pretraining and high-fidelity fine-tuning strategy?
  Would similar ranking-calibration trade-offs be expected under joint multi-fidelity training or
  multi-head approaches? Additional comparisons, or a more detailed discussion of the expected
  differences, would help establish whether these findings reflect a broader property of
  multi-fidelity learning.
