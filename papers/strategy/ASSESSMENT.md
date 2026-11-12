# Assessment of the work against the current literature

Written 2026-11-12 from: the ICLR 2027 manuscript with its revision appendix
(`ICLR27-UQ-MF/iclr2027_conference.tex`, paper commit `1c358d9`), the posted rebuttal, the ledger
`results/results.md`, [`FINDINGS.md`](FINDINGS.md), the ICLR reviews, and a literature scan of the
same day ([`LITERATURE.md`](LITERATURE.md) Section 4, search level). Results are referred to
qualitatively; every number lives in the ledger. Opinions here are the assessor's (an AI agent
working for the first author), to be challenged by the authors.

## 1. What the work is

A controlled empirical study: ensembles of ten mean-variance MACE models, trained on ANI-1x with
DFT (low fidelity, LF) and CCSD(T)/CBS (high fidelity, HF) energies, under three protocols (LF-only,
HF-only, LF pretraining then HF fine-tuning), two distribution shifts (unseen systems, higher
energies), five data splits as replicates. Uncertainty is split into the mean predicted variance
("AU") and the member disagreement ("EU"), evaluated for ranking (AUSE) and calibration (ENCE), at
the selected checkpoints and along training. A single-round acquisition experiment closes the
paper. The rebuttal added: a pretrain-only baseline, learning-rate sweeps and step counts,
mechanism ablations (variance-head reset, beta-NLL, fine-tuning learning rate), and estimator
baselines (MSE ensembles, single model, ensemble size). 27 experiments exist today (`uq list`).

## 2. Is it sound?

Yes, and this is its main asset.

- Replication and statistics are above the norm of the field: five independent splits as the unit,
  ten members per ensemble, paired contrasts, intervals over splits, a ledger with provenance.
  Much of the related literature reports one split and one ensemble.
- The data design is genuinely controlled: same geometries at both fidelities, shifts defined from
  geometry and LF energy only, a geometry-only support filter instead of trimming on the evaluated
  quantity.
- No reviewer at ICLR disputed a result. The weak points found by the project's own audits
  (`docs/known-issues.md`) were fixed or disclosed.

Points where a careful referee can still push (assessor's reading, none is an error):

1. **Selected checkpoints of LF->HF are almost not fine-tuned.** Validation NLL selects LF->HF
   members after one to three fine-tuning epochs (paper Sec. 3), because the NLL is hit by the
   variance collapse the paper describes. The "converged model" comparison therefore depends on the
   selection criterion, and the stored checkpoints (every fifth epoch plus each member's best) do
   not resolve the first epochs, where everything happens. Not yet analysed: selection by
   validation RMSE or a fixed budget, and sub-epoch dynamics.
2. **The calibration contrast between protocols is split- and learning-rate-dependent**
   (`FINDINGS.md` Section 2.2), while the abstract states it without that qualification.
3. **The speed of the AU collapse is not monotone in the fine-tuning learning rate**
   (`FINDINGS.md` Section 2.1); finding (iii) "faster after pretraining" is true at the main setting
   and explained by the inherited head, but it is a statement about one optimization setting.
4. **The acquisition gain is small in absolute terms** and comes from one round on a pool cut from
   the test set (proof of concept by the paper's own description).

## 3. Position in the literature (November 2026)

Three neighbouring lines exist; the paper sits where they meet, and that meeting point is still
empty at search level.

| Line | State of the art (keys in `LITERATURE.md`) | Where this work stands |
|---|---|---|
| UQ for interatomic potentials | Deep and committee ensembles as reference; mean-variance ensembles with recalibration (busk2023graph); single-model alternatives not consistently better (tan2023single); 2025-26: shallow ensembles with force NLL, evidential and Bayesian-last-layer **force** UQ, conformal-style recalibration under shift, heterogeneous ensembles for foundation models | Uses the reference estimator, but on **energies only**; the field's evaluation has moved to forces |
| Multi-fidelity / transfer learning | Sequential fine-tuning is established (ANI-1ccx and successors); 2024-26: fidelity-conditioned joint training (kim2024data, ko2025data, UMA-style task embeddings), multi-head (gardner2025understanding), fine-tuning strategy benchmarks for foundation models (tompa2026fine), "LF forces + HF energies" (messerly2025multi) | Studies the established protocol; does not cover the joint / multi-head alternatives nor forces in the LF stage |
| UQ of fine-tuned / foundation models | bilbrey2025uncertainty (readout ensembles overconfident by orders of magnitude); 2026: fine-tuned members from one checkpoint lose diversity (arXiv:2606.30870); fine-tuning shallow ensembles keeps calibration (arXiv:2602.15747) | **The only work found that follows the two variance terms through a change of fidelity and along training, with baselines that isolate the estimator** |

So the novelty claim ("first systematic analysis of how multi-fidelity training affects uncertainty
dynamics") survives the scan. What has changed since the first draft is the surroundings: the
motivation the rebuttal leaned on (fine-tuning foundation models) is now a crowded, fast area, and
readers from it bring two expectations the study does not meet.

## 4. Gaps in external validity (the real weaknesses)

Ranked by how much they limit the conclusions for a 2026-27 reader.

1. **Energies-only training.** Practical potentials are trained on forces; multi-fidelity studies
   single out force labels as decisive (gardner2025understanding, messerly2025multi), and
   energy-only objectives are reported to miscalibrate force uncertainty (arXiv:2602.15747, search
   level). ANI-1x has no CC forces, but it **does have DFT forces for every configuration with a TZ
   energy** (`data/raw/ani1x-release.h5`: `wb97x_tz.forces`, 4,617,229 configurations; counted
   2026-11-12), which the current splits do not export (`docs/datasets.md`). The realistic protocol
   "LF energies + forces, HF energies" is therefore testable with no new labels.
2. **Every member has its own pretrained parent.** Member (s, m) of LF->HF is fine-tuned from LF
   member (s, m): ten independent pretrainings per ensemble. A practitioner fine-tuning a
   foundation model has **one** checkpoint, and at least one 2026 report finds that ensembles built
   that way have almost no disagreement (arXiv:2606.30870, search level). The paper's finding that
   pretraining turns EU into a competitive ranking signal has not been tested in that regime. It
   costs only fine-tuning compute to test.
3. **One architecture** (raised by all three ICLR reviewers).
4. **One dataset and a small-data regime**: 50k DFT and 5k CC configurations per split, out of 4.6M
   and 490k available in the raw file; no data-scale axis. The removed water study behaved
   differently (no AU overconfidence), plausibly because its test set is close to its training set;
   that would *support* the mechanism if shown on aligned splits, and today it is simply absent.
5. **One multi-fidelity strategy** (two of three ICLR reviewers).

## 5. What is strong and should lead

The rebuttal experiments, more than the submitted main text, contain the paper's sharpest and
least expected content (ledger entries of 2026-11-07/08/10, interpretation in `FINDINGS.md`):

- The ranking gain from pretraining is **tied to the mean-variance members**: it is several times
  larger for the mean-variance ensemble than for an MSE-trained ensemble with the same parents
  protocol, and one mean-variance model ranks errors better than ten MSE members.
- A **re-initialized variance head ranks worse at every epoch** and loses calibration later and
  less: the pretrained head carries transferable information about *where* errors are large, and
  an inherited *scale* that is wrong for the new fidelity from the first epoch.
- The pretrain-only ensemble (reference energies refit) ranks in-distribution CC errors about as
  well as HF-only training but is far less calibrated.
- Training-set AU stays calibrated or conservative in every variant while held-out AU becomes
  overconfident.

Together these say one thing: **across a change of fidelity, the ordering of the uncertainty
transfers and its scale does not.** The submitted paper states four findings side by side; this
single statement explains all four (better ranking after pretraining, calibration lost in the HF
stage, the decoupling under energy extrapolation, the larger acquisition gain, since acquisition
needs only the ordering). It is also falsifiable by intervention, which is what Q2LT asked for and
what makes an observational paper acceptable at a method-oriented venue.

## 6. Verdict

- **Validity:** high. The results are reproducible, replicated and honestly qualified.
- **Novelty:** real but narrow as currently framed; the niche is unoccupied.
- **Significance as submitted:** limited by the setting (energies only, independent parents, one
  model) more than by the findings. This is what "informativeness 1-2" at ICLR measures.
- **Potential:** good. The missing pieces are identifiable, mostly cheap (fine-tuning-scale
  compute), and the most expensive one (forces in pretraining) needs no new data. The work does not
  need a new direction; it needs its setting brought to where the field is, and its best result
  moved from the appendix to the title.
- **Risk to state plainly:** the two validity tests (forces; a single shared parent) can change the
  conclusions. If EU's ranking gain disappears with a shared parent, or the AU collapse changes
  with force-trained parents, the paper's story changes. Both outcomes are publishable and more
  useful to readers than the current one, but they must be run before the story is rewritten.

The development plan that follows from this is [`PLAN.md`](PLAN.md).
