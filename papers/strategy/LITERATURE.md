# Literature: verified references, novelty map, theory notes

The maintained literature file of the project (all venues). Sections 1-3 were written during the
ICLR 2027 rebuttal (2026-11-06 to 11-10) and moved here unchanged on 2026-11-12 from
`papers/iclr27/rebuttal/draft_rebuttal.md` (its Sections 7, 1 and 6). Section 4 is the landscape
scan of 2026-11-12 behind [`ASSESSMENT.md`](ASSESSMENT.md).

Rules: a statement about prior work in a paper, rebuttal or reply cites a key of Section 1 whose
"Check" column says how it was verified. Section 4 entries are **search-level only** (title,
abstract or a search summary): read the full text and move the entry to Section 1 before citing it.
Numbers in Section 2 are the project's own and carry their ledger entry.

## 1. Verified references (key, citation, how it was checked)

`bib` = already in `ICLR27-UQ-MF/iclr2027_conference.bib` (used in the submitted paper).
`web 2026-11-06` = checked against the publisher/arXiv page on that date (title, authors, venue;
quoted phrases copied from the abstract). `memory` = not yet checked: verify before use.

| Key | Reference | Check |
|---|---|---|
| smith2019approaching | Smith et al., Approaching coupled cluster accuracy with a general-purpose neural network potential through transfer learning, Nat. Commun. 10, 2903 (2019) | bib; web 2026-11-06 (DFT training then transfer learning to CCSD(T)/CBS; ~5M DFT, ~500k CC); PMC6602931 2026-11-09: two of four hidden layers frozen in TL |
| smith2020ani | Smith et al., The ANI-1ccx and ANI-1x data sets, Sci. Data (2020) | bib |
| smith2018less | Smith, Nebgen, Lubbers, Isayev, Roitberg, Less is more: sampling chemical space with active learning, J. Chem. Phys. 148, 241733 (2018) | web 2026-11-06 (QBC with ensemble disagreement built ANI-1x) |
| kaser2021transfer | Käser, Boittier, Upadhyay, Meuwly, Transfer learning to CCSD(T): accurate anharmonic frequencies from machine learning models, J. Chem. Theory Comput. 17(6) (2021) | web 2026-11-06 |
| kaser2023transfer | Käser, Meuwly, Transfer-learned potential energy surfaces: towards microsecond-scale molecular dynamics simulations in the gas phase at CCSD(T) quality, J. Chem. Phys. (2023), arXiv:2303.11685 | web 2026-11-06; full text (arXiv v1) read 2026-11-09: PhysNet, all parameters updated in TL, see 2.1; DOI 10.1063/5.0151266 (volume/pages to add); **not in the paper bib** |
| zaverkin2023transfer | Zaverkin et al., Transfer learning for chemically accurate interatomic neural network potentials, PCCP (2023) | bib |
| chen2023data | Chen et al., Data-efficient machine learning potentials from transfer learning of periodic correlated electronic structure methods: liquid water at AFQMC, CCSD, and CCSD(T) accuracy, JCTC (2023) | bib; web 2026-11-06 (<=200 CC energies); full text + SI (arXiv 2211.16619) read 2026-11-09: n2p2 committee of 8, retrained from LL weights on CC energies only, see 2.1 |
| kim2024data | Kim et al., Data-efficient multifidelity training for high-fidelity machine learning interatomic potentials, JACS 147, 1042-1054 (2024/2025) | bib; web 2026-11-06 (SevenNet-MF, one-hot fidelity encoding, simultaneous training) |
| ko2025data | Ko et al., Data-efficient construction of high-fidelity graph deep learning interatomic potentials, npj Comput. Mater. (2025) | bib |
| cui2025multi | Cui, Reuter, Margraf, Multi-fidelity transfer learning for quantum chemical data using a robust density functional tight binding baseline, Mach. Learn.: Sci. Technol. 6, 015071 (2025) | web 2026-11-06 |
| messerly2025multi | Messerly et al., Multi-fidelity learning for interatomic potentials: low-level forces and high-level energies are all you need, Mach. Learn.: Sci. Technol. (2025), arXiv:2505.01590 | web 2026-11-06 (quote "most high-accuracy datasets ... often do not contain atomic forces") |
| gardner2025understanding | Gardner, Schulz, Helie, Sun, Simm, Understanding multi-fidelity training of machine-learned force-fields, arXiv:2506.14963 (v2 2026) | bib; web 2026-11-06 (quotes "critically, the inclusion of force labels"; "marginally reduce model performance in most cases") |
| tompa2026fine | Tompa, Varga-Umbrich, Batatia, Elena, Bernstein, Csányi, Fine-tuning MLIP foundation models: strategies for accuracy and transferability, arXiv:2606.12704 (2026) | web 2026-11-06 (quotes "increasingly important practice"; naive FT "best convergence for single-system applications"; multihead replay preserves OOD robustness) |
| radova2025fine | Radova et al., Fine-tuning foundation models of materials interatomic potentials with frozen transfer learning, npj Comput. Mater. (2025) | bib |
| bilbrey2025uncertainty | (in the paper bib since 2026-11-10, cited in Sec. 1) Bilbrey, Firoz, Lee, Choudhury, Uncertainty quantification for neural network potential foundation models, npj Comput. Mater. 11, 109 (2025), doi:10.1038/s41524-025-01572-y | full text read 2026-11-07 (mace-leo, publisher PDF): 7-member readout ensemble (interaction layers frozen, Huber loss, different 80k MPtrj subsets; fine-tuned ensembles differ by seed only) + 5th/95th-percentile quantile readouts; MPtrj, HEA25, H-ZSM-5 + water (n = 8, 16 held out); quote "the ensemble is overconfident in its predictions ... lower than the error by orders of magnitude" (Introduction); Table 1 coverage ensemble 11/21/64%, quantile 87/88/86% |
| bocus2025operando | Bocus et al., The operando nature of isobutene adsorbed in zeolite H-SSZ-13 unraveled by machine learning potentials beyond DFT accuracy, Angew. Chem. (2025), doi:10.1002/ange.202413637, PubMed 39480685 | web 2026-11-10 (abstract: transfer learning to RI-RPA quality for enhanced-sampling MD, 0.3 meV/atom RMSE, an order of magnitude fewer post-HF single points); full author list not checked; **not in the paper bib** (rebuttal only) |
| chen2021learning | Chen, Zuo, Ye, Li, Ong, Learning properties of ordered and disordered materials from multi-fidelity data, Nat. Comput. Sci. 1, 46-53 (2021) | web 2026-11-06 |
| shiota2024taming | Shiota et al., Taming multi-domain, -fidelity data: towards foundation models for atomistic scale simulations, arXiv:2412.13088 | web 2026-11-06 (total energy alignment across fidelities; optional) |
| hullermeier2021aleatoric | Hüllermeier, Waegeman, Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods, Mach. Learn. 110, 457-506 (2021) | bib; web 2026-11-06 (context-dependence: changing the representation turns aleatoric into epistemic and vice versa) |
| pozdnyakov2020incompleteness | Pozdnyakov et al., Incompleteness of atomic structure representations, Phys. Rev. Lett. 125, 166001 (2020) | web 2026-11-06 |
| lahlou2023deup | Lahlou et al., DEUP: Direct epistemic uncertainty prediction, TMLR (2023) | web 2026-11-06 |
| busk2022calibrated | Busk et al., Calibrated uncertainty for molecular property prediction using ensembles of message passing neural networks, Mach. Learn.: Sci. Technol. 3, 015012 (2022) | web 2026-11-06 (aleatoric + epistemic on QM9/PC9); web 2026-11-09: MLST 3(1), 015012 (2022), doi:10.1088/2632-2153/ac3eb3 |
| busk2023graph | Busk et al., Graph neural network interatomic potential ensembles with calibrated aleatoric and epistemic uncertainty on energy and forces, PCCP (2023) | bib |
| tan2023single | Tan et al., Single-model uncertainty quantification in neural network potentials does not consistently outperform model ensembles, npj Comput. Mater. (2023) | bib |
| valdenegro2022deeper | Valdenegro-Toro, Saromo, A deeper look into aleatoric and epistemic uncertainty disentanglement, CVPRW (2022) | bib |
| seitzer2022pitfalls | Seitzer et al., On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks, ICLR 2022 (arXiv:2203.09168) | bib; web 2026-11-09: ICLR 2022 (iclr.cc poster 6755, OpenReview aPOpXlnV1T): bib entry updated from arXiv to ICLR 2022 |
| sluijterman2024optimal | Sluijterman, Cator, Heskes, Optimal training of mean variance estimation neural networks, Neurocomputing 597, 127929 (2024) | bib; web 2026-11-06 (warm-up with fixed variance; separate regularization) |
| nix1994estimating, skafte2019reliable, stirn2023faithful | NLL variance-head literature | bib |
| lakshminarayanan2017simple, kendall2017uncertainties, depeweg2018decomposition | deep ensembles, decomposition | bib |
| krogh1994neural | ambiguity decomposition | bib |
| rasmussen2023uncertain, lind2024uncertainty, kuleshov2018accurate, levi2022evaluating | UQ metrics, calibration | bib |
| batatia2022mace, batatia2025design | MACE | bib |
| batatia2025design (content) | ACE and NequIP unified in one design space (ACE recast as one layer of a multi-layer architecture; linearised NequIP as a sparsification of a larger polynomial model), arXiv:2205.06643 | web 2026-11-09 (abstract) |
| batatia2023foundation | Batatia et al., A foundation model for atomistic materials chemistry (MACE-MP-0), arXiv:2401.00096 (2023) | web 2026-11-09 (quote "can be fine-tuned on just a handful of application-specific data points to reach ab initio accuracy"; not in bib) |
| kovacs2025maceoff | Kovács et al., MACE-OFF: short-range transferable machine learning force fields for organic molecules, JACS (2025), doi:10.1021/jacs.4c07099 | web 2026-11-09 (not in bib) |
| chmiela2018towards | Chmiela, Sauceda, Müller, Tkatchenko, Towards exact molecular dynamics simulations with machine-learned force fields, Nat. Commun. 9, 3887 (2018) | web 2026-11-06 (CCSD(T)-level sGDML; candidate 2nd dataset with CC forces, WP7) |
| kennedy2000predicting | Kennedy, O'Hagan, Predicting the output from a complex computer code when fast approximations are available, Biometrika 87(1), 1-13 (2000) | web 2026-11-09 (title, authors, Biometrika 87(1), 2000; GP model combining expensive code runs with cheap approximations; page range from memory) |
| ramakrishnan2015big | Ramakrishnan, Dral, Rupp, von Lilienfeld, Big data meets quantum chemistry approximations: the Delta-machine learning approach, J. Chem. Theory Comput. 11(5), 2087-2096 (2015), doi:10.1021/acs.jctc.5b00099 | web 2026-11-09 (arXiv:1503.04987; ML corrections added to cheap approximate methods) |
| christensen2020role | Christensen, von Lilienfeld, On the role of gradients for machine learning of molecular energies and forces (rMD17), Mach. Learn.: Sci. Technol. (2020), doi:10.1088/2632-2153/abba6f, arXiv:2007.09593 | web 2026-11-09 (title, authors, journal; rMD17; article number 045018 from memory) |
| kovacs2021linear | Kovács et al., Linear atomic cluster expansion force fields for organic molecules: beyond RMSE (3BPA), J. Chem. Theory Comput. 17(12), 7696-7711 (2021), doi:10.1021/acs.jctc.1c00647 | web 2026-11-09 (MD17, ISO17 and the 3BPA molecule) |
| smith2020ani (content) | ANI-1x: ~5M DFT calculations, C/H/N/O, GDB-11 + ChEMBL + generated amino acids and dipeptides, MD/normal-mode/dimer/torsion active-learning sampling; ANI-1ccx ~500k CCSD(T)*/CBS, mostly <20 atoms | web 2026-11-07 (PMC7195467) |

## 2. Novelty map: paper finding vs closest known result

Written for the ICLR rebuttal (Q3, G1); reusable in a related-work section. Values are from the
ledger entries named in brackets.

| Finding in the paper | Closest known general result | What is new here (MF-specific) |
|---|---|---|
| Ranking and calibration can disagree | Metrics capture different properties (rasmussen2023uncertain, lind2024uncertainty); calibration vs sharpness (kuleshov2018accurate) | They move in **opposite directions along the energy-extrapolation axis**, and pretraining *amplifies* the split (Energy-OOD TU ENCE 1.637 HF-only vs 2.969 LF->HF while TU AUSE 0.186 vs 0.143) `[ledger 2026-09-23, results/tables/iclr27_main_text_numbers.txt]`; in-domain isotonic recalibration does not repair it `[paper App. "Post-hoc isotonic recalibration"]` |
| AU unreliable under distribution shift | valdenegro2022deeper | AU also collapses **in domain**, and the collapse is located **in the HF stage** and accelerated by pretraining (OOD AU ENCE, end of pretraining -> end of FT: system 0.993 -> 81.913, energy 10.863 -> 207.071) `[ledger 2026-09-23]`; on the training set AU is never overconfident (train ENCE below 1, mean AU above the mean squared training error: conservative, not "calibrated") `[ledger 2026-09-23, interpretation checks]` |
| NLL variance heads are fragile | nix1994estimating, skafte2019reliable, seitzer2022pitfalls, stirn2023faithful, sluijterman2024optimal | These works study single-task training; we show that the failure emerges during HF optimization in both protocols, is accelerated by pretraining, and quantify it per epoch; our ablations show that held-out AU becomes overconfident while training-set AU stays calibrated or conservative in every variant, that the collapse re-emerges after a variance-head reset (later and about five times weaker), develops more slowly but still develops at a smaller or larger fine-tuning learning rate, and is not removed by beta-NLL `[ledger 2026-11-07, "Fine-tuning ablations of the AU collapse"; "Head reset: AU calibration along fine-tuning"; ledger 2026-11-08, "AU along fine-tuning for all LF->HF ablations"]` |
| Overconfidence of MACE ensemble UQ | bilbrey2025uncertainty: readout ensembles of MACE-MP-0 and fine-tuned models are overconfident by orders of magnitude; quantile readouts are not | Bilbrey et al. retrain only readouts of a foundation model (ensemble and quantile model separately) and do not track AU/EU of a deep ensemble across an LF->HF transition or through training; we do, and find that EU, not AU, carries the improvement after pretraining |
| Pretraining changes the decomposition | - (not found in prior work) | HF-only: EU is the weakest ranking signal on every test set; LF->HF: EU ranks as well as or better than AU on 3 of 4 (AUSE) `[paper Sec. 4.1; ledger 2026-09-23, results/tables/iclr27_main_text_numbers.txt]`; the gain is tied to the mean-variance members: pretraining improves the EU ranking of an MSE-trained ensemble 2-6x less (AUSE) `[ledger 2026-11-08, system and energy splits]` |
| Pretrained UQ is a better acquisition function | QBC with ensemble disagreement (smith2018less), committee AL (schran2020committee) | Despite starting more accurate, the pretrained ensemble benefits more from uncertainty-guided CC acquisition: gain over random 2.316 / 2.497 / 2.347 vs 0.807 / 1.603 / 1.140 meV/atom (AU/EU/TU) `[ledger 2026-09-23]` |

## 3. Theory notes (background for writing; not for posting verbatim)

- **Law of total variance and the ensemble.** TU = AU + EU is the variance of the mixture of the
  members' Gaussians (lakshminarayanan2017simple, depeweg2018decomposition); TU is the only signal
  the observed error can calibrate (paper App. "Technical background").
- **Ambiguity decomposition** (krogh1994neural): e^2 = mean_m e_m^2 - EU per configuration, so
  TU - e^2 = (AU - mean_m e_m^2) + 2 EU; TU can only be overconfident where the heads underestimate
  their own members' errors by more than 2 EU (paper App. "Technical background"). Use this against "AU is meaningless":
  the heads have an observable target, each member's own squared error.
- **Why NLL heads collapse on held-out data.** For fixed mean, the NLL optimum is the conditional
  mean squared residual (nix1994estimating); fitted on training data it inherits training residual
  scale (skafte2019reliable, seitzer2022pitfalls, stirn2023faithful); sluijterman2024optimal recommend
  a warm-up of the mean with fixed variance (an extra ablation option: MSE warm-up then NLL).
- **Multi-fidelity as structured discrepancy.** The classic autoregressive model writes
  f_HF(x) = rho f_LF(x) + delta(x) (kennedy2000predicting); pretraining learns f_LF, fine-tuning
  learns rho and delta from scarce HF data, so the HF-stage uncertainty is about delta, which is
  smaller and smoother than f_HF. This motivates why EU becomes informative after pretraining
  (hypothesis; do not claim as shown).
- **Excess risk.** lahlou2023deup define epistemic uncertainty as excess risk, which includes model
  misspecification; with deterministic labels all error is excess risk. Be careful: this reference
  supports the reviewer's semantic point; we use it to *agree* and to frame our AU finding.

## 4. Landscape scan of 2026-11-12 (search-level; verify before citing)

Found with eleven web searches on 2026-11-12 (UQ of fine-tuned/foundation MLIPs, multi-fidelity
training, force uncertainty, paired-fidelity datasets). Not exhaustive: absence from this list is
not evidence that a paper does not exist. "Relevance" is this project's reading of the abstract.

| Work (as found) | What it does (abstract level) | Relevance to this project |
|---|---|---|
| Schäfer, Kellner, Kästner, Ceriotti, *How to train a shallow ensemble*, arXiv:2602.15747 (Feb 2026; a listing gives J. Chem. Theory Comput. 2026, not confirmed on the journal page) | Shallow (last-layer) ensembles: energy-only objectives give miscalibrated **force** uncertainties; an NLL term on forces is needed; fine-tuning a shallow ensemble instead of training from scratch costs little calibration and saves up to 96% of training time | Closest "UQ x fine-tuning" result; supports reviewer ge6X's point that energy-only training is the weak spot; a cheap-ensemble baseline a reviewer may ask for |
| Liu et al., *Heterogeneous ensemble enables a universal uncertainty metric for atomistic foundation models*, npj Comput. Mater. 12, 34 (2026), arXiv:2507.21297 | Uncertainty from disagreement between existing, different pretrained MLIPs; ranks configuration-level risk; used for distillation and fine-tuning data selection | Foundation-model UQ by *cross-architecture* disagreement; competitor to same-architecture ensembles when all members share pretrained weights |
| *Adaptive fine-tuning of foundation models for crystal structure prediction* (DPA-3), arXiv:2606.30870 (2026) | Reports that members fine-tuned from one pretrained checkpoint are nearly identical (ensemble variance below 1 meV/atom even with a re-initialized fitting network), so variance-based active learning fails; uses a second architecture's disagreement instead | **Direct threat to the generality of our finding that EU becomes a good ranking signal after pretraining**: our members have independent pretrained parents; practitioners have one. Motivates work package P1 of `PLAN.md` |
| *Multi-head committees enable direct uncertainty prediction for atomistic foundation models*, arXiv:2508.09907 (2025) | Committee of readout heads on MACE foundation models, energies and forces | Cheap-ensemble alternative on the same architecture family |
| Rensmeyer et al., *On-the-fly fine-tuning of foundational neural network potentials: a Bayesian neural network approach*, Digital Discovery (2026) | Bayesian fine-tuning of foundation NNPs with uncertainty-triggered on-the-fly learning | UQ during fine-tuning as a workflow; Digital Discovery publishes this topic |
| *Flexible uncertainty calibration for machine-learned interatomic potentials*, arXiv:2510.00721 (2025) | Conformal-prediction-style recalibration of force uncertainties, tested under distribution shift on OMol25 | Post-hoc calibration under shift; relevant to our negative recalibration result under energy extrapolation |
| *Equivariant evidential deep learning for interatomic potentials* (e2IP), arXiv:2602.10419 (2026); *Bayesian last layer for neural force fields*, ICML 2026 | Single-model force UQ (evidential 3x3 covariance; Bayesian last layer) claimed competitive with deep ensembles | Trend toward single-model, force-level UQ; ensembles remain the reference baseline |
| Tompa et al., arXiv:2606.12704 (already in Section 1 as `tompa2026fine`) | Seven fine-tuning strategies for MACE foundation models; naive fine-tuning best for single systems, multihead replay for broad scope; E0 initialization and hyperparameters matter more than the strategy | Defines the fine-tuning strategies a reviewer now expects (naive, frozen, LoRA, multihead replay); no uncertainty analysis in the abstract |
| Messerly et al., MLST 2025, arXiv:2505.01590 (Section 1, `messerly2025multi`) | Low-level forces + high-level energies in one multi-fidelity model | The protocol to adopt for forces: DFT forces in the low-fidelity stage, CC energies only in the high-fidelity stage |
| Oerder et al., *Multi-fidelity learning for atomistic models via trainable data embeddings*, ChemRxiv (2025); UMA task embeddings; *Exascale multi-task graph foundation models for imbalanced, multi-fidelity atomistic data* (HydraGNN), arXiv:2604.15380; *Cross learning between electronic structure theories*, arXiv:2510.25380 | Joint training with fidelity embeddings or fidelity-specific heads at scale | The joint / fidelity-conditioned alternative that reviewers j1p7 and Q2LT asked about; none reports calibration per fidelity (search level) |
| *Reactive chemistry at unrestricted coupled cluster level*, arXiv:2509.10872 (2025) | Energies **and forces** at UCCSD(T) for ~3k organic reaction configurations | A dataset with CC forces (small); candidate for a force-level check at high fidelity |
| MatPES (Kaplan et al.), arXiv:2503.04070 | PBE and r2SCAN energies, forces and stresses for ~400k materials structures (the two sets are similar, not identical, per the listings) | Materials-side fidelity pair with forces at both levels; a different paper's worth of work |
| Own record: `neurips.cc/virtual/2026/174030` | Listing of the RPS workshop paper | The workshop version is publicly indexed under the NeurIPS 2026 site |

Reading of the scan (2026-11-12): the intersection "uncertainty decomposition x change of fidelity x
training dynamics" is still unoccupied at search level; the neighbouring work has moved to forces,
to foundation models with a single pretrained checkpoint, and to cheap single-model estimators.
