# ICLR 2027 rebuttal: plan (Submission 18667)

Entry point and standing rules: [`README.md`](README.md). This file holds *what* we do and *how we
present it*; *where* jobs run: [`RUNS.md`](RUNS.md); numbers: `results/results.md` only.
Condensed 2026-11-07 (user: stop adding ablations, evaluate, plan the presentation); the long
versions of earlier designs and logs are in Git history (`git log -p papers/iclr27/rebuttal/PLAN.md`).

Markers: `[ ]` open, `[~] (agent: <name>, since <time>)` claimed, `[x]` done (with ledger entry or
output), `[!]` blocked, `[-]` dropped/paused by the user. Keep IDs stable.

## 1. Situation and deadlines

- Ratings 1 / 2 / 2 (ge6X clear reject; j1p7, Q2LT weak reject); presentation rated clear by all.
  Attacks: motivation/novelty (G1, Q3), generality (G2, J1, Q2), missing baselines/ablations
  (G4, G7, Q1, J2), "aleatoric" framing (G5). Most winnable: **Q2LT** (concrete ablations), then j1p7.
- **Responses and revised PDF due 2026-11-18** (main text <= 10 pages, appendix unlimited, pdfdiff
  against the submission; word limit per comment, any number of comments). Post in waves, early
  (user 2026-11-08: shortened, reviewers need time to answer; nothing new after Nov 14):
  Wave A **Nov 10**, Wave B **Nov 12**, revised PDF + Wave C **Nov 14**, Nov 15-18 discussion replies only.
- Budget: Leonardo `IscrC_UQ-MF` 80,000 local h (1 GPU-h = 8 local h); used ~4,750 by 2026-11-07
  noon (sacct), ~2,500 more committed by queued jobs; DGX: 3 GPUs max.

## 2. Reviewer asks -> answer -> status

Type: W writing, A analysis of existing outputs, E new training.

| ID | Ask (short) | Answer | Type | Status |
|---|---|---|---|---|
| G1 | "HF fine-tuning is not common" | DFT->CC transfer is the standard route to CC MLIPs (ANI-1ccx and others, draft 2.1) | W | draft |
| G2 / J1 / Q2 (data, model) | only MACE, only ANI-1x | MACE design space spans descriptor and equivariant MP models; ANI-1x = thousands of systems (3114 formulas; 1246 System-OOD); single-system benchmarks are common; cost of a controlled multi-architecture study | W | draft (2026-11-07); citations to verify |
| G3 | energies only | CC labels in ANI-1x have no forces | W (+E opt., D3) | draft |
| G4 | compute, LRs, model size, pretrain-only | WP4.1 steps; WP4.2/4.3 LR sweeps; WP2 pretrain-only | A + E | **4.1, 4.2 recorded**; 4.3 energy in, system evaluating |
| G5 | "aleatoric" with deterministic labels | operational definition; rename (D5) | W | draft |
| G6 | white band Fig. 1 | Safari soft-mask bug, fixed file exists | W | to apply (9.1) |
| G7 | LF-only not reported | WP2: LF-only on CC labels after E0 refit; DFT-label results already in the paper's appendix | A | **recorded** |
| J2 / Q2 (strategies) | fidelity as input, multi-head, joint | written only: interesting alternatives (multi-head, Delta-learning, joint); we study FT, the most used | W | draft; experiments paused (user) |
| J3 | better arguments for choices | "Design choices and scope" appendix | W | open (9.2) |
| Q1 / Q4 | validate mechanism; reset head; other objectives | WP3.2 reset, 3.3 beta-NLL, 3.5 slow FT, 3.6 MSE; (d) via existing held-out recalibration (App.) | E | **recorded** (reset, beta-NLL, FT lr both splits); MSE LF->HF training |
| Q2 / Q5 (estimators) | one estimator | single MVE, ensemble size, MSE ensembles | A + E | HF-only recorded; MSE LF->HF training |
| Q3 | novelty; Bilbrey 2025 | novelty table; Bilbrey read in full | W | draft |

## 3. Presentation plan (how results go into the rebuttal and the revision)

Principles: (1) main text stays qualitative, numbers in appendix figures/tables (authors' prose kept);
(2) each reply quotes at most 2-3 numbers, from the ledger, and points to the appendix; (3) all
contrasts paired over the five splits, 95% t-intervals, TU unless the claim is about AU or EU;
(4) robustness checks are grouped into one "does not change the conclusions" appendix instead of one
section per sweep; (5) each appendix item names the reviewer question it answers.

### 3.1 New appendix "Additional baselines and ablations" (revised PDF)

| Item | Content (one figure or table each) | Source | Answers |
|---|---|---|---|
| A. Pretrain-only baseline | Table: 4 tests x RMSE and TU Spearman/AUSE/ENCE for LF-only (E0 refit), HF-only, LF->HF; one sentence that the DFT-label LF-only results are App. reliability/dynamics | `results/tables/lf_only_vs_hf.csv` (ledger WP2) | G7, G4 |
| B. Learning rate and compute | **One compact figure**: rows = split (system, energy), columns = RMSE, TU Spearman, TU ENCE; x = lr per protocol (HF-only 1e-2 / 3e-3 / 1e-3; LF->HF 1e-4 / 1e-3 / 1e-2), ID and OOD markers, five-split intervals; caption sentence on gradient steps (WP4.1). Message: lr changes HF-only little (Spearman +0.012 to +0.035, RMSE <= 6%), LF->HF stays ~2x more accurate ID and ranks better in every cell against the best-tuned HF-only; calibration contrasts are split-dependent. Small table of the LF->HF vs best-HF-only contrasts. | `ablations_vs_reference.csv` + aggregate summaries (ledger WP4.2; WP4.3 pending) | G4 |
| C. Mechanism of the AU collapse | **Forest plot** of paired contrasts vs the LF->HF reference, one row per ablation (variance-head reset, beta-NLL, FT lr 1e-4), columns AU ENCE ratio, AU Spearman diff, RMSE ratio, ID/OOD, both splits; text maps each row to predictions (b) reset does not prevent the collapse, (c) slower FT delays it, (d) held-out fit calibrates. MSE LF->HF (EU only) as the "no variance head" row when done. | `ablations_vs_reference.csv` (ledger entries to write) | Q1/Q4, Q2 |
| D. Estimators | existing: single MVE model, ensemble size M = 1..10, MSE ensembles | ledger 2026-11-02/03/04 (+ MSE LF->HF) | Q2/Q5 |
| E. Design choices and scope | rewritten limitations: MACE, ANI-1x, energies only, sequential FT; other MF strategies as future work | text | G2, G3, J1, J2, J3, Q2 |

Main text: one sentence in Results pointing to App. A-C; terminology change for G5 (D5); Fig. 1
fix (G6); related work (Bilbrey 2025, Tompa 2026, Gardner 2025, Messerly 2025). Mark changes with
`\revise{}`.

### 3.2 Rebuttal posts

- **General response** (all reviewers + AC): one table "new experiment | question | does the paper's
  conclusion hold?" (rows A-D above), plus the list of text changes (E, G5 renaming, related work).
- **Per reviewer**: short answers in `draft_rebuttal.md`, each ending with "see App. X / general response";
  ge6X answered point by point for the AC; Q2LT gets the mechanism figure first.
- Waves (shortened by the user 2026-11-08): **A (Nov 10)**: G1, G3, G5, G6, Q3, J1-J3 text + WP2, WP4.1-4.3;
  **B (Nov 12)**: mechanism ablations (C) and MSE LF->HF; **C (Nov 14)**: revised PDF + anything optional;
  Nov 15-18 replies only. Order of work (user 2026-11-08): revised appendix PDF first, then the rebuttal.

### 3.3 Figure / table work (CPU only, `paper-figures` skill)

User rules (2026-11-08): AUSE + ENCE only (no Spearman); final models as Fig. 2 dot plots (`dotplot.py`), dynamics as
Fig. 5 line plots; always show the training dynamics. Interpretation and phrasing of every new result: `FINDINGS.md`.

- [x] 3.3a `papers/iclr27/figures/figA_lr_sensitivity.py` (item B): RMSE / TU Spearman / TU ENCE vs lr, both protocols and splits, from the aggregate summaries (mace-leo-2, `03e6b94`).
- [x] 3.3b `papers/iclr27/figures/figA_ablations.py` (item C, forest plot: reset, beta-NLL, FT lr 1e-4/1e-2 x 4 tests; RMSE, AU Spearman, AU/TU ENCE) (mace-leo-2, `03e6b94`). Add an MSE LF->HF row when WP3.6 is recorded.
- [x] 3.3c `papers/iclr27/figures/tabA_lf_only.py` -> `out/tabA_lf_only.tex` (item A, booktabs; LF-only / HF-only / LF->HF x 4 tests) (mace-leo-2, `03e6b94`).
- [x] 3.3d General-response table (Markdown, in `draft_rebuttal.md` Section 5; MSE LF->HF row: system and energy filled 2026-11-08) (mace-leo-2, mace-leo-3).
- [x] 3.3g (mace-leo-3, `7463f08`) `figA_pretrain_only.py` and `figA_estimators.py` (Fig. 2 style, replace tabA_* in the
  appendix), `figA_ablation_dynamics.py` (Fig. 5 style; Energy-OOD column added 2026-11-08 16:50, ledger "... on Energy-OOD"), AUSE instead of
  Spearman in `figA_ablations.py` / `figA_lr_sensitivity.py`.
- [x] 3.3e `papers/iclr27/figures/tabA_estimators.py` -> `out/tabA_estimators_{spearman,ence}.tex` (item D; energy MSE LF->HF filled 2026-11-08 17:10) (mace-leo-3, `97da6e4`; mace-leo-4).
- [~] 3.3f (mace-leo-3, 2026-11-08) Revised appendix drafted in `ICLR27-UQ-MF/iclr2027_conference.tex`: new App. "Additional baselines and ablations" (M.1-M.5 = items A-E) after Broader impact, all in `\revise{}`, open items as red `\pending{}`; one pointer sentence in Results intro; the Limitations sentence "we do not ablate ... squared loss, single model, reset head" replaced (old text kept as a comment). Figures copied from `03e6b94`. Compiles (pdflatex, 38 pp., main text still ends p. 9); preview `ICLR27-UQ-MF/iclr2027_conference.pdf` (untracked). **Committed and pushed to the paper repo (Overleaf) 2026-11-09, `38241ec`, at the user's request.** Still to fill: MSE LF->HF row in figA_ablations + energy in tabA_estimators, ensemble-size figure (or text only), design-choices arguments and citations (rebuttal 2.1, 2.2, 3.1-3.3).

## 4. Work packages: status and what is left

### Done (ledger)
- [x] WP2 LF-only on CC labels, E0 refit (G7, G4).
- [x] WP4.1 gradient steps per protocol (G4).
- [x] WP4.2 HF-only lr 1e-3 / 3e-3, both splits (G4).
- [x] WP1.3 Bilbrey 2025 read; WP10.1 ICLR rules; WP11 Leonardo setup (env, `uq train-slurm`,
      `uq evaluate-slurm`, `uq sync-back`).

### In progress (no new launches; finish, evaluate, record)
- [x] **Recorded 2026-11-07** (ledger "Fine-tuning ablations of the AU collapse" + "Fine-tuning learning rate on the system split", mace-leo-2: all 24 pairs). System FT lr: same pattern as energy (1e-4: selected epoch 2 -> 25, AU more overconfident, TU slightly better; 1e-2: worse ranking, TU underconfident). Finding: with a reset head the collapse re-emerges along FT, ~5-8x slower; the selected checkpoint (epochs 2-19) precedes it, so final AU looks better calibrated but ranks worse; consistent with the paper's (i) and (iii). WP3.2 variance-head reset, WP3.3 beta-NLL, WP3.5/4.3 FT lr 1e-4 / 1e-2:
      energy split synced and aggregated 2026-11-07; system: reset-var re-sync and beta-NLL syncing,
      FT lr evaluating (~18:00). Then `compare_ablations`, one ledger entry per WP. Note: reset-var
      and beta-NLL on the system split need the FT-lr system twins only for item B, not for C.
- [x] (agent: mace-80 -> mace-leo -> mace-leo-2 -> mace-leo-3 -> 2026-11-10 session) WP3.6 MSE ensembles LF-only / LF->HF: **system recorded**
      (ledger 2026-11-08: pretraining improves the MVE EU ranking ~5x more than the MSE EU; at LF->HF the MVE ensemble
      ranks better on every metric, accuracy equal; `results/tables/mse_vs_mve_pretraining.csv`). Energy: 100/100
      trained; final models recorded 2026-11-08; full evaluation, aggregates and ID/train dynamics recorded 2026-11-10
      (EU AUSE flat during fine-tuning, ENCE jumps at the switch to CC and recovers). Dynamics figure
      `figA_mse_dynamics` (MSE EU vs MVE EU/TU, 4 tests, Energy-OOD filtered) recorded 2026-11-10; in App. M.4 as
      Fig. 25 (paper repo, user request).

### Open, needs a user decision (after looking at the energy results)
- [!] WP3.1 AU vs training MSE per epoch (prediction (a)): the stored per-epoch means are inconclusive;
      a fair test needs robust per-configuration statistics (GPU re-run of `epoch_quality`). Do not cite.
- [ ] WP5.2 forces in pretraining (D3) and WP7.2 second dataset (D4): budget allows (~70k local h
      left), time does not if decided after ~Nov 9 (LF-scale ~19 h per chain + evaluation).

### Dropped or paused by the user
- [-] WP3.4 held-out-fitted variance head (dropped, user 2026-11-07): it is a learned post-hoc recalibration on
      frozen features (a linear model of the log-variance), close to the isotonic held-out recalibration already in
      the appendix (repairs in-domain scale, not energy extrapolation); the mechanism is already supported by the
      reset dynamics, beta-NLL and FT lr. In Q1, do not claim (d) as tested; cite the recalibration appendix.
      The MSE ensembles (WP3.6) cover the "no variance head" objective. Design in Git history.
- [-] WP6 fidelity-aware / multi-head training (2026-11-07): written answer only (Section 2, J2/Q2).
- [-] Further learning-rate sweeps and WP4.4 model size (2026-11-07: "enough ablations on the LR").

### Writing (all reviewers)
- [ ] WP1.1/1.2/1.4 literature table, novelty table, motivation paragraph (G1, Q3).
- [ ] WP5.1 forces argument (G3); WP7.1 dataset/architecture argument (done in draft 2.2, verify citations).
- [ ] WP8.1 "aleatoric" paragraph, WP8.2 renaming (D5) (G5).
- [ ] WP9 revision: 9.1 Fig. 1 Safari fix (G6); 9.2 "Design choices and scope" (J3); 9.3 related work;
      9.4 appendix of Section 3.1; 9.5 `\revise{}` marks + changelog.
- [x] (mace-leo-5, 2026-11-09) Consistency pass of draft_rebuttal.md and App. M against ledger/FINDINGS: AUSE instead of Spearman everywhere (values from the ledger tables `ablations_vs_reference.csv`, `mse_vs_mve*.csv`, `iclr27_main_text_numbers.txt`); 4.1 held-out effect + (c) caveats (lr 1e-2, Energy-OOD); MSE energy rows; every promise about the PDF that does not exist yet marked `[NOT YET IN PDF]` (novelty table, Bilbrey related work, Fig. 1 fix, D5 rename, main-text mechanism paragraph, model size); 6 references verified (Section 7). App. M: estimator paragraph, MACE and sequential-FT paragraphs with citations (bib: +ramakrishnan2015big, seitzer2022pitfalls -> ICLR 2022), counts verified (1246 held out in all 5 splits), M.4 Energy-OOD single-AU caveat, MSE row of figA_ablations replaced by a pointer to M.4 (no AU, so no row in AU panels). Recompiled: 40 pp., main text ends p. 9, no undefined refs; paper repo still uncommitted. Remaining `\pending`: ensemble-size figure vs text (user).
- [ ] WP10.2-10.5 general response, per-reviewer replies (replace every `[PENDING]` or drop it), final
      check (numbers -> ledger, prior work -> verified reference, no author names), post by wave.

## 5. Decisions

Decided: D1/D1b MSE energy on the DGX, system on Leonardo; D7 3-GPU cap on the DGX only; D8 code to
Leonardo from `iclr_rebuttal`; D9 prune epoch checkpoints after evaluation; D11/D12 Leonardo Miniconda
+ ToS; **D2 multi-head paused (2026-11-07)**; **no further LR sweeps (2026-11-07)**.

Open (**the user decides on 2026-11-09**, together with the Q1-rewrite review): **D3** forces (WP5.2); **D4** second
dataset/architecture (WP7.2); **D5** rename AU/EU; **D10** use of the remaining Leonardo budget; WP3.1 (GPU re-run
or drop). **D6** revised PDF: yes (user 2026-11-08: wants to see the revised appendix PDF before the rebuttal);
WP3.4 dropped (user, 2026-11-07).

## 6. Log (newest last; earlier entries in Git history)

- 2026-11-06 — reviews received; branch `iclr_rebuttal`; Leonardo access and setup (mace-80).
- 2026-11-07 01:00-10:00 — mace-leo: all FT/HF ablations and the system MSE launched on Leonardo;
  WP2, WP4.1, WP4.2 (energy lr 1e-3) recorded; Bilbrey read; ICLR rules; night summary.
- 2026-11-07 10:30-12:30 — mace-leo: certificate renewed (user); Leonardo healthy (no unexpected failures);
  sync-back of 9 finished experiments (one interrupted, retried); WP4.2 complete and recorded (`5c16649`);
  draft G2/J1/J2/Q2/G7 rewritten per user directions; WP6 paused; PLAN condensed with the presentation plan.
- 2026-11-07 18:15-18:40 — mace-leo-2: took over; system FT lr synced and recorded (24/24 pairs); 3.3a-c done (`03e6b94`);
  3.3d general-response table; Q1 (draft 4.1) rewritten from the recorded ablations (to review by the user).
- 2026-11-09 — mace-leo-5: energy MSE evaluation relaunched (`uq-mse-eval4`, RUNS.md); consistency pass of the rebuttal draft and App. M (see Writing).
- 2026-11-09 — mace-leo-5: Fig. 22 (App. M.3, `figA_ablations`) rebuilt as a Fig. 2-style plot of raw values (user); old forest plot kept as `figA_ablations_contrasts.py`; caption + one TU clause in M.3; plotted values match the ledger, `paper_numbers.py` unchanged.
- 2026-11-09 12:05 — mace-leo-5: user plans to post a first rebuttal (wave A) **tonight, 2026-11-09**, after correcting the draft and the PDF. Before posting: resolve every `[NOT YET IN PDF]` in `draft_rebuttal.md` (apply or drop the sentence), decide D5, D3/D4 wording in G2/G3, trim per OpenReview limits, no author names; commit the paper repo after the user's review of App. M.
- 2026-11-09 12:15 — mace-leo-5: paper repo committed and pushed (`38241ec`: App. M, Limitations, bib, 5 appendix figures); Overleaf had no newer commits.
- 2026-11-09 evening — user's revisions: Fig. 21 (`figA_ablation_dynamics`) dots at each variant's median selected epoch (50 members, `select_best_checkpoints`); Fig. 23 (`figA_estimators`) tinted rows + legend mark our estimator (10-member MVE ensemble; new `highlight` option in `dotplot.draw`, other dot plots pixel-identical); captions updated, recompiled (40 pp.). Rebuttal assembled as Markdown in `ICLR27-UQ-MF/rebuttal.tex` (6 posts, all < 4k characters) from `draft_rebuttal.md` + user/co-author directions (foundation-model framing, MACE "one of the most used", closing question with options); MACE-MP-0 (arXiv:2401.00096) and MACE-OFF (JACS 2025) checked on the web. Paper repo pushed.
- 2026-11-09 late — user: no general response, one post per reviewer (<= 5000 characters each: ge6X 4908, j1p7 3502, Q2LT 4592); `ICLR27-UQ-MF/rebuttal.tex` is now compilable LaTeX (Markdown in fvextra Verbatim boxes) with `rebuttal.pdf` committed (paper repo `9de546d`). Revised PDF: Fig. 1 Safari fix applied (arXiv `f342f6b`, same source file, md5 checked), red `\pending` removed from App. M.4. **Open (kept only here):** App. M.4 ensemble-size result is text only; a five-split figure of the M = 1..10 curves (ledger 2026-11-02) can still be added. Rebuttal promises still to apply in a later PDF: Bilbrey related-work paragraph (Q3), main-text summary of the mechanism (Q1). D3/D4 assessment for the user: forces-in-pretraining on the energy split (50 LF + 50 FT members, 5 x 10) is feasible on Leonardo by ~Nov 14-15 if launched Nov 10 (needs DFT forces re-extracted from the ANI-1x h5 into a new dataset variant, NLL+force loss check, smoke test; LF members > 24 h with forces -> resume chains; November quota nearly used -> lower priority); second architecture not realistic (MVE head in a new model + 300 members).
- 2026-11-09 late — rebuttal posts revised (user: new work must be explicit): each opens with "What we added for this review" (bullets per App. M item) and marks new results "new", existing ones "already in the submission" (paper repo 03b7ca1).
- 2026-11-09 late — rebuttal reworded (user: more natural, no (a)/(b) lists in the closing questions); reviewer-specific openings (ge6X 'missed a trend', 'not on your side' for Fig. 1). Final counts ge6X 4926, j1p7 3836, Q2LT 4753. Push failed (VS Code credential socket gone): paper repo 03b7ca1 + 2d8616c and this repo unpushed until the user pushes.
- 2026-11-09 late — rebuttal closing questions tailored to each reviewer (ge6X: forces / model size; j1p7: fidelity as input; Q2LT: direct mechanism test / joint or multi-head); counts ge6X 4949, j1p7 3923, Q2LT 4786 (paper repo 0b295e4).
- 2026-11-09 late — rebuttal style pass after the user's Overleaf edits (64812e2): topic headings instead of G/J/Q IDs, no semicolons outside citations, fewer adjectives; verbatim quotes paraphrased (ge6X, j1p7). Counts ge6X 4841, j1p7 3764, Q2LT 4759.
- 2026-11-09 late — rebuttal per user: opening 'what we added' blocks removed (one sentence on App. M kept); MACE design space (batatia2025design, abstract checked) and single-system benchmarks (rMD17, 3BPA, water) in ge6X and j1p7; j1p7 choices argued inline; Q2LT states what each ablation shows and how it supports the paper (+ ensemble-size result, ledger 2026-11-02). Counts ge6X 4935, j1p7 4490, Q2LT 4884.
- 2026-11-09 23:20 — supervisor's request: Käser & Meuwly 2023 and Chen et al. 2023 checked on full texts; both fine-tune all weights of a LL-pretrained model on ~100-200 CC labels, like us (ANI-1ccx instead freezes 2 of 4 layers). Proposed ge6X Motivation with both added (two variants, 4975 / 4998 characters) in `draft_rebuttal.md` 2.1; **not applied to rebuttal.tex** (user decision). Käser 2023 not in the paper bib.
- 2026-11-09 23:12 — DGX energy-MSE eval queue moved from `uq-mse-eval4` to `uq-mse-eval5` (dynamic GPU placement against OOM); details in RUNS.md log.
- 2026-11-10 — after the "draft 2" pull (user + Luigi): Fig. 23 split into Fig. 23 HF-only / Fig. 24 LF→HF (`figA_estimators_{hf,lfhf}`, CPU, numbers unchanged: same `mse_vs_mve.csv`), tint removed, "(ours)" in the legend; rebuttal: thanks before each answer, TODO citations filled from the reference table (Zaverkin 2023, Käser & Meuwly 2023, Chen 2023; Batatia 2023, Radova 2025), typos, closings for j1p7/Q2LT. Open: ge6X (~5.6k) and Q2LT (~6.4k chars) exceed the 5000-character limit; Bocus et al. 2025 not in Section 7 (unverified). Paper-repo changes left uncommitted for the user.
- 2026-11-10 13:10 — user follow-up: Bocus et al. 2025 verified (Angew. Chem., transfer learning to RI-RPA, Section 7 row added). Promises turned into edits (blue): Sec. 1 motivation sentence (radova2025fine, bilbrey2025uncertainty, new bib entry) and a sentence on how findings (i)-(iv) extend single-fidelity work (rasmussen2023uncertain, valdenegro2022deeper, seitzer2022pitfalls); Sec. 3 one sentence on ANI-1x/MACE (-> App. M.5). Rebuttal says "we have updated..."; every closing asks whether there is anything else we can do. Main text still ends on p. 9; Figs. 23/24 now p. 40. Box lengths ge6X 5707, j1p7 3022, Q2LT 6624 (limit 5000). Paper repo still uncommitted.
- 2026-11-10 — energy MSE evaluation finished on the DGX (`uq-mse-eval5`, 08:13), aggregated + pruned; ledger entry "MSE ensembles, ANI-1x energy split: full evaluation ..."; WP3.6 closed. DGX idle for this project.
- 2026-11-10 16:40 — `figA_mse_dynamics` (item D dynamics) built and recorded; candidate for App. M next to Fig. 23/24, not copied to the paper repo.
- 2026-11-10 evening — after the user's "final rebuttal" pull (paper `f8b4309`): consistency check (compiles, no undefined refs, all section/figure numbers cited in rebuttal.tex match, promised intro/Sec. 3/M.5 edits present, figure PDFs identical to `figures/out/` except fig6 creation date); `figA_mse_dynamics` added to App. M.4 as Fig. 25 (paper commit, not pushed). ge6X 5462 and Q2LT 5832 characters: fine, posted as two OpenReview comments each (user, 2026-11-10).
- 2026-11-10 — **responses posted on OpenReview by the user** (rebuttal.tex as in paper commit `f8b4309`, unchanged; ge6X and Q2LT as two comments each). Next: replies to reviewers only.
