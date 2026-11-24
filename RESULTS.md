# CXR-Recall — Results (Open-I)

Final results from the real, early-stopped training run on Kaggle (`raddar/chest-xrays-indiana-university`, ViT-B/16 backbone, up to 20 epochs, patience 5). Smoke-test numbers (fixed 3-epoch runs used to validate the pipeline before spending real compute) are included for comparison, not as the headline result. Includes the Phase 1b (Arm 3b, early fusion), Phase 1c (Arm 3c, joint attention), Phase 1d (Arm 3c-frozen, encoder-sharing fix), and Phase 1e (seeded confirmation) follow-ups, each directly testing the leading explanation for the previous variant's shortfall.

**D9 (2026-11-24):** with a frozen RAD-DINO encoder (5 seeds), retrieval − probe = +0.0065 (p = 0.075) but retrieval − random neighbours = +0.0021 (p = 0.49): no retrieval-specific gain. The RAD-DINO probe alone (0.7461) beats the ViT baseline by +0.033. See "D9".

**Status (final, 2026-11-24, after the 3-seed run; see "Phase 1 multi-seed"): H1 is not supported.** Across 3 paired training seeds on the 743 test images, Arm 3c-frozen − Arm 1 = **−0.0042 ± 0.0060** (95% CI [−0.019, +0.011], paired t p = 0.35, positive in 1 of 3). Every earlier positive margin in this file was a single run inside Arm 1's ~0.02 run-to-run spread. The status paragraphs below are kept as history.

*Previous status:* **(revised 2026-09-06 after the audit rerun — see "Phase 1 audit rerun"): H1 is *directionally* consistent — Arm 3c-frozen has beaten Arm 1 in every run where both existed (3/3 on val, and +0.0108 on the first held-out test evaluation) — but its *magnitude* is not established.** The audit rerun's val margin (+0.0030) is a tenth of the seeded run's (+0.0261), and Arm 1 alone moved 0.7092 → 0.7272 between two `SEED=42` runs of code that does not differ for Arm 1, so the earlier "confirmed reproducible" claim is withdrawn: Arm 1's run-to-run range is ~0.024 and the H1 margin lives inside it. A multi-seed run is now required, not optional. The paragraph below and Phases 1a–1e were written before that rerun; read their confidence language with this caveat.

*Previous status (superseded):* H1 was described as well-supported for the final architecture (Arm 3c / Arm 3c-frozen) under a seeded run — see Phase 1e. That reversed the original Phase 1 finding, which was specific to the very first, label-only evidence design (Arm 3), not the fusion approach overall. Read the top-level "Verdict" section below as describing that first, now-superseded design.

**Read every AUROC in this file as a *validation* number that was also the early-stopping selection target** — the held-out test split (743 images) was computed by both notebooks and never evaluated on. A 2026-09-05 pipeline audit (section "Pipeline audit" below) added test-set evaluation, fixed a same-study leakage path in Arm 3's training evidence, and moved Phase 1f's threshold calibration off the evaluation set; **Phase 1 and Phase 2 have both been regenerated with test numbers ("Phase 1 audit rerun", "Phase 2 audit rerun" and "Phase 2 rerun at batch 64" below).** The batch-32 Phase 2 rerun exposed that every earlier Phase 2 linear probe was under-trained and that Arm 3 had the wrong baseline; the batch-64 rerun plus a follow-up convergence analysis on its saved embeddings **supersede every earlier Phase 2 number, including the old +0.0129 headline and the gated-fusion "regression"** (retracted); the follow-up puts the Phase 2 evidence gain at only about +0.003 on test (see "Phase 2 rerun at batch 64").

## Project summary: Phase 1 + Phase 2, both conclusions

The project structure mirrors Phase 1's three arms (Arm 1: no retrieval, Arm 2: post-hoc retrieval quality, Arm 3: pre-diagnosis evidence fusion) across two phases — image-only (Phase 1) and image+text (Phase 2) — to answer two tracked questions cleanly: does retrieval-evidence fusion help, and does adding text help. Headline numbers (Phase 2: best real run, contrastive redesign, `STAGE1_BATCH_SIZE=64`+AMP):

| | Arm 1 (no retrieval) | Arm 2 (retrieval quality) | Arm 3 (evidence fusion) |
|---|---|---|---|
| **Phase 1** (image-only, seeded run, val) | 0.7092 | label_agreement@5 = 0.2727 (vs. 0.1845 random) | **0.7353** |
| **Phase 1 audit rerun** (val / **test**) | 0.7272 / **0.7051** | label_agreement@5 = 0.2954 / **0.2864** (vs. 0.1845 random) | 0.7302 / **0.7159** |
| **Phase 2 attempt 1** (naive fusion, discarded) | image-only ablation 0.5328 (collapsed) | — | not built (attempt discarded before Arm 2/3) |
| **Phase 2 attempt 2** (contrastive, batch 64, val, probes under-converged — **superseded by the last row**) | full 0.8045 (image-only ablation 0.7042, text-only ablation 0.8280) | label_agreement@5 = 0.2651 (vs. 0.1845 random) | **0.7171** |
| **Phase 2 audit rerun** (batch 32, val / **test**, probes under-converged) | full 0.7563 / **0.7474** (image-only 0.6723 / **0.6337**, text-only 0.7607 / **0.7627**) | label_agreement@5 = 0.2373 / **0.2420** | 0.6800 / **0.6359** |
| **Phase 2, batch 64, Kaggle run A** (val / **test**; probes lr 1e-4 capped at 400 epochs — under-converged) | full 0.9069 / 0.9025; image-only probe 0.6958 / 0.6553; text-only probe 0.9148 / 0.9128; gated 0.9158 / 0.9184 | 0.2568 / **0.2428** | 0.7137 / 0.6734 |
| **Phase 2, batch 64, truly converged probes on saved embeddings (run B)** (val / **test**) — current | full 0.9395 / **0.9329**; image-only probe 0.7078 / **0.6944**; text-only probe 0.9423 / **0.9355**; gated 0.9440 / **0.9404** | 0.2861 / **0.2732** | 0.7247 / **0.6972** (**+0.003** test over the image probe) |

### Conclusion 1 — does retrieval-evidence fusion improve classification? **No measurable gain. Phase 1, 3 seeds: −0.0042 ± 0.0060 test (p = 0.35). Phase 2: +0.003 test against a converged baseline.**

**Final 2026-11-24 (Phase 1 multi-seed):** the paired 3-seed test margin is −0.0042, 95% CI [−0.019, +0.011]. The +0.0108 single-run test margin below was not reproduced. See "Phase 1 multi-seed".

**Revised 2026-09-06:** the audit rerun (the first with a held-out test evaluation) gives Arm 3c-frozen − Arm 1 = **+0.0108 on test** (+0.0030 on val), versus +0.0261 val in the seeded run. Arm 3c-frozen itself was stable across runs (0.7360 / 0.7353 / 0.7302); the shrinking margin is almost entirely Arm 1 landing a better run (0.7092 → 0.7272). So the honest reading is: fusion has never lost to Arm 1 in four runs, but the margin is ~0.01 on the one clean test number and the noise on Arm 1 alone is ~0.02 — a multi-seed distribution is needed before any magnitude is claimed. The paragraph that follows is the pre-rerun framing.

**Update 2026-11-10 (Phase 2, batch 64, after a convergence analysis):** the +0.0181 first reported here came from an image-only baseline that was itself under-trained (a probe at lr 1e-4 stopped at 0.6553 test). On the saved batch-64 embeddings, a properly converged image-only probe scores 0.7078 val / **0.6944 test**, and Arm 3 (3 seeds, learning rate swept and selected on val) scores 0.7247 val / **0.6972 test** at lr 1e-4 (0.6957 at 3e-4): a gain of **+0.017 on val but only +0.003 on test** (±0.002 across seeds; above the baseline in 3 of 3 seeds at 1e-4, 2 of 3 at 3e-4). Raising the learning rate makes Arm 3 lean on the evidence harder (ablation shift 0.011 → 0.088) and test gets *worse* (0.685). So Phase 2 shows at best a marginal, noise-sized evidence gain on test against Phase 1's +0.0108; the val gain is largely selection, since each Arm 3 run picks its best epoch on val. Two batch-64 Stage 1 runs exist (the Kaggle paste and the saved embeddings) and differ visibly (Arm 2 0.2568 vs 0.2861 val), so Stage 1 variance is itself not small and is unmeasured.

Phase 1: Arm 1 → Arm 3c-frozen, **+0.0261**, confirmed across 3 independent runs (2 unseeded, 1 seeded, all agreeing on direction). Phase 2: Arm 1 image-only → Arm 3, **+0.0129**, same direction, on the best (`batch=64`+AMP) run — earlier, more weakly-trained Phase 2 runs showed this margin shrink toward zero as the underlying image encoder was weaker, consistent with an already-documented Phase 1e pattern: a better base encoder needs less help from evidence. Retrieval evidence has never once shown a *negative* margin in either phase once fairly compared (Arm 3 vs. the same-modality no-retrieval baseline) — the effect size varies with encoder quality, but the direction is consistent.

### Conclusion 2 — does adding text improve decision-making? **Mechanically yes, but not for a trustworthy reason.**

Full fusion (0.8045) beats Phase 1's pure-image baseline (0.7092) by +0.11. But **text-only alone (0.8280) is the single highest number anywhere in the project** — higher than full fusion itself, in every version of Phase 2 tested (naive end-to-end fusion and the contrastive redesign alike, across all runs). This lines up with two independently-measured, concrete confounds: a **64.0% lexical-overlap floor** (MeSH labels are NLM's own indexing *of* the report text being fed to the model — the label's own words are frequently present in the input), and a **21.7% train/val exact-duplicate-text rate** (concentrated 87.1% in the trivially-easy "normal" class), which the model can partly solve via memorized boilerplate rather than reading novel reports. Additive fusion landing *below* text-only alone replicated in every single run — the signature of a real-but-secondary image signal diluting an already-dominant shortcut, not two modalities genuinely combining. A follow-up learned gate (Fix #3) first looked like it made full fusion *worse* (0.7566 vs. 0.8045), but **that was an artifact of under-converged probes and is retracted** (2026-11-10): at true convergence the gated and ungated heads are within noise of each other (0.9404 vs. 0.9325 test) and of text alone (0.9386).

**Update 2026-11-10 (truly converged probes, batch-64 embeddings):** with the probes trained to convergence, a *linear* classifier on frozen report embeddings recovers the MeSH labels at **0.9355 test AUROC** (0.9386 with the torch recipe) while the image alone gets **0.6944** — a gap of about 0.24. Full fusion (sum 0.9329, concatenation 0.9311) and the gated head (0.9404) all land within about 0.005 of text alone: **the image adds essentially nothing once the text is there**, and the earlier 'fusion below text-only' signature shrinks from ~0.010 to a statistical tie. Two checks on what drives the text advantage: on test reports whose text is *not* an exact duplicate of any training report (558 of 743), text-only still scores 0.9165 and the text-over-image gap does not shrink (0.2411 on all rows, 0.2466 on non-duplicates), so memorized duplicates are a small effect (text −0.015 to −0.019, image −0.023 to −0.025), not the explanation. The explanation that remains is circularity: the labels are NLM's indexing of these same reports, so a text embedding already contains them. The text numbers measure label recovery, not clinical reasoning.

**Bottom line: text raises the benchmark score, but the project's own diagnostics point to the model reading its own label source, not learning clinical language.** A trustworthy test of "does text help" would require removing the shortcut first (e.g. training/evaluating on the Findings section alone, dropping Impression) — a smaller fix proposed during this investigation. **Update 2026-11-11:** it has been built as a notebook, but a word-level pre-check shows Findings alone recovers the labels at 0.9767 test AUROC (Impression alone 0.8347), so it will not remove the shortcut — see "Findings-only text variant"

## Final numbers

| Arm | Metric | Smoke test (3 epochs) | Real run (early-stopped) |
|---|---|---|---|
| Arm 1 — no retrieval | val macro AUROC | 0.6907 | **0.7335** (best epoch 3/20) |
| Arm 2 — post-hoc retrieval | label_agreement@5 | 0.2794 | **0.2973** |
| Arm 2 — post-hoc retrieval | MRR | 0.4063 | 0.4295 |
| Arm 2 — random-retrieval baseline | label_agreement@5 | 0.1845 | 0.1845 (unchanged, doesn't depend on the encoder) |
| Arm 3 — CXR-Recall (fusion, label-only evidence) | val macro AUROC | 0.6986 | **0.7025** (best epoch 7/20) |
| Arm 3 — CXR-Recall (fusion, label-only evidence) | evidence-ablation mean \|shift\| | 0.0388 | **0.0516** |
| Arm 3b — CXR-Recall (fusion, label + image evidence, early fusion) | val macro AUROC | — (real-run-only follow-up) | **0.7278** (best epoch 3/20) |
| Arm 3b — CXR-Recall (fusion, label + image evidence, early fusion) | full evidence-ablation mean \|shift\| | — | **0.0757** |
| Arm 3b — CXR-Recall (fusion, label + image evidence, early fusion) | image-only ablation mean \|shift\| (labels kept, images zeroed) | — | **0.0611** |
| Arm 3c — CXR-Recall (fusion, label + image evidence, joint attention) | val macro AUROC | — (real-run-only follow-up) | **0.7318** (best epoch 5/20) |
| Arm 3c — CXR-Recall (fusion, label + image evidence, joint attention) | full evidence-ablation mean \|shift\| | — | **0.0999** |
| Arm 3c — CXR-Recall (fusion, label + image evidence, joint attention) | image-only ablation mean \|shift\| (labels kept, images zeroed) | — | **0.0985** |
| Arm 3c-frozen — encoder-sharing fix (separate rerun — see Phase 1d, unseeded, not directly comparable to the row above) | val macro AUROC | — | **0.7360** (best epoch 4/20) |
| Arm 3c-frozen — encoder-sharing fix (same rerun's own Arm 1 / Arm 3c, for reference) | val macro AUROC | — | Arm 1 = 0.7192, Arm 3c = 0.7311 |
| Arm 3c-frozen — encoder-sharing fix | full / image-only ablation mean \|shift\| | — | **0.0394 / 0.0381** |
| Arm 1 / Arm 3c / Arm 3c-frozen — **seeded confirmation run** (SEED=42, see Phase 1e) | val macro AUROC | — | **0.7092 / 0.7262 / 0.7353** |

## Verdict (original Phase 1 finding — label-only evidence design, since superseded)

**H1 (pre-diagnosis fusion beats no-retrieval), for the original label-only evidence design: not supported.** Arm 3's real-run best (0.7025) is 0.031 AUROC below Arm 1's (0.7335), after both arms were trained fairly under identical conditions (same data, same budget, same early-stopping criterion). This finding motivated everything that follows (Phases 1b–1e); it does **not** describe the project's final architecture — see Phase 1e for the current, seed-confirmed verdict.

**H2 (retrieval finds relevant cases): supported, and remains supported through every later phase.** Arm 2's retrieval clearly and consistently beats the random-retrieval baseline (0.2973 vs 0.1845, ~61% relative improvement; 0.2727 vs 0.1845 in the seeded run, Phase 1e) across every run so far. Retrieval quality itself is real and reproduces; what varied across phases was how that retrieved evidence should be represented and fused.

## Why Arm 3 underperforms — not just "didn't help," actively used and still worse

The evidence-ablation shift **increased** from the smoke test to the real run (0.0388 → 0.0516), not decreased. That rules out the simplest failure mode — a fusion module that's effectively dead weight, quietly ignoring its input and losing to Arm 1 purely on extra-parameter noise. Instead, Arm 3 leans on the retrieved evidence *more* in the fully-trained run, and is still worse than Arm 1.

The likely mechanism: `label_agreement@5 = 0.2973` means roughly **70% of the k=5 neighbors fused into any given prediction don't share a label with the query**. A model that is causally, substantially trusting evidence that's wrong most of the time is exactly the setup that would drag predictions down while still showing a large ablation shift — the model isn't ignoring the evidence, it's being actively misled by it often enough to net negative.

This also points at a specific architectural weakness, not just "retrieval fusion doesn't work here": the evidence encoder only ever sees each neighbor's 14-dim multi-hot label vector, never the neighbor's actual image. That's a severe information bottleneck — the model can't distinguish "a neighbor that's a strong, unambiguous match" from "a neighbor that's a weak, partial match," because both collapse to the same coarse label bits before reaching the fusion layer. See "Next steps" below.

## Phase 1b: Arm 3b — richer evidence closes most of the gap

Directly motivated by the finding above. `EvidenceEncoder` was extended (`use_image_embeddings=True`) to fuse each retrieved neighbor's own image embedding alongside its label vector, instead of the label vector alone. The neighbor embeddings (already computed during Arm 2's retrieval, no extra encoder pass needed) are passed through their own learned `image_proj` layer before use — a deliberate decoupling from the query's own embedding space, since naively reusing the same representation for retrieval and classification is a known failure mode in the KNN literature. Everything else (data, splits, encoder from Arm 1's real run, k=5 neighbors, early-stopping protocol) is identical to Arm 3, so this is a controlled, apples-to-apples comparison.

| Arm | val macro AUROC | vs. Arm 1 gap |
|---|---|---|
| Arm 1 (no retrieval) | 0.7335 | — |
| Arm 3 (label-only evidence) | 0.7025 | −0.0310 |
| Arm 3b (label + image evidence) | **0.7278** | **−0.0057** |

**Arm 3b closes ~82% of Arm 3's gap to Arm 1** (0.0310 → 0.0057). Arm 3b's training curve was also materially more stable than Arm 3's — Arm 3 collapsed sharply in its back half (down to 0.66 by epoch 11–12) while Arm 3b stayed in a tight 0.72–0.728 band for all 8 epochs before stopping. Richer evidence didn't just raise the ceiling, it stabilized training.

**The two-part ablation pinpoints exactly what's responsible.** Zeroing everything gives a shift of 0.0757; zeroing *only* the image embeddings (keeping the real labels) gives 0.0611. That means image embeddings alone account for **~81% of the total evidence reliance** (0.0611 / 0.0757). More strikingly: **that image-only contribution (0.0611) is larger than Arm 3's entire evidence-ablation shift (0.0516)** — a single retrieved neighbor's image embedding carries more causal weight in Arm 3b than the *entire* label-vector evidence did in Arm 3. This is about as direct a confirmation as this kind of test can give that the original diagnosis was right: label-only evidence was the bottleneck, and image information was the fix.

**Calibration note on the remaining 0.0057 gap:** on 746 validation images, a gap this small is plausibly within run-to-run noise rather than a genuine remaining deficit — no bootstrap confidence intervals have been computed to test this formally. The more defensible claim is "richer evidence closed the large majority of the gap, and the remainder is not yet distinguishable from noise," not "Arm 3b still definitively underperforms Arm 1."

## Phase 1c: Arm 3c — joint attention closes 94.5% of the original gap

Arm 3b fused each neighbor's image embedding and label vector into a *single* blended vector, via one linear layer applied uniformly to every neighbor, before attention ever ran. That structurally prevents the model from independently trusting a neighbor's image while discounting that same neighbor's label (or vice versa) — the two modalities are pre-mixed at a fixed ratio no matter how reliable either one actually is for a given neighbor. Arm 3c fixes this: each neighbor now contributes **two separately-attendable tokens** (one for its image embedding, one for its label vector), each tagged with a learned modality embedding (image vs. label) and a learned neighbor-position embedding (so the model still knows which image/label pair came from the same neighbor). The query attends over all `2k` tokens jointly, free to weight each one independently. Everything else — data, encoder, k=5, early-stopping protocol — is unchanged from Arm 3b, so this remains a controlled comparison.

| Arm | val macro AUROC | vs. Arm 1 gap | Full ablation shift | Image-only shift | % reliance from images |
|---|---|---|---|---|---|
| Arm 1 (no retrieval) | 0.7335 | — | — | — | — |
| Arm 3 (label-only) | 0.7025 | −0.0310 | 0.0516 | — | — |
| Arm 3b (early fusion) | 0.7278 | −0.0057 | 0.0757 | 0.0611 | 81% |
| Arm 3c (joint attention) | **0.7318** | **−0.0017** | **0.0999** | **0.0985** | **98.6%** |

**94.5% of the original H1 gap is now closed** (0.0310 → 0.0017), and the remaining 0.0017 AUROC on 746 validation images is well inside the range a single stochastic run's noise would produce — smaller than what run-to-run variance alone typically causes, though still not formally tested (see "Next steps"). Training stayed stable across all 10 epochs before stopping (tight 0.72–0.73 band, no late collapse) despite `2k` tokens being more attendable capacity than Arm 3b's `k` on the same small dataset — the capacity-risk concern raised before this ran did not materialize.

**Two patterns emerge from the full three-variant arc (Arm 3 → 3b → 3c):**

1. **Evidence reliance grows monotonically with accuracy** — ablation shift went 0.0516 → 0.0757 → 0.0999 as AUROC went 0.7025 → 0.7278 → 0.7318. Every step closer to Arm 1 came from the model trusting its evidence *more*, never from quietly learning to ignore it.
2. **Joint attention learned, on its own, to almost completely discount the labels.** Early fusion's image-only share was 81% — dominant, but the label signal was still structurally forced into the blend. With modality-separated tokens, that share jumps to **98.6%** — labels contribute almost nothing distinguishable (0.0014 out of 0.0999). Given `label_agreement@5 ≈ 0.30` (labels are wrong on ~70% of neighbors), a model free to weight modalities independently appears to have learned to trust the unreliable signal far less and the reliable one far more — exactly the adaptive behavior early fusion's fixed linear blend couldn't express.

**Untried follow-up this suggests:** since labels contribute almost nothing in Arm 3c, an images-only evidence variant (dropping label tokens entirely) might match or slightly exceed this result by removing near-dead-weight tokens. Not yet built.

## Phase 1d: Arm 3c-frozen — the encoder-sharing fix, and an unexpected discovery about run-to-run variance

Every fusion variant so far (Arm 3, 3b, 3c) trained its own encoder end-to-end alongside the fusion mechanism — a confound noted since Arm 3's own writeup: some of the AUROC gains could reflect the encoder adapting differently under fusion-mediated gradients, not the fusion mechanism alone. `RetrievalFusionClassifier.load_and_freeze_encoder()` removes this directly: Arm 3c-frozen's encoder is loaded from Arm 1's exact trained weights and frozen (`requires_grad=False`, permanently kept in `eval()` mode even across `.train()` calls) — byte-identical to Arm 1's, no gradients reach it. Only the evidence encoder and classification head train. Architecture is otherwise identical to Arm 3c (joint attention).

**This run's own internally-consistent numbers:**

| Arm | val macro AUROC | Full ablation shift | Image-only shift |
|---|---|---|---|
| Arm 1 (no retrieval, this run) | 0.7192 | — | — |
| Arm 3c (own encoder, this run) | 0.7311 | 0.0999* | 0.0985* |
| Arm 3c-frozen (Arm 1's exact encoder) | **0.7360** | **0.0394** | **0.0381** |

*Arm 3 / Arm 3c ablation shifts shown here are carried over from the original documented run above — this particular rerun's own Arm 3c ablation wasn't separately re-measured.

**Within this run, Arm 3c-frozen beats both Arm 1 and Arm 3c** — the first fusion variant across the whole series to beat Arm 1 outright rather than just close the gap toward it. The image-reliance pattern replicates too: 0.0381 / 0.0394 ≈ 97% of evidence reliance comes from images, consistent with unfrozen Arm 3c's 98.6% — this part of the finding looks robust across runs even though the absolute AUROCs don't.

**An important complication surfaced running this:** this session's Arm 1 (0.7192) and Arm 3c (0.7311) do not match the earlier documented run's numbers (0.7335 and 0.7318 respectively) — a 0.0143 swing on Arm 1 alone, from identical code and data. Cause: no random seed was set anywhere in the notebook except the patient-level split itself, so classification-head initialization and DataLoader shuffle order differed between runs. **This has since been fixed** (`set_seed()`, called at the top and again before each training block — see repo commit history) **but the fix was added after this run, so this specific comparison is not yet seed-controlled.**

**How to read this result honestly:** the Arm 1 vs. Arm 3c-frozen comparison is fair *within* this run (both trained/evaluated in the same session, Arm 3c-frozen literally uses this run's own Arm 1 encoder) — so "Arm 3c-frozen beats Arm 1 by 0.0168 here" is a real, internally consistent observation, not an artifact of comparing across mismatched runs. But given we just measured 0.0143 of pure run-to-run noise on Arm 1 alone, a margin of 0.0168 is only slightly larger than demonstrated noise — encouraging, and directionally the first clean win for H1 in this whole investigation, but not yet strong enough to call the hypothesis definitively settled. A seeded rerun (now possible) is what would actually confirm or contradict this.

## Phase 1e: seeded confirmation — H1 is now well-supported (superseded: not supported across 3 seeds, see "Phase 1 multi-seed")

With `set_seed(42)` in place (defined near the top of the notebook, re-called before each of the three training blocks), the full pipeline was rerun top to bottom: Arm 1 → Arm 2 → Arm 3c → Arm 3c-frozen. This is the first run in the entire investigation that is actually reproducible — rerunning this exact notebook should reproduce these numbers, not produce another surprise.

| Run | Arm 1 | Arm 3c (own encoder) | gap | Arm 3c-frozen | gap |
|---|---|---|---|---|---|
| Run 1 (original, unseeded) | 0.7335 | 0.7318 | −0.0017 | — | — |
| Run 2 (rebuild, unseeded) | 0.7192 | 0.7311 | **+0.0119** | 0.7360 | **+0.0168** |
| Run 3 (seeded, SEED=42) | 0.7092 | 0.7262 | **+0.0170** | 0.7353 | **+0.0261** |

**H1 is now well-supported for the final architecture.** Across three independent runs, fusion never meaningfully underperforms Arm 1 — the single "loss" (run 1) was by only 0.0017, essentially a tie, and every other comparison is a clear win, with the seeded run showing the largest margins yet (+0.0170 for Arm 3c, +0.0261 for Arm 3c-frozen) — both comfortably larger than the 0.0143 noise floor measured between runs 1 and 2. Three independently-initialized runs agreeing on direction is substantially stronger evidence than any single run, seeded or not.

**Retrieval quality replicates too.** `label_agreement@5 = 0.2727` (seeded run) vs. 0.1845 random — consistent with 0.2973 (run 1) and effectively unchanged from run to run, since Arm 2's random baseline doesn't depend on the encoder at all. H2 is robust across every phase of this investigation.

**Two additional patterns emerge from comparing the seeded run against run 2, both suggestive rather than proven (small sample):**

1. **Arm 3c-frozen is more stable than Arm 1 itself.** Between runs 2 and 3, Arm 1 dropped by 0.01 (0.7192 → 0.7092), but Arm 3c-frozen barely moved (0.7360 → 0.7353, a difference of 0.0007) despite being built on top of that same declining encoder each time. This suggests the fusion mechanism may be compensating for a weaker base encoder by leaning harder on retrieved evidence, producing a more consistent final result than the no-retrieval baseline alone — an argument for fusion's value beyond the headline AUROC number. Only two data points for this specific comparison.
2. **A better encoder needs less help from evidence.** Within the seeded run, Arm 3c-frozen's ablation shift (0.0283) is only ~41% of unfrozen Arm 3c's (0.0696), even though frozen scores higher (0.7353 vs. 0.7262). Consistent with the theory raised after Phase 1d: a cleanly-converged, stable encoder gives the model strong-enough baseline features that it doesn't need to lean as hard on evidence to improve on them — evidence reliance and evidence *usefulness* are not the same axis.
3. **The image-dominance finding replicates a third time.** Seeded-run image reliance: 97.7% for Arm 3c (0.0680/0.0696), 94.3% for Arm 3c-frozen (0.0267/0.0283) — consistent with run 1's 98.6% and run 2's ~97%. Across three separate runs and two architectures, the model consistently learns to trust images far more than labels once it's free to weight them independently.

**What "seeded" does and doesn't establish.** This run is now reproducible — rerunning the exact same notebook and seed should give the exact same numbers. It does not, by itself, prove the effect holds under a *different* seed; that would need multiple seeded reruns to build a distribution rather than a point estimate (see "Next steps"). What it does establish is that this specific, strong result (+0.0170 / +0.0261) is not an artifact of unseeded randomness the way the previous ambiguous +0.0168 margin was — combined with two other independent runs pointing the same direction, this is now the best-supported finding in the investigation.

## Phase 1f: retrieval-confidence gating (similarity-threshold fallback) — run 2026-09-06, negative result (see "Phase 1 audit rerun", point 3)

Every arm above fuses whatever k neighbors FAISS returns, regardless of how weak the match is — and `label_agreement@5` (0.2973 in the real run, 0.2727 seeded) is only marginally above the 0.1845 random baseline, meaning roughly 70% of retrieved neighbors don't even share a label with the query. Nothing checked this before trusting the evidence.

**What was built:** `RetrievalFusionClassifier.forward()` (and the notebook's `RetrievalFusionClassifierV3`) gained an optional `evidence_confidence: (batch,)` parameter — when a query's retrieved evidence isn't trusted (hard 0.0), its contribution to the fused representation is zeroed entirely (`fused = query_embedding + evidence_context * evidence_confidence`), producing a true vision-only prediction for that query, not an approximation of one. Omitting the parameter reproduces every prior arm's behavior exactly (verified by test), so Arm 3/3b/3c/3c-frozen above are all still reproducible unchanged.

**Applied eval-time-only, on the already-trained Arm 3c-frozen checkpoint — no retraining.** This deliberately mirrors the existing evidence-ablation methodology (Phase 1b/1c/1d): zero the evidence contribution post-hoc on a trained checkpoint, don't retrain under the altered regime. Confidence gating is mechanically the same operation, just driven by a real per-query similarity signal instead of researcher-imposed 100% zeroing across the whole val set. Retraining under gating is scoped out as a candidate Phase 1g (see Next steps) — bundling it in here would reopen the exact encoder-adaptation confound `load_and_freeze_encoder()` was built to eliminate.

**Threshold calibration, not a guess — and fit on the train side, not on the set it's evaluated on.** Two methods, both computed from `search()`'s per-neighbor cosine similarity scores (previously discarded everywhere in the pipeline, now kept as `train_top1_sim`/`val_scores`/`test_scores`): a quintile-bucket table of mean `label_agreement@5` vs. top-1 similarity, sanity-checked against a random baseline, plus an ROC/Youden-J threshold treating "is the single nearest neighbor itself label-relevant" as a binary target and top-1 similarity as its score. Both are fit on **train-vs-train retrieval** (self- and same-study-excluded, Section 8 of the notebook) and then *applied* to val and test. The first draft of this section fit Youden's J on `val_labels` and then measured gated AUROC on the same val set — caught in the 2026-09-05 audit before it ever ran, so no number was contaminated. Val's bucket table is still printed as an out-of-sample check of the train-calibrated cutoff. If the two methods disagree substantially, that's flagged as a finding, not silently resolved.

**Run 2026-09-06 (audit rerun):** threshold 0.9919, 64.9% val / 66.9% test flagged, gated AUROC 0.6912 val / 0.6797 test vs 0.7302 / 0.7159 ungated — gating hurts, and the diagnosis (zeroed evidence is out-of-distribution for a head trained only with evidence) is in the audit-rerun section. Original write-up of the design follows. New code: `src/retrieval/confidence.py` (`confidence_mask`, `flag_low_confidence`), `src/retrieval/metrics.py:label_agreement_by_similarity_bucket`, 12 new unit tests (all passing locally, including a corrected invariant: zero-confidence must exactly equal `head(encoder(images))`, not a zeroed-evidence-tensor forward pass — those are *not* equivalent, since `label_proj`'s bias means a zeroed evidence tensor still produces a non-zero attention context). Notebook Section 12 (`cxr_recall_arm1_kaggle.ipynb`) is ready to run on top of an already-completed Section 10.

## Phase 1 audit rerun (2026-09-06): first held-out test numbers, reproducibility claim withdrawn, gating negative

Full rerun of `cxr_recall_arm1_kaggle.ipynb` (Sections 8–12, `SEED=42`, torch 2.10.0+cu128, ~3.8 h) under the audit fixes: test-set evaluation of every reloaded best checkpoint, study-level exclusion of training evidence, train-side threshold calibration. Nothing in any training loop, loss, model, or hyperparameter changed.

| Arm | val macro AUROC (selected on) | **TEST macro AUROC (held out)** |
|---|---|---|
| Arm 1 — no retrieval | 0.7272 (best epoch 5/20, stopped at 10) | **0.7051** |
| Arm 2 — retrieval quality, label_agreement@5 / MRR | 0.2954 / 0.4059 (random 0.1845) | **0.2864 / 0.3866** |
| Arm 3c — joint attention, own encoder | 0.7304 (best epoch 7/20, stopped at 12) | **0.7098** |
| Arm 3c-frozen — Arm 1's encoder, frozen | 0.7302 (best epoch 7/20, stopped at 12) | **0.7159** |
| margin: Arm 3c − Arm 1 | +0.0031 | **+0.0047** |
| margin: Arm 3c-frozen − Arm 1 | +0.0030 | **+0.0108** |
| evidence-ablation shift, Arm 3c (full / images-zeroed) | 0.0816 / 0.0810 → 99% image reliance | |
| evidence-ablation shift, Arm 3c-frozen (full / images-zeroed) | 0.0418 / 0.0397 → 95% image reliance | |
| `LEAKAGE CHECK` (same-study view in old top-5) | 52/5977 = 0.9% (matches the local measurement exactly) | |
| Phase 2 attempt 1 replication (Section 11) | full 0.9946 / image-only 0.5322 / text-only 0.9946 | |

**1. The seed does not reproduce Arm 1, and the "confirmed reproducible" claim in Phase 1e is withdrawn.** Arm 1's code path is identical between the seeded run and this one (the audit changed nothing before or inside its training loop), yet it scored 0.7092 then and 0.7272 now — a 0.018 swing, larger than the 0.0143 "noise floor" Phase 1d measured between two *unseeded* runs. `set_seed()` pins initialization and shuffle order but not GPU kernel non-determinism (ViT attention/matmul kernels, atomics), and the Kaggle image has moved to torch 2.10 since the seeded run. Arm 1's val AUROC across all four real runs: 0.7335, 0.7192, 0.7092, 0.7272 — a range of 0.024. Every margin this project reports must be read against that.

**2. H1 is directionally consistent, but the magnitude is inside that noise.** Arm 3c-frozen − Arm 1 on val across runs: +0.0168, +0.0261, +0.0030; Arm 3c − Arm 1: −0.0017, +0.0119, +0.0170, +0.0031. Fusion has never lost, and on the one held-out test number it wins by +0.0108 — but Arm 3c-frozen has barely moved between runs (0.7360 / 0.7353 / 0.7302, range 0.006) while Arm 1 swings four times as much, so the margin is mostly a function of which Arm 1 you happened to get. That is the same "fusion is stable, Arm 1 is not" pattern Phase 1e noted as a secondary finding; it now cuts the other way, because a good Arm 1 run makes the margin nearly vanish. The image-dominance ablation finding replicates a fourth time (95–99%). **Multi-seed evaluation (Next steps, item 5) is now the highest-priority Phase 1 item, not optional.**

**3. Retrieval-confidence gating (Phase 1f) is a negative result — and the reason is instructive.** Calibrated on train, Youden's J landed at top-1 similarity **0.9919**, which flags **64.9% of val and 66.9% of test** queries for vision-only fallback. Gated Arm 3c-frozen: **0.6912 val / 0.6797 test**, versus 0.7302 / 0.7159 ungated — and *below Arm 1 itself* (0.7051 test). That last fact is the tell: falling back to "vision-only" here means zeroing the evidence context *inside* Arm 3c-frozen, whose head has only ever seen `query + evidence_context`; a zero context is an out-of-distribution input for that head, exactly the failure Phase 2's gated fusion showed when handed a zero vector it never trained on. So this measures the head's response to an input regime it never saw, not the value of dropping weak evidence. Two things survive as real findings: (a) the bucket tables are monotonic and out-of-sample — label agreement rises 0.26 → 0.51 across train quintiles and 0.21 → 0.43 across val quintiles — so top-1 similarity *is* a genuine relevance signal; (b) even the lowest-similarity quintile sits above the random baseline (0.21 vs 0.18), so "below threshold retrieval looks like chance" is false, and a threshold that flags two-thirds of queries is not a usable policy. The usable similarity range is ~0.97–1.00 (see the audit's compressed-space note). **Follow-up built, not yet run:** a hybrid fallback cell (Section 12) that routes flagged queries to *Arm 1's own prediction* instead of a zeroed Arm 3 — the operationally honest meaning of "fall back to vision-only" — plus train-time gating (item 13) as the principled fix.

**4. Cost note.** Section 11 (Phase 2 attempt 1, superseded) consumed ~1.2 h of the 3.8 h run; it is now behind a `RUN_PHASE2_ATTEMPT1 = False` flag so future reruns skip it.

## Phase 2 attempt 1 (superseded): naive end-to-end fusion

Adds report text as a second modality alongside the image (`MultimodalFusionClassifier`, `src/models/multimodal_classifier.py`, notebook Section 11): `ImageEncoder` + `TextEncoder` (Bio_ClinicalBERT), fused via residual sum (`image_embedding + text_embedding`), trained end-to-end, with a decomposed ablation (zero each modality post-hoc) rather than three separately-trained models — the same methodology already validated for Arm 3's evidence-ablation.

**Two real problems, both confirmed with numbers, not suspected.**

1. **Circularity.** IU X-ray's `mesh_tags` labels are NLM's own MeSH indexing *of the report text itself*. A lexical-overlap diagnostic found **64.0%** of (report, label) pairs have the label's own word literally present in the report — a floor, undercounts due to naive substring matching (MeSH normalizes surface forms the check can't catch), so true overlap is likely higher.
2. **Modality collapse.** Full fusion val macro AUROC = **0.9958**. Text-only ablation (image zeroed) = **0.9958** — identical to 4 decimal places, meaning the image contributes zero marginal AUROC. Image-only ablation (text zeroed) = **0.5328** — barely above chance, and *worse* than Arm 1's standalone **0.7239**. Diagnosis: end-to-end joint training let the trivial text shortcut dominate so completely that the image branch never received gradient pressure to develop useful features, and degraded below its own standalone quality. (Shift magnitudes: zeroing text moves predictions by 0.1022 mean |Δ|; zeroing image, only 0.0689 — consistent with text dominating the fused representation.)

**Verdict: this result is not trustworthy as evidence that text helps.** It's near-certain the 0.9958 is the model reading label words directly out of the report, not learning multimodal diagnostic reasoning. Kept documented here, not deleted, per this project's practice of keeping every result — including superseded ones — reproducible and visible. Motivated the redesign below.

## Phase 2 attempt 2: contrastive dual-tower pretraining (ConVIRT/CLIP-style)

Rebuilds Phase 2 the way the published literature actually does image+text (ConVIRT/CLIP/GLoRIA/BioViL/CheXzero): separate image and text towers (`ImageEncoder`, `TextEncoder`, unchanged), each with a new linear projection head into a shared **512-dim** embedding space, aligned with a symmetric contrastive (InfoNCE) loss — not label classification from text directly. No label-prediction gradient exists during this pretraining stage, so there's no mechanism for one modality to starve the other.

**Confirmed design decisions:** retrieval (Arm 2/3) stays **image-only** — same FAISS mechanism as Phase 1's Arm 2/3, just on the new contrastively-pretrained image embeddings, isolating exactly one variable (the image encoder's pretraining objective) between phases, same discipline as Arm 3c-frozen. Stage 2 classification arms use **frozen** towers (linear-probe style, standard practice in the papers being replicated) — this is what structurally prevents modality collapse from recurring, since a frozen encoder cannot degrade during classifier training.

New code: `src/models/contrastive.py` (`ProjectionHead`, `DualTowerContrastiveModel`, `clip_contrastive_loss`, `ImageTowerOnly`), `src/models/frozen_dual_tower_classifier.py` (Arm 1), `src/models/frozen_fusion_classifier.py` (Arm 3, a new lean class rather than extending `RetrievalFusionClassifier` — that class always owns and runs its own `ImageEncoder`, machinery for a structurally different situation than an encoder frozen and cached *before* this class ever runs), `extract_dual_embeddings` (`src/retrieval/embeddings.py`). 18 new unit tests, all passing locally — including the corrected ablation-equivalence pattern learned in Phase 1f. A separate, self-contained notebook (`notebooks/cxr_recall_phase2_kaggle.ipynb`) is used — kept apart from the Phase 1 notebook so each Kaggle session stays focused; Phase 1's and Phase 2 attempt 1's already-real numbers are hardcoded as reference constants in it rather than recomputed, since regenerating them would cost real GPU time for no new information.

### First run: STAGE1_BATCH_SIZE=16 (reused from Arm 1's config) — batch too small, image tower didn't train

Stage 1 contrastive learning needs many negatives per step to be informative; batch=16 gave only 15. Result: `image→text recall@1 = 0.0014`, `text→image recall@1 = 0.0000`, against a random baseline of `~0.0014` (n=736) — **exactly chance**. Val contrastive loss never improved past epoch 3 while train loss kept dropping (classic overfitting on an uninformative task), so early stopping reverted to a checkpoint barely 3 epochs removed from raw ImageNet+ClinicalBERT init. Downstream: Arm 1 image-only ablation = 0.5745 (still far below Phase 1 Arm 1's 0.7092, though clearly better than attempt 1's collapsed 0.5328), Arm 2 retrieval = 0.2113 (vs. Phase 1's 0.2727), full fusion (0.6907) landed *below* text-only alone (0.7026) — summing a near-random image term with a strong text term actively hurt. **A bug in the notebook's own comparison logic was also caught and fixed here:** Arm 3's query is image-only (mirrors Arm 3c-frozen), so comparing it against Arm 1's *full* image+text number was apples-to-oranges; corrected to compare against Arm 1's image-only ablation. Once corrected, Arm 3 beat Arm 1 image-only by +0.0262 — almost exactly Phase 1's own +0.0261 margin.

Attempting `STAGE1_BATCH_SIZE=64` next OOM'd on a single Kaggle GPU (~14.5GB) — ViT-B/16 + Bio_ClinicalBERT trained jointly, both fully unfrozen, is memory-heavy, and 64 used the entire budget (`Tried to allocate 192.00 MiB. GPU 0 has ... 116.81 MiB free`).

### Second run: STAGE1_BATCH_SIZE=32 + TextEncoder max_length 256→192 — real numbers

Backed off to a batch that fits with headroom, and trimmed `TextEncoder`'s `max_length` (BERT attention memory scales with sequence length squared; 192 vs. 256 only truncates 0.9% of reports, checked against the real token-length distribution: p99=186, max=365, median=64).

**Stage 1:** recall@1 moved off exact chance — `image→text = 0.0027`, `text→image = 0.0068` (vs. random `~0.0014`) — real but still modest; best checkpoint still reverted to epoch 3 (val loss still overfits quickly on this small, ~6k-example dataset). Absolute recall@1 likely has an inherent low ceiling here regardless of training quality: many IU X-ray reports (especially "normal" cases) describe genuinely similar findings in similar language, making exact-match retrieval out of 736 candidates a hard task independent of representation quality.

**Arm 1 (frozen linear probe):**

| | image-only | text-only | full |
|---|---|---|---|
| Phase 2 attempt 1 (naive, collapsed) | 0.5328 | 0.9958 | 0.9958 |
| Phase 2 attempt 2 | **0.6600** | 0.7655 | 0.7506 |
| Phase 1 Arm 1 (reference) | 0.7092 | — | — |

Modality collapse is genuinely resolved (image-only cleared the notebook's own >0.1-improvement threshold over attempt 1, and moved well off chance). But **full fusion is below text-only alone**, replicating the same pattern from the first run (0.7506 < 0.7655, was 0.6907 < 0.7026) — unweighted additive fusion consistently fails to reach its best single modality's ceiling; the weaker modality drags the combined score down rather than being harmlessly ignored.

**Arm 2 (retrieval):** `label_agreement@5 = 0.2155` (vs. Phase 1's 0.2727, still clearly above the 0.1845 random baseline) — barely moved from the first run's 0.2113, despite Arm 1's classification numbers jumping substantially. Plausible reason: a linear probe only needs *some* linearly-decodable signal (can down-weight noisy dimensions); nearest-neighbor retrieval needs the *whole* embedding space to cluster by diagnosis, a stronger requirement that a lightly-trained Stage 1 (still only ~3 real epochs) may not yet satisfy.

**Arm 3 (evidence fusion): retrieval evidence's benefit vanished — 0.6594 vs. Arm 1 image-only's 0.6600, essentially zero (−0.0006), reversing the first run's +0.0262.** Evidence-ablation shift also shrank (0.0059 → 0.0033). Not a contradiction of Phase 1's finding — a direct replication of an already-documented pattern from **Phase 1e**: *"a better encoder needs less help from evidence — evidence reliance and evidence usefulness are not the same axis."* As Arm 1's own image representation strengthened, there was less gap for neighbors to fill; combined with retrieval quality still trailing Phase 1's (0.2155 vs. 0.2727), the evidence available no longer clears the bar. Whether evidence would help on top of a more fully-converged Stage 1 remains open.

### Verdict on the two tracked conclusions, as of this (second) run — superseded by the third run below, kept for the record

1. **Does retrieval-evidence fusion improve classification?** Phase 1: yes, clearly (+0.0261). Phase 2 (this run): no, negligible (−0.0006) — explained above as encoder-quality-dependent, not a contradiction. **The third run below reverses this specific number to +0.0129, positive again**, once Stage 1 was better trained — supporting the "encoder-quality-dependent margin" explanation directly rather than leaving it as a hypothesis.
2. **Does adding text improve decision-making?** Mechanically yes — full fusion (0.7506) beats Phase 1's pure-image baseline (0.7092) by +0.0414. But **text-only alone (0.7655) is the single highest number in the entire comparison**, exceeding full fusion itself — the same signature expected from a model exploiting the 64.0% lexical-overlap circularity floor (unchanged, unaffected by any of this redesign) rather than genuine multimodal clinical reasoning. Additionally, ~21.7% of val reports are exact-duplicate text also present in train (concentrated 87.1% in the trivially-easy "normal" class vs. 21.5% among non-duplicates) — a smaller, secondary memorization confound bounded to roughly +0.01–0.015 of macro AUROC given equal-weight averaging across 14 labels, not the primary driver but a real, additional reason to distrust the raw text-only number as evidence of clinical reasoning. **This pattern (full below text-only) held again in the third run and is now considered robust, not run-specific noise.**

**Bottom line (second run):** the contrastive redesign successfully fixed modality collapse (the specific failure mode it was built to fix) and gave a real, if still-developing, image representation. It did not, and structurally could not, fix circularity.

### Third run: STAGE1_BATCH_SIZE=64 + mixed precision (AMP) — best run, plus a gated-fusion test

`STAGE1_BATCH_SIZE=96` (with AMP) still OOM'd (AMP's real savings were smaller than the ~40-50% textbook estimate here — Adam's optimizer state doesn't shrink under autocast, only activations do). Backed off to 64+AMP, which fit. Separately, a dedicated lower LR + higher patience for the image tower (`LR_STAGE1_IMAGE=1e-5`, `PATIENCE_STAGE1=10`) was tried and **reverted** — a real negative result: val loss still peaked at epoch 3 and then climbed steadily for 7 more epochs with no recovery, the same ceiling as every prior run, just with more wasted compute confirming it. This is most likely a property of the ~6k-example dataset size, not something LR/patience tuning on the image tower alone can move.

**Stage 1:** `recall@1 = 0.0057` both directions (~4x the random baseline) — the best cross-modal alignment across every run, though the contrastive loss itself still overfit by epoch 3-4 as usual. The key insight: even though *when* it overfits doesn't change, *how much useful signal per epoch* scales with batch size — more negatives per step produced meaningfully better representations within the same limited window before overfitting.

**Arm 1 (ungated):**

| | image-only | text-only | full |
|---|---|---|---|
| Phase 2 attempt 2, run 3 (batch=64+AMP) | **0.7042** | 0.8280 | 0.8045 |
| Phase 1 Arm 1 (reference) | 0.7092 | — | — |

Image-only (0.7042) now essentially matches Phase 1 Arm 1 (0.7092, gap −0.0050) — the first time contrastive pretraining produced an image encoder as good as Phase 1's dedicated supervised fine-tuning. Full fusion (0.8045) again lands below text-only alone (0.8280) — the third consecutive run to show this pattern.

**Arm 2:** `label_agreement@5 = 0.2651`, nearly matching Phase 1's 0.2727 (up from ~0.21 in both prior runs) — confirms the stronger Stage 1 representation improved retrieval geometry, not just linear-probe separability.

**Arm 3:** `0.7171`, a gain of **+0.0129** over Arm 1 image-only — the first Phase 2 run where retrieval evidence showed a clean, positive margin (previous runs: +0.0262 with a badly undertrained Stage 1 that also happened to be apples-to-oranges before a bug fix, then ≈0 with a moderately-trained Stage 1). Consistent direction with Phase 1's +0.0261.

**Gated fusion (Fix #3) — a real result, and a caught misinterpretation.** Trained alongside the ungated model in the same run: full=0.7566, image-only ablation=0.6983, text-only ablation=0.5858. The notebook's own automated comparison declared this "fixed" the additive-fusion weakness, since 0.7566 beats both ablation numbers — **but that comparison is misleading.** Gated text-only (0.5858, barely above chance) uses the *same* text embeddings that scored 0.8280 ungated; the only difference is the gate's response to a zeroed image vector, and the gate never saw a zeroed modality during training (both modalities were always present in every training example). Its behavior on that input is out-of-distribution, exactly the caveat documented when the gate was built (`src/models/frozen_dual_tower_classifier.py`) — this run supplies concrete evidence the caveat was real, not just a theoretical risk.

**The comparison that isn't corrupted by this — gated full vs. ungated full, both using real, non-ablated inputs — shows gating made real fusion performance *worse*: 0.7566 vs. 0.8045.** ~~Fix #3, as built, is a regression on the metric that matters, not an improvement.~~ **Retracted 2026-11-10:** that comparison used linear probes that had hit the 20-epoch cap while still improving, and the gated head has about 1,000 extra parameters, so it converges more slowly. With truly converged probes the two heads are within noise (val 0.9440 vs 0.9406, test 0.9404 vs 0.9325) — no regression; see "Phase 2 rerun at batch 64". A trustworthy gate would need to see zeroed-modality inputs during training too (modality dropout) so its behavior there is in-distribution — this connects to the smaller fix (modality dropout) proposed earlier in the investigation and never built; it would need to be combined with gating, not just gating alone, to actually test whether adaptive weighting helps.

**Bottom line (all three runs):** Fix #1 (bigger batch + AMP) is confirmed to be the lever that mattered — recall@1, Arm 1 image-only, Arm 2 retrieval, and Arm 3's evidence gain all improved monotonically with batch size (16 → 32 → 64) despite Stage 1's own loss always overfitting by epoch ~3-4. Fix #2 (LR/patience tuning) was a genuine negative result. Fix #3 (gated fusion) first looked like a negative result once measured correctly (**retracted 2026-11-10:** that was under-converged probes; see "Phase 2 rerun at batch 64"). None of the three fixes touch circularity, which remains the dominant, unaddressed explanation for text's apparent advantage across every run.

## Pipeline audit (2026-09-05): three fixes, rerun pending

A full walk-through of both notebooks and `src/`, with the data-level claims checked against the raw `indiana_reports.csv`/`indiana_projections.csv` rather than assumed. The `src/` code and all tests were clean; every issue was in the notebooks' pipeline design. Three were fixed immediately (no retraining needed for the first and third); the numbers above have **not** been regenerated yet.

**1. The test split was never used — every reported AUROC is a val number selected on val.** Both notebooks build `test_df` (743 images) and discard it. Each arm reports `best_auroc`, the maximum val AUROC over up to 20 epochs, chosen by early stopping on that same val set — an optimistic selection bias on the order of +0.01–0.02, the same size as the margins the conclusions rest on. Val is also thin per label (Diaphragm has 17 positives, Cicatrix 22). *Fix:* both notebooks now evaluate each arm's reloaded best checkpoint exactly once on `test_df` (`held_out_auroc`), and every comparison cell prints val and TEST side by side with the test margin labelled as the number to report. Arm 2's retrieval metrics and Phase 1f's gated numbers are reported on test too. Phase 2 attempt 1 (superseded) was not extended. **Phase 1 rerun done 2026-09-06** — test numbers and the reproducibility finding are in "Phase 1 audit rerun"; Phase 2 rerun still pending.

**2. Same-study leakage into Arm 3's training evidence.** 3,210 of 3,851 studies have two views (frontal + lateral) and 181 have three, all sharing one report and one label vector. Train-vs-train retrieval excluded only the query's own row (`j != i`), so a query's study-mate stayed in the pool — a neighbor whose label token *is* the query's label. Val/test queries can never retrieve a study-mate (the split is study-level), so the evidence encoder trained under a regime it never sees at eval time. *Fix:* exclusion is now by `patient_id` (`src/retrieval/faiss_index.py:search_excluding_study`, mirrored inline in both notebooks), and both notebooks print a `LEAKAGE CHECK` line reporting how many train queries actually had a same-study view in their old top-5. **Measured for Phase 1 (2026-09-06)** from the saved `arm2_train_embeddings_real.npy` / `arm2_retrieval_real.index` of the completed seeded run (rows verified identical to the local `SEED=42` train split, labels equal row-for-row, Arm 3c-frozen's backbone verified byte-identical to Arm 1's): **52 of 5,977 train queries (0.9%) had a study-mate in their top-5; 52 of 29,885 evidence slots (0.2%) change under the fix.** When it did fire the study-mate was the #1 neighbor 54% of the time, with similarity 0.9888 vs 0.9877 for the best legitimate neighbor — so the leak was a label-perfect neighbor (agreement 1.00 vs 0.35 for legitimate neighbors) but it reached only ~1% of training queries: frontal and lateral views land far apart in ViT space. Phase 1's Arm 3 conclusions are not materially affected. Phase 2's rate (contrastive image space, 512-dim) is still unmeasured — it could differ, since that space was trained to align with report text shared by both views. `same_study_rate()` is the reusable diagnostic; 6 new tests.

**3. Phase 1f calibrated its threshold on val labels, then evaluated on val.** Never ran, so nothing was contaminated. *Fix:* calibrate on train-vs-train retrieval (`train_top1_sim`), apply to val and test (see Phase 1f above).

Also corrected in the same pass: the Phase 2 notebook's gated-fusion auto-verdict compared gated-full against the *corrupted* gated-ablation numbers (the exact misreading documented in the third-run section above) — it now compares gated-full vs ungated-full on test.

**Found, not yet fixed** (in priority order): (a) Stage 1's contrastive loss has in-batch false negatives — two views of one study in a batch get identical text, and InfoNCE penalizes image A for matching text B; ~16% of batch-32 batches contain such a pair (ConVIRT samples one view per study for this reason). The recall@1 diagnostic's "random ≈ 1/n" baseline is also wrong for the same reason. (b) 40 images have an empty `report_text` (25 reports with both Findings and Impression missing) and enter Stage 1 as pairs with an empty string. (c) The **64.0% lexical-overlap number is not a floor**: substring matching counts `normal` inside `abnormal` (808 of 2,142 normal-tagged train reports contain "abnormal"); word-boundary matching gives 53.9%. It is simultaneously a genuine *under*count for MeSH-normalized tags (Pulmonary Atelectasis, Calcinosis, Cicatrix score 0.0 because reports say "atelectasis"/"calcification"/"scarring"). Treat it as a rough estimate in both directions — the circularity conclusion survives (`normal` is 0.90 even word-bounded), the "floor" framing used above does not. (d) `build_tags` splits on `,` when a `Problems` value has no `;`, breaking single MeSH terms — "Catheters, Indwelling", "Fractures, Bone", "Aorta, Thoracic", "Hernia, Hiatal" (64 reports) — into two bogus tags. (e) The "patient-level" split is really study-level: `uid` is a report ID and the Kaggle mirror has no cross-study patient linkage. (g) Three exact-duplicate image pairs exist in the train split (6 rows: `1015_IM-0001-1001`≡`1015_IM-0013-1001`, `1015_IM-0001-2001`≡`1015_IM-0013-2001`, `3245_IM-1537-1001`≡`3245_IM-1537-4001` — identical embeddings, found while measuring the leakage rate), all within one study, so the study-level exclusion already handles them; a cross-split duplicate check has not been run. (h) The Arm-1 fine-tuned embedding space is extremely compressed: the top non-self neighbor's cosine similarity has median 0.990 and 1st percentile 0.979 across all 5,977 train queries. Retrieval still beats chance (Arm 2), but the usable similarity range is ~0.02 wide — Phase 1f's threshold will sit in that narrow band, which is why its calibration uses quantile buckets rather than fixed similarity cut-offs. (f) `train.py`/`setup_iu_xray.py` have drifted from the notebooks (no seed, no early stopping, last-epoch checkpoint, and labels from NLM XML `major`/`minor` tags rather than the Kaggle `Problems` column) despite README saying they mirror the same logic.

## Phase 2 audit rerun (2026-09-06): batch-32 configuration, first Phase 2 test numbers, and two notebook defects found

Rerun of `cxr_recall_phase2_kaggle.ipynb` (Sections 7–11) under the audit fixes. **This was the `STAGE1_BATCH_SIZE=32` configuration** (first-epoch contrastive loss 3.44 ≈ ln 32) — the value restored to the file at the user's request — *not* the batch-64+AMP configuration behind the headline numbers in the Project summary, which was run by editing Kaggle directly. It reproduces the documented batch-32 run closely (image-only 0.6723 vs 0.6600 then; text-only 0.7607 vs 0.7655; Arm 2 0.237 vs 0.216), so the headline configuration still has no held-out numbers.

| | val (selected on) | **TEST** |
|---|---|---|
| Stage 1 best val contrastive loss | 3.2896 (epoch 3, stopped at 8); recall@1 0.0041 / 0.0027 (n=736) | |
| Arm 1 full (image+text), ungated | 0.7563 (**epoch 20/20, still rising**) | **0.7474** |
| Arm 1 image-only ablation | 0.6723 | **0.6337** |
| Arm 1 text-only ablation | 0.7607 | **0.7627** |
| Arm 1 gated full | 0.7294 (epoch 20/20, still rising) | 0.7299 |
| Arm 2 label_agreement@5 / MRR | 0.2373 / 0.3719 | **0.2420 / 0.3667** (Phase 1 audit rerun: 0.2864 test) |
| `LEAKAGE CHECK` (contrastive image space) | 44/5977 = 0.7% | |
| Arm 3 (image query + evidence) | 0.6800 (**best epoch 1**, stopped at 6); ablation shift **0.0034** | **0.6359** |
| Arm 3 − Arm 1 image-only ablation | +0.0076 | +0.0023 |

**What holds:** modality collapse stays resolved (image-only 0.63–0.67 vs attempt 1's 0.53); text-only beats full fusion *on test too* (0.7627 vs 0.7474), the fifth run in a row with that signature; the gated head looked like a regression on test as well (0.7299 vs 0.7474) — **retracted 2026-11-10**, an artifact of the same under-converged probes; Phase 2's same-study leakage rate is as negligible as Phase 1's.

**Two defects in the notebook, found from this log, fixed for the next run:**

1. **The Phase 2 linear probes were undertrained in every run so far.** The ungated probe ran all 20 epochs and was still climbing at +0.0015 AUROC/epoch when it hit the cap (0.7549 → 0.7563 in the last epoch); the gated one at +0.004/epoch. Each epoch costs 0.6 s. The cap came from the shared `EPOCHS_REAL=20`, tuned for ViT fine-tuning. *Fix:* probes and Arm 3 now get `EPOCHS_PROBE=400, PATIENCE_PROBE=25` — early stopping decides, and the loop warns if the cap is ever hit. Every Phase 2 Arm 1 number above (and in the earlier runs) should be expected to move up once the probes converge.
2. **Arm 3 was compared against the wrong baseline.** "Arm 1 image-only" was the *ablation of a head trained with text present* — weights tuned for `image + text`, then handed images alone — not a probe trained on images. At the same time Arm 3 peaked at epoch 1 with an evidence-ablation shift of 0.0034 (Phase 1's frozen arm: 0.04), i.e. it barely used the evidence at all. A +0.0023 test "gain" between those two objects is not evidence of anything. *Fix:* dedicated image-only and text-only probes are now trained (seconds each) and Arm 3 is compared to the image-only probe; the ablation numbers stay printed for continuity, with the gap between ablation and probe reported explicitly.

Also new: the cached Stage 1 embeddings are now saved to `/kaggle/working/phase2_stage1_embeddings.npz`, so any future probe-side experiment (including the duplicate-text robustness check and the Findings-only variant's *evaluation*) can run in seconds without repeating Stage 1.

**Val→test gap is larger for image-based Phase 2 numbers** (image-only −0.039, Arm 3 −0.044) than for text (+0.002) or for Phase 1's arms (−0.02). The image-only ablation was not itself selected on, so this is not selection bias; it is either a genuinely harder test split for image models or per-label noise (four test labels have < 30 positives). Worth remembering when reading any single Phase 2 image number.

**Next Phase 2 run (done 2026-11-10, next section):** set `STAGE1_BATCH_SIZE = 64` (the headline configuration; AMP is already on) so the held-out numbers are for the run the Project summary reports, and let the converged probes and the proper baseline decide Conclusion 1 for Phase 2.

## Phase 2 rerun at batch 64 (2026-11-10): capped probes, then a convergence analysis

Two parts, because the first one mislabeled its own probes.

**Part 1 — Kaggle run A** (`STAGE1_BATCH_SIZE=64` + AMP, probes with Adam lr 1e-4, 400-epoch cap, patience 25; output pasted from Section 8). The per-epoch logs show the full, gated and text-only probes **hit the 400-epoch cap still improving** (the notebook printed its own warning), and the image-only probe stopped on a plateau. I initially described these as "converged"; that was wrong.

| Run A (lr 1e-4, capped) | val | TEST |
|---|---|---|
| Full | 0.9069 | 0.9025 |
| Image-only probe | 0.6958 | 0.6553 |
| Text-only probe | 0.9148 | 0.9128 |
| Gated full | 0.9158 | 0.9184 |
| ablation of full: image-only / text-only | 0.6599 / 0.9099 | 0.6439 / 0.9090 |
| Arm 2 label_agreement@5 / MRR | 0.2568 / 0.3795 | 0.2428 / 0.3750 |
| Arm 3 | 0.7137 | 0.6734 |
| `LEAKAGE CHECK` | 85/5977 = 1.4% | |

**Part 2 — convergence analysis on the saved embeddings.** The `phase2_stage1_embeddings.npz` the notebook saves comes from a *different* batch-64 Stage 1 run than the pasted output (retrieval over identical embeddings is deterministic, yet Arm 2 differs: 0.2861 vs 0.2568 val; leakage 1.7% vs 1.4%). Call it run B; its rows match the local `SEED=42` split exactly. On run B the probes were retrained two independent ways — sklearn L-BFGS logistic regression (convex, solved to tolerance, C chosen on val, 0 convergence warnings) and the same torch heads at lr 1e-2 (all early-stopped after 80–200 epochs) — and the two agree to within about 0.002 val AUROC:

| Run B, converged | val | TEST |
|---|---|---|
| Image-only probe (sklearn / torch) | 0.7078 / 0.7075 | **0.6944** / 0.6930 |
| Text-only probe (sklearn / torch) | 0.9423 / 0.9437 | **0.9355** / 0.9386 |
| Full, image+text summed (sklearn / torch) | 0.9395 / 0.9406 | **0.9329** / 0.9325 |
| Full, concatenated (sklearn) | 0.9428 | 0.9311 |
| Gated full (torch) | 0.9440 | **0.9404** |
| Arm 2 label_agreement@5 / MRR | 0.2861 / 0.4173 | **0.2732 / 0.3917** |
| Arm 3, lr 1e-4, 3 seeds | 0.7247 | **0.6972** ± 0.0018 |
| Arm 3, lr 3e-4, 3 seeds | 0.7247 | 0.6957 ± 0.0028 |
| Arm 3, lr 1e-3, 3 seeds | 0.7235 | 0.6846 ± 0.0067 |
| `LEAKAGE CHECK` | 102/5977 = 1.7% | |

For reference, the same run-B probes at lr 1e-4 with the 400-epoch cap scored full 0.9125, text 0.9259, gated 0.9295, image 0.7018 val — the cap cost the text and fusion probes roughly 0.015 to 0.027 val AUROC.

**Findings.**

1. **Every earlier Phase 2 number is superseded**, including the old 0.8045/0.8280/+0.0129 headline and the run-A numbers above. The converged text-only probe is 0.9355 test.
2. **Conclusion 1 for Phase 2: no clear gain.** Against the converged image probe (0.7078 val / 0.6944 test), Arm 3 gains +0.017 val but only **+0.003 test** (lr 1e-4; +0.001 at 3e-4; −0.010 at 1e-3). Higher learning rates make Arm 3 rely on the evidence far more (ablation shift 0.011 → 0.088) and test gets worse. The val gain is mostly selection. Phase 1's +0.0108 test margin is the larger effect.
3. **The text advantage is circularity, not memorization.** The text − image gap is 0.2411 on all test rows and 0.2466 on the 558 test reports whose text is not an exact duplicate of any training report (val 0.2345 vs 0.2423). Text-only still scores 0.9165 on those unseen reports; duplicates cost the text probe only 0.015–0.019 AUROC and the image probe 0.023–0.025.
4. **The image adds essentially nothing once text is present:** sum 0.9329, concatenation 0.9311, gated 0.9404 against text-only 0.9355 — all within about 0.005. The "fusion below text-only" signature shrinks from ~0.010 to a tie.
5. **Gating: no regression** (retracted). Gated 0.9404 vs ungated 0.9325 on test is within noise. The earlier "regression" and the intermediate "gating helps" were both artifacts of probes that had not finished training.
6. **The frozen contrastive image tower is close to Phase 1's Arm 1:** converged image probe 0.6944 test vs Arm 1's 0.7051 (−0.011, inside Arm 1's own ~0.024 run-to-run range). The earlier "weaker by 0.05" and "near parity" claims both rested on unconverged probes; this is the first converged comparison.
7. **Contrastive pretraining did not improve retrieval:** Arm 2 0.2732 test (run B) and 0.2428 (run A) vs Phase 1's 0.2864. The 0.03 spread between two runs at the same setting is the real size of Stage 1 noise.
8. **Same-study leakage stays small:** 1.4% (run A) and 1.7% (run B).

**What I got wrong along the way.** I called run A's probes "converged" twice, retracted the gated "regression", then asserted "gating helps" — each time from a probe that had not finished training. The notebook now trains probes at lr 1e-2 with a 3000-epoch budget and patience 50 (verified locally to early-stop), and the cap warning is to be treated as a hard flag: if it fires, the printed number is a lower bound, not a result.

## Findings-only text variant (2026-11-11): the premise did not hold up

The Findings-only variant — proposed as the one remaining lever on circularity, on the assumption that the Impression is the section that most directly restates the diagnosis — was built as `notebooks/cxr_recall_phase2_findings_only_kaggle.ipynb`. Before building it, a word-level check on the real `SEED=42` split (TF-IDF unigrams+bigrams, one logistic regression per label, C chosen on validation, macro AUROC) showed the assumption is wrong for this dataset:

| Report text | val | TEST | TEST, rows that have Findings |
|---|---|---|---|
| Findings only | 0.9693 | **0.9767** | 0.9892 |
| Impression only | 0.8172 | **0.8347** | 0.8179 |
| Both | 0.9919 | **0.9930** | 0.9937 |

**The label signal lives in the Findings, not the Impression.** The Impression is short (median 5 words against 29 for the Findings) and generic ("no acute disease"); the detailed terms the MeSH tags were assigned from (spine, thoracic vertebrae, calcified granuloma, diaphragm, aorta) are in the Findings. Per-label test AUROC from Findings alone is at least 0.940 for all 14 labels; the Impression wins on only one (pleural effusion, 0.991 vs 0.942). Dropping the Impression costs 0.0163 AUROC at word level. Two more facts: the word "normal" appears in 76.9% of Findings but only 36.9% of Impressions, and 997 of 7,466 rows (13.4%) have no Findings at all, spread evenly across normal and abnormal studies (13.0% vs 13.6%).

**Consequence:** Findings-only text will not remove the shortcut, and a text-only classifier on it should still score far above any image result. More generally, report-derived MeSH labels cannot test whether text helps, whichever section is used; a fair test would need labels that are not derived from the report text, which Open-I does not offer. The experiment is still worth one Kaggle run as a controlled measurement, not as a fix.

**The notebook** (31 cells, smoke-tested locally end to end with stand-in models; not yet run on Kaggle, about 1 hour of GPU) puts the cheap checks first: Section 4 reproduces the word-level table above in seconds; Section 5 repeats it through a frozen, non-contrastive Bio_ClinicalBERT for the three text variants (the controlled same-encoder comparison); Sections 6–9 run the full Phase 2 pipeline with `TEXT_VARIANT` (`'findings'`, `'impression'` or `'full'`) as the report text, with converged probes, scores on all rows / rows with text / rows with non-duplicate text, and Arm 2. Design choices: the 997 rows without Findings stay in the split (so numbers remain comparable) but are excluded from Stage 1 contrastive pairs; Stage 1 masks same-study pairs out of the negatives (the audit fix); `TEXT_VARIANT = 'full'` gives a same-recipe control run. Because of those two Stage 1 changes and Stage 1's run-to-run noise (about 0.01 on probes, 0.03 on retrieval), comparison with the earlier full-text numbers is confounded; the clean comparisons are Section 5 and the control run. Arm 3 is deliberately not rebuilt (it does not depend on the text variant).

## Selective-feature retrieval (2026-11-23): negative, and an oracle shows why

**Idea:** plain retrieval compares the whole image embedding with cosine similarity, so every dimension counts equally, including dimensions that encode other findings, view or devices. Compare only the features that matter for the query instead. `src/retrieval/feature_weighting.py` takes a linear probe fit on TRAIN embeddings and scores each dimension's relevance to the query's predicted findings, `sum_c p_c |W_cd|`. It then retrieves with a weighted cosine in three variants: **soft** weighting (`beta`), **hard** selection (keep only the top fraction of dimensions), and a two-stage **re-rank** of the plain top-20/50 shortlist. Unit tests (`tests/test_feature_weighting.py`, 8) check that `beta=0` and keeping 100% of dimensions both reproduce plain cosine exactly.

**Setup:** run locally on the cached Phase 2 batch-64 image embeddings (run B, `phase2_stage1_embeddings.npz`; the Phase 1 embeddings are no longer on disk). Val/test queries retrieve from the train index, so no study exclusion is needed (the split is study-level). The probe's C (0.1) and the variant were chosen on val; test was used once. Script: `scripts/eval_feature_weighting.py`, log `feature_weighting_p2.log`. Plain cosine reproduces the documented Arm 2 numbers exactly (0.2861 val / 0.2732 test), so the harness matches the notebook.

| label_agreement@5 (val, 746 queries; random 0.1918) | |
|---|---|
| plain cosine | 0.2861 |
| soft beta 0.5 / 1 / 2 | 0.2853 / 0.2810 / 0.2764 |
| hard top 10% / 25% / 50% of dims | 0.2786 / 0.2836 / **0.2869** |
| re-rank plain top-20 or top-50 by hard 50% | 0.2869 (identical to full-corpus hard 50%) |

**Test, chosen variant (hard 50%) vs plain:** 0.2697 vs 0.2732, diff −0.0035, 95% CI [−0.0113, +0.0046], Wilcoxon p=0.44; MRR +0.0001, p=0.96. **No effect.**

**Why it fails, in two checks:**
1. **The weights are almost the same for every query.** Mean pairwise correlation of the per-query weight vectors is 0.976. The probe leans on the same dimensions for all 14 labels, so query-adaptive weighting collapses to one fixed global reweighting.
2. **An oracle doesn't help either.** Computing the weights from the query's *true* labels (not deployable, an upper bound) gives at most 0.2887 vs 0.2861 on val, and nothing on the 484 abnormal-only queries (0.1793 plain, 0.1736–0.1839 for any variant, probe or oracle).

**Conclusion:** in this embedding, finding information isn't held in separate dimensions, so there are no finding-specific features to select, whoever chooses them. A gain would have to come from a better representation (e.g. Phase 1's label-fine-tuned ViT, or region/patch-level features instead of one pooled vector), not from re-weighting the pooled one. **Caveat:** tested on the Phase 2 contrastive image tower only. Phase 1's Arm 1 encoder was fine-tuned directly on the labels and could hold more label-specific dimensions; checking that needs its embeddings saved from a Kaggle run.

## Phase 1 multi-seed (2026-11-24): H1 not supported

`notebooks/cxr_recall_phase1_multiseed_kaggle.ipynb`, run on Kaggle with training seeds 0, 1, 2. The split is fixed at seed 42, so every seed scores the same 743 test images and the margins are paired. Raw per-seed output: `results/phase1_multiseed_results.jsonl`.

| Test | Seed 0 | Seed 1 | Seed 2 | Mean ± std |
|---|---|---|---|---|
| Arm 1 (no retrieval) | 0.7142 | 0.7026 | 0.7226 | 0.7131 ± 0.0100 |
| Arm 3c-frozen | 0.7167 | 0.6936 | 0.7166 | 0.7090 ± 0.0133 |
| **Margin** | +0.0025 | −0.0090 | −0.0060 | **−0.0042 ± 0.0060** |
| Arm 2 label_agreement@5 | 0.2886 | 0.3074 | 0.2913 | 0.2958 |
| Image share of evidence reliance (shift_img / shift_full) | 97.6% | 95.4% | 97.5% | ~97% |
| Same-study leakage rate (excluded) | 1.37% | 1.57% | 1.22% | |
| Best epoch, Arm 1 / Arm 3c-frozen | 4 / 4 | 7 / 3 | 6 / 13 | |

- **Margin:** 95% CI [−0.019, +0.011], paired t-test p = 0.35, positive in 1 of 3 seeds. The val margins were −0.0007, +0.0051, −0.0046.
- **Convergence:** every model early-stopped before the 20-epoch cap (the latest was seed 2's Arm 3c-frozen, best epoch 13, stopped at 18).
- **Seed 0 was run twice.** The first run (baseline 0.7099, Arm 3c-frozen 0.7123, margin +0.0024, 116 min) and the re-run used in the aggregate (0.7142 / 0.7167 / +0.0025, 94 min) differ by GPU nondeterminism. With the first run, the mean margin is still −0.0042 (p = 0.34).
- **Why seeds 3–4 were not run:** the pre-agreed rule was to add them if the signs were mixed. They were, but even two more seeds at the best observed margin (+0.0025) would leave the 5-seed mean near −0.0015. No outcome could support a gain, so the extra ~4 GPU hours would only narrow a null.
- **What holds:** the image-over-label reliance (95–98%) replicated in all 3 seeds. Arm 3c-frozen is not more stable than Arm 1 across seeds (std 0.0133 vs 0.0100), which retracts Phase 1e pattern 1.
- **Interpretation:** the evidence pathway is used (ablation shift 0.03–0.05) but does not improve accuracy. With only ~30% of retrieved neighbours sharing a finding, the evidence is mostly noise. A better retrieval encoder is the obvious next test (`notebooks/cxr_recall_retrieval_encoder_comparison_kaggle.ipynb`).

## Retrieval encoder comparison (2026-11-24): RAD-DINO beats the current encoder for abnormal cases

`notebooks/cxr_recall_retrieval_encoder_comparison_kaggle.ipynb`, run on a second Kaggle account. Same split (5,977 / 746 / 743), gallery = train, k = 5, relevance = at least one shared tag. Strata: tagged (any top-14 tag), normal, abnormal (a tag other than normal), untagged (no top-14 tag; can never match). The reference is a freshly trained seed-0 Arm 1 (val 0.7205, test 0.7155, epoch 7; a third seed-0 run, again different by GPU nondeterminism). Paired bootstrap 95% CI and Wilcoxon against the reference on test. No extraction errors.

| Test | P@5 all | tagged | normal | abnormal | MRR | Jaccard abnormal | median top-1 cos |
|---|---|---|---|---|---|---|---|
| random | 0.195 | 0.241 | 0.358 | 0.131 | | | |
| oracle | 0.808 | 1.0 | 1.0 | 1.0 | | | |
| imagenet_vit_cls | 0.269 | 0.333 | 0.510 | 0.166 | 0.395 | 0.070 | 0.860 |
| imagenet_vit_patchmean | 0.278 | 0.345 | 0.528 | 0.172 | 0.385 | 0.078 | 0.956 |
| **arm1_vit_cls (current)** | 0.293 | 0.363 | 0.542 | 0.194 | 0.385 | 0.081 | 0.987 |
| arm1_vit_patchmean | 0.291 | 0.361 | 0.525 | 0.206 | 0.383 | 0.085 | 0.994 |
| xrv_densenet_chex | 0.255 | 0.316 | 0.468 | 0.173 | 0.379 | 0.071 | 0.979 |
| **raddino_cls** | **0.321** | **0.398** | 0.531 | **0.273** | **0.444** | **0.116** | 0.526 |
| raddino_patchmean | 0.285 | 0.353 | 0.532 | 0.184 | 0.408 | 0.087 | 0.962 |

Paired vs arm1_vit_cls on test (diff, 95% CI, Wilcoxon p):

| | tagged | abnormal |
|---|---|---|
| imagenet_vit_cls | −0.030 [−0.058, −0.004], p = 0.045 | −0.029 [−0.056, −0.001], p = 0.15 |
| imagenet_vit_patchmean | −0.018 [−0.045, +0.007], p = 0.24 | −0.022 [−0.049, +0.005], p = 0.30 |
| arm1_vit_patchmean | −0.002 [−0.020, +0.015], p = 0.74 | +0.012 [−0.012, +0.036], p = 0.51 |
| xrv_densenet_chex | −0.047 [−0.073, −0.022], p = 0.001 | −0.021 [−0.050, +0.006], p = 0.31 |
| **raddino_cls** | **+0.035 [+0.007, +0.061], p = 0.0095** | **+0.078 [+0.046, +0.111], p = 2.4×10⁻⁶** |
| raddino_patchmean | −0.010 [−0.037, +0.015], p = 0.50 | −0.010 [−0.039, +0.019], p = 0.89 |

- **RAD-DINO CLS is the only encoder that beats the current one,** used zero-shot with no Open-I training. The abnormal-query gain (+0.078, ~40% relative) survives a Bonferroni correction for the 12 comparisons (threshold 0.0042); the tagged gain (p = 0.0095) does not. It was also best on val (abnormal 0.254 vs 0.234, tagged 0.385 vs 0.361), so it was not picked on test.
- **Fine-tuning helps the ImageNet ViT** (abnormal 0.166 → 0.194), but not as much as chest-X-ray self-supervised pretraining.
- **The CheXpert-supervised DenseNet is the worst chest-X-ray encoder,** significantly below the current one on tagged queries. Domain pretraining alone isn't enough; its 14 CheXpert classes don't line up with Open-I's MeSH tags.
- **Patch-mean pooling doesn't help** (the external advice to try it is not supported), and for RAD-DINO it is much worse than CLS.
- **Similarity spread:** the current encoder's top-1 cosines cluster at 0.99, so a cosine score carries almost no information for a reviewer. RAD-DINO's median is 0.53, a usable range.
- **Still far from the ceiling:** abnormal P@5 0.27 against 1.0. Whether better retrieval turns into a classification gain is untested (ARCHITECTURE.md D9).

## D9 (2026-11-24): RAD-DINO as the frozen encoder; retrieval ≈ random neighbours

`notebooks/cxr_recall_d9_raddino_retrieval_kaggle.ipynb`. Frozen RAD-DINO [CLS] embeddings (768-d, standardized with train statistics) shared by all arms. Arms: `probe` (linear head), `retrieval` (Arm 3c-frozen design: top-5 RAD-DINO neighbours with study exclusion, joint attention, residual fusion), `random_control` (same, 5 random train cases outside the query's study, redrawn per seed). LR swept over 1e-4/3e-4/1e-3 per arm and seed, selected on val; up to 200 epochs, patience 15, batch 64. Full output: `results/d9_results.json`. Retrieval P@5 val 0.3252 / test 0.3211 (matches the encoder comparison exactly). Leakage check 5.4% (excluded). No selected run hit the cap. ~1 min per seed after a one-off extraction.

| Test | Seed 0 | 1 | 2 | 3 | 4 | Mean ± std |
|---|---|---|---|---|---|---|
| probe | 0.7416 | 0.7506 | 0.7480 | 0.7426 | 0.7475 | 0.7461 ± 0.0038 |
| retrieval | 0.7432 | 0.7498 | 0.7550 | 0.7558 | 0.7590 | 0.7526 ± 0.0062 |
| random_control | 0.7324 | 0.7455 | 0.7520 | 0.7598 | 0.7626 | 0.7505 ± 0.0121 |
| retrieval − probe | +0.0016 | −0.0008 | +0.0070 | +0.0132 | +0.0115 | **+0.0065**, CI [−0.0010, +0.0140], p = 0.075 |
| retrieval − random | +0.0108 | +0.0043 | +0.0030 | −0.0040 | −0.0036 | **+0.0021**, CI [−0.0055, +0.0097], p = 0.49 |
| random − probe | −0.0092 | −0.0051 | +0.0040 | +0.0172 | +0.0151 | +0.0044, CI [−0.0102, +0.0190], p = 0.45 |
| image share (retrieval) | 94.6% | 95.8% | 96.3% | 96.4% | 96.5% | 95.9% |
| chosen LR / best epoch: probe | 1e-4 / 73 | 3e-4 / 20 | 1e-4 / 78 | 1e-3 / 5 | 1e-3 / 5 | |
| retrieval, random_control best epoch | 2, 2 | 3, 3 | 3, 3 | 3, 3 | 3, 3 | |

- **Retrieval is not distinguishable from random neighbours.** The retrieval − probe trend (+0.0065, 4/5 positive) is matched by random evidence in seeds 3–4. Whatever gain the evidence pathway gives is not from similarity.
- **Val overstates the evidence models:** retrieval − probe is +0.0091 on val (all 5 positive) but +0.0065 on test. They peak at epoch 2–3, so selection picks a val spike.
- **Probe LR selection is close to a coin flip:** the three LRs' val scores differ by < 0.001, but the 1e-3 runs (seeds 3–4) test ~0.004 lower than the 1e-4 runs. That pulls the probe down in exactly the seeds where retrieval looks best.
- **The encoder is the real lever:** the RAD-DINO probe beats the fine-tuned ViT Arm 1 (0.7131 ± 0.0100, Phase 1 multi-seed) by +0.033 (Welch p = 0.02, unpaired), with a third of the seed spread.
- **Random-control ablation shift is not ~0 (0.009–0.013):** the model stops at epoch 3, before it could learn to ignore random evidence, and zeroed evidence is out of distribution at eval (as in Phase 1f). Its accuracy, not its shift, is the meaningful control.

## Reliability-aware retrieval, step 0 (2026-11-24): NO-GO

**Question:** a reliability-aware system would decide per query whether to trust its retrieved neighbours. That only helps if retrieval helps some queries and hurts others. `notebooks/cxr_recall_reliability_step0_kaggle.ipynb` repeated D9 exactly (replicated: probe 0.7461, retrieval 0.7526, random 0.7505) and saved every arm's per-query val/test probabilities for 5 seeds, plus six inference-time reliability features per query: top-1 / mean similarity, similarity gap, neighbour label consensus (mean pairwise Jaccard), probe-neighbour agreement, probe entropy. Follow-up checks ran locally on the saved outputs (`scripts/reliability_step0_followup.py`). Results: `results/reliability_step0_results.json`, `results/reliability_step0_followup.json`.

1. **No reliability feature predicts when retrieved neighbours beat random ones.** The per-query loss reduction of retrieval over random neighbours is tiny but real overall (+0.0018, 95% CI [+0.0005, +0.0031]). Its Spearman correlation with every feature is between −0.08 and +0.02, and no feature is significant in the same direction on both val and test.
2. **The apparent calibration win was an undertrained baseline.** Raw test ECE: probe 0.294, retrieval 0.028, random neighbours 0.030. Random neighbours calibrate as well as retrieval, so retrieval isn't the cause. The probe's output layer starts at p = 0.5 and, at its selected LR (1e-4), hadn't moved its probabilities down to the labels' low prevalence. AUROC had converged; the probabilities had not. After Platt scaling per label on val: probe 0.0153, retrieval 0.0169, random 0.0167. Retrieval is slightly *worse* calibrated (+0.0016, p = 0.01).
3. **The ensemble gain is not from retrieval.** Averaging calibrated probabilities (test macro AUROC, 5 seeds):

   | Ensemble | Test AUROC |
   |---|---|
   | probe alone | 0.7460 |
   | probe + probe (another seed) | 0.7518 |
   | probe + **retrieval** | 0.7645 |
   | probe + **random neighbours** | 0.7643 |
   | retrieval + retrieval (another seed) | 0.7637 |

   Probe + retrieval vs probe + random: +0.0002 (p = 0.92). The ~+0.018 gain comes from combining two different model types (a linear probe and an attention model), whatever neighbours the attention model sees. A seed ensemble of the retrieval model gets the same.
4. **Gates don't help.** Routing each query to the probe or the retrieval model by one feature (threshold and direction chosen on val, calibrated probabilities): best is neighbour consensus, +0.0022 over retrieval (p = 0.36). None is significant. (The first, uncalibrated version of this check was unfair: routing mixed the probe's inflated probabilities with the retrieval model's, which breaks the ranking AUROC depends on.)

**Verdict:** NO-GO for reliability-aware retrieval on this data. There is no subset of queries where retrieval reliably helps, so a reliability score or gate has nothing to exploit. Contributions 1, 2 and 4 of the proposal are unsupported; 3 (relevant vs random vs none) is D9. Side lesson, added to the convergence checklist: check that probabilities converge, not only AUROC, before comparing calibration.

## Known deviations from the original proposal's clean experimental design

(Carried forward from earlier in the project, still true and relevant to interpreting these numbers.)

- **All AUROCs reported so far are val-set numbers that were also the early-stopping selection target; the held-out test split was never evaluated on.** Fixed in the 2026-09-05 audit (test evaluation added to both notebooks), rerun pending. See "Pipeline audit" above.
- The split is study-level (`uid` = report ID), not patient-level — the best available on this dataset, but "patient-level" overstates it.

- Arm 3/3b/3c each train their own encoder end-to-end rather than sharing/freezing Arm 1's weights, so their results alone aren't a fully isolated test of the fusion mechanism. **Addressed in Phase 1d** (encoder-sharing fix) **and confirmed under a seeded rerun in Phase 1e.**
- Text (ClinicalBERT) is now exercised (Phase 2 attempts 1 and 2, above) rather than unused, but Phase 1 (Arms 1/2/3 and variants) remains image-only.
- No knowledge-graph/anatomical grounding (ASG-Rad's Stage A was never built, blocked on MIMIC-CXR/Chest ImaGenome).
- A literature check found the core mechanism (fuse retrieved cases into a classifier before it predicts) is established prior art (RAC 2022, X-TRA 2023 specifically for chest X-ray, retrieval-augmented multi-label text classification 2023, RAMDS 2024) predating REMEMBER by 2-3 years — the "first cross-domain extension" framing from the original proposal doesn't hold up. See conversation history / project memory for the full breakdown.

## Next steps (not yet implemented)

1. **~~Richer evidence representation~~ — done, see Phase 1b above.** Fusing neighbor image embeddings alongside labels closed ~82% of Arm 3's gap to Arm 1.
2. **~~Joint (non-pre-fused) attention~~ — done, see Phase 1c above.** Keeping image/label tokens separate closed 94.5% of the original gap and revealed the model self-selecting to trust images over labels (98.6% vs. 81% under early fusion).
3. **~~Encoder-sharing fix~~ — done, see Phase 1d above.** Freezing the encoder to Arm 1's exact weights produced the first outright win for a fusion variant over Arm 1.
4. **~~Seeded rerun~~ — done, see Phase 1e above.** Confirmed under `set_seed(42)`: Arm 3c beats Arm 1 by +0.0170, Arm 3c-frozen by +0.0261, both well above the previously-measured noise floor. H1 is now well-supported for the final architecture.
5. **~~Multi-seed distribution~~ — done 2026-11-24, see "Phase 1 multi-seed": margin −0.0042 ± 0.0060 over 3 seeds, H1 not supported.** Original item: The audit rerun showed `SEED=42` does not reproduce Arm 1 (0.7092 → 0.7272 on identical code) and that the H1 margin (+0.0030 val / +0.0108 test) sits inside Arm 1's ~0.024 run-to-run range. Run Arm 1 and Arm 3c-frozen under 3–5 seeds, report mean ± std and per-seed test margins; only then state a magnitude for H1. **Built:** `notebooks/cxr_recall_phase1_multiseed_kaggle.ipynb` — fixed `seed=42` split (same 743 test images every seed, margins paired), Arm 1 → retrieval → Arm 3c-frozen per seed (~1.7 h/seed), `SEEDS=[0,1,2]` per session with `PRIOR_RESULTS` for merging a second session's `[3,4]`, aggregate cell prints mean ± std, 95% CI, sign count and a paired t-test on the test margin. Control flow smoke-tested locally on fake data; not yet run on Kaggle.
6. **Images-only evidence variant.** Since labels contribute only ~1-3% of evidence reliance across all three runs where Arm 3c/3c-frozen were tested, dropping label tokens entirely might match or slightly exceed these results by removing near-dead-weight tokens. Cheap to test, not yet built.
7. **~~Phase 2 (naive fusion)~~ — done, see "Phase 2 attempt 1" above.** Real numbers obtained; diagnosed as modality collapse + a 64% lexical-overlap circularity floor, not evidence text helps.
8. **~~Phase 2 attempt 2: contrastive dual-tower pretraining~~ — done, see above.** Real numbers obtained across two runs (batch-size-fix iteration included). Modality collapse resolved. Retrieval-evidence fusion (Conclusion 1) did not replicate Phase 1's positive result in this run — explained by the "better encoder needs less evidence" pattern from Phase 1e plus retrieval quality still trailing Phase 1's, not a contradiction. Text still appears to help (Conclusion 2) but the circularity floor (64.0%) plus a newly-quantified train/val duplicate-text overlap (21.7%, 87.1% concentrated in the "normal" class) both remain unaddressed by this redesign — text-only alone is still the single highest number in the whole comparison.
9. **~~Stage 1 undertraining~~ — resolved for Fix #1, a genuine negative result for Fix #2.** Best checkpoint reverted to epoch 3-4 in every run (val contrastive loss overfits quickly on ~6k examples) — this itself never went away, but the *quality* reached before overfitting scaled with batch size. **Fix #1 (mixed precision + bigger batch) — kept, confirmed working across 3 batch sizes.** `STAGE1_BATCH_SIZE=64` (no AMP) OOM'd; added `torch.amp.autocast`/`GradScaler`; 96 still OOM'd (AMP's real savings were smaller than the ~40-50% textbook estimate — Adam's optimizer state doesn't shrink under autocast, only activations do); landed on 64+AMP. **Real result, monotonic with batch size (16→32→64): recall@1 0.0014→0.0068→0.0057 (all directions), Arm 1 image-only 0.5745→0.6600→0.7042 (now essentially matching Phase 1 Arm 1's 0.7092), Arm 2 retrieval 0.2113→0.2155→0.2651 (nearly matching Phase 1's 0.2727), Arm 3 evidence gain +0.0262→−0.0006→+0.0129.** See "Phase 2 attempt 2, third run" above. **Fix #2 (`LR_STAGE1_IMAGE=1e-5` + `PATIENCE_STAGE1=10`) — tried, reverted, a genuine negative result.** Val loss still peaked at epoch 3 and climbed steadily for 7 more epochs with no recovery — same ceiling as every other run, just more wasted compute confirming it. Reverted to the shared `LR`/`PATIENCE`.
9b. **~~Additive-fusion weakness~~ — fix built and tested; first read as a negative result, RETRACTED 2026-11-10 (see "Phase 2 rerun at batch 64"): the "regression" came from under-converged probes, and at true convergence gated ≈ ungated (0.9404 vs 0.9325 test). Original, superseded write-up follows.** `FrozenDualTowerClassifier` gained an opt-in `use_gating` parameter (`src/models/frozen_dual_tower_classifier.py`, 4 new passing tests, default `use_gating=False` reproduces prior behavior byte-for-byte). Real run: full=0.7566, image-only ablation=0.6983, text-only ablation=0.5858 — the notebook's own automated comparison called this "fixed" since 0.7566 beats both ablation numbers, **but that's a false positive**: gated text-only collapsed to 0.5858 (barely above chance) on the *same* text embeddings that scored 0.8280 ungated, because the gate never saw a zeroed modality during training (both modalities were always present) and its response to one is out-of-distribution — exactly the documented caveat, now confirmed with concrete evidence. **The uncorrupted comparison — gated full (0.7566) vs. ungated full (0.8045), both on real inputs — shows gating made real fusion performance *worse*.** A trustworthy gate would need modality dropout during training (zeroing one modality some fraction of training steps, not just at eval-time ablation) so its zero-vector behavior is in-distribution — not built, a candidate follow-up if gating is revisited.
10. **Findings-only text variant — built 2026-11-11 (`cxr_recall_phase2_findings_only_kaggle.ipynb`), not yet run on Kaggle; a word-level pre-check says it will not remove the shortcut (Findings alone recovers the labels at 0.9767 test AUROC, Impression alone 0.8347 — see "Findings-only text variant").** Since contrastive pretraining structurally cannot fix circularity, the only remaining lever is changing what text the model sees — dropping the Impression section (closest to directly restating the diagnosis) and training/evaluating on Findings alone, as a new comparison point alongside both Phase 2 attempts.
11. **Duplicate-text robustness check (proposed, not yet built).** Re-evaluate Phase 2 Arm 1's text-only/full AUROC restricted to the ~584 val examples whose report text is *not* an exact duplicate of anything in train, to isolate genuine generalization from the 21.7%-of-val memorization risk quantified above.
12. **~~Retrieval-confidence gating~~ — run 2026-09-06, negative (see Phase 1 audit rerun, point 3).** Eval-time zeroing of evidence on a head trained only with evidence is out-of-distribution; the result measures that, not the value of gating. 12b. **Hybrid Arm-1 fallback (built, not yet run):** Section 12 now also scores flagged queries with Arm 1's own model instead of a zeroed Arm 3 — runs in the next Phase 1 rerun at no training cost.
14. **~~Rerun Phase 1 under the audit fixes~~ — done 2026-09-06. ~~Rerun Phase 2~~ — done at batch 32 (see Phase 2 audit rerun; leakage 0.7%). ~~Phase 2 at `STAGE1_BATCH_SIZE=64` with converged probes~~ — done 2026-11-10 (Kaggle run plus a convergence analysis).** Against a properly converged image probe the Arm 3 gain is only about +0.003 on test (see "Phase 2 rerun at batch 64").
15. **Remaining audit items** (see "Pipeline audit"): mask same-study pairs in Stage 1's contrastive loss, drop empty-text pairs, replace the substring lexical-overlap check with word-boundary matching and reword "floor", fix `build_tags` comma-splitting, rename the split as study-level, and sync or re-describe `train.py`/`setup_iu_xray.py`.
16. **~~Selective-feature retrieval~~ — run 2026-11-23 on Phase 2 embeddings, negative (see "Selective-feature retrieval").** Open: the same check on Phase 1 Arm 1 embeddings, which need to be saved from a Kaggle run first.
13. **Phase 1g (candidate, now motivated by 12's negative result): train-time confidence gating.** Phase 1f only gates at eval time on an already-trained checkpoint. Training Arm 3c-frozen with gated evidence from the start (so the model learns under the same mixed confident/unconfident regime it sees at inference) is a legitimate follow-up, deliberately not built alongside Phase 1f to avoid reopening the encoder-adaptation confound `load_and_freeze_encoder()` exists to eliminate.
