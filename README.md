# CXR-Recall

A retrieval-augmented chest X-ray classifier. Before it predicts, the model
retrieves the five most similar past cases and reads their images and
diagnoses as evidence, then returns a 14-label diagnosis together with the
cases it relied on.

The evidence-encoder idea follows
[REMEMBER](https://arxiv.org/abs/2504.09354) (Can et al., ACM Multimedia 2025),
a retrieval-based framework for Alzheimer's diagnosis from brain MRI. This
project applies the idea to chest X-rays and tests whether retrieval actually
helps.

**Design:** frozen ViT-B/16 encoder, then FAISS retrieval (leakage-safe),
then joint cross-attention over separate image and label tokens for each
neighbour, then residual fusion, then a linear head. See
**[ARCHITECTURE.md](ARCHITECTURE.md)** for the component diagram, data flow,
and a decision record for every design choice, each backed by an experiment.
The full experiment logs are in [RESULTS.md](RESULTS.md).

**Key design findings**
- Neighbours must carry their **images**, not just labels: label-only
  evidence hurt accuracy, and adding image embeddings closed ~82% of the gap.
- Given separate image and label tokens, the model puts **~95-99% of its
  evidence reliance on neighbour images**, replicated in every run.
- **Freezing the baseline's encoder** isolates the fusion mechanism and gave
  the most stable variant across runs.
- Report text is **excluded by design**: Open-I's labels are derived from the
  reports, so text models read their own answer (0.9355 test AUROC from text
  alone, with the image adding nothing).
- **Rejected after testing:** evaluation-time confidence gating (hurt
  accuracy) and selective-feature retrieval (no gain, even with an oracle).
- **Multi-seed check:** across 3 seeds the full design does not beat the
  same encoder without retrieval (see below).

**Measured performance** (743 held-out test images, 3 training seeds,
paired): baseline 0.7131 ± 0.0100 vs CXR-Recall 0.7090 ± 0.0133 macro
AUROC; gain −0.0042, 95% CI [−0.019, +0.011], p = 0.35. **Retrieval gives
no measurable gain on this dataset.** Earlier single-run wins of up to
+0.026 were inside the baseline's run-to-run spread. The likely limit is
retrieval quality: only ~30% of retrieved neighbours share a finding with
the query. Details in ARCHITECTURE.md section 4.

**Better encoder (D9):** swapping in frozen RAD-DINO (a chest-X-ray
foundation model, not trained on Open-I) lifts the plain classifier to
0.7461 ± 0.0038 (+0.033 over the ViT, 5 seeds). Retrieval on top of it gives
+0.0065 (p = 0.075), but a control with **random** neighbours does about as
well (retrieval − random +0.0021, p = 0.49). So the encoder matters, while
finding similar cases does not measurably help the diagnosis on this dataset.

**Scope:** Indiana University chest X-rays (Open-I) only, with no
credentialing required. Labels are Open-I's MeSH `Problems` tags
(`src/data/label_schema.py:build_mesh_vocab`). Training runs on **Kaggle**
(`notebooks/`, dataset `raddar/chest-xrays-indiana-university`). The local
`train.py`/`retrieve.py` scripts mirror the same logic for local GPU use.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the unit tests (no data required):

```bash
pytest tests/
```

Fetch IU X-ray and build its manifest (downloads ~2GB from the public NLM
Open-i archive):

```bash
python scripts/setup_iu_xray.py --data-dir data/iu_xray
```

Smoke-test the full pipeline on a small subset before a real run:

```bash
python train.py --config configs/arm1_iu_xray.yaml --epochs 1 --limit 50
```

Then a real run:

```bash
python train.py --config configs/arm1_iu_xray.yaml
```

Build the Arm-2 retrieval index and evaluate retrieval quality against a
trained Arm-1 checkpoint:

```bash
python retrieve.py --config configs/arm1_iu_xray.yaml \
    --checkpoint checkpoints/arm1_iu_xray/arm1_final.pt --k 5
```

## Layout

```
src/data/       dataset classes, label schema, patient-level splitting
src/models/     image/text encoders, Arm-1 classifier, Arm-3 evidence fusion,
                Phase 2 attempt 1 (multimodal_classifier.py, superseded),
                Phase 2 attempt 2 contrastive dual-tower (contrastive.py,
                frozen_dual_tower_classifier.py, frozen_fusion_classifier.py)
src/retrieval/  FAISS index, embedding extraction (incl. dual-modality),
                retrieval-quality metrics, similarity-confidence gating
scripts/        one-off setup/download scripts (never run automatically)
notebooks/      self-contained Kaggle notebooks — this is where training
                actually runs; local scripts mirror the same logic.
                cxr_recall_arm1_kaggle.ipynb: Phase 1 + Phase 2 attempt 1 +
                retrieval-confidence gating. cxr_recall_phase2_kaggle.ipynb:
                Phase 2 attempt 2 (contrastive), standalone/self-contained.
                cxr_recall_phase1_multiseed_kaggle.ipynb: Arm 1 vs Arm 3c-frozen
                over several seeds, paired test margins with CI.
                cxr_recall_phase2_findings_only_kaggle.ipynb: Phase 2 with
                Findings-only / Impression-only report text, plus lexical-ceiling
                diagnostics
configs/        one YAML per experiment
tests/          unit tests runnable without any downloaded data
```

## Roadmap (maps to the proposal's phases, Section 7, adapted to the Open-I-only scope)

- [x] Phase 0/1 bootstrap: repo, label schema, patient splits, IU X-ray dataset
- [x] Arm 1 (no-retrieval) model — real run, best val macro AUROC **0.7335** (early-stopped, epoch 3/20)
- [x] Arm 2: post-hoc retrieval index (FAISS) + retrieval-quality metrics — label_agreement@5 **0.2973** vs. 0.1845 random baseline (clearly beats chance); display-only, does not influence the diagnosis, matching RadFuse's design
- [x] Arm 3: CXR-Recall — evidence encoder + cross-attention fusion — best val macro AUROC **0.7025** (early-stopped, epoch 7/20), does **not** beat Arm 1
- [x] Causal evidence-ablation test — mean |shift| **0.0516**, confirms the model genuinely uses the retrieved evidence (not ignoring it) even though doing so doesn't help accuracy
- [x] Full 3-way real-training comparison — see [RESULTS.md](RESULTS.md) for the complete write-up and interpretation
- [x] Arm 3b: richer evidence, early fusion (neighbor image embeddings + labels blended before attention) — best val macro AUROC **0.7278** (early-stopped, epoch 3/20), closes ~82% of Arm 3's gap to Arm 1; two-part ablation confirms image embeddings account for ~81% of total evidence reliance
- [x] Arm 3c: joint attention (image and label kept as separate, tagged tokens, not pre-blended) — best val macro AUROC **0.7318** (early-stopped, epoch 5/20), closes **94.5%** of Arm 3's gap to Arm 1; ablation shows the model self-selected to rely on images for **98.6%** of its evidence reliance
- [x] Arm 3c-frozen: encoder-sharing fix (`load_and_freeze_encoder()`, byte-identical to Arm 1's weights) — best val macro AUROC **0.7360**, first fusion variant to beat Arm 1 outright *within its own run* (Arm 1 that run: 0.7192, Arm 3c that run: 0.7311)
- [x] Discovered and fixed a reproducibility gap: no seed was set anywhere except the data split, so Arm 1 alone scored 0.7335 vs. 0.7192 across two runs of identical code — `set_seed()` now called at notebook start and before each training block
- [x] **Seeded rerun — H1 confirmed.** SEED=42: Arm 1 = 0.7092, Arm 3c = 0.7262 (**+0.0170**), Arm 3c-frozen = 0.7353 (**+0.0261**) — both margins well above the 0.0143 noise floor, and consistent in direction with the two prior unseeded runs. Three independent runs now agree: fusion beats no-retrieval for the final architecture.
- [ ] **Multi-seed distribution — now required** (2026-09-06: the seed does not reproduce Arm 1, and the H1 margin +0.0030 val / +0.0108 test is inside Arm 1's ~0.024 run-to-run range; 3–5 seeds of Arm 1 + Arm 3c-frozen, report mean ± std) -- built as `notebooks/cxr_recall_phase1_multiseed_kaggle.ipynb`, not yet run
- [ ] Images-only evidence variant (drop label tokens entirely, since they contribute only ~1-3% of evidence reliance across every run tested) — cheap to test, not yet built
- [x] Phase 2 attempt 1 (superseded): naive end-to-end text fusion -- `MultimodalFusionClassifier` (`src/models/multimodal_classifier.py`, notebook Section 11). Real numbers: full=0.9958, text-only=0.9958, image-only=0.5328 (worse than Arm 1's 0.7239) -- diagnosed as modality collapse, plus a 64.0% lexical-overlap circularity floor. Not deleted, motivated the redesign below -- see RESULTS.md.
- [x] Phase 2 attempt 2: contrastive dual-tower pretraining (ConVIRT/CLIP-style), frozen linear-probe classification arms mirroring Phase 1's Arm 1/2/3 -- `src/models/contrastive.py`, `frozen_dual_tower_classifier.py`, `frozen_fusion_classifier.py`, standalone `notebooks/cxr_recall_phase2_kaggle.ipynb`. Best real run (batch=64+AMP): modality collapse resolved, image-only 0.7042 now essentially matches Phase 1 Arm 1's 0.7092; retrieval-evidence fusion +0.0129 (positive, same direction as Phase 1's +0.0261); text still the single highest number (text-only 0.8280 > full fusion 0.8045), consistent with the 64.0% circularity floor and 21.7% train/val duplicate-text overlap dominating over genuine multimodal reasoning. A gated-fusion follow-up (opt-in `use_gating`) first looked like a regression, but that was an artifact of under-converged probes -- at true convergence gated and ungated fusion are within noise (0.9404 vs 0.9325 test). **Superseded 2026-11-10 (batch-64 embeddings, truly converged probes, held-out test):** image-only probe 0.6944, text-only probe 0.9355, full 0.9329, gated 0.9404; Arm 3 gains only about +0.003 over the image probe (Phase 1: +0.0108) -- the earlier 0.7042/0.8280/+0.0129 and +0.0181 figures came from under-trained probes. See RESULTS.md for the full project summary and both conclusions.
- [ ] Retrieval-confidence gating: similarity-threshold fallback to vision-only + human-review flag for weak retrieval matches -- `evidence_confidence` param on `RetrievalFusionClassifier`, calibration via quintile-bucket + ROC/Youden threshold fit on train-side retrieval (`src/retrieval/confidence.py`, `src/retrieval/metrics.py`), notebook Section 12 ready to run on Kaggle; no real results yet -- see RESULTS.md Phase 1f
- [x] **Pipeline audit (2026-09-05)** -- found that every reported AUROC was a val number selected on val (test split never evaluated), that Arm 3's training evidence could include the query's own study-mate (same report, same labels -- 85% of studies have 2+ views), and that Phase 1f's threshold was to be calibrated on its own eval set. All three fixed in both notebooks (`held_out_auroc` test evaluation per arm, `search_excluding_study`/`same_study_rate` in `src/retrieval/faiss_index.py` with 6 tests, train-side calibration). **Numbers in RESULTS.md are not yet regenerated under these fixes** -- see RESULTS.md "Pipeline audit" for the rerun plan and the remaining unfixed items
- [x] Phase 1 rerun under the audit fixes (2026-09-06): test AUROC Arm 1 **0.7051**, Arm 3c **0.7098**, Arm 3c-frozen **0.7159**; `LEAKAGE CHECK` 0.9%; gating negative (0.6797 test, 67% flagged) -- see RESULTS.md "Phase 1 audit rerun"
- [x] Phase 2 rerun under the audit fixes (batch 32, then batch 64 with a probe-convergence analysis, 2026-11-10) -- see RESULTS.md "Phase 2 rerun at batch 64"
- [ ] Findings-only text variant -- `notebooks/cxr_recall_phase2_findings_only_kaggle.ipynb` built and smoke-tested, not yet run. A word-level pre-check already shows the premise is wrong: Findings alone recovers the labels at 0.9767 test AUROC against 0.8347 for the Impression alone, so dropping the Impression will not remove the circularity (RESULTS.md "Findings-only text variant")
- [ ] Rare-label evaluation on Open-I's own long-tail MeSH terms (scaled down from the original ChestX-ray14-scale few-shot design)
