
**On YOLO-World:** Not new — CVPR 2024, released January 2024. You absolutely know about it and so does everyone in the field. It's not a "new thing to try," it's a well-established baseline you'd be expected to include. That's fine, it just means framing it as a baseline comparison, not a novelty.

**On your TENT + DUA + SAR + custom method plan:** This is solid and the right instinct. That's exactly the comparison table structure that makes an arXiv preprint credible. Let me build the full roadmap around it.

---

## Full Roadmap

### Phase 0 — Clean Foundation (1 week)

Before rerunning anything, settle the experiment design so you don't rerun twice.

**Decisions to lock in now:**

- **Backbone:** YOLOv8m stays as primary. Add YOLO11m as secondary for ablation. Both trained on SH17 at 100 epochs.
- **Target domains:** SHWD (primary — highest sample count, most reliable numbers), Pictor-PPE, CHV. All three, same as before.
- **Evaluation metric:** mAP50 primary, mAP50-95 secondary, per-class AP (hard\_hat, no\_hard\_hat, person) mandatory. The per-class breakdown is where your cold start story shows up most clearly.
- **Batch protocol:** Fix adapt\_batch=16, val\_batch=64 across all methods. This is what makes your comparison fair and reproducible.
- **Add YOLO-World as a reference point:** Run YOLO-World-L zero-shot with text prompts "hard hat", "person", "no hard hat" on all three target domains. No adaptation. This answers the question "does a foundation model just solve this?" — it's one eval run, gives you a strong baseline row in your table.

---

### Phase 1 — Baseline Grid (1 week)

Run and document cleanly:

| Experiment | What it measures |
|---|---|
| SH17 val (source) | Confirms model didn't degrade |
| Zero-shot → SHWD, Pictor, CHV | Domain gap magnitude |
| YOLO-World zero-shot → all three | Foundation model baseline |

This gives you your Table 1: domain gap quantification. The story here is: "YOLOv8m loses ~85% relative mAP under cross-factory shift. Even YOLO-World, despite open-vocabulary pretraining, shows X% degradation, confirming the gap is not solved by scale alone."

---

### Phase 2 — TTA Method Comparison (2 weeks)

Run all four methods with a consistent sweep (steps × lr grid, same as your existing sweep\_tent.py pattern):

**TENT** — you have this already. Rerun cleanly with fixed protocol for reproducibility.

**DUA** (Mirza et al. 2022) — adapts momentum in a new normalization layer, doesn't rely purely on entropy. Important because it handles small batches better than TENT. Implementation exists in the TTA literature repos.

**SAR** (Niu et al. 2023) — sharpness-aware entropy minimization + sample selection. Filters out unreliable samples before updating. This is specifically relevant to your cold start problem because SAR's sample selection should theoretically detect that there are no reliable samples — worth seeing what it does when the entire batch is unreliable.

**ETA/EATA** — I'd suggest swapping CoTTA for EATA here. EATA filters redundant samples and has a Fisher regularization term that prevents collapse. It's been compared with TENT and SAR extensively in recent literature and is a cleaner addition to your table than CoTTA which is more complex.

This gives you Table 2: the main comparison. The expected pattern — which is actually your finding — is that all entropy-based methods either collapse or show negligible recovery under your severe shift regime. That pattern IS the contribution. It's not a failed experiment, it's an empirical characterization of the cold start failure mode across methods.

---

### Phase 3 — Your Custom Method (2-3 weeks)

This is the part that goes from "benchmark paper" to "method paper." Based on your data and what the literature doesn't cover, here's the most viable custom contribution:

**Detection-Aware Warm-Start TTA**

The core insight from your existing results: TENT has nothing to grip onto because the model produces near-zero confident predictions under severe shift. Entropy minimization requires some signal to minimize. The fix isn't a better entropy objective — it's a warm-up stage that gives the model enough signal to start adapting.

Concretely:

**Stage 1 — Warm-start alignment:** Before entropy minimization, align the target domain's BN statistics to the source statistics using the incoming batch. This is essentially resetting BN running mean/variance to match the target distribution's first moments, without any gradient update. It's cheap, parameter-free, and gives the model a better starting point before Stage 2.

**Stage 2 — Detection-aware entropy minimization:** Instead of averaging entropy over all predictions (which collapses when there are no confident predictions), compute entropy only over predictions above an IoU-coherence threshold — i.e., proposals where the model's spatial predictions are internally consistent. This filters the cold-start noise. Inspired directly by the VLOD-TTA IoU-weighted objective, but applied to your closed-set YOLO setup.

This is two-stage, interpretable, and directly motivated by your empirical failure analysis. You can ablate each stage independently (Stage 1 only, Stage 2 only, both together), which gives you Table 3.

**Why this is achievable:** Stage 1 requires ~30 lines of code modifying your existing tent.py. Stage 2 requires implementing the IoU-coherence filter on YOLO's output, which is more involved but your existing eval infrastructure handles YOLO outputs already.

---

### Phase 4 — Write-Up (1-2 weeks)

**Structure:**

1. **Introduction** — cross-factory deployment problem, why TTA is the right framework, cold start failure mode as the gap
2. **Related Work** — TTA methods (TENT, DUA, SAR, EATA), TTA for detection (STFAR, MemCLR, VLOD-TTA), industrial safety CV
3. **Problem Setup** — datasets, domain gap characterization, label space, evaluation protocol
4. **Experiments** — Tables 1, 2, 3 as above
5. **Method** — your two-stage warm-start TTA
6. **Analysis** — per-class breakdown, ablation, cold start failure characterization
7. **Conclusion**

Target: arXiv preprint, 8 pages. Not submitting to a venue initially — just getting it online so you have a link for cold emails.

---

### Timeline Summary

| Phase | Duration | Output |
|---|---|---|
| 0 — Design lock | Week 1 | Experiment spec, YOLO-World baseline |
| 1 — Baseline grid | Week 1-2 | Table 1, domain gap numbers |
| 2 — TTA comparison | Week 2-4 | Table 2, TENT/DUA/SAR/EATA |
| 3 — Custom method | Week 4-6 | Table 3, warm-start TTA ablation |
| 4 — Write-up | Week 6-8 | arXiv preprint |

Eight weeks total on your A10s is realistic. The write-up can overlap with Phase 3 — start writing Section 1-3 while experiments run.
