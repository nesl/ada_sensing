# Adaptive Camera-Sensor Selection — Analysis & Findings

An investigation into **why the sensor-selection policy cannot beat a fixed camera
config** on ImageNet-ES-Diverse, and what *can*.

All numbers below are **locally measured** (no figures taken from prior slides):
downstream **top-1 accuracy**, **ResNet-50** downstream classifier, on the
**seed-0 3/1/1 split by reference image** (train 3,600 / val 1,200 / **test 1,200**
scenes). Test noise floor ≈ ±1.4%. Each "scene" = one reference image under one of
6 lighting conditions, with 27 parameter-controlled candidate captures.

---

## TL;DR

1. **Every single-shot policy we tried (the original A–G, and two of ours with a new
   label) lands at 29.5–34.4% — none beats the fixed-config baseline (34.0%).**
2. The wall is an **information** limit, not a label limit: the per-scene optimal
   config is driven by image **content**, not lighting. Knowing the lighting
   perfectly is worth only **+0.17%** over a global fixed config.
3. Max-softmax confidence (VisiT) is **badly miscalibrated** under this shift — a
   >90%-confident pick is still wrong **36%** of the time.
4. A new **feature-distance-to-clean** label is a much better *oracle* (40.8% vs
   VisiT's 32.2%), but requires the clean image, so it's a **training label**, not a
   deployable selector — and it does **not** rescue single-shot policies.
5. The **only deployable method that beats a fixed config is a "relative" scorer**
   that sees all 27 captures jointly (self-attention) and ranks them:
   **36.4%** (+2.4 over fixed, +4.3 over VisiT). Its top-5 shortlist contains a
   correct capture 47% of the time (≈92% of all recoverable scenes).
6. The deployable ceiling here is ~36.4%; the clean-image oracle is 40.8%; the hard
   ceiling (any of 27 correct) is 51.0%.

---

## Full results table

Downstream top-1 accuracy, ResNet-50, seed-0 test split (1,200 scenes), ±1.4%.

| # | Method / Config | Details | Captures @ deploy | Test acc |
|---|---|---|---|---:|
| — | Oracle-S (ceiling) | any of 27 correct | — | **51.0%** |
| — | feature-oracle | closest-to-clean (needs clean image) | 27 + clean | 40.8% |
| 1 | **Relative scorer** | self-attention over 27 candidates | 27 | **36.4%** |
| 2 | Policy H (ours) | feature-dist label · fixed-exposure input | 1 | 34.4% |
| 3 | Independent scorer | per-candidate learned scorer | 27 | 34.2% |
| 4 | Fixed-config baseline | best single constant config | 1 | 34.0% |
| 5 | Policy **F** | Oracle / full-ft / hard | 1 | 33.7% |
| 6 | Policy **G** | Oracle / full-ft / soft | 1 | 33.1% |
| 7 | Policy **A** | Lens / full-ft / hard | 1 | 32.8% |
| 8 | Policy **B** | Lens / head / hard | 1 | 32.4% |
| 9 | VisiT | max-softmax over 27 | 27 | 32.2% |
| 10 | Policy I (ours) | feature-dist label · AE input | 1 | 31.6% |
| 11 | Policy **C** | Oracle / head / hard | 1 | 31.0% |
| 12 | Policy **E** | Oracle / partial / soft | 1 | 29.8% |
| 13 | Policy **D** | Oracle / head / soft | 1 | 29.5% |

**Policy configs (single AE input, MobileNetV3-Small, 50 epochs, batch 16):**

| Cfg | Label | Scope | Loss |
|---|---|---|---|
| A | Lens (max-confidence) | full-finetune | hard-CE |
| B | Lens | head-only | hard-CE |
| C | Oracle (correctness) | head-only | hard-CE |
| D | Oracle | head-only | soft-KL |
| E | Oracle | partial-unfreeze | soft-KL (resumes D) |
| F | Oracle | full-finetune | hard-CE |
| G | Oracle | full-finetune | soft-KL |
| H (ours) | feature-distance-to-clean | full-finetune | soft-KL, fixed-exposure input |
| I (ours) | feature-distance-to-clean | full-finetune | soft-KL, AE input |

---

## What we did, and what we found

### 1. VisiT confidence-rank distribution
For each scene, rank the 27 candidates by confidence and find the rank of the first
*correct* one. **VisiT top-1 = 32.0%; Oracle-S = 49.9%; 50% of scenes are
unrecoverable** (no config yields a correct prediction). Of the recoverable half,
VisiT's top pick is correct only 64% of the time — the rest is smeared across ranks
2–26. → `measure_visit_rank.py`

### 2. Confidence calibration
Distribution of VisiT's selected confidence in 5% bins, with per-bin accuracy.
**A >90%-confident pick is wrong 36% of the time; a >80%-confident pick is wrong 45%
of the time.** Confidence magnitude is an untrustworthy correctness signal under this
covariate shift. → `analyze_confidence_hist.py`

### 3. Exposure / visibility gate
Hypothesis: the confidently-wrong picks are blown-out / pitch-black captures.
Partly true (they have ~3× more clipped highlights, lower contrast/entropy) but the
effect is a *tail* — a gate buys only **+1 point** (contrast/entropy help; raw
brightness barely does). → `build_combined_cache.py`, `analyze_visibility.py`,
`analyze_gate.py`

### 4. Luminance-matching selector
Idea: pick the config whose luminance matches the training images. **Fails: 27.2%
at the training target, 29.4% even at the best-possible target — worse than VisiT.**
Confirms the paper's point that model-optimal exposure ≠ human/natural luminance.
→ `analyze_luminance_match.py`

### 5. Feature-distance-to-clean label (the good idea)
Pick the config whose capture is closest to the **clean original image in ResNet-50
feature space**. As an oracle: **40.8%** (+8.6 over VisiT). Pixel/structure closeness
is *worse* (27.9%) — must be feature space. This needs the clean image, so it's a
**training label**, not a test-time selector. → `build_fidelity_cache.py`,
`analyze_fidelity.py`

### 6. Single-shot policy on the new label
Train the policy to predict the feature-distance-optimal config from one input.
**AE input: 31.6%; fixed-exposure input: 34.4%** — still at the fixed-config wall,
even with the better label. → `train_fidelity_policy.py`

### 7. The settling diagnostic — why single-shot is capped
- Global best fixed config → **34.0%**
- Per-lighting best fixed config, *with oracle lighting knowledge* → **34.2%** (+0.17)

**Knowing the lighting is worth almost nothing.** Within each lighting, the optimal
config is spread across many options (entropy 2.5–3.2 bits). → the per-scene optimal
config is **content-driven**, and content isn't legible from one degraded capture
without solving classification itself. This is a *data property*, not a training bug.

### 8. Learned all-27 selectors
A deployable selector *can* see all 27 captures (like VisiT does) and use a learned
scorer instead of max-softmax.
- **Independent** per-candidate scorer: fits train to 40–42% but **tests at 34%** —
  overfits content-specific patterns. → `train_scorer.py`
- **Relative** scorer (self-attention over the 27, listwise loss): **36.4% test**,
  and the train→test gap shrinks from ~7 to ~2.5. Scoring candidates *against each
  other* generalizes across content far better than absolute scoring. Its ranking is
  near-oracle in the top-k: top-3 = 43.6%, top-5 = 47.2%, top-10 = 50.7% (≈ Oracle-S).
  → `train_relative_scorer.py`

### 9. Using the higher-ceiling correctness labels
Supervising the relative scorer with the Oracle correctness labels (ceiling 51 vs the
distance label's 40.8) did **not** raise top-1 (all ~35.5–36.4, within noise) — the
correctness target overfits more. Confirms we are **generalization-limited, not
label-limited**. Small gain on recall@5 (48.1 vs 47.2). → `train_relative_scorer.py --target correct`

### 10. Policy grid A–G (reproduction of the original approach)
All 7 original configs, run locally with downstream metric (see table). Best is
**F = 33.7%**, all at/below the fixed-config baseline. Full-finetune > head-only;
Oracle-soft (D, E) is the weakest. → `train_policy_grid.py`

---

## Conclusions & implications

- **Single-shot selection is near-fundamentally capped at ~34% on this dataset**
  (9 policy variants confirm it), because the optimal config is content-driven and
  content is not recoverable from one blind input.
- **The path past the wall is either:**
  1. **See all 27 and score them relatively** (the relative scorer, 36.4%, deployable
     but expensive — needs all 27 captures), or
  2. A **sequential / budgeted multi-shot policy** (probe → observe → adapt) that
     *reduces* captures while still observing the scene — the natural next step, and
     the bridge to the CARLA / video direction where config-optimality is driven by
     *observable* physics (lighting extremes, motion blur, HDR) rather than fine
     content.
- The dataset itself is a bottleneck: lighting is not extreme enough to make the
  optimal config lighting-dependent, and 50% of scenes are unrecoverable by any config.

---

## Reproduction

Scripts live in `scripts/`. They were run **from inside the dataset directory**
`data/ImageNet-ES-Diverse/` (gitignored), where they expect to sit alongside the
extracted dataset and the caches they build (`*.npz`). To reproduce, copy a script
next to the data or adjust the `here = Path(__file__).parent` line.

Dataset: `Edw2n/ImageNet-ES-Diverse` (HuggingFace), extracted to
`data/ImageNet-ES-Diverse/es-diverse-test/{param_control,auto_exposure,sampled_tin_no_resize2}`.

Rough order:
1. `build_fidelity_cache.py` — per-candidate conf, correctness, feature-distance-to-clean (also builds feature cache inputs).
2. `make_group_meta.py` — scene metadata (AE paths, split keys) aligned to the caches.
3. `extract_gtprob.py`, `build_feature_cache.py` — GT-class prob + 2048-d features.
4. Analyses: `measure_visit_rank.py`, `analyze_confidence_hist.py`, `analyze_gate.py`, `analyze_luminance_match.py`, `analyze_fidelity.py`.
5. Policies / scorers: `train_policy_grid.py` (A–G), `train_fidelity_policy.py` (H, I), `train_scorer.py` (independent), `train_relative_scorer.py` (relative).

---

## Appendix: how much of the 51% is actually recoverable?

(Added after the main writeup. Nothing above is changed — this refines what
"Oracle-S = 51%" really means.)

**Oracle-S = 51% is the only *hard* ceiling** — it is the fraction of test scenes
where at least one of the 27 configs classifies correctly; the other **49.0% are
impossible** for any method. But 51% badly overstates what a real selector can
reach, for two reasons: (1) it assumes an oracle that already knows the GT label,
and (2) many "recoverable" scenes are recoverable only via a **low-confidence
correct config** — a lucky guess the model itself is unsure about.

### What our two signals reach (and an important caveat)

Two deployable selectors and the fraction of **all 1,200 test scenes** on which
each lands on a correct config:

| Selector | picks | finds a correct config on |
|---|---|---:|
| confidence (VisiT) | argmax max-softmax | 32.2% |
| feature-distance | argmin distance-to-clean | 40.8% |
| either, picked perfectly | union of the two | 42.8% |
| **missed by BOTH** | — | **8.2%** |

**Caveat — 40.8% / 42.8% are NOT a "learnable ceiling."** They only describe the
two signals we happened to test. A different signal (better features, a separate
degradation model, an ensemble, cues the classifier's confidence doesn't capture)
could recover some of the 8.2% "missed-by-both" scenes. The **only proven hard
ceiling is Oracle-S (51%)**; the true learnable ceiling is unknown, bounded below
by our best deployable method (relative scorer, **36.4%**) and above by 51%.

### A confidence-grounded estimate of "reliably recoverable"

Treat a scene as *reliably* recoverable only if it has a correct config the model
is actually **confident** about (≥50%). Scenes whose only correct config fires
**below 50% confidence** are lucky guesses — the correct classification is a fluke,
statistically indistinguishable from the sea of wrong low-confidence captures, so
no confidence-based selector can reliably find it.

**Table 1 — max confidence of the correct config, per recoverable scene, in 5%
buckets (% of all 1,200 test scenes):**

| Correct config's max confidence | % of all test | Cumulative |
|---|---:|---:|
| [0, 5) | 0.42% | 0.42% |
| [5, 10) | 3.08% | 3.50% |
| [10, 15) | 2.67% | 6.17% |
| [15, 20) | 1.50% | 7.67% |
| [20, 25) | 1.83% | 9.50% |
| [25, 30) | 1.00% | 10.50% |
| [30, 35) | 1.50% | 12.00% |
| [35, 40) | 2.08% | 14.08% |
| [40, 45) | 1.17% | 15.25% |
| [45, 50) | 1.25% | **16.50%** |
| [50, 55) | 1.75% | 18.25% |
| [55, 60) | 0.58% | 18.83% |
| [60, 65) | 1.83% | 20.67% |
| [65, 70) | 1.08% | 21.75% |
| [70, 75) | 0.75% | 22.50% |
| [75, 80) | 1.42% | 23.92% |
| [80, 85) | 1.83% | 25.75% |
| [85, 90) | 1.92% | 27.67% |
| [90, 95) | 2.42% | 30.08% |
| [95, 100) | **20.92%** | 51.00% |

*(the other 49.0% — no correct config — is not in this table)*

The cumulative column crosses **16.5% at the 50% mark**: that's the share of all
test scenes whose *only* correct config is below 50% confidence. The big pile at
**[95,100) = 20.9%** is the easy core; below 50% it's a thin, flat tail of flukes.

**Table 2 — decomposition of all test scenes:**

| Band | % of all test | Interpretation |
|---|---:|---|
| Impossible (no config correct) | **49.0%** | hard ceiling (Oracle-S = 51% recoverable) |
| Lucky guess (correct config < 50% conf) | **16.5%** | probably unrecoverable |
| Reliably recoverable (≥50%-confident correct config exists) | **34.5%** | defensible confidence-grounded target |

**Takeaway:** don't benchmark against 51%. Only **34.5%** of scenes have a
confident-correct config, and our best methods already sit right at it
(fixed-config 34.0%, relative scorer 36.4%). The remaining **16.5%** are lucky
guesses we treat as probably unrecoverable, and **49.0%** are impossible. The true
learnable ceiling somewhere in (36.4%, 51%] remains open, but 34.5% is the honest,
confidence-grounded anchor.
