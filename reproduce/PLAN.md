# Evidence-grounded execution plan

This plan targets Table 1 of *Adaptive Camera Sensor for Vision Models*. A
paper value is never copied into a reproduced field, and an undocumented
library default is never silently promoted to the primary protocol.

## 1. Lock the target

- **Complete:** Transcribe all 12 model rows and the three reported columns
  (`IN`, ImageNet-ES `AE`, ImageNet-ES-Diverse `AE`) directly from Table 1.
- **Complete:** Record field-level provenance in
  `evidence/model_evidence.csv`, with every claim tagged as `paper`,
  `official_code_or_readme`, `official_weight_config`, `inferred`, or
  `unknown`.
- **Gate:** Any future model identifier change requires a new evidence entry;
  it may not silently overwrite the locked registry.

## 2. Lock and validate data

- **Complete:** Validate the local Diverse set: 200 classes, 1,000 clean
  references, and 30 AE settings × 1,000 images.
- **Complete:** Acquire the original ImageNet-ES archive and validate 10 AE
  settings × 1,000 images.
- **Complete:** Hash-compare the original and Diverse clean test reference
  sets with `python -m reproduce_in_ae.compare_references`. The original
  ImageNet-ES test reference remains the default; all 1,000 evaluation images
  are byte-identical between the two packages.
- **Failure policy:** Missing/extra classes, images, or settings stop the run.

## 3. Lock software and checkpoints

- **Complete:** Record a publication-era reference environment: Python 3.10,
  PyTorch 2.0.1, torchvision 0.15.2, timm 0.9.7, and NumPy 1.x.
- **Locked local execution choice:** Run with the pre-existing `lens`
  environment and record its actual versions in every result. Do not modify
  the base environment.
- **Complete:** Isolate Torch and Hugging Face caches under `checkpoints/`.
- **Complete:** Acquire all official weights. Each result stores package
  versions, requested identifier, resolved cache location where exposed by the
  loader, parameter count, and device. Full SHA-256 values for every acquired
  file—including multi-file DINOv2 checkpoints—are stored in
  `evidence/checkpoint_manifest.json`.
- **Gate:** A model must return exactly 1,000 logits before 200-way slicing.

## 4. Execute the frozen evaluation matrix

- **Complete:** For each of 12 models, run `IN`, original `AE`, and Diverse
  `AE` (36 production artifacts).
- Use the official evaluator's common transform: RGB, short-edge resize 256,
  center crop 224, PIL bilinear, `[0,1]`, ImageNet normalization.
- Keep weights frozen, use FP32 inference, and perform no target-domain
  adaptation, prompt fitting, fine-tuning, or calibration.
- Slice 1,000 logits to the 200 sorted Tiny-ImageNet WNIDs before argmax.
- Save one atomic, resumable JSON artifact per model × dataset.
- Smoke artifacts have a separate filename and are excluded from reporting.

## 5. Aggregate and audit

- **Complete:** `IN`: micro top-1 over 1,000 images.
- Original `AE`: micro top-1 over 2 environments × 5 AE shots × 1,000 images.
- Diverse `AE`: micro top-1 over 6 environments × 5 AE shots × 1,000 images.
- Retain per-setting correct/total/accuracy and the macro setting mean.
- **Complete:** Build one final row per paper model in CSV, JSON, and Markdown.
- **Complete:** Compare at one decimal place, but retain six decimals in
  reproduced fields. The strict artifact audit passes 36/36.

## Blocking and unresolved details

1. Lens does not publish its complete evaluator or exact package/checkpoint
   hashes; exact model identifiers are inherited from the official
   ImageNet-ES evaluator.
2. The DeepAugment+AugMix pretraining description conflicts between the Lens
   table (`IN-21K`) and checkpoint publisher (`ImageNet classifier`).
3. OpenCLIP table metadata omits the ImageNet-1K fine-tuning encoded by the
   official `.laion2b_ft_in1k` identifiers.
4. Native model transforms conflict with the common transform hard-coded by
   official ImageNet-ES evaluation code. The common transform is primary;
   native transforms are diagnostic only.
5. DINOv2 Hub source was unpinned upstream; this project pins the source code
   commit while retaining official backbone/head URLs.
6. ResNet-50 gives 86.3% under both paper-era and current torchvision, versus
   86.0% in Lens. Reference-set byte identity rules out a difference between
   the two currently downloaded packages.
7. The ImageNet-ES Hub archive was first uploaded in January 2026 and has no
   published 2024 checksum. The current snapshot/etag is locked, but identity
   with the authors' publication-time private copy cannot be proven.
