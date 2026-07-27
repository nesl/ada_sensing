# Reproducing the IN and AE columns

This folder reproduces the complete `IN` and both `AE` columns in Table 1 of
[Adaptive Camera Sensor for Vision Models](https://openreview.net/pdf?id=He2FGdmsas).
The paper table is the source of truth: it contains 12 model rows, including
SwinV2-S (which was absent from the older ImageNet-ES paper table).

## Exact target values

| Model | IN | AE ImageNet-ES | AE ImageNet-ES Diverse |
| --- | ---: | ---: | ---: |
| ResNet-50 | 86.0 | 32.1 | 17.6 |
| ResNet-50 + DeepAugment + AugMix | 87.0 | 53.2 | 36.2 |
| ResNet-152 | 87.8 | 41.1 | 21.9 |
| EfficientNet-B0 | 88.2 | 51.8 | 21.8 |
| EfficientNet-B3 | 88.1 | 62.0 | 33.6 |
| SwinV2-T | 90.6 | 54.1 | 26.5 |
| SwinV2-S | 91.7 | 59.9 | 30.8 |
| SwinV2-B | 91.9 | 60.0 | 30.8 |
| OpenCLIP-b | 94.3 | 66.1 | 38.8 |
| OpenCLIP-h | 94.9 | 79.0 | 45.5 |
| DINOv2-b | 93.6 | 74.5 | 44.5 |
| DINOv2-g | 94.7 | 84.3 | 62.8 |

The final deliverable has one row per model, as requested:

`model, checkpoint, IN_paper, IN_reproduced, AE_ImageNet_ES_paper,
AE_ImageNet_ES_reproduced, AE_Diverse_paper, AE_Diverse_reproduced,
evaluation_label_space, AE_aggregation`.

## Locked evaluation protocol

- `IN` is the exact 1,000-image clean reference set (5 images × 200 classes),
  not the full ImageNet validation set.
- Classification is closed-set 200-way. Every model retains its 1,000-output
  ImageNet classifier; logits are sliced to the 200 Tiny-ImageNet WNIDs, in
  sorted WNID order, before argmax.
- `AE ImageNet-ES` is evaluated over all 10,000 images:
  2 environments × 5 AE shots × 1,000 references.
- `AE Diverse` is evaluated over all 30,000 images:
  6 environments × 5 AE shots × 1,000 references.
- The primary value is micro accuracy over all images. Each setting has the
  same size, so it equals the mean of the 10 or 30 per-setting accuracies;
  both values and all underlying counts are saved.
- All weights are frozen, inference uses FP32, and no target-domain adaptation,
  prompt fitting, calibration, or fine-tuning is performed.

The primary run deliberately follows the official ImageNet-ES evaluation code:
PIL RGB → short-edge bilinear resize to 256 while preserving aspect ratio →
224 center crop → `ToTensor` in `[0,1]` → ImageNet mean/std normalization.
This common transform overrides the native weight transforms. Native configs
are retained in the model registry/evidence because EfficientNet-B3 expects
300px, SwinV2 expects 256px, and OpenCLIP has CLIP normalization. Running native
transforms would be a useful diagnostic, but those numbers must not be placed
in the paper-reproduction columns.

## Run

```bash
cd /mnt/hdd1/yuyang/adaptive_sensing/Lenz/reproduce
conda activate lens

bash scripts/download_imagenet_es.sh
bash scripts/download_checkpoints.sh
python -m reproduce_in_ae.validate_data
bash scripts/run_all.sh
```

Per the local execution requirement, actual runs use the existing `lens`
environment. `environment.yml` remains a publication-era reference rather than
the active runtime; every result JSON records the actual package versions.

Runs are resumable at model × dataset granularity under `results/raw/`. To run
a small non-authoritative smoke test:

```bash
PYTHONPATH=src python -m reproduce_in_ae.evaluate \
  --models resnet50 --datasets in --device cpu --workers 0 --max-samples 8
```

`python -m reproduce_in_ae.report` writes the final CSV, JSON, and Markdown
files to `results/in_ae_reproduction.*`. Smoke-test results are intentionally
excluded from the final reproduced columns.

For a model too large to evaluate efficiently on one GPU, deterministic
interleaved shards can be run on separate devices and merged by exact
`correct`/`total` counts:

```bash
PYTHONPATH=src python -m reproduce_in_ae.evaluate \
  --models dinov2_g --datasets ae_imagenet_es_diverse \
  --device cuda:0 --shard-index 0 --shard-count 2
PYTHONPATH=src python -m reproduce_in_ae.evaluate \
  --models dinov2_g --datasets ae_imagenet_es_diverse \
  --device cuda:1 --shard-index 1 --shard-count 2
PYTHONPATH=src python -m reproduce_in_ae.merge_shards \
  --model dinov2_g --dataset ae_imagenet_es_diverse --shard-count 2
```

The two-way split sends even/odd global sample indices to separate GPUs. Since
every setting has exactly 1,000 images, each shard contributes 500 images per
setting; merging exactly recovers all 30,000 predictions.

## Current execution status

- Actual execution uses `/mnt/hdd1/yuyang/install/conda_envs/lens`.
- Complete: all 12 models × 3 datasets have formal results; the strict audit
  passes 36/36 artifacts and all 7 unit tests pass.
- Final deliverables are `results/in_ae_reproduction.csv`, `.json`, and `.md`;
  the machine-readable audit is `results/audit.json`.
- A full ResNet-50 `IN` run produced `86.3%` (863/1,000) under both the
  paper-era environment and the pre-existing newer environment, versus `86.0%`
  in Lens Table 1. This is retained as a real reproduced value, not adjusted.
- The original ImageNet-ES and Diverse datasets both pass strict count checks.
  The saved count audit is in `evidence/data_validation.json`. Their 1,000
  clean reference images are byte-identical; the comparison is in
  `evidence/reference_comparison.json`.
- Two RTX 3090 GPUs are available to the `lens` environment from the host
  execution context.

## Evidence status and unresolved details

The source catalog is [evidence/sources.json](evidence/sources.json). The full
per-field evidence table is [evidence/model_evidence.csv](evidence/model_evidence.csv),
and exact hashes of every acquired weight file are in
[evidence/checkpoint_manifest.json](evidence/checkpoint_manifest.json).
The exact active `lens` runtime is saved in
[evidence/runtime_environment.json](evidence/runtime_environment.json).
Each field is labeled as one of `paper`, `official_code_or_readme`,
`official_weight_config`, `inferred`, or `unknown`.

Known unresolved or conflicting details are not silently resolved:

1. The Lens paper publishes model libraries but not exact package versions or
   checkpoint hashes. The environment uses the unchanged publication-era
   ImageNet-ES code requirements (`torch 2.0.1`, `torchvision 0.15.2`,
   `timm 0.9.7`) and records runtime versions/checkpoint identity.
2. The paper says the DeepAugment+AugMix ResNet-50 was pretrained on IN-21K,
   while the linked checkpoint publisher calls it an ImageNet classifier and
   the checkpoint has a 1,000-class head. This remains a source conflict.
3. The paper describes OpenCLIP pretraining as LAION-2B, while the exact
   official code identifiers end in `.laion2b_ft_in1k`; they include supervised
   ImageNet-1K fine-tuning and a 1,000-class head.
4. Appendix E.3.1 does not spell out preprocessing. The older official
   ImageNet-ES evaluator hard-codes one 224px/ImageNet-normalized transform for
   every architecture, conflicting with several native weight configs.
5. Lens does not publish its full evaluation source. Exact model identifiers
   are inherited from the official ImageNet-ES evaluator, whose core files are
   byte-identical between the May 2024 publication commit and current main.
6. Several Lens Table 1 values differ by 0.1–0.3 points from the older
   ImageNet-ES Table 3. This project targets Lens Table 1 because it is the
   paper linked in the request.
7. The original official DINOv2 Hub ref was unpinned. This project pins Hub
   code to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; the backbone and
   linear-head checkpoint URLs are separately recorded.
8. The local clean reference yields ResNet-50 `86.3%` with the exact V1
   checkpoint under both paper-era and current torchvision, versus `86.0%` in
   Lens. The original and Diverse packages' current reference images are
   byte-identical, so choosing between those two current copies and changing
   torchvision version have both been ruled out. Identity with the authors'
   unpublished publication-time image bytes has not been established.
9. The downloadable ImageNet-ES Hub repository was created in January 2026,
   after both papers, and exposes no publication-era archive hash. The acquired
   snapshot is commit `daa6a83600cc4a13e8e59a6375987cd18981ff59` with archive
   object etag
   `f6d4b5f1291d773ea7088482f0e66761ec6c3487e6fba8d81fbf5e61769b1ba0`.
   Whether its image bytes are identical to the authors' 2024 local archive is
   unknowable from published artifacts.
