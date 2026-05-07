# OpenCLIP Downstream Policy Experiments

This folder is intentionally separate from the existing `policy_network/results_*`
experiments. It reuses the dataset and policy-network code, but writes OpenCLIP
labels and results under `openclip_ds_policy/`.

## Setup

- Dataset: `data/ImageNet-ES-Diverse`
- Split: existing corrected `3/1/1` split from `data/ImageNet-ES-Diverse/policy_labels`
- Class space: the 200 ImageNet classes present in the split, using the original
  ImageNet label IDs from the manifest.
- Text prompt: `a photo of a {class_name}.`
- OpenCLIP score: L2-normalized image/text cosine similarity over the 200 classes.

## Baselines

Run:

```bash
bash openclip_ds_policy/scripts/eval_openclip_baselines.sh
```

The baseline table contains:

- `AE`: auto-exposure image.
- `Random`: average top-1 accuracy over all 27 options.
- `Oracle-S`: per-sample upper bound. If at least one candidate is correct, select
  a correct candidate; otherwise select the candidate with highest GT similarity.
- `Oracle-F`: best fixed option over the whole test split.
- `Lens`: select the candidate with the highest OpenCLIP softmax confidence.

## Soft Labels

Run:

```bash
bash openclip_ds_policy/scripts/generate_openclip_soft_labels.sh
```

For each sample:

```text
gt_score[k] = cosine_similarity(image_k, text_gt)
soft_target = softmax(gt_score / tau)
```

The default `tau` is `0.05`.

## Policy Training

Run the dual-input fixed-k sweep:

```bash
bash openclip_ds_policy/scripts/train_dual_fixedk_full_sweep.sh
```

Each run uses:

- input: `AE + fixed option k`
- `k = 0..26`
- backbone: `mobilenet_v3_small`
- pretrained ImageNet initialization
- trainable scope: `full_finetune`
- loss: `soft_kl`

## Downstream Evaluation

Run:

```bash
bash openclip_ds_policy/scripts/eval_dual_fixedk_full_sweep.sh
```

This reports only policy-selected OpenCLIP top-1 downstream accuracy.
