# Adaptive Sensing Repo
This is an adaptive sensing research codebase for selecting the best image
capture option from a fixed set of camera-parameter candidates. For each
ImageNet-ES-Diverse sample, the dataset provides an auto-exposure image and 27
parameter-controlled candidate images. The main task is to train a lightweight
policy network that predicts which candidate should be selected to maximize
downstream recognition performance.

## Repository Layout

| Path | Description |
| --- | --- |
| `data/ImageNet-ES-Diverse/` | Dataset metadata, manifests, ImageNet class index, Lens labels, oracle labels, OpenCLIP labels, and derived label subsets. |
| `lens/` | Baseline utilities for manifest construction, image loading, classifier-based candidate selection, and Lens expert-label generation. |
| `policy_network/static_pred/` | Main policy-learning code: dataset loader, policy model, oracle-label generation, training entry point, organized experiment scripts, and debug tools. |
| `policy_network/results/` | Checkpoints, JSON metrics, cached downstream results, and experiment summaries. |
| `policy_network/vis_results/` | Generated plots and visualizations for option/index distributions and multiview probes. |
| `openclip_ds_policy/` | OpenCLIP-based downstream experiments, including soft-label generation, baselines, policy evaluation, and xlsx result summaries. |

## Core Workflow

1. Build or load `manifest_all.json`, where each sample contains 27 candidate
   capture options.
2. Generate supervision labels:
   - Lens labels select the candidate with the highest classifier confidence.
   - Oracle labels select candidates based on downstream correctness using the
     ground-truth class.
   - OpenCLIP labels use image-text similarity over the 200-class subset.
3. Train a policy network with single, dual, or multiview inputs.
4. Evaluate both policy-index accuracy and downstream top-1/top-k accuracy.
5. Analyze results with distribution plots, ablations, heatmaps, and probe
   experiments.

## Main Entry Points

| File or Folder | Purpose |
| --- | --- |
| `policy_network/static_pred/train_policy.py` | General policy training entry point. |
| `policy_network/static_pred/generate_oracle_policy_labels.py` | Generates oracle hard/soft labels from downstream classifier behavior. |
| `policy_network/static_pred/scripts/` | Organized training, evaluation, label-generation, and visualization shell scripts. See `policy_network/static_pred/scripts/README.md`. |
| `policy_network/static_pred/debug/` | Organized analysis and debug tools. See `policy_network/static_pred/debug/README.md`. |
| `openclip_ds_policy/` | OpenCLIP downstream pipeline and documentation. |

## Policy Inputs

The policy network supports three input modes:

- `single`: one image, either AE/baseline or a fixed candidate option.
- `dual`: AE/baseline plus one fixed candidate option.
- `multiview`: multiple fixed candidate options, optionally including AE.

The training code supports MobileNetV3-Small, ResNet18, tiny convolutional
models, and DINOv2 through a shared `SensorPolicyNetwork` interface.

## Notes

- Most scripts are intended to be run from the repository root.
- Shell scripts usually expose overrides such as `PYTHON_BIN`, `DEVICE`,
  `BATCH_SIZE`, `NUM_WORKERS`, `RESULTS_DIR`, and sweep ranges.
- The code assumes a Python environment with PyTorch, torchvision, timm, and
  related scientific Python packages. OpenCLIP experiments additionally require
  `open_clip_torch`.
