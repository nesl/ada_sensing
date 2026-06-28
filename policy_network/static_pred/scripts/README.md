# Static Policy Scripts

Run these scripts from the repository root unless a script-specific override is noted.
Most scripts also accept environment-variable overrides such as `PYTHON_BIN`,
`RESULTS_DIR`, `DEVICE`, `NUM_WORKERS`, `BATCH_SIZE`, and sweep bounds.

## Label generation

| Script | Purpose |
| --- | --- |
| `generate_oracle_policy_labels_v2_allwrong_uniform_w01.sh` | Regenerates oracle policy labels using confidence-weighted correct candidates, uniform soft targets for all-wrong samples, and sample weight `0.1` for all-wrong samples. |

## Training

| Script | Purpose |
| --- | --- |
| `train_policy.sh` | General A-G experiment driver for Lens/oracle labels with configurable backbone, input mode, and input variant. The active block currently trains `D_oracle_head_soft`; other blocks are commented for manual selection. |
| `train_policy_dual_backbone_control.sh` | Runs dual-input AE + fixed-option experiments for DINOv2 and tiny-conv backbones across Lens/oracle hard/soft label settings. |
| `train_policy_dual_mobilenet_fixedk_sweep_F.sh` | Sweeps fixed option IDs `0..26` for dual-input MobileNetV3-Small using oracle hard labels and full finetuning, writing `F_oracle_full_hard` runs. |
| `train_policy_dual_mobilenet_fixedk_sweep_G2_allwrong_uniform_w01.sh` | Sweeps fixed option IDs for dual-input MobileNetV3-Small using v2 oracle soft labels with all-wrong uniform targets, writing `G2_oracle_full_soft_allwrong_uniform_w01` runs. |
| `train_policy_single_mobilenet_fixedk_sweep_F.sh` | Sweeps single-input fixed candidate images for MobileNetV3-Small using oracle hard labels and full finetuning. |
| `train_policy_single_fixedk_l1_l7_option2_25.sh` | Sweeps single-input fixed candidate images on the reduced `l1/l7` and option `2/25` oracle-label subset. |
| `run_hist3_multiview_experiments.sh` | Trains multiview MobileNetV3-Small experiments using histogram-selected three-option probes, AE + hist3, and random three-option probe sets. |

## Evaluation and visualization

| Script | Purpose |
| --- | --- |
| `eval_policy.sh` | Legacy/manual evaluation driver for selected A-G policy runs; most sections are commented and the active block evaluates `D_oracle_head_soft`. |
| `eval_policy_dual_mobilenet_fixedk_sweep_F.sh` | Evaluates best and last checkpoints for the dual-input MobileNetV3-Small `F_oracle_full_hard` fixed-k sweep with top-k downstream analysis. |
| `eval_policy_dual_mobilenet_fixedk_sweep_G2_top1.sh` | Evaluates best checkpoints for the v2 all-wrong-uniform dual-input sweep with top-1 downstream analysis. |
| `eval_policy_dual_tiny_full.sh` | Evaluates tiny-conv dual-input runs and writes downstream/top-k plus train-history distribution summaries. |
| `eval_hist3_multiview_downstream.sh` | Evaluates multiview hist3/random3 checkpoints with downstream top-k analysis. |
| `visualize_hist3_multiview_results.sh` | Produces train-history plots, probe image visualizations, and predicted-vs-ground-truth index distribution plots for hist3/random3 runs. |

Debug-only and one-off analysis shell scripts remain under `policy_network/static_pred/debug/`
next to their corresponding Python tools and experiment-specific README files.
