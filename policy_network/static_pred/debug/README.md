# Debug Script Map

This directory contains policy-network evaluation, analysis, and visualization
helpers. Top-level scripts are grouped by test function; experiment-specific
folders keep their existing local structure and outputs.

## Folder Overview

| Folder | Purpose |
| --- | --- |
| `downstream_eval/` | Downstream classifier evaluation for selected policy outputs, top-k candidate analysis, acquisition baselines, oracle upper-bound checks, counterfactual forced-option tests, and fixed-k/result summarization. |
| `index_prediction/` | Policy-index prediction accuracy and distribution analysis, including predicted-vs-ground-truth index plots, index-by-environment summaries, and oracle index distribution plots. |
| `number_probe/` | Number/probe experiments that train and evaluate MLP probes from lighting/class features, then plot probe history and predicted-index distributions. |
| `multiview_probes/` | Probe-selection utilities for multiview experiments, including brightness-histogram probe selection and visualization of selected multiview images. |
| `policy_visualization/` | Qualitative policy visualizations, selected case grids, and train-history distribution plots. |
| `dataset_checks/` | Dataset and label-maintenance checks, including split consistency validation and filtered lighting/option subset construction. |
| `random_input_ablation_inference/` | Random-input ablation evaluation for single/dual policy inputs across fixed-k settings and noise seeds. |
| `l1_l7_correct_16_25/` | Focused reduced-subset experiment for selected lighting conditions and option IDs. |
| `co_winner_matrix/` | Co-winner matrix analysis for pairs of downstream-correct options. |
| `index_cooccurrence_matrix/` | Top-k oracle-option co-occurrence matrices from soft targets. |
| `lighting_correct_index_heatmap/` | Lighting-by-option downstream-correct count heatmaps. |
| `oracle_soft_label_param_heatmaps/` | Oracle soft-label top-1 parameter heatmaps, with and without fallback targets. |
| `raw_csv_generation/` | Raw downstream-correctness CSV/matrix generation and histogram artifacts. |

Most scripts are intended to run from the repository root. Shell wrappers use
`ROOT_DIR` internally and usually support overrides such as `PYTHON_BIN`,
`DEVICE`, `BATCH_SIZE`, `NUM_WORKERS`, and sweep ranges.
