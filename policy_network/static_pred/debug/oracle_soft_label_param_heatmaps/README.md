# Oracle Soft Label Parameter Heatmaps

Generate all-set oracle-soft-label top-1 count heatmaps over camera parameters.

Metrics:

- `with_fallback_top1_count`: uses `data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_all_labels.json`.
  Downstream-all-wrong samples have one-hot fallback labels and are counted.
- `no_fallback_top1_count`: uses `data/ImageNet-ES-Diverse/oracle_policy_labels_v2_allwrong_uniform_w01/oracle_policy_all_labels.json`.
  Downstream-all-wrong samples are skipped because they have no non-fallback top-1.

Run from the repo root:

```bash
/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python \
  policy_network/static_pred/debug/oracle_soft_label_param_heatmaps/plot_oracle_soft_label_param_heatmaps.py
```

Outputs are written under `outputs/`.
