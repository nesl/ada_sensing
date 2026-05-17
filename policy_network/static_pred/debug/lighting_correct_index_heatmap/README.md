# Lighting Correct Index Heatmap

Goal: build a `6 x 27` matrix where each row is a lighting/env and each
column is an option/index.  The value is the number of samples under that
lighting for which the option is downstream-correct.

Counting rule:

- Source field: `oracle_correct_option_ids` from oracle policy labels.
- If multiple option ids are downstream-correct for one sample, every listed
  option id receives `+1`.
- Samples with no downstream-correct candidate contribute zero counts.

Default input:

```bash
data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_all_labels.json
```

Run:

```bash
/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python \
  policy_network/static_pred/debug/lighting_correct_index_heatmap/plot_lighting_correct_index_heatmap.py
```

Outputs are written to `outputs/`:

- `lighting_correct_index_counts.json`
- `lighting_correct_index_counts.csv`
- `lighting_correct_index_counts.npy`
- `lighting_correct_index_heatmap.png`

