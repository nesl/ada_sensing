# Index Co-occurrence Matrix

This debug experiment builds top-5 oracle option co-occurrence matrices from:

`data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_all_labels.json`

For each sample, the top-k set is computed from positive `soft_target` entries,
sorted by descending weight and then by `option_id`.

Two all-set experiments are generated:

- `include_fallback`: use every sample directly from `soft_target`.
- `correct_only`: skip samples where `oracle_had_correct_candidate` is false.

In the original oracle labels, all-wrong samples use a one-hot fallback target.
The script checks that `oracle_had_correct_candidate` agrees with
`oracle_num_correct_candidates > 0` before producing results.

For each mode, the matrix definition is:

`M[i, j] = count of samples where option i and option j co-occur in the same top-5 set`

The diagonal is therefore:

`M[i, i] = count of samples where option i appears in the top-5 set`

Run with the lens conda environment:

```bash
/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python \
  policy_network/static_pred/debug/index_cooccurrence_matrix/analyze_index_cooccurrence.py
```

Outputs are written to `outputs/`:

- `include_fallback_counts.npy`
- `include_fallback_counts.json`
- `include_fallback_heatmap.png`
- `correct_only_counts.npy`
- `correct_only_counts.json`
- `correct_only_heatmap.png`
