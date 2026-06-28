# Co-winner Matrix

Goal: build a `27 x 27` matrix where:

`M[i, j] = count of samples where both option/index i and option/index j are downstream-correct`

Counting rule:

- Source fields: `oracle_had_correct_candidate`,
  `oracle_num_correct_candidates`, and `oracle_correct_option_ids` from oracle
  policy labels.
- Downstream-all-wrong samples are skipped even when old labels contain a
  loss-min fallback option in `oracle_correct_option_ids`.
- Each sample contributes `+1` to every ordered pair `(i, j)` in its unique
  downstream-correct option set where `i != j`.
- The diagonal is set to `0` by construction for convenience.
- Samples with fewer than two downstream-correct options contribute no pair
  counts.

Default input:

```bash
data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_all_labels.json
```

Run:

```bash
/mnt/hdd1/yuyang/install/conda_envs/lens/bin/python \
  policy_network/static_pred/debug/co_winner_matrix/build_co_winner_matrix.py
```

Outputs are written to `outputs/`:

- `co_winner_matrix.csv`
- `co_winner_matrix.json`
- `co_winner_matrix.npy`
- `co_winner_heatmap.png`
- `single_index_correct_counts.csv`
- `top_co_winner_pairs.csv`
