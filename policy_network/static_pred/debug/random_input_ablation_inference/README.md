# Random Input Ablation Inference

Run from the repo root:

```bash
policy_network/static_pred/debug/random_input_ablation_inference/run_eval.sh
```

Default output:

```bash
policy_network/results/results_random_input_ablation_inference/summary_best.csv
```

The script evaluates `NOISE_SEEDS=0..19` and `fixed_k=0..26` by default.

Existing completed outputs are skipped by default. Set `SKIP_EXISTING=0` to force
reruns.

Real-only settings do not depend on `NOISE_SEED`, so by default they only run for
`seed0`. Set `RUN_REAL_ONLY_SEED0=0` to run them for every seed.

Before the loop starts, the script builds or reuses a downstream classifier cache:

```bash
policy_network/results/downstream_candidate_cache/resnet50_im224_oracle_test.json
```

The cache stores downstream correctness for each real `(sample_id, option_id)`.
Noise still affects the policy input and the policy-selected `option_id`; the
downstream classifier result is then looked up from this cache.

## Evaluated Settings

`single input` models receive only the fixed candidate image for `fixed_k`.

- `real_fixed_input`: use the real fixed candidate image.
- `random_fixed_input`: replace the fixed candidate image with deterministic random noise.

`dual input` models receive two images: the AE/baseline image and the fixed candidate image for `fixed_k`.

- `real_ae_real_fixed_input`: use the real AE/baseline image and the real fixed candidate image.
- `random_ae_real_fixed_input`: replace only the AE/baseline image with deterministic random noise.
- `real_ae_random_fixed_input`: replace only the fixed candidate image with deterministic random noise.
- `random_ae_random_fixed_input`: replace both images with deterministic random noise.

## Summary Columns

The final CSV contains one row per `tag` and `input_index`.

- `tag`: `single input` or `dual input`
- `input_index`: the fixed candidate index, i.e. `fixed_k`
- `n_seeds`: number of noise seeds found for that row
- `*__index_acc__avg/std`: top-1 policy-index accuracy across noise seeds
- `*__downstream_top1_acc__avg/std`: top-1 downstream accuracy across noise seeds
