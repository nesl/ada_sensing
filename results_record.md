# Policy Training Results Record

This file records the current policy-training settings and their corresponding results.

Common setup:
- Policy model backbone: `MobileNetV3-Small`
- Downstream classifier used for evaluation: `resnet50`
- Test split size: `600`
- Lens downstream reference accuracy on test split: `29.5% (177 / 600)`
- Upper bound downstream with `resnet50`: All set - `49.93333333333333`; Test set - `49.333333333333336`

## Result Table

| ID | Supervision Label | Input Sampling | Backbone Freeze | Unfrozen Parts | Loss | Best Val Index Acc | Test Index Acc | Test Downstream Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Lens label | baseline only | no explicit freeze record | not recorded | hard label | not recorded | `17.67%` | `32.00%` |
| B | Lens label | random candidate state augmentation | yes | `policy_head` only | hard label | `21.83%` | `19.00%` | `27.33%` |
| C | Oracle label | random candidate state augmentation | yes | `policy_head` only | hard label | `17.67%` | not separately recorded | `29.00%` |
| D | Oracle label | random candidate state augmentation | yes | `policy_head` only | soft label (`soft_kl`) | `17.83%` | not separately recorded | `29.33%` |
| E | Oracle label | random candidate state augmentation | partial freeze | `backbone[9:12]` + `feature_proj` + `policy_head` | soft label (`soft_kl`) | `18.33%` | `20.33%` | `30.17%` |

## Fixed Input-13 Result Table

These runs keep the training input fixed to candidate `option_id = 13` instead of using random-candidate state augmentation.

| ID | Supervision Label | Input Sampling | Backbone Freeze | Unfrozen Parts | Loss | Best Val Index Acc | Test Index Acc | Test Downstream Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-fixed13 | Lens label | fixed `option_id=13` | yes | `policy_head` only | hard label | `24.83%` | `22.00%` | `30.50%` |
| C-fixed13 | Oracle label | fixed `option_id=13` | yes | `policy_head` only | hard label | `18.67%` | `19.83%` | `31.33%` |
| D-fixed13 | Oracle label | fixed `option_id=13` | yes | `policy_head` only | soft label (`soft_kl`) | `16.50%` | `18.50%` | `29.33%` |
| E-fixed13 | Oracle label | fixed `option_id=13` | partial freeze | `backbone[9:12]` + `feature_proj` + `policy_head` | soft label (`soft_kl`) | `16.83%` | `19.83%` | `31.67%` |
| F-fixed13 | Oracle label | fixed `option_id=13` | no | full model | hard label | `18.67%` | `18.00%` | `28.83%` |
| G-fixed13 | Oracle label | fixed `option_id=13` | no | full model | soft label (`soft_kl`) | `16.50%` | `17.67%` | `29.50%` |

## Setting Notes

### A. Lens Label Baseline
- Label source: Lens confidence selection
- Input image: fixed baseline image
- Freeze status: not recorded in current project notes
- Test index imitation result comes from [`policy_network/results/test_best_index_analysis.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/test_best_index_analysis.json)
- Downstream result comes from [`policy_network/results/test_policy_selected_visit_acc.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/test_policy_selected_visit_acc.json)

### B. Lens Label + Freeze + State Augmentation
- Label source: Lens confidence selection
- Input image: random candidate from the same scene
- Freeze status: yes
- Frozen part: most of backbone
- Trainable part: `policy_head` only at that stage
- Best checkpoint: [`policy_network/results/policy_net_freeze.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/policy_net_freeze.pth)
- Test index result: [`policy_network/results/test_analysis_freeze_ver.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/test_analysis_freeze_ver.json)
- Downstream result: [`policy_network/results/test_freeze_policy_selected_visit_acc.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/test_freeze_policy_selected_visit_acc.json)

### C. Oracle Label + Freeze + Hard Label
- Label source: Oracle label
- Input image: random candidate from the same scene
- Freeze status: yes
- Trainable part: `policy_head` only
- Loss: hard label
- Best checkpoint: [`policy_network/results_oracle_policy/policy_net_freeze.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/policy_net_freeze.pth)
- Downstream result: [`policy_network/results_oracle_policy/test_oracle_compare.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/test_oracle_compare.json)

### D. Oracle Label + Freeze + Soft Label
- Label source: Oracle label
- Input image: random candidate from the same scene
- Freeze status: yes
- Trainable part: `policy_head` only
- Loss: soft label with `soft_kl`
- Best checkpoint: [`policy_network/results_oracle_policy/policy_net_freeze_soft.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/policy_net_freeze_soft.pth)
- Downstream result: [`policy_network/results_oracle_policy/test_oracle_soft_compare.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/test_oracle_soft_compare.json)

### E. Oracle Label + Soft Label + Stage 2 Partial Unfreeze
- Label source: Oracle label
- Input image: random candidate from the same scene
- Freeze status: partial freeze
- Frozen part: lower backbone layers
- Trainable part: `backbone[9:12]`, `feature_proj`, and `policy_head`
- Learning rates:
  `backbone_lr = 5e-6`
  `head_lr = 1e-4`
- Best checkpoint: [`policy_network/results_oracle_policy/policy_net_part_freeze_soft.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/policy_net_part_freeze_soft.pth)
- Test index result: [`policy_network/results_oracle_policy/test_result.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/test_result.json)
- Downstream result: [`policy_network/results_oracle_policy/test_oracle_part_soft_compare.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/test_oracle_part_soft_compare.json)

## Fixed Input-13 Setting Notes

### B-fixed13. Lens Label + Freeze + Fixed Input-13
- Label source: Lens confidence selection
- Input image: fixed candidate `option_id=13` during training
- Freeze status: yes
- Trainable part: `policy_head` only
- Best checkpoint: [`policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/B_lens_head_hard/policy_net_fixed13_head_hard.pth)
- Train history: [`policy_network/results_fixed_input13/B_lens_head_hard/train_history.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/B_lens_head_hard/train_history.json)
- Test index result: [`policy_network/results_fixed_input13/B_lens_head_hard/test_analysis_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/B_lens_head_hard/test_analysis_fixed13.json)
- Downstream result: [`policy_network/results_fixed_input13/B_lens_head_hard/test_policy_selected_visit_acc_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/B_lens_head_hard/test_policy_selected_visit_acc_fixed13.json)

### C-fixed13. Oracle Label + Freeze + Hard Label + Fixed Input-13
- Label source: Oracle label
- Input image: fixed candidate `option_id=13` during training
- Freeze status: yes
- Trainable part: `policy_head` only
- Loss: hard label
- Best checkpoint: [`policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/C_oracle_head_hard/policy_net_fixed13_head_hard.pth)
- Train history: [`policy_network/results_fixed_input13/C_oracle_head_hard/train_history.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/C_oracle_head_hard/train_history.json)
- Test index result: [`policy_network/results_fixed_input13/C_oracle_head_hard/test_analysis_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/C_oracle_head_hard/test_analysis_fixed13.json)
- Downstream result: [`policy_network/results_fixed_input13/C_oracle_head_hard/test_oracle_compare_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/C_oracle_head_hard/test_oracle_compare_fixed13.json)

### D-fixed13. Oracle Label + Freeze + Soft Label + Fixed Input-13
- Label source: Oracle label
- Input image: fixed candidate `option_id=13` during training
- Freeze status: yes
- Trainable part: `policy_head` only
- Loss: soft label with `soft_kl`
- Best checkpoint: [`policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/D_oracle_head_soft/policy_net_fixed13_head_soft.pth)
- Train history: [`policy_network/results_fixed_input13/D_oracle_head_soft/train_history.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/D_oracle_head_soft/train_history.json)
- Test index result: [`policy_network/results_fixed_input13/D_oracle_head_soft/test_analysis_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/D_oracle_head_soft/test_analysis_fixed13.json)
- Downstream result: [`policy_network/results_fixed_input13/D_oracle_head_soft/test_oracle_soft_compare_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/D_oracle_head_soft/test_oracle_soft_compare_fixed13.json)

### E-fixed13. Oracle Label + Soft Label + Stage 2 Partial Unfreeze + Fixed Input-13
- Label source: Oracle label
- Input image: fixed candidate `option_id=13` during training
- Freeze status: partial freeze
- Trainable part: `backbone[9:12]`, `feature_proj`, and `policy_head`
- Learning rates:
  `backbone_lr = 5e-6`
  `head_lr = 1e-4`
- Best checkpoint: [`policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/E_oracle_partial_soft/policy_net_fixed13_part_soft.pth)
- Train history: [`policy_network/results_fixed_input13/E_oracle_partial_soft/train_history.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/E_oracle_partial_soft/train_history.json)
- Test index result: [`policy_network/results_fixed_input13/E_oracle_partial_soft/test_analysis_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/E_oracle_partial_soft/test_analysis_fixed13.json)
- Downstream result: [`policy_network/results_fixed_input13/E_oracle_partial_soft/test_oracle_part_soft_compare_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/E_oracle_partial_soft/test_oracle_part_soft_compare_fixed13.json)

### F-fixed13. Oracle Label + Full Finetune + Hard Label + Fixed Input-13
- Label source: Oracle label
- Input image: fixed candidate `option_id=13` during training
- Freeze status: no
- Trainable part: full model (`full_finetune`)
- Loss: hard label
- Best checkpoint: [`policy_network/results_fixed_input13/F_oracle_full_hard/policy_net_fixed13_full_hard.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/F_oracle_full_hard/policy_net_fixed13_full_hard.pth)
- Train history: [`policy_network/results_fixed_input13/F_oracle_full_hard/train_history.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/F_oracle_full_hard/train_history.json)
- Test index result: [`policy_network/results_fixed_input13/F_oracle_full_hard/test_analysis_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/F_oracle_full_hard/test_analysis_fixed13.json)
- Downstream result: [`policy_network/results_fixed_input13/F_oracle_full_hard/test_oracle_full_hard_compare_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/F_oracle_full_hard/test_oracle_full_hard_compare_fixed13.json)

### G-fixed13. Oracle Label + Full Finetune + Soft Label + Fixed Input-13
- Label source: Oracle label
- Input image: fixed candidate `option_id=13` during training
- Freeze status: no
- Trainable part: full model (`full_finetune`)
- Loss: soft label with `soft_kl`
- Best checkpoint: [`policy_network/results_fixed_input13/G_oracle_full_soft/policy_net_fixed13_full_soft.pth`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/G_oracle_full_soft/policy_net_fixed13_full_soft.pth)
- Train history: [`policy_network/results_fixed_input13/G_oracle_full_soft/train_history.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/G_oracle_full_soft/train_history.json)
- Test index result: [`policy_network/results_fixed_input13/G_oracle_full_soft/test_analysis_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/G_oracle_full_soft/test_analysis_fixed13.json)
- Downstream result: [`policy_network/results_fixed_input13/G_oracle_full_soft/test_oracle_full_soft_compare_fixed13.json`](/mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/G_oracle_full_soft/test_oracle_full_soft_compare_fixed13.json)

## Current Best Downstream Result

- Best policy downstream accuracy currently recorded: `32.00%`
- This result comes from setup A, which is the earlier Lens-label baseline.

## Current Best Oracle-Based Result

- Best Oracle-based downstream accuracy currently recorded: `30.17%`
- This result comes from setup E, the soft-label Stage 2 partial-unfreeze model.

## Current Best Fixed Input-13 Result

- Best fixed-input-13 downstream accuracy currently recorded: `31.67%`
- This result comes from setup E-fixed13, the Oracle soft-label Stage 2 partial-unfreeze model trained with fixed `option_id=13` input.

## Fixed Input-13 Result Table After Fixing The Dataset Problem

These runs use the corrected `3/1/1` split with `1200` test samples.

| ID | Supervision Label | Input Sampling | Backbone Freeze | Unfrozen Parts | Loss | Best Val Index Acc | Test Index Acc | Test Downstream Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A-fixed13-newsplit | Lens label | fixed `option_id=13` | no | full model | hard label | `21.00%` | `18.58%` | `33.25%` |
| B-fixed13-newsplit | Lens label | fixed `option_id=13` | yes | `policy_head` only | hard label | `20.33%` | `19.50%` | `34.58%` |
| C-fixed13-newsplit | Oracle label | fixed `option_id=13` | yes | `policy_head` only | hard label | `19.92%` | `19.25%` | `33.67%` |
| D-fixed13-newsplit | Oracle label | fixed `option_id=13` | yes | `policy_head` only | soft label (`soft_kl`) | `17.50%` | `16.50%` | `31.33%` |
| E-fixed13-newsplit | Oracle label | fixed `option_id=13` | partial freeze | `backbone[9:12]` + `feature_proj` + `policy_head` | soft label (`soft_kl`) | `18.25%` | `19.00%` | `33.33%` |
| F-fixed13-newsplit | Oracle label | fixed `option_id=13` | no | full model | hard label | `18.42%` | `19.25%` | `33.92%` |
| G-fixed13-newsplit | Oracle label | fixed `option_id=13` | no | full model | soft label (`soft_kl`) | `17.58%` | `16.58%` | `30.42%` |

## Current Best Result After Fixing The Dataset Problem

- Best policy downstream accuracy on the corrected split: `34.58%`
- This result comes from setup B-fixed13-newsplit, the Lens-label head-only model trained with fixed `option_id=13` input.

## Current Best Oracle-Based Result After Fixing The Dataset Problem

- Best Oracle-based downstream accuracy on the corrected split: `33.92%`
- This result comes from setup F-fixed13-newsplit, the Oracle hard-label full-finetune model trained with fixed `option_id=13` input.

## Current Best Fixed Input-13 Result After Fixing The Dataset Problem

- Best fixed-input-13 downstream accuracy on the corrected split: `34.58%`
- This result comes from setup B-fixed13-newsplit, the Lens-label head-only model trained with fixed `option_id=13` input.

## Fixed Input-13 Downstream Best-vs-Last Summary

This table is read directly from `policy_network/results_fixed_input13/*/downstream_test_best.json` and `downstream_test_last.json`.

| ID | Supervision Label | Freeze | Loss | Test Index Acc | Downstream Best Top-1 | Downstream Best Top-5 Cumulative | Downstream Last Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Lens label | no, full model | hard label | `15.08%` | `29.25%` | `45.50%` | `24.58%` |
| B | Lens label | yes, `policy_head` only | hard label | `15.08%` | `29.50%` | `44.58%` | `27.42%` |
| C | Oracle hard label | yes, `policy_head` only | hard label | `18.42%` | `31.75%` | `45.42%` | `26.50%` |
| D | Oracle soft label | yes, `policy_head` only | soft label (`soft_kl`) | `17.92%` | `30.83%` | `45.00%` | `26.33%` |
| E | Oracle soft label | partial freeze, `backbone[9:12]` + `feature_proj` + `policy_head` | soft label (`soft_kl`) | `18.42%` | `31.42%` | `45.08%` | `26.67%` |
| F | Oracle hard label | no, full model | hard label | `19.33%` | `33.00%` | `45.67%` | `24.17%` |
| G | Oracle soft label | no, full model | soft label (`soft_kl`) | `19.25%` | `33.33%` | `45.08%` | `27.83%` |

## ResNet18 Downstream Best-vs-Last Summary

This table is read directly from `policy_network/results_resnet18/*/downstream_test_best.json`, `downstream_test_last.json`, and `index_test_result.json`.

| ID | Supervision Label | Freeze | Loss | Test Index Acc | Downstream Best Top-1 | Downstream Best Top-5 Cumulative | Downstream Last Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Lens label | no, full model | hard label | `17.58%` | `30.58%` | `45.00%` | `25.75%` |
| B | Lens label | yes, `policy_head` only | hard label | `16.83%` | `31.50%` | `45.33%` | `30.58%` |
| C | Oracle hard label | yes, `policy_head` only | hard label | `18.75%` | `30.33%` | `45.75%` | `30.58%` |
| D | Oracle soft label | yes, `policy_head` only | soft label (`soft_kl`) | `18.50%` | `31.50%` | `46.42%` | `29.42%` |
| E | Oracle soft label | partial freeze, `layer4` + `policy_head` | soft label (`soft_kl`) | `18.50%` | `31.50%` | `46.42%` | `28.33%` |
| F | Oracle hard label | no, full model | hard label | `16.33%` | `29.67%` | `45.42%` | `28.17%` |
| G | Oracle soft label | no, full model | soft label (`soft_kl`) | `15.58%` | `29.08%` | `44.42%` | `27.92%` |

## Dual MobileNetV3-Small Downstream Best-vs-Last Summary

This table is read directly from `policy_network/results_dual_mobilenet_v3_small/*/downstream_test_best.json`, `downstream_test_last.json`, and `index_test_result.json`.

| ID | Supervision Label | Freeze | Loss | Test Index Acc | Downstream Best Top-1 | Downstream Best Top-5 Cumulative | Downstream Last Top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Lens label | no, full model | hard label | `19.00%` | `34.00%` | `47.08%` | `29.08%` |
| B | Lens label | yes, `policy_head` only | hard label | `19.67%` | `34.67%` | `46.50%` | `31.58%` |
| C | Oracle hard label | yes, `policy_head` only | hard label | `19.58%` | `34.50%` | `48.00%` | `33.50%` |
| D | Oracle soft label | yes, `policy_head` only | soft label (`soft_kl`) | `18.83%` | `33.58%` | `47.42%` | `31.50%` |
| E | Oracle soft label | partial freeze, `backbone[9:12]` + `feature_proj` + `policy_head` | soft label (`soft_kl`) | `18.83%` | `33.58%` | `47.42%` | `31.67%` |
| F | Oracle hard label | no, full model | hard label | `21.33%` | `34.75%` | `47.42%` | `32.25%` |
| G | Oracle soft label | no, full model | soft label (`soft_kl`) | `19.58%` | `33.83%` | `47.67%` | `31.00%` |
