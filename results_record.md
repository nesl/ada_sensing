# Policy Training Results Record

This file records the current policy-training settings and their corresponding results.

Common setup:
- Policy model backbone: `MobileNetV3-Small`
- Downstream classifier used for evaluation: `resnet50`
- Test split size: `600`
- Lens downstream reference accuracy on test split: `29.5% (177 / 600)`

## Result Table

| ID | Supervision Label | Input Sampling | Backbone Freeze | Unfrozen Parts | Loss | Best Val Index Acc | Test Index Acc | Test Downstream Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | Lens label | baseline only | no explicit freeze record | not recorded | hard label | not recorded | `17.67%` | `32.00%` |
| B | Lens label | random candidate state augmentation | yes | `policy_head` only | hard label | `21.83%` | `19.00%` | `27.33%` |
| C | Oracle label | random candidate state augmentation | yes | `policy_head` only | hard label | `17.67%` | not separately recorded | `29.00%` |
| D | Oracle label | random candidate state augmentation | yes | `policy_head` only | soft label (`soft_kl`) | `17.83%` | not separately recorded | `29.33%` |
| E | Oracle label | random candidate state augmentation | partial freeze | `backbone[9:12]` + `feature_proj` + `policy_head` | soft label (`soft_kl`) | `18.33%` | `20.33%` | `30.17%` |

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

## Current Best Downstream Result

- Best policy downstream accuracy currently recorded: `32.00%`
- This result comes from setup A, which is the earlier Lens-label baseline.

## Current Best Oracle-Based Result

- Best Oracle-based downstream accuracy currently recorded: `30.17%`
- This result comes from setup E, the soft-label Stage 2 partial-unfreeze model.

