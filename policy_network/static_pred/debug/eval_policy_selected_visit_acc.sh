python3 /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/static_pred/debug/eval_policy_selected_visit_acc.py \
  --manifest /mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/policy_net_part_freeze_soft.pth \
  --data_json /mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_oracle_policy/test_oracle_part_soft_compare.json \
  --device cuda
