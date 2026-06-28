python3 /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/static_pred/debug/downstream_eval/eval_policy_selected_visit_acc.py \
  --manifest /mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input13/A_lens_ft_all_hard/best_checkpoint.pth \
  --data_json /mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_json /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_fixed_input/lens_downstream.json \
  --device cuda
