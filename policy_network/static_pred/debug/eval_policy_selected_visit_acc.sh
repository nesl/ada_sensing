python3 /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/static_pred/debug/eval_policy_selected_visit_acc.py \
  --manifest /mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/best_policy_net.pth \
  --data_json /mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_json /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results/test_policy_selected_visit_acc.json \
  --device cuda
