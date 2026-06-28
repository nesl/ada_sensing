# python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
#   --manifest data/ImageNet-ES-Diverse/manifest_all.json \
#   --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/A_lens_ft_all_hard/best_checkpoint.pth \
#   --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
#   --output_dir policy_network/results_dual_mobilenet_v3_small/A_lens_ft_all_hard/vis_cases \
#   --device cuda \
#   --num_samples 30 \
#   --filter_mode all

# # B
# python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
#   --manifest data/ImageNet-ES-Diverse/manifest_all.json \
#   --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/B_lens_head_hard/best_checkpoint.pth \
#   --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
#   --output_dir policy_network/results_dual_mobilenet_v3_small/B_lens_head_hard/vis_cases \
#   --device cuda \
#   --num_samples 30 \
#   --filter_mode all

# # C
# python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
#   --manifest data/ImageNet-ES-Diverse/manifest_all.json \
#   --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/C_oracle_head_hard/best_checkpoint.pth \
#   --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
#   --output_dir policy_network/results_dual_mobilenet_v3_small/C_oracle_head_hard/vis_cases \
#   --device cuda \
#   --num_samples 30 \
#   --filter_mode all

# D
python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/D_oracle_head_soft/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_dir policy_network/results_dual_mobilenet_v3_small/D_oracle_head_soft/vis_cases \
  --device cuda \
  --num_samples 30 \
  --filter_mode all

# # E
# python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
#   --manifest data/ImageNet-ES-Diverse/manifest_all.json \
#   --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/E_oracle_partial_soft/best_checkpoint.pth \
#   --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
#   --output_dir policy_network/results_dual_mobilenet_v3_small/E_oracle_partial_soft/vis_cases \
#   --device cuda \
#   --num_samples 30 \
#   --filter_mode all

# F
python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/F_oracle_full_hard/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_dir policy_network/results_dual_mobilenet_v3_small/F_oracle_full_hard/vis_cases \
  --device cuda \
  --num_samples 30 \
  --filter_mode all

# G
# python3 policy_network/static_pred/debug/policy_visualization/visualize_policy_cases.py \
#   --manifest data/ImageNet-ES-Diverse/manifest_all.json \
#   --checkpoint /mnt/hdd1/yuyang/adaptive_sensing/Lenz/policy_network/results_dual_mobilenet_v3_small/G_oracle_full_soft/best_checkpoint.pth \
#   --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
#   --output_dir policy_network/results_dual_mobilenet_v3_small/G_oracle_full_soft/vis_cases \
#   --device cuda \
#   --num_samples 30 \
#   --filter_mode all
