python3 policy_network/static_pred/debug/visualize_pred_vs_gt_index_distribution.py \
  --checkpoint policy_network/results_random_noise/G_oracle_full_soft/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --output_png policy_network/vis_results/random_noise_G_pred_vs_gt_index_distribution.png \
  --output_json policy_network/results_debug/random_noise_G_pred_vs_gt_index_distribution.json \
  --device cuda