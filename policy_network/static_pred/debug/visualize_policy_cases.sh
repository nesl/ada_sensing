python3 policy_network/static_pred/debug/visualize_policy_cases.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_resnet18/C_oracle_head_hard/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_dir policy_network/results_resnet18/C_oracle_head_hard/vis_cases \
  --device cuda \
  --num_samples 10 \
  --filter_mode all
