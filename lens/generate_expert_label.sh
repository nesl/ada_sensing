python3 lens/generate_expert_label.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --output_dir data/ImageNet-ES-Diverse/policy_labels \
  --model resnet50 \
  --train_groups_per_class 3 \
  --val_groups_per_class 1 \
  --test_groups_per_class 1 \
  --expected_num_classes 200 \
  --seed 0
