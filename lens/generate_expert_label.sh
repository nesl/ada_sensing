python3 generate_expert_label.py \
  --manifest ../data/ImageNet-ES-Diverse/manifest_all.json \
  --output_dir ../data/ImageNet-ES-Diverse/policy_labels \
  --model resnet50 \
  --baseline_option_id 13 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 0