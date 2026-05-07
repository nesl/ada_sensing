# this is for dual input to policy network, AE + fixed option_id=13

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# generate OpenCLIP soft label
python3 openclip_ds_policy/generate_openclip_soft_labels.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --class_index_json data/ImageNet-ES-Diverse/imagenet_class_index.json \
  --split_dir data/ImageNet-ES-Diverse/policy_labels \
  --output_dir data/openclip_labels/tau_0p05 \
  --openclip_model ViT-B-32 \
  --openclip_pretrained openai \
  --prompt_template "a photo of a {class_name}." \
  --tau 0.05 \
  --num_candidates 27 \
  --device cuda

# baselines
python3 openclip_ds_policy/eval_openclip_baselines.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --data_json data/openclip_labels/tau_0p05/openclip_soft_test_labels.json \
  --class_index_json data/ImageNet-ES-Diverse/imagenet_class_index.json \
  --output_json openclip_ds_policy/results/baselines/openclip_baselines_test.json \
  --openclip_model ViT-B-32 \
  --openclip_pretrained openai \
  --prompt_template "a photo of a {class_name}." \
  --device cuda

# train policy network with dual input: AE + fixed option_id=13
python3 policy_network/static_pred/train_policy.py \
  --train_json data/openclip_labels/tau_0p05/openclip_soft_train_labels.json \
  --val_json data/openclip_labels/tau_0p05/openclip_soft_val_labels.json \
  --test_json data/openclip_labels/tau_0p05/openclip_soft_test_labels.json \
  --save_dir openclip_ds_policy/results/dual_mobilenet_v3_small_fixedk_full_t0p05/fixed_k_13/oracle_soft_full \
  --manifest_json data/ImageNet-ES-Diverse/manifest_all.json \
  --image_size 224 \
  --num_candidates 27 \
  --backbone mobilenet_v3_small \
  --input_mode dual \
  --input_variant real \
  --env_option_id 13 \
  --batch_size 16 \
  --epochs 50 \
  --lr 2e-5 \
  --backbone_lr 1e-6 \
  --weight_decay 5e-4 \
  --device cuda \
  --num_workers 4 \
  --seed 0 \
  --pretrained \
  --trainable_scope full_finetune \
  --loss_type soft_kl

# evaluate policy-selected candidate with OpenCLIP top-1 downstream accuracy
python3 openclip_ds_policy/eval_policy_openclip_downstream.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --data_json data/openclip_labels/tau_0p05/openclip_soft_test_labels.json \
  --checkpoint openclip_ds_policy/results/dual_mobilenet_v3_small_fixedk_full_t0p05/fixed_k_13/oracle_soft_full/best_checkpoint.pth \
  --output_json openclip_ds_policy/results/dual_mobilenet_v3_small_fixedk_full_t0p05/fixed_k_13/oracle_soft_full/openclip_downstream_test_best.json \
  --class_index_json data/ImageNet-ES-Diverse/imagenet_class_index.json \
  --openclip_model ViT-B-32 \
  --openclip_pretrained openai \
  --prompt_template "a photo of a {class_name}." \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda

# update xlsx result summary
python3 openclip_ds_policy/summarize_results_xlsx.py \
  --results_dir openclip_ds_policy/results \
  --output_xlsx openclip_ds_policy/results/results_summary.xlsx
