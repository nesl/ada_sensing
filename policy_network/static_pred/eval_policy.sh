#!/usr/bin/env bash

set -euo pipefail

# Downstream accuracy of using policy network: Top1 - Top5 for both best and last pth

## A
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/A_lens_ft_all_hard/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/A_lens_ft_all_hard/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/A_lens_ft_all_hard/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/A_lens_ft_all_hard/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

## B
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/B_lens_head_hard/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/B_lens_head_hard/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/B_lens_head_hard/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/policy_labels/policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/B_lens_head_hard/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

## C
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/C_oracle_head_hard/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/C_oracle_head_hard/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/C_oracle_head_hard/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/C_oracle_head_hard/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

## D
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/D_oracle_head_soft/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/D_oracle_head_soft/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/D_oracle_head_soft/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/D_oracle_head_soft/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

## E
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/E_oracle_partial_soft/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/E_oracle_partial_soft/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/E_oracle_partial_soft/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/E_oracle_partial_soft/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

## F
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/F_oracle_full_hard/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/F_oracle_full_hard/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/F_oracle_full_hard/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/F_oracle_full_hard/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

## G
python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/G_oracle_full_soft/best_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/G_oracle_full_soft/downstream_test_best.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5

python3 policy_network/static_pred/debug/analyze_topk_downstream_candidates.py \
  --manifest data/ImageNet-ES-Diverse/manifest_all.json \
  --checkpoint policy_network/results_fixed_input13/G_oracle_full_soft/last_checkpoint.pth \
  --data_json data/ImageNet-ES-Diverse/oracle_policy_labels/oracle_policy_test_labels.json \
  --output_json policy_network/results_fixed_input13/G_oracle_full_soft/downstream_test_last.json \
  --image_size 224 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --topk 5


# Visualize the training history distribution of the policy network

## A
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/A_lens_ft_all_hard/train_history.json \
  --output_png policy_network/results_fixed_input13/A_lens_ft_all_hard/train_history_distribution_summary.png

## B
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/B_lens_head_hard/train_history.json \
  --output_png policy_network/results_fixed_input13/B_lens_head_hard/train_history_distribution_summary.png

## C
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/C_oracle_head_hard/train_history.json \
  --output_png policy_network/results_fixed_input13/C_oracle_head_hard/train_history_distribution_summary.png

## D
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/D_oracle_head_soft/train_history.json \
  --output_png policy_network/results_fixed_input13/D_oracle_head_soft/train_history_distribution_summary.png

## E
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/E_oracle_partial_soft/train_history.json \
  --output_png policy_network/results_fixed_input13/E_oracle_partial_soft/train_history_distribution_summary.png

## F
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/F_oracle_full_hard/train_history.json \
  --output_png policy_network/results_fixed_input13/F_oracle_full_hard/train_history_distribution_summary.png

## G
python3 policy_network/static_pred/debug/visualize_train_history_distribution.py \
  --history_json policy_network/results_fixed_input13/G_oracle_full_soft/train_history.json \
  --output_png policy_network/results_fixed_input13/G_oracle_full_soft/train_history_distribution_summary.png
