# Report in responce of Debugging Experiments
## Verify Possible Results Mix-up
[Results for the suspicious random noise](policy_network/results_random_noise/G_oracle_full_soft)

[Results for all the baselines](policy_network/results/acquisition_baselines_test.json)
1. checked the random noise input index predictions

[policy_network/static_pred/debug/summarize_downstream_selected_indices.py](policy_network/static_pred/debug/summarize_downstream_selected_indices.py)

    ```
    total=1200
    downstream_acc=33.92%
    selected_equals_target=17.75% 

    option_id=24 param_1   count=978
   
    option_id=8  param_11  count=66

    option_id=12 param_2   count=139

    option_id=18 param_21  count=7

    option_id=21 param_10  count=6

    option_id=9  param_5   count=2

    option_id=15 param_15  count=2
    ```

2. Diagonse the index pred of noise input
    How about we manually set all the other option id to index 24, will at get the same downstream accuracy?  
    [policy_network/static_pred/debug/counterfactual_force_option.py](policy_network/static_pred/debug/counterfactual_force_option.py)

    Result: we got accuracy at 408/1200 (34.0%), while the baseline got 407/1200 (33.92%). 
    Reason: baseline run on cpu while random noise experiment run on cuda. 
    Fix: rerun the baselines on cuda

    Fixed result on cuda: [policy_network/results/acquisition_baselines_test_cuda.json](policy_network/results/acquisition_baselines_test_cuda.json) and it aligns with previous results.

3. A side note: the option_id to param mapping  
[policy_network/results_random_noise/G_oracle_full_soft/downstream_selected_index_stats.json](policy_network/results_random_noise/G_oracle_full_soft/downstream_selected_index_stats.json)

    The `option_name_map` field defines the mapping from each `option_id` to its corresponding sensor parameter setting.

