# Lenz Project Notes
Results are logged in `results_record.md`

## Problems and Confusions Encountered

### Environment mismatch during assistant-side execution
The assistant environment initially did not use the correct conda environment.

Observed issue:
- missing `timm` in the default environment

User clarified that the correct environment is:
- `(/mnt/hdd1/yuyang/install/conda_envs/lens)`

As a result:
- code execution from the assistant side should not be trusted unless the correct `lens` conda env is active
- future runs should be done by the user or only after explicitly activating the correct env

### User preference
1. the user prefer continuous bash script instead of "case" style. e.g. the user preffered format is like `python3 ..` then directly list all scripts and arguments.
2. After writing a code, point out the path and the command (or bash script), do not directly run the code unless requested.

## Model Diagnosis
1. New training input
As we believe that the input index should be fixed to ensure stable training, we now fix the index to be 13, and got all the results under folder `policy_network/results_fixed_input13`. We edited the corresponding files: `policy_dataset.py`, `train_policy.py`

2. Analyze index prediction
Introduce the top5 index prediction accuracy in file `debug/analyze_best_index_predictions`, showing that top5 prediction as a much higher hit accuracy.

3. New training
Introduce new setting F&G.
- F: full fintune with hard oracle label. Results in folder `results_fixed_input13/F_oracle_full_hard`
- G: full fintune with soft oracle label. Results in folder `results_fixed_input13/G_oracle_full_soft`

4. Discovered dataset bias
The old splittign strategy has a bias on ImageNet classes. New dataset splited by:
For 5 images in each class: split 3/1/1 for train/val/test.
I fixed the dataset problem and got the new one under `data/oracle_policy_labels` and `data/policy_labels`

5. Problem of input image
Original we use fixed index 13 as the parameter setting of inup image for policy network. Bu† we discovered that the image is too dark for policy network to extract useful information. Currently we use the auto_exposure param1 as the input for policy network. Rectified correspoing scripts: `policy_dataset.py` `train_policy.py`and all the dataset label generation scripts.

6. Debug by changing the input image of policy network to random noise. also tested on random noise data. We ended up getting similar accuracy comparing to using auto exposure / fixed index images as input. This possibilly proves that the policy network is only learning the index distribution without prio (the image's information of both lightness and classes) 

7. Debug by sending the lighting and class information directly to the MLP in. order to select the index for downstream task. Enocde the lightiing and class by sinusodal encoder

8. Important note: we should run all the experiment on cuda instead of cpu!
