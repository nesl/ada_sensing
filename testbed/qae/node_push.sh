#!/usr/bin/env bash
# Copy the QAE scripts and calibration to a node (ssh alias as $1, default chi-qae), then run node_setup.sh there in the background.
set -euo pipefail
H=${1:-chi-qae}
T=$(dirname "$0")
ssh -o ConnectTimeout=25 "$H" 'mkdir -p ~/qae ~/qa/qae_batch/precheck ~/qa/qae_batch/sweeps ~/qa_scoring'
scp -o ConnectTimeout=25 -q "$T"/vlm_score.py "$T"/vlm_loop.py "$T"/model_text.py "$T"/label_curve.py "$T"/start_loop.sh "$T"/node_setup.sh "$H":~/qae/
scp -o ConnectTimeout=25 -q "$T"/../qa_scoring/jepa_qa.py "$T"/../data/qa_calib/layoutB.json "$H":~/qa/
ssh -o ConnectTimeout=25 "$H" 'chmod +x ~/qae/*.sh; nohup bash ~/qae/node_setup.sh > ~/qae/setup.log 2>&1 < /dev/null & echo "setup started pid $!"'
