#!/usr/bin/env bash
# Start (once) the VLM scoring loop on the node; pidfile-based so repeated calls are safe.
cd ~/qae && source ~/venv/bin/activate
if [ -f vlm_loop.pid ] && kill -0 "$(cat vlm_loop.pid)" 2>/dev/null; then echo "loop already running (pid $(cat vlm_loop.pid))"; exit 0; fi
pkill -f "[p]ython vlm_loop.py" 2>/dev/null; sleep 1
mkdir -p ~/qa/qae_batch/precheck ~/qa/qae_batch/sweeps
nohup python vlm_loop.py ~/qa/qae_batch/precheck ~/qa/qae_batch/sweeps --crop 602,854,2687,2027 > vlm_loop.log 2>&1 < /dev/null &
echo $! > vlm_loop.pid; echo "loop started pid $!"
