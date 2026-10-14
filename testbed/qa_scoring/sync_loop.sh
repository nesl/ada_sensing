#!/usr/bin/env bash
# Mac side: push completed scenes to the GPU node and trigger incremental scoring.
cd /Users/malikh/Desktop/ada_sensing/testbed
while true; do
  for i in 1 2 3; do
    rsync -az -e "ssh -o ConnectTimeout=25" --exclude '*.log' data/qa_dataset/ chi-gigaio:~/qa/dataset/ && \
    ssh -o ConnectTimeout=25 chi-gigaio 'bash ~/qa/node_incr.sh' && break
    sleep 10
  done
  echo "$(date +%H:%M) synced $(ls data/qa_dataset/*/manifest.json 2>/dev/null | wc -l | tr -d ' ') scenes"
  pgrep -f '[q]a_collect.py' >/dev/null || { echo "collector finished"; break; }
  sleep 480
done
