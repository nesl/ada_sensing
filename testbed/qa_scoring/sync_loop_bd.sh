#!/usr/bin/env bash
cd /Users/malikh/Desktop/ada_sensing/testbed
while true; do
  for i in 1 2 3; do
    rsync -az -e "ssh -o ConnectTimeout=25" --exclude '*.log' data/qa_dataset_bd/ chi-gigaio:~/qa/dataset_bd/ && \
    ssh -o ConnectTimeout=25 chi-gigaio 'bash ~/qa/node_incr_bd.sh' && break
    sleep 10
  done
  echo "$(date +%H:%M) synced $(ls data/qa_dataset_bd/*/manifest.json 2>/dev/null | wc -l | tr -d ' ') scenes"
  pgrep -f '[q]a_collect.py' >/dev/null || { echo "collector finished"; break; }
  sleep 300
done
