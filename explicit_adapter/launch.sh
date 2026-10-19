#!/bin/bash
# usage: launch.sh <specs.json> <gpu> <comma-separated spec names> <log name>
# Detached run of run.py on one GPU; log in _out/logs/<log name>.log
cd "$(dirname "$0")" && mkdir -p _out/logs
setsid nohup ~/miniforge3/envs/qae/bin/python run.py "$1" --gpu "$2" --only "$3" > "_out/logs/$4.log" 2>&1 < /dev/null &
echo "launched pid $! -> _out/logs/$4.log"
