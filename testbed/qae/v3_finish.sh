#!/usr/bin/env bash
# Mac side, unattended tail of the v3 batch: wait for the pre-check phase, then for every pre-check to have a
# verdict and every passing scene to be swept; stop the sweep loop; wait for all sweep scores; run the node's
# model + label-curve script; pull results. Logs to data/qae_v3_finish.log.
cd "$(dirname "$0")/.." || exit 1
log() { echo "$(date '+%H:%M:%S') $*"; }
until grep -q PRECHECK_PHASE_DONE data/qae_batch_precheck_v3.log 2>/dev/null; do sleep 120; done
log "pre-check phase done"
while true; do
  python3 - <<'EOF' && break
import json, sys
from pathlib import Path
P = Path("data/qae_batch/precheck"); S = Path("data/qae_batch/sweeps")
pre = [d for d in P.glob("v3_*") if (d / "manifest.json").exists()]
nov = [d.name for d in pre if not (d / "verdict.json").exists()]
passing = [d.name for d in pre if (d / "verdict.json").exists() and json.loads((d / "verdict.json").read_text())["pass"]]
unswept = [n for n in passing if not (S / n / "manifest.json").exists()]
print(f"{len(pre)} pre-checks, {len(nov)} without verdict, {len(passing)} passing, {len(unswept)} unswept", flush=True)
sys.exit(0 if not nov and not unswept else 1)
EOF
  sleep 180
done
log "all passing scenes swept; stopping the sweep loop"
pkill -f "[b]atch.py sweep" ; sleep 2
while true; do
  n=$(ls -d data/qae_batch/sweeps/v3_* 2>/dev/null | wc -l | tr -d ' '); m=$(ls data/qae_batch/sweeps/v3_*/vlm.json 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] && [ "$n" -eq "$m" ] && break
  log "sweep scores: $m / $n"; sleep 120
done
log "all $n v3 sweeps scored; display check"
python3 qae/check_display.py data/qae_batch/sweeps --thr 0 > data/qae_v3_display_check.txt 2>&1
ssh -o ConnectTimeout=25 chi-qae 'nohup bash ~/qae/run_v3.sh > ~/qae/run_v3.out 2>&1 < /dev/null & echo started'
log "run_v3.sh started on the node"
until ssh -o ConnectTimeout=25 -o BatchMode=yes chi-qae 'grep -q CURVE_V3_DONE ~/qae/results_v3/curve.log' 2>/dev/null; do sleep 300; done
mkdir -p qae/results_v3
scp -o ConnectTimeout=25 -q "chi-qae:~/qae/results_v3/*.json" "chi-qae:~/qae/results_v3/*.log" qae/results_v3/
log "V3_FINISH_DONE"
