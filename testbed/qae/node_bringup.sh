#!/usr/bin/env bash
# Mac side: wait for the lease, create the bare-metal instance, attach a floating IP, add the ssh alias,
# then push the QAE files and start node_setup.sh on the node.
#   bash qae/node_bringup.sh <lease name> <instance name> <ssh alias> [cloud]
set -uo pipefail
LEASE=$1; INST=$2; ALIAS=$3; CLOUD=${4:-tacc_chi_241284}
export PATH=$HOME/chi-venv/bin:$PATH OS_CLIENT_CONFIG_FILE=~/.config/openstack/clouds-all.yaml OS_CLOUD=$CLOUD
T=$(dirname "$0")
log() { echo "$(date '+%H:%M:%S') $*"; }

for i in $(seq 1 60); do
  st=$(openstack reservation lease show "$LEASE" -f value -c status 2>/dev/null)
  [ "$st" = "ACTIVE" ] && break
  log "lease $LEASE is $st, waiting"; sleep 30
done
RID=$(openstack reservation lease show "$LEASE" -f value -c reservations 2>/dev/null | python3 -c "import re,sys; print(re.search(r'\"id\": \"([0-9a-f-]+)\"', sys.stdin.read()).group(1))")
log "lease active, reservation $RID"

if ! openstack server show "$INST" >/dev/null 2>&1; then
  openstack server create --image CC-Ubuntu22.04-CUDA --flavor baremetal --key-name berth-key \
    --network sharednet1 --hint reservation="$RID" "$INST" >/dev/null && log "server create issued" || { log "server create FAILED"; exit 1; }
fi
for i in $(seq 1 60); do
  st=$(openstack server show "$INST" -f value -c status 2>/dev/null)
  [ "$st" = "ACTIVE" ] && break
  [ "$st" = "ERROR" ] && { log "server ERROR"; openstack server show "$INST" -f value -c fault; exit 1; }
  log "server $st"; sleep 30
done
IP=$(openstack server show "$INST" -f value -c addresses 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | grep -vE '^10\.' | head -1)
if [ -z "$IP" ]; then
  IP=$(openstack floating ip create public -f value -c floating_ip_address)
  openstack server add floating ip "$INST" "$IP" && log "floating ip $IP attached"
fi
log "instance $INST at $IP"

python3 - "$ALIAS" "$IP" <<'EOF'
import re, sys, pathlib
alias, ip = sys.argv[1:3]
p = pathlib.Path.home() / ".ssh" / "config"; s = p.read_text()
block = f"Host {alias}\n    HostName {ip}\n    User cc\n    IdentityFile ~/.ssh/id_ed25519\n    StrictHostKeyChecking accept-new\n"
s2, n = re.subn(rf"Host {re.escape(alias)}\n(?:[ \t]+.*\n)*", block, s)
p.write_text((s2 if n else s.rstrip("\n") + "\n\n" + block))
EOF
ssh-keygen -R "$IP" >/dev/null 2>&1
for i in $(seq 1 40); do
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$ALIAS" 'echo ssh_ok' 2>/dev/null && break
  log "ssh not up yet"; sleep 30
done
bash "$T/node_push.sh" "$ALIAS" && log "push + setup started"
for i in $(seq 1 120); do
  out=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$ALIAS" 'tail -1 ~/qae/setup.log 2>/dev/null' 2>/dev/null)
  echo "$out" | grep -q "SETUP_DONE" && { log "SETUP_DONE on $ALIAS"; break; }
  log "setup: ${out:0:80}"; sleep 60
done
log "BRINGUP_DONE"
