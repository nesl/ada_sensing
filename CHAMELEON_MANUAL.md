# Chameleon GPU manual (from the Berth work, Oct 2026)

Everything we learned about getting a GPU node on Chameleon and working on it, so that another session can do it
without re-discovering the traps. Times in this file are Pacific; Chameleon's API speaks UTC.

## 1. Access (already set up on this Mac)

- CLI: the OpenStack client lives in a venv: `export PATH=$HOME/chi-venv/bin:$PATH` (plain `openstack` is not
  on the PATH otherwise; `timeout` does not exist on macOS either).
- Credentials: `export OS_CLIENT_CONFIG_FILE=~/.config/openstack/clouds-all.yaml`. One cloud entry per site,
  all for project 241284: `tacc_chi_241284` (CHI@TACC), `uc_chi_241284` (CHI@UC), `ncar_chi_241284`
  (CHI@NCAR), also `nu_chi_241284`, `nrp_chi_241284`, `edge_chi_241284`, `kvm_chi_241284`. Select with
  `export OS_CLOUD=tacc_chi_241284`.
- The token comes from `ccauth` (device-flow login in the browser). If a command fails with an auth error, run
  `ccauth` again (the user has to approve in the browser) — ask the user to do it.
- Every `openstack reservation ...` call prints a harmless warning (`Could not load 'reservation_host_unset'`).
- SSH key pair on every site: `berth-key` (private key `~/.ssh/id_ed25519`). Instances log in as user `cc`.

Quick check that everything works:
```
export PATH=$HOME/chi-venv/bin:$PATH OS_CLIENT_CONFIG_FILE=~/.config/openstack/clouds-all.yaml OS_CLOUD=tacc_chi_241284
openstack reservation lease list -f value -c name -c start_date -c end_date -c status
```

## 2. What GPUs exist where (verify, the inventory moves)

```
openstack reservation host list -f value -c node_name -c node_type -c gpu.gpu_model 2>/dev/null | grep -i "liqid\|gpu\|gh200"
```
- CHI@TACC: `compute_liqid` nodes (liqid02, 04, 06, 07, 08, ...): **one A100 PCIe 40 GB** each, AMD EPYC 7763,
  256 GB RAM, 420 GB disk. This is where our paper machine is (liqid06). About 4 SU/h.
- CHI@UC: `gpu_a100_pcie` nodes: **4× A100 PCIe 80 GB**. About 16 SU/h. Also `gpu_rtx_6000` and others.
- CHI@NCAR: `compute_gh200` nodes (gh02, ...): GH200 (Grace + H100 96 GB). Often booked weeks ahead.
- Allocation: a few thousand SU; the project page on chameleoncloud.org shows the balance.

## 3. Leases: the rules that bit us

- A lease is at most **7 days**. It can be **extended only within 4 days of its end** (Blazar: "can only be
  extended within 345600 seconds"), and only if the node is free after it:
  `openstack reservation lease set --end-date "2026-10-20 16:00" <lease>` (UTC).
- To make sure the node is still yours for the next week, create a **placeholder lease on the same node** for
  the following period now (pending leases cost nothing), then on the day the extension becomes possible delete
  the placeholder and extend the real lease. Blazar wants a few minutes' gap between two leases on one node
  (a 1-minute gap is refused, 5 minutes works).
- A **new lease on the same node re-provisions it: the disk is wiped.** Extending keeps the instance. Plan the
  setup (section 5) as scripts so that a re-provision costs an afternoon, not days.
- Lease start times in the past are rejected; "now" means a minute or two ahead. A lease that is pending cannot
  be used until it starts; cancelling it before the start is free.
- Reserve a specific node by name:
```
openstack reservation lease create --start-date "2026-10-13 16:05" --end-date "2026-10-20 16:00" \
  --reservation 'min=1,max=1,resource_type=physical:host,resource_properties=["==","$node_name","liqid06"]' <lease name>
```
  or any node of a type: `resource_properties=["==","$node_type","compute_liqid"]`.
- "Not enough resources available" = that node/type is taken in that window (or your own placeholder holds it).
- Useful: `openstack reservation lease show <lease> -f value -c start_date -c end_date -c status -c reservations`
  (the reservation `id` inside is what the instance needs).

Our leases now: `berth-liqid` (TACC, liqid06, ACTIVE to Oct 13 9:00 am; instance `berth-a100-40`,
129.114.108.212), `berth-liqid-hold` (same node, Oct 13 9:05 am → Oct 20, placeholder), `berth-gh200` (NCAR
gh02, Oct 16 5:06 pm → Oct 23, pending). **Do not touch liqid06 / `berth-a100-40` while the matrix runs there.**

## 4. Instance

```
openstack server create --image CC-Ubuntu22.04-CUDA --flavor baremetal --key-name berth-key \
  --network sharednet1 --hint reservation=<reservation id from the lease> <instance name>
openstack floating ip create public -f value -c floating_ip_address        # then
openstack server add floating ip <instance name> <that address>
openstack server list -f value -c Name -c Status -c Networks
```
- Provisioning takes 10–15 minutes (bare metal boot). `Status` goes BUILD → ACTIVE; ssh works a few minutes
  after ACTIVE. Images: `CC-Ubuntu22.04-CUDA` (what we use: driver 580.178.04, CUDA 12.8 at `/usr/local/cuda`
  with ncu and nsys, Docker not installed), plus CentOS/older CUDA images; `openstack image list | grep CUDA`.
- Add an ssh alias on the Mac (`~/.ssh/config`): `Host <alias>`, `HostName <floating ip>`, `User cc`,
  `IdentityFile ~/.ssh/id_ed25519`, `StrictHostKeyChecking accept-new`.

## 5. Node setup (what the paper machine has; script it)

```
sudo apt-get update && sudo apt-get install -y python3-venv cmake patchelf gdb unrar
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install torch torchvision transformers numpy pyyaml pandas      # torch 2.8 cu128 wheels work with driver 580
# CPU power management (mandatory for latency work): performance governor, idle states deeper than C1 off
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee $f >/dev/null; done
for f in /sys/devices/system/cpu/cpu*/cpuidle/state[2-9]/disable; do echo 1 | sudo tee $f >/dev/null; done
sudo nvidia-smi -pm 1
```
- `sudo` works without a password (`sudo -n`).
- Docker + NVIDIA container toolkit (for Tally): see `berth/third_party/baselines/tally/BUILD.md`; the Tally
  image is 223 GB on disk, an hour to pull.
- Builds we have recipes for: Berth shim/daemon (`make shim CUDA=/usr/local/cuda` with
  `CPLUS_INCLUDE_PATH=/usr/local/cuda/extras/CUPTI/include LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64`),
  XSched, TGS, Orion, Tally: each under `berth/third_party/baselines/<name>/BUILD.md`.
- The suite's per-host steps: `python suite/run.py host --out D`, `calibrate`, `suite/mig_cal.py`,
  `suite/orion_cal.py` (see `berth/HANDOFF.md` section 6).

## 6. Working with the node from a session (the traps)

- **SSH is flaky** (connection resets, occasional 30-s stalls). Wrap every call in a retry loop:
  `for i in 1 2 3; do ssh -o ConnectTimeout=25 <alias> '...' && break; sleep 10; done`.
- **Never chain `A && nohup B &` over ssh**: the session hangs. Write a script, `scp` it, run it with one short
  ssh, and start long work with `nohup bash script.sh > log 2>&1 &` inside that script; poll the log.
- `pkill -f <pattern>` over ssh matches the ssh shell itself and kills your session: use bracket patterns
  (`pkill -f '[t]enant.py'`) or PIDs.
- **Process names are truncated to 15 characters**: `pgrep -x nvidia-cuda-mps-server` never matches (the name is
  `nvidia-cuda-mps`). Use `pgrep -f '[n]vidia-cuda-mps-(server|control)'`. A stale MPS server silently
  survives otherwise and blocks MIG mode changes and pollutes "no-MPS" runs.
- Editing a bash script while it runs breaks it (bash reads scripts incrementally). Copy, then start.
- MIG: `sudo nvidia-smi -i 0 -mig 1` works without a reboot only when nothing holds the GPU; instance profiles
  with `nvidia-smi mig -lgip`; create largest-first (`-cgi 9,14,14 -C` works, `14,14,9` does not); 4g+3g is
  accepted on driver 580. Always `-dci; -dgi; -mig 0` afterwards.
- Time slice: `sudo nvidia-smi compute-policy -i 0 --set-timeslice=N` (0 default, 1 short, 2 medium, 3 long);
  the option order matters.
- `ncu` needs `sudo` (performance counters) and, under sudo, `$HOME` is root's: pass absolute paths.
- Clock: the node is on UTC; subtract 7 h for Pacific.
- Leave nothing behind: results live only on the node's disk. `scp -r` result folders to the Mac
  (`berth/results/<date>_<what>_<gpu>_<site>/`) before the lease ends; the instance is destroyed with it.

## 7. Costs and etiquette

- SU are charged per node-hour while the lease is active (whether or not you use the GPU): TACC single-A100
  node ≈ 4 SU/h (≈ 670 SU/week), UC 4×A100 ≈ 16 SU/h. Check the balance before a week-long lease.
- Delete leases you will not use (`openstack reservation lease delete <lease>`), especially pending ones that
  block a node for others.
