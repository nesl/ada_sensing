#!/usr/bin/env bash
# Test: drive ManualCam over adb (AE + 3 manual settings), pull the photos,
# and record the exposure the camera actually used for each one.
set -uo pipefail
S=${S:-192.168.1.177:42091}
OUT=${OUT:-/Users/malikh/Desktop/ada_sensing/testbed/data/qa_test}
mkdir -p "$OUT"
A="adb -s $S"
ctrl() { $A shell am broadcast -a com.malikh.manualcam.CTRL "$@" >/dev/null; }

$A shell input keyevent KEYCODE_WAKEUP
ctrl --es mode photo
$A logcat -c

shoot() {  # label, then optional ctrl args
  local label=$1; shift
  [ $# -gt 0 ] && ctrl "$@"
  sleep 1.5                      # let the setting settle on the preview
  local before; before=$($A shell ls /sdcard/DCIM/ManualCam 2>/dev/null | sort)
  ctrl --ez photo true
  for _ in $(seq 1 20); do       # wait for the new file to appear
    sleep 0.5
    new=$(comm -13 <(echo "$before") <($A shell ls /sdcard/DCIM/ManualCam 2>/dev/null | sort) | tail -1)
    [ -n "$new" ] && break
  done
  if [ -z "${new:-}" ]; then echo "$label: NO FILE"; return; fi
  sleep 1                        # let the write finish
  $A pull "/sdcard/DCIM/ManualCam/$new" "$OUT/${label}.jpg" >/dev/null
  echo "$label -> $new"
  sleep 1.1                      # filenames have 1-second resolution
}

shoot ae         --ez ae true
shoot iso100_1-1000 --ez ae false --ei iso 100  --el shutter_ns 1000000
shoot iso400_1-60   --ez ae false --ei iso 400  --el shutter_ns 16666667
shoot iso1600_1-8   --ez ae false --ei iso 1600 --el shutter_ns 125000000

ctrl --ez ae true
echo "--- actual exposures reported by the camera (last lines per setting) ---"
$A logcat -d | grep -E "actual=" | awk '{print $(NF-3), $(NF-2), $(NF-1), $NF}' | uniq | tail -12
ls -la "$OUT"
