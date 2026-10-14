#!/usr/bin/env bash
# Live view of the RPi Camera 2 (imx219) on the Pi 3, tunnelled over SSH
# via the ThinkPad jump host. Requires the `rpi` Host entry in ~/.ssh/config.
#
#   ./rpi_cam_stream.sh              # 1280x720 @ 30fps
#   ./rpi_cam_stream.sh 1640 1232 25 # full-FOV binned mode
set -euo pipefail

W=${1:-1280}
H=${2:-720}
FPS=${3:-30}

echo "Streaming ${W}x${H} @ ${FPS}fps from rpi — ctrl-C to stop"

ssh rpi "libcamera-vid -t 0 --width $W --height $H --framerate $FPS \
    --codec h264 --inline --nopreview -o - 2>/dev/null" \
  | ffplay -hide_banner -loglevel warning \
      -fflags nobuffer -flags low_delay -framedrop \
      -window_title "rpi cam (${W}x${H})" -i -
