#!/usr/bin/env python3
"""Homography calibration: locate the iPad screen quad in a white-screen capture
and save the four corners for ffmpeg's perspective filter. Unlike the old
bounding-box crop, this registers the capture to stimulus pixel coordinates
exactly, correcting tilt and keystone.

Usage:
  python3 calibrate_homography.py record          # record + detect + save
  python3 calibrate_homography.py detect <video>  # detect from existing capture
  python3 calibrate_homography.py filter          # print the ffmpeg filter string
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FFMPEG = "/opt/homebrew/bin/ffmpeg"
ADB = "/opt/homebrew/share/android-commandlinetools/platform-tools/adb"
HERE = Path(__file__).parent
HOMOGRAPHY_FILE = HERE / "homography.json"
SCRATCH = Path("/private/tmp/claude-501/-Users-malikh-Desktop/"
               "a49234e6-7c60-42a4-b26f-63809ce6157a/scratchpad")


def sh(*cmd):
    return subprocess.run([str(c) for c in cmd], check=True,
                          capture_output=True, text=True)


def detect(video):
    """Find the bright screen quad's corners in a white-screen capture."""
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", "2", "-i", str(video), "-frames:v", "1",
         "-vf", "tmix=frames=5,format=gray", "-f", "rawvideo", "-"],
        check=True, capture_output=True).stdout
    # captures are 1920x1080 after the rotation fix
    w, h = 1920, 1080
    img = np.frombuffer(raw[:w * h], dtype=np.uint8).reshape(h, w).astype(np.float64)

    lo, hi = np.percentile(img, [5, 95])
    mask = img > (lo + hi) / 2
    ys, xs = np.nonzero(mask)
    if len(xs) < 10000:
        sys.exit("no bright screen found - is white.mp4 playing?")

    # quad corners as extreme points; robust to rotation unlike a bounding box
    s = xs + ys
    d = xs - ys
    corners = {
        "tl": [int(xs[s.argmin()]), int(ys[s.argmin()])],
        "tr": [int(xs[d.argmax()]), int(ys[d.argmax()])],
        "bl": [int(xs[d.argmin()]), int(ys[d.argmin()])],
        "br": [int(xs[s.argmax()]), int(ys[s.argmax()])],
    }
    data = {"corners": corners, "src_w": w, "src_h": h}
    HOMOGRAPHY_FILE.write_text(json.dumps(data, indent=2))
    print(f"corners: {corners} -> {HOMOGRAPHY_FILE}")
    return data


def filter_string(data=None):
    """ffmpeg perspective filter mapping the screen quad to the full frame."""
    data = data or json.loads(HOMOGRAPHY_FILE.read_text())
    c = data["corners"]
    return ("perspective="
            f"{c['tl'][0]}:{c['tl'][1]}:{c['tr'][0]}:{c['tr'][1]}:"
            f"{c['bl'][0]}:{c['bl'][1]}:{c['br'][0]}:{c['br'][1]}:"
            "interpolation=linear:sense=source")


def record_and_detect():
    sys.path.insert(0, str(HERE))
    from run_ae_sweep import cue, newest_capture

    def ctrl(*args):
        sh(ADB, "shell", "am", "broadcast", "-a", "com.malikh.manualcam.CTRL", *args)

    ctrl("--es", "mode", "video", "--ez", "ae", "true")
    cue("play", "white.mp4", loop=True)
    time.sleep(4)
    before = newest_capture()
    ctrl("--es", "record", "start")
    time.sleep(6)
    ctrl("--es", "record", "stop")
    time.sleep(2.5)
    cue("black")
    cap = newest_capture()
    if cap is None or cap == before:
        sys.exit("no capture recorded")
    local = SCRATCH / "homography_cal.mp4"
    sh(ADB, "pull", f"/sdcard/DCIM/ManualCam/{cap}", local)
    sh(ADB, "shell", "rm", f"/sdcard/DCIM/ManualCam/{cap}")
    detect(local)


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record_and_detect()
    elif sys.argv[1] == "detect":
        detect(sys.argv[2])
    elif sys.argv[1] == "filter":
        print(filter_string())
