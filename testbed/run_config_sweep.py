#!/usr/bin/env python3
"""Config sweep: record the SAME clip as video once per exposure config
(ISO x shutter grid + one AE baseline), each capture cropped, aligned, and
saved under a config-labeled name.

Usage: python3 run_config_sweep.py <output_dir> <clip_path>
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from run_ae_sweep import (ADB, FFMPEG, CROP, HERE, cue, duration, gray_frames,
                          content_start_offset, newest_capture, sh)
from qr_overlay import qr_start_offset
from calibrate_homography import filter_string

# Coarse grid for the first pass (full-stop native-app values, video-mode range)
ISOS = [50, 200, 800, 3200]
SHUTTER_DENOMS = [2000, 500, 125, 30]  # 1/n s; video cap is 1/30 at 30 fps
SETTLE_S = 1.0          # let the pipeline apply a new config before recording
MAX_RECORD_S = 45


def ctrl(*args):
    sh(ADB, "shell", "am", "broadcast", "-a", "com.malikh.manualcam.CTRL", *args)


def one_capture(clip, rec_s, label, out_dir, manifest):
    cue("black")
    time.sleep(1.5)
    before = newest_capture()
    ctrl("--es", "record", "start")
    time.sleep(0.7)
    cue("play", clip.name, loop=False)
    time.sleep(rec_s)
    ctrl("--es", "record", "stop")
    time.sleep(2.5)

    cap = newest_capture()
    if cap is None or cap == before:
        print(f"  !! no capture for {label}", flush=True)
        return

    raw = out_dir / f"{label}_raw.mp4"
    sh(ADB, "pull", f"/sdcard/DCIM/ManualCam/{cap}", raw)
    sh(ADB, "shell", "rm", f"/sdcard/DCIM/ManualCam/{cap}")

    # Perspective-rectify the screen quad to full 1920x1080 stimulus coordinates
    cropped = out_dir / f"{label}_tmp.mp4"
    sh(FFMPEG, "-v", "error", "-i", raw,
       "-vf", filter_string(),
       "-c:v", "libx264", "-crf", "18", "-an", "-y", cropped)
    raw.unlink()

    original = out_dir / f"original{clip.suffix}"
    # Marker strip is the primary aligner (exposure-proof); NCC is the fallback
    # for clips without the overlay.
    offset, score = qr_start_offset(cropped)
    method = "qr"
    if offset is None:
        offset, score = content_start_offset(cropped, original)
        method = "ncc"
    offset = max(0.0, offset)
    captured = out_dir / f"{label}_captured.mp4"
    sh(FFMPEG, "-v", "error", "-ss", f"{offset:.3f}", "-i", cropped,
       "-t", f"{duration(original):.3f}",
       "-c:v", "libx264", "-crf", "18", "-an", "-y", captured)
    cropped.unlink()

    manifest.append({"config": label, "device_file": cap,
                     "align_offset_s": round(offset, 3),
                     "align_method": method,
                     "align_score": round(score, 3)})
    print(f"  ok -> {captured.name} ({method} offset {offset:.2f}s, "
          f"score {score:.2f})", flush=True)


def main():
    out_dir = Path(sys.argv[1])
    clip = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    original = out_dir / f"original{clip.suffix}"
    if not original.exists():
        shutil.copy2(clip, original)

    rec_s = min(duration(clip) + 4.0, MAX_RECORD_S)
    configs = [("ae", None, None)] + [
        (f"iso{i}_s{d}", i, d) for i in ISOS for d in SHUTTER_DENOMS]
    print(f"{len(configs)} configs x {rec_s:.0f}s on {clip.name}", flush=True)

    # video mode + warmup once (Safari caches the clip; recordings replay it)
    ctrl("--es", "mode", "video", "--ez", "ae", "true", "--ez", "af", "true")
    cue("play", clip.name, loop=False)
    time.sleep(rec_s + 8)

    manifest = []
    for n, (label, iso, denom) in enumerate(configs):
        print(f"[{n+1}/{len(configs)}] {label}", flush=True)
        if iso is None:
            ctrl("--ez", "ae", "true")
        else:
            ctrl("--ez", "ae", "false", "--ei", "iso", str(iso),
                 "--el", "shutter_ns", str(round(1e9 / denom)))
        time.sleep(SETTLE_S)
        one_capture(clip, rec_s, label, out_dir, manifest)

    ctrl("--ez", "ae", "true")
    cue("black")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"clip": clip.name, "rectify": filter_string(), "isos": ISOS,
         "shutter_denoms": SHUTTER_DENOMS, "captures": manifest}, indent=2))
    print(f"done: {len(manifest)}/{len(configs)} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
