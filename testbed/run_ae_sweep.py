#!/usr/bin/env python3
"""AE-baseline sweep: for each stimulus clip, restart it on the iPad, record it
with ManualCam (current settings, e.g. AE on), pull, crop, and save results.

Produces per clip in the output dir:
  <clip>_original.<ext>   the stimulus file
  <clip>_captured.mp4     cropped phone capture
  <clip>_sbs.mp4          side-by-side comparison (original | captured)

Usage: python3 run_ae_sweep.py <output_dir> [clip ...]
       (no clips listed -> all clips in data/bdd100k_videos and data/imagenet_vid mp4s)
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
ADB = "/opt/homebrew/share/android-commandlinetools/platform-tools/adb"
HERE = Path(__file__).parent
CROP = json.loads((HERE / "crop.json").read_text())
SERVER = "http://localhost:8765"
RECORD_TAP = (1769, 862)  # action button in ManualCam's landscape UI
MAX_RECORD_S = 45


def sh(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True,
                          capture_output=True, text=True, **kw)


def cue(action, video=None, loop=False):
    body = json.dumps({"action": action, "video": video, "loop": loop})
    sh("curl", "-s", "-X", "POST", f"{SERVER}/api/cmd", "-d", body)


def duration(path):
    out = sh(FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path).stdout.strip()
    return float(out)


def newest_capture():
    out = sh(ADB, "shell", "ls", "-t", "/sdcard/DCIM/ManualCam/").stdout.split()
    vids = [f for f in out if f.endswith(".mp4")]
    return vids[0] if vids else None


def tap(x, y):
    sh(ADB, "shell", "input", "tap", x, y)


def gray_frames(path, w=160, h=90, seek=None, count=None, fps=None):
    """Decode video to small grayscale frames as a (n, h, w) uint8 array."""
    cmd = [FFMPEG, "-v", "error"]
    if seek is not None:
        cmd += ["-ss", str(seek)]
    cmd += ["-i", str(path)]
    vf = f"scale={w}:{h}"
    if fps:
        vf = f"fps={fps}," + vf
    cmd += ["-vf", vf + ",format=gray"]
    if count:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w)


def content_start_offset(captured, original, search_s=10, fps=30):
    """Seconds into `captured` where the original's first frame appears.

    Uses zero-normalized cross-correlation so AE brightness shifts don't matter;
    flat frames (blackout, load flash) have ~zero variance and are skipped.
    """
    ref = gray_frames(original, count=1)[0].astype(np.float64)
    ref -= ref.mean()
    ref_norm = np.linalg.norm(ref)
    frames = gray_frames(captured, count=search_s * fps, fps=fps).astype(np.float64)
    best_i, best_score = 0, -2.0
    for i, f in enumerate(frames):
        f = f - f.mean()
        denom = np.linalg.norm(f) * ref_norm
        if denom < 1e-6 or f.std() < 4:
            continue
        score = float((f * ref).sum() / denom)
        if score > best_score:
            best_i, best_score = i, score
    return best_i / fps, best_score


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 2:
        clips = [Path(c) for c in sys.argv[2:]]
    else:
        clips = sorted((HERE / "data" / "bdd100k_videos").glob("*.mov")) + \
                sorted((HERE / "data" / "imagenet_vid").glob("*.mp4"))

    manifest = []
    for i, clip in enumerate(clips):
        name = clip.stem
        # +4s margin: ~1-2s recorder start latency at the head is trimmed away by
        # alignment, and the tail must still cover the full clip afterwards
        rec_s = min(duration(clip) + 4.0, MAX_RECORD_S)
        print(f"[{i+1}/{len(clips)}] {name}: warmup, then record {rec_s:.0f}s", flush=True)

        # Warmup pass (not recorded): Safari buffers the whole file the first time
        # it plays a clip (~20 s stall on a fresh 20 MB file); replaying from cache
        # starts almost instantly.
        cue("play", clip.name, loop=False)
        time.sleep(rec_s + 8)

        cue("black")
        time.sleep(1.5)
        before = newest_capture()
        tap(*RECORD_TAP)              # start recording
        time.sleep(0.7)
        cue("play", clip.name, loop=False)
        time.sleep(rec_s)
        tap(*RECORD_TAP)              # stop
        time.sleep(2.5)

        cap = newest_capture()
        if cap is None or cap == before:
            print(f"  !! no new capture found for {name}, skipping", flush=True)
            continue

        raw = out_dir / f"{name}_raw.mp4"
        sh(ADB, "pull", f"/sdcard/DCIM/ManualCam/{cap}", raw)

        cropped = out_dir / f"{name}_cropped_tmp.mp4"
        sh(FFMPEG, "-v", "error", "-i", raw,
           "-vf", f"crop={CROP['w']}:{CROP['h']}:{CROP['x']}:{CROP['y']}",
           "-c:v", "libx264", "-crf", "18", "-an", "-y", cropped)
        raw.unlink()

        original = out_dir / f"{name}_original{clip.suffix}"
        if not original.exists():
            shutil.copy2(clip, original)

        # Align: trim capture to where the clip's first frame appears, and cut
        # to the original's duration so both start and end together
        offset, score = content_start_offset(cropped, original)
        print(f"  content starts at {offset:.2f}s in capture (match {score:.2f})",
              flush=True)
        captured = out_dir / f"{name}_captured.mp4"
        sh(FFMPEG, "-v", "error", "-ss", f"{offset:.3f}", "-i", cropped,
           "-t", f"{duration(original):.3f}",
           "-c:v", "libx264", "-crf", "18", "-an", "-y", captured)
        cropped.unlink()

        sbs = out_dir / f"{name}_sbs.mp4"
        sh(FFMPEG, "-v", "error", "-i", original, "-i", captured,
           "-filter_complex",
           "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack",
           "-c:v", "libx264", "-crf", "20", "-an", "-y", sbs)

        manifest.append({"clip": name, "device_file": cap,
                         "record_seconds": rec_s, "crop": CROP,
                         "align_offset_s": round(offset, 3),
                         "align_score": round(score, 3)})
        print(f"  ok -> {captured.name}, {sbs.name}", flush=True)

    cue("black")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(manifest)}/{len(clips)} clips -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
