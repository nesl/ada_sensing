#!/usr/bin/env python3
"""Detect the iPad screen rectangle in a white-screen calibration capture and
save it as the crop geometry for all subsequent captures.

Usage:
  python3 calibrate_crop.py <calibration.mp4>     # writes crop.json
  python3 calibrate_crop.py --apply <in.mp4> <out.mp4>   # crop using crop.json
"""
import json
import subprocess
import sys
from pathlib import Path

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
CROP_FILE = Path(__file__).parent / "crop.json"
THRESH = 96  # luma above this = "screen" in a white-screen capture


def video_size(path):
    """Displayed (post-autorotation) dimensions."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:side_data=rotation", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = st["width"], st["height"]
    rot = 0
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


def detect(path):
    w, h = video_size(path)
    # average middle frames to suppress noise, then read raw grayscale
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", "2", "-i", str(path), "-frames:v", "1",
         "-vf", "tmix=frames=5,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    if len(raw) < w * h:
        sys.exit("could not read frame")
    # column/row bright-pixel counts
    cols = [0] * w
    rows = [0] * h
    for y in range(h):
        base = y * w
        for x in range(w):
            if raw[base + x] > THRESH:
                cols[x] += 1
                rows[y] += 1
    # screen spans where at least 30% of the cross dimension is bright
    xs = [x for x in range(w) if cols[x] > h * 0.3]
    ys = [y for y in range(h) if rows[y] > w * 0.3]
    if not xs or not ys:
        sys.exit("no bright region found - is the white clip playing?")
    # inset by 4px to stay safely inside the screen edge/bezel blur
    x0, x1 = min(xs) + 4, max(xs) - 4
    y0, y1 = min(ys) + 4, max(ys) - 4
    cw, ch = (x1 - x0) & ~1, (y1 - y0) & ~1  # even dims for yuv420
    crop = {"x": x0, "y": y0, "w": cw, "h": ch,
            "src_w": w, "src_h": h,
            "fraction_of_frame": round(cw * ch / (w * h), 3)}
    CROP_FILE.write_text(json.dumps(crop, indent=2))
    print(f"crop: {cw}x{ch}+{x0}+{y0} of {w}x{h} "
          f"({crop['fraction_of_frame']*100:.0f}% of frame) -> {CROP_FILE}")


def apply(src, dst):
    c = json.loads(CROP_FILE.read_text())
    subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(src),
         "-vf", f"crop={c['w']}:{c['h']}:{c['x']}:{c['y']}",
         "-c:v", "libx264", "-crf", "18", "-c:a", "copy", "-y", str(dst)],
        check=True)
    print(f"cropped -> {dst}")


if __name__ == "__main__":
    if sys.argv[1] == "--apply":
        apply(sys.argv[2], sys.argv[3])
    else:
        detect(sys.argv[1])
