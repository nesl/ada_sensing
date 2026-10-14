#!/usr/bin/env python3
"""Frame-index marker overlay for stimulus clips, and its decoder.

Encoding: a 64px strip along the bottom of the 1920x1080 video with 20 blocks
of 96px: 4 sync blocks (W,B,W,B) + 12-bit frame index (MSB first) + 4-bit
checksum (XOR of the three index nibbles). White=1, black=0. Big blocks +
relative thresholding survive extreme under/over-exposure, blur, and noise.

CLI:
  python3 qr_overlay.py <clip>            # -> data/overlaid/<name>_qr.mp4 (CFR 30)
  python3 qr_overlay.py --check <capture> # decode first seconds, print indices
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
HERE = Path(__file__).parent

W, H = 1920, 1080
STRIP_H = 64
BLOCKS = 20
BLOCK_W = W // BLOCKS
FPS = 30
SYNC = [1, 0, 1, 0]


def encode_bits(idx):
    data = [(idx >> (11 - i)) & 1 for i in range(12)]
    chk = (idx & 0xF) ^ ((idx >> 4) & 0xF) ^ ((idx >> 8) & 0xF)
    chkb = [(chk >> (3 - i)) & 1 for i in range(4)]
    return SYNC + data + chkb


def strip_frame(idx):
    bits = encode_bits(idx)
    row = np.zeros((STRIP_H, W), dtype=np.uint8)
    for b, bit in enumerate(bits):
        if bit:
            row[:, b * BLOCK_W:(b + 1) * BLOCK_W] = 255
    return row


def make_overlay(clip, out=None, ev=0):
    """ev < 0 darkens the CONTENT by 2^ev in linear light (simulated display
    brightness); the marker strip is overlaid afterwards at full contrast so
    alignment still decodes under the darkest variants."""
    clip = Path(clip)
    out_dir = HERE / "data" / "overlaid"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if ev == 0 else f"_ev{ev}"
    out = Path(out) if out else out_dir / f"{clip.stem}_qr{tag}.mp4"
    # encoded-domain factor for a linear-light multiply through gamma 2.2
    f = (2.0 ** ev) ** (1 / 2.2)
    darken = (f"colorchannelmixer=rr={f:.4f}:gg={f:.4f}:bb={f:.4f},"
              if ev != 0 else "")

    dur = float(subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(clip)], capture_output=True, text=True,
        check=True).stdout.strip())
    n = int(dur * FPS) + 2

    proc = subprocess.Popen(
        [FFMPEG, "-v", "error",
         "-i", str(clip),
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{STRIP_H}",
         "-r", str(FPS), "-i", "-",
         "-filter_complex",
         f"[0:v]fps={FPS},scale={W}:{H},{darken}format=yuv420p[base];"
         f"[base][1:v]overlay=0:{H - STRIP_H}:shortest=1",
         "-c:v", "libx264", "-crf", "18", "-an", "-y", str(out)],
        stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(strip_frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit("ffmpeg overlay failed")
    print(f"overlaid -> {out} ({n} frames @ {FPS}fps)")
    return out


def decode_strip(gray_frame):
    """Decode frame index from a full grayscale capture frame (any size).
    Returns index or None. The strip occupies the bottom STRIP_H/H of the frame."""
    h, w = gray_frame.shape
    sh = max(4, int(h * STRIP_H / H))
    band = gray_frame[h - sh:h - max(1, sh // 8), :].astype(np.float64)
    bw = w / BLOCKS
    means = []
    for b in range(BLOCKS):
        x0 = int(b * bw + bw * 0.25)
        x1 = int(b * bw + bw * 0.75)
        means.append(band[:, x0:x1].mean())
    means = np.array(means)
    lo, hi = means.min(), means.max()
    if hi - lo < 2.0:  # no contrast at all -> nothing to decode
        return None
    bits = (means > (lo + hi) / 2).astype(int)
    if list(bits[:4]) != SYNC:
        return None
    idx = 0
    for b in bits[4:16]:
        idx = (idx << 1) | int(b)
    chk = 0
    for b in bits[16:20]:
        chk = (chk << 1) | int(b)
    if chk != (idx & 0xF) ^ ((idx >> 4) & 0xF) ^ ((idx >> 8) & 0xF):
        return None
    return idx


def gray_frames_full(path, seek=0, count=300, fps=30, w=480, h=270):
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(seek), "-i", str(path),
         "-vf", f"fps={fps},scale={w}:{h},format=gray",
         "-frames:v", str(count), "-f", "rawvideo", "-"],
        check=True, capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w)


def qr_start_offset(captured, search_s=10, fps=30):
    """Find where stimulus frame 0 plays in the capture via the marker strip.

    Returns (offset_seconds, decoded_fraction) or (None, frac) if undecodable.
    Uses a linear fit over all decoded (capture_frame, stimulus_idx) pairs so a
    missed frame 0 doesn't matter.
    """
    frames = gray_frames_full(captured, count=search_s * fps, fps=fps)
    pairs = [(i, decode_strip(f)) for i, f in enumerate(frames)]
    good = [(i, d) for i, d in pairs if d is not None]
    frac = len(good) / max(1, len(frames))
    if len(good) < 10:
        return None, frac
    # Each decoded pair directly measures the offset; the median needs no
    # extrapolation and shrugs off false decodes and display frame jitter.
    offsets = np.array([i / fps - d / FPS for i, d in good])
    return float(np.median(offsets)), frac


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        frames = gray_frames_full(sys.argv[2], count=90)
        decoded = [decode_strip(f) for f in frames]
        ok = [d for d in decoded if d is not None]
        print(f"decoded {len(ok)}/{len(decoded)} frames; "
              f"first: {decoded[:12]}")
        off, frac = qr_start_offset(sys.argv[2])
        print(f"start offset: {off if off is None else round(off, 3)}s "
              f"(decode rate {frac:.0%})")
    else:
        make_overlay(sys.argv[1])
