#!/usr/bin/env python3
"""Stream the ImageNet-VID tar.gz from the HF mirror and extract the first
N_SNIPPETS video snippets' frames, then stop (no full 86 GB download).

Usage: python3 fetch_imagenet_vid.py [N_SNIPPETS]
"""
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/guanxiongsun/imagenetvid/resolve/main/"
       "ILSVRC2015_VID.tar.gz.aa")
OUT_DIR = Path(__file__).parent / "data" / "imagenet_vid"
DATA_PREFIX = "ILSVRC2015/Data/VID/train/"


class CountingStream:
    def __init__(self, resp):
        self.resp = resp
        self.n = 0
        self.reported = 0

    def read(self, size):
        buf = self.resp.read(size)
        self.n += len(buf)
        if self.n - self.reported > 500 * 1024 * 1024:
            self.reported = self.n
            print(f"... streamed {self.n / 1e9:.1f} GB", flush=True)
        return buf


def main():
    n_snippets = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(URL)
    with urllib.request.urlopen(req) as resp:
        stream = CountingStream(resp)
        tf = tarfile.open(fileobj=stream, mode="r|gz")
        snippets = []  # snippet dir names in order of appearance
        frames = 0
        for m in tf:
            if not m.name.startswith(DATA_PREFIX) or not m.isfile():
                continue
            # name: .../train/<set>/<snippet>/<frame>.JPEG
            parts = m.name.split("/")
            if len(parts) < 6:
                continue
            snippet = parts[-2]
            if snippet not in snippets:
                if len(snippets) >= n_snippets:
                    break  # a new snippet beyond quota -> done
                snippets.append(snippet)
                print(f"extracting snippet {snippet}", flush=True)
            dest = OUT_DIR / snippet / parts[-1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            with open(dest, "wb") as out:
                out.write(src.read())
            frames += 1
    print(f"done: {len(snippets)} snippets, {frames} frames, "
          f"{stream.n / 1e9:.1f} GB streamed -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
