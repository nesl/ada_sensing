#!/usr/bin/env python3
"""Extract a few video clips from the remote BDD100K zip via HTTP range requests,
without downloading the full 1.8 TB archive.

Usage: python3 fetch_bdd_clips.py [N_CLIPS]
"""
import io
import sys
import zipfile
import urllib.request
from pathlib import Path

URL = "http://dl.yf.io/bdd100k/bdd100k_videos.zip"
OUT_DIR = Path(__file__).parent / "data" / "bdd100k_videos"
BLOCK = 1 << 20  # 1 MiB fetch granularity


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with block caching."""

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.length = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}  # block index -> bytes
        self.fetched = 0

    def _block(self, idx):
        if idx not in self.cache:
            start = idx * BLOCK
            end = min(start + BLOCK, self.length) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req) as r:
                self.cache[idx] = r.read()
            self.fetched += len(self.cache[idx])
        return self.cache[idx]

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.length - self.pos
        n = max(0, min(n, self.length - self.pos))
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, BLOCK)
            chunk = self._block(idx)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hf = HttpFile(URL)
    print(f"archive size: {hf.length / 1e12:.2f} TB", flush=True)
    zf = zipfile.ZipFile(hf)
    movs = [i for i in zf.infolist() if i.filename.endswith(".mov")]
    print(f"central directory read: {len(movs)} videos listed "
          f"({hf.fetched / 1e6:.0f} MB fetched so far)", flush=True)

    # spread picks across the archive rather than taking neighbors
    step = max(1, len(movs) // n_clips)
    picks = [movs[i * step] for i in range(n_clips)]
    for info in picks:
        dest = OUT_DIR / Path(info.filename).name
        if dest.exists():
            print(f"skip (exists): {dest.name}", flush=True)
            continue
        print(f"extracting {info.filename} ({info.file_size / 1e6:.1f} MB)", flush=True)
        with zf.open(info) as src, open(dest, "wb") as out:
            while True:
                buf = src.read(1 << 16)
                if not buf:
                    break
                out.write(buf)
    print(f"done. total transferred: {hf.fetched / 1e6:.0f} MB -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
