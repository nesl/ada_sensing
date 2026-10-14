"""
Did each photo show the scene it was supposed to? For v2/v3 scenes the white
window position is known from targets.json; a shot of the wrong scene has the
expected window region dark. Reports, per shot, the mean luminance inside the
expected window divided by the mean over the rest of the iPad crop.

  python3 qae/check_display.py data/qae_batch/sweeps data/qae_batch/precheck [--thr 3.0]
Writes data/qae_batch/display_check.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CROP = (602, 854, 2687, 2027)
DS = 4


def window_mask(win_canvas, H, shape):
    x0, y0, x1, y1 = win_canvas
    p = cv2.perspectiveTransform(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])[None], np.array(H, np.float64))[0]
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(p / DS)], 1)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--thr", type=float, default=3.0)
    ap.add_argument("--calib", default=str(ROOT / "data" / "qa_calib" / "layoutB.json"))
    a = ap.parse_args()
    H = json.loads(Path(a.calib).read_text())["H_canvas_to_camera"]
    out, bad = {}, []
    for root in a.dirs:
        for d in sorted(Path(root).glob("*")):
            t = d / "targets.json"
            if not t.exists() or d.name.startswith("cur_"):
                continue
            meta = json.loads(t.read_text())
            if "window_canvas" not in meta:   # sweep dirs: geometry is in the matching pre-check dir
                pre = d.parent.parent / "precheck" / d.name / "targets.json"
                if not pre.exists():
                    continue
                meta = json.loads(pre.read_text())
            rows = {}
            for jpg in sorted(d.glob("*.jpg")):
                im = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                im = im[::DS, ::DS].astype(np.float32)
                wm = window_mask(meta["window_canvas"], H, im.shape)
                cx0, cy0, cx1, cy1 = (c // DS for c in CROP)
                crop = np.zeros(im.shape, bool); crop[cy0:cy1, cx0:cx1] = True
                inside = im[wm & crop].mean(); outside = im[crop & ~wm].mean()
                r = float(inside / max(outside, 1e-3))
                rows[jpg.name] = round(r, 2)
                if r < a.thr:
                    bad.append((d.name, jpg.name, round(r, 2)))
            out[d.name] = rows
    Path(a.dirs[0]).parent.joinpath("display_check.json").write_text(json.dumps(out, indent=1))
    n = sum(len(v) for v in out.values())
    print(f"{n} shots in {len(out)} scenes; {len(bad)} below ratio {a.thr} ({len(bad)/max(n,1):.1%})")
    ratios = np.array([r for v in out.values() for r in v.values()])
    print("ratio percentiles 1/5/25/50/75:", np.percentile(ratios, [1, 5, 25, 50, 75]).round(2))
    by_scene = {}
    for s, j, r in bad:
        by_scene.setdefault(s, []).append(f"{j}:{r}")
    for s in sorted(by_scene):
        print(f"  {s}: {len(by_scene[s])} bad  {' '.join(by_scene[s][:6])}")


if __name__ == "__main__":
    main()
