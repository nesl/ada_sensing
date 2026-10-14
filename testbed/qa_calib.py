"""
iPad -> camera mapping for automatically generated targets.

Shows four ArUco markers at known canvas positions on the iPad, takes one manual
photo, detects the markers and fits a homography from canvas pixels to camera
pixels. Re-run whenever the iPad, phone, or zoom moves.

  python3 qa_calib.py <layout_name> [--iso 400 --shutter 0.25]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import qa_sweep as Q

CANVAS = (2388, 1668)            # generated images are this size (iPad landscape aspect)
MARK = 260                       # marker side in canvas px
MARGIN = 140
SERVER = "http://127.0.0.1:8765/api/cmd"
DATA = Path(__file__).resolve().parent / "data"
CALIB_DIR = DATA / "qa_calib"
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_positions():
    W, H = CANVAS
    xs = [MARGIN, W - MARGIN - MARK]
    ys = [MARGIN + 60, H - MARGIN - MARK]   # +60 keeps clear of the iPad status bar
    return {0: (xs[0], ys[0]), 1: (xs[1], ys[0]), 2: (xs[1], ys[1]), 3: (xs[0], ys[1])}


def make_pattern(path):
    W, H = CANVAS
    canvas = np.zeros((H, W), np.uint8)
    for mid, (x, y) in marker_positions().items():
        m = cv2.aruco.generateImageMarker(DICT, mid, MARK)
        pad = 30   # white quiet zone so the black border is detectable on a black screen
        canvas[y - pad:y + MARK + pad, x - pad:x + MARK + pad] = 255
        canvas[y:y + MARK, x:x + MARK] = m
    Image.fromarray(canvas).convert("RGB").save(path)


def show(name):
    import urllib.request
    req = urllib.request.Request(SERVER, data=json.dumps(
        {"action": "image", "video": name, "color": "#000"}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def canvas_corners():
    """Canvas coords of each marker's 4 corners, in ArUco corner order (TL,TR,BR,BL)."""
    out = {}
    for mid, (x, y) in marker_positions().items():
        out[mid] = np.array([[x, y], [x + MARK, y], [x + MARK, y + MARK], [x, y + MARK]], np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("--iso", type=int, default=400)
    ap.add_argument("--shutter", type=float, default=0.25)
    ap.add_argument("--focus-cm", type=float, default=30.0)
    a = ap.parse_args()

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    pat = DATA / "qa_aruco_pattern.png"
    make_pattern(pat)
    show(pat.name)
    time.sleep(2.5)

    Q.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    Q.ctrl(mode="photo")
    Q.ctrl(af=False, focus_diopters=round(100.0 / a.focus_cm, 3))
    Q.ctrl(ae=False, iso=a.iso, shutter_ns=int(round(a.shutter * 1e9)))
    r = Q.shoot(CALIB_DIR, f"{a.layout}_markers", settle_s=Q.settle_for(1.0, a.shutter),
                expect=(a.iso, a.shutter))
    img = cv2.imread(str(CALIB_DIR / r["file"]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    found = [] if ids is None else ids.ravel().tolist()
    print("markers found:", found)
    if len(found) < 3:
        raise SystemExit("need at least 3 markers; check the iPad is visible and exposure is reasonable")

    ref = canvas_corners()
    src = np.concatenate([ref[i] for i in found])
    dst = np.concatenate([c.reshape(4, 2) for c in corners])
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
    proj = cv2.perspectiveTransform(src[None], Hm)[0]
    err = float(np.linalg.norm(proj - dst, axis=1).mean())
    print(f"homography fitted on {len(found)} markers, mean reprojection error {err:.1f} px")

    vis = img.copy()
    W, H = CANVAS
    quad = cv2.perspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]])[None], Hm)[0]
    cv2.polylines(vis, [quad.astype(np.int32)], True, (255, 0, 255), 8)
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.imwrite(str(CALIB_DIR / f"{a.layout}_check.jpg"), cv2.resize(vis, (1600, 900)))

    out = {"layout": a.layout, "canvas": CANVAS, "H_canvas_to_camera": Hm.tolist(),
           "markers_found": found, "reproj_err_px": err, "photo": r["file"],
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (CALIB_DIR / f"{a.layout}.json").write_text(json.dumps(out, indent=1))
    Q.ctrl(ae=True)
    print("saved", CALIB_DIR / f"{a.layout}.json")


if __name__ == "__main__":
    main()
