"""
Exposure sweep for the query-aware camera experiment.

Drives ManualCam over adb: AE shot, a grid of manual ISO x shutter shots with
focus locked, then a closing AE shot (drift check). Pulls every photo to
data/qa_sweeps/<name>/ and writes manifest.json with the requested and the
camera-reported (actual) exposure for each shot.

  python3 qa_sweep.py <scene_name> [--focus-cm 30] [--note "..."]
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

SERIAL = "192.168.1.177:42091"
ISOS = [50, 100, 200, 400, 800, 1600, 3200]
SHUTTERS = [1 / 4000, 1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4, 1.0]
OUT_ROOT = Path(__file__).resolve().parent / "data" / "qa_sweeps"
REMOTE_DIR = "/sdcard/DCIM/ManualCam"
ACTUAL_RE = re.compile(r"requested=(\d+)ns actual=(\d+)ns iso_req=(\d+) iso_actual=(\d+)")


def adb(*args, check=True, capture=True):
    t0 = time.time()
    r = subprocess.run(["adb", "-s", SERIAL, *args], capture_output=capture, text=True,
                       errors="replace", timeout=60)  # logcat can emit non-UTF-8 bytes
    dt = time.time() - t0
    if dt > 3:
        print(f"  slow adb ({dt:.0f}s): {' '.join(args)[:70]}", flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def ctrl(**kw):
    args = ["shell", "am", "broadcast", "-a", "com.malikh.manualcam.CTRL"]
    for k, v in kw.items():
        if isinstance(v, bool):
            args += ["--ez", k, "true" if v else "false"]
        elif isinstance(v, int):
            args += ["--el" if k == "shutter_ns" else "--ei", k, str(v)]
        elif isinstance(v, float):
            args += ["--ef", k, str(v)]
        else:
            args += ["--es", k, str(v)]
    adb(*args)


def remote_files():
    return set(adb("shell", "ls", REMOTE_DIR, check=False).split())


def last_actual():
    """Most recent camera-reported exposure from ManualCam's log."""
    log = adb("logcat", "-d", "-t", "400", check=False)
    hits = ACTUAL_RE.findall(log)
    if not hits:
        return None
    req_ns, act_ns, iso_req, iso_act = map(int, hits[-1])
    return {"exposure_s": act_ns / 1e9, "iso": iso_act}


def read_exif(path):
    from PIL import Image
    ex = Image.open(path).getexif().get_ifd(0x8769)
    return {"exposure_s": float(ex.get(0x829A, 0)), "iso": ex.get(0x8827)}


def exif_matches(exif, iso, shutter_s):
    return (abs(exif["exposure_s"] / shutter_s - 1) < 0.05
            and abs(exif["iso"] / iso - 1) < 0.05)


def shoot(out_dir, label, settle_s, expect=None, retries=3, resend=None):
    """Take one photo. expect=(iso, shutter_s) verifies the saved JPEG's EXIF and
    retakes on mismatch (the camera can still have frames of the previous, longer
    exposure in flight). resend() re-sends the settings broadcast before a retake:
    ManualCam occasionally drops it and keeps its previous manual values."""
    for attempt in range(retries + 1):
        if attempt and resend is not None:
            resend()
        before = remote_files()
        time.sleep(settle_s)
        actual = last_actual()
        ctrl(photo=True)
        new = None
        for _ in range(60):
            time.sleep(0.5)
            diff = sorted(remote_files() - before)
            if diff:
                new = diff[-1]
                break
        if new is None:
            raise RuntimeError(f"{label}: no photo appeared on the phone")
        time.sleep(1.0)  # let the JPEG finish writing
        dst = out_dir / f"{label}.jpg"
        adb("pull", f"{REMOTE_DIR}/{new}", str(dst))
        time.sleep(1.1)  # ManualCam filenames have 1-second resolution
        exif = read_exif(dst)
        if expect is None or exif_matches(exif, *expect):
            return {"file": dst.name, "phone_file": new, "actual": actual, "exif": exif,
                    "attempts": attempt + 1}
        print(f"  {label}: EXIF {exif} != expected {expect}, retaking", flush=True)
        settle_s += 2.0
    raise RuntimeError(f"{label}: settings never applied after {retries + 1} attempts")


def settle_for(prev_s, new_s):
    """Wait long enough to flush frames of the previous exposure from the pipeline."""
    longest = max(prev_s, new_s)
    if longest >= 0.9:   # 1 s frames linger: a 4.6 s wait still yields stale shots
        return 9.0
    return max(1.2, 4.0 * longest + 0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--focus-cm", type=float, default=30.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--redo", nargs="*", default=None,
                    help="only reshoot these labels in an existing sweep (updates its manifest)")
    a = ap.parse_args()

    out_dir = OUT_ROOT / a.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    ctrl(mode="photo")
    ctrl(af=False, focus_diopters=round(100.0 / a.focus_cm, 3))
    adb("logcat", "-c", check=False)

    grid = [(f"iso{iso}_s{sh:.6f}".rstrip("0").rstrip("."), iso, sh) for iso in ISOS for sh in SHUTTERS]
    t0 = time.time()

    if a.redo is not None:
        manifest = json.loads((out_dir / "manifest.json").read_text())
        by_label = {s["label"]: s for s in manifest["shots"]}
        prev = 1.0
        for label in a.redo:
            if label.startswith("ae"):
                ctrl(ae=True)
                r = shoot(out_dir, label, settle_s=settle_for(prev, 0.1) + 2.0)
                by_label[label].update(r)
                prev = r["exif"]["exposure_s"]
            else:
                _, iso, sh = next(g for g in grid if g[0] == label)
                ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
                r = shoot(out_dir, label, settle_s=settle_for(prev, sh), expect=(iso, sh))
                by_label[label].update(r)
                prev = sh
            print(f"redo {label}: exif={r['exif']} attempts={r['attempts']}", flush=True)
        manifest["shots"] = list(by_label.values())
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return

    shots = []
    ctrl(ae=True)
    shots.append({"label": "ae_start", "mode": "ae", **shoot(out_dir, "ae_start", settle_s=3.0)})
    print("ae_start", shots[-1]["exif"], flush=True)

    prev = shots[-1]["exif"]["exposure_s"]
    for i, (label, iso, sh) in enumerate(grid, 1):
        done = out_dir / f"{label}.jpg"
        if done.exists() and exif_matches(read_exif(done), iso, sh):  # resume after a crash
            r = {"file": done.name, "phone_file": None, "actual": None, "exif": read_exif(done), "attempts": 0}
            shots.append({"label": label, "mode": "manual", "iso": iso, "shutter_s": sh, **r})
            continue
        ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
        r = shoot(out_dir, label, settle_s=settle_for(prev, sh), expect=(iso, sh))
        shots.append({"label": label, "mode": "manual", "iso": iso, "shutter_s": sh, **r})
        print(f"[{i}/{len(grid)}] {label} exif={r['exif']} attempts={r['attempts']}", flush=True)
        prev = sh

    ctrl(ae=True)
    shots.append({"label": "ae_end", "mode": "ae",
                  **shoot(out_dir, "ae_end", settle_s=settle_for(prev, 0.1) + 2.0)})
    print("ae_end", shots[-1]["exif"], flush=True)

    manifest = {"scene": a.scene, "note": a.note, "focus_cm": a.focus_cm, "serial": SERIAL,
                "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                "duration_s": round(time.time() - t0, 1), "isos": ISOS, "shutters_s": SHUTTERS,
                "shots": shots}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(shots)} shots in {manifest['duration_s']}s -> {out_dir}")


if __name__ == "__main__":
    main()
