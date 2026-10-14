"""
Unattended dataset collection for the query-aware exposure work.

Loop: generate a random iPad scene -> show it -> AE shot + 7x7 ISO x shutter grid
(+ closing AE) -> batch-pull photos -> verify every shot's EXIF, retake failures ->
write the scene's manifest (settings, EXIF, target boxes in camera pixels).

Resumable: completed scenes (manifest present) are skipped.

  python3 qa_collect.py --layout layoutA --n 100 --tag white-towel
"""
import argparse
import json
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

import qa_scenegen as G
import qa_sweep as Q

ROOT = Path(__file__).resolve().parent / "data"
OUT = ROOT / "qa_dataset"
SERVER = "http://127.0.0.1:8765/api/cmd"


def show(name):
    req = urllib.request.Request(SERVER, data=json.dumps(
        {"action": "image", "video": name, "color": "#000"}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def ensure_phone():
    """Reconnect wireless adb if the phone dropped (it changes port on reconnect)."""
    r = subprocess.run(["adb", "-s", Q.SERIAL, "get-state"], capture_output=True, text=True)
    if r.stdout.strip() == "device":
        return
    svc = subprocess.run(["adb", "mdns", "services"], capture_output=True, text=True).stdout
    for line in svc.splitlines():
        if "_adb-tls-connect" in line:
            addr = line.split()[-1]
            subprocess.run(["adb", "connect", addr], capture_output=True, text=True)
            Q.SERIAL = addr
            print("reconnected to", addr, flush=True)
            return
    raise RuntimeError("phone not reachable over adb")


def take(label, settle_s):
    """Fire one photo; return its phone filename (not pulled yet)."""
    before = Q.remote_files()
    time.sleep(settle_s)
    Q.ctrl(photo=True)
    for _ in range(60):
        time.sleep(0.4)
        new = sorted(Q.remote_files() - before)
        if new:
            time.sleep(1.05)  # ManualCam filenames have 1-second resolution
            return new[-1]
    raise RuntimeError(f"{label}: no photo appeared")


def sweep_scene(sdir):
    # snake order: shutter ascends on one ISO row and descends on the next, so the
    # camera never jumps straight from a 1 s exposure to 1/4000 s (stale frames)
    grid = []
    for r, iso in enumerate(Q.ISOS):
        for sh in (Q.SHUTTERS if r % 2 == 0 else Q.SHUTTERS[::-1]):
            grid.append((f"iso{iso}_s{sh:.6f}".rstrip("0").rstrip("."), iso, sh))
    plan = [("ae_start", None, None)] + grid + [("ae_end", None, None)]
    phone = {}
    prev = 1.0
    for label, iso, sh in plan:
        if iso is None:
            Q.ctrl(ae=True)
            phone[label] = take(label, Q.settle_for(prev, 0.1) + 2.0)
            prev = 0.1
        else:
            Q.ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
            phone[label] = take(label, Q.settle_for(prev, sh))
            prev = sh
    # batch pull
    for label, fname in phone.items():
        Q.adb("pull", f"{Q.REMOTE_DIR}/{fname}", str(sdir / f"{label}.jpg"))
    # verify, retake failures with a long settle
    shots, retakes = [], 0
    for label, iso, sh in plan:
        exif = Q.read_exif(sdir / f"{label}.jpg")
        if iso is not None and not Q.exif_matches(exif, iso, sh):
            Q.ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
            r = Q.shoot(sdir, label, settle_s=Q.settle_for(1.0, sh), expect=(iso, sh))
            exif, retakes = r["exif"], retakes + 1
        shot = {"label": label, "file": f"{label}.jpg", "phone_file": phone[label], "exif": exif,
                "mode": "ae" if iso is None else "manual"}
        if iso is not None:
            shot.update(iso=iso, shutter_s=sh)
        shots.append(shot)
    Q.ctrl(ae=True)
    return shots, retakes


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="first scene index (seed)")
    ap.add_argument("--tag", default="", help="free-text physical condition, e.g. panel color")
    ap.add_argument("--focus-cm", type=float, default=30.0)
    ap.add_argument("--mode", default="random", choices=["random", "bd"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--isos", default=",".join(map(str, Q.ISOS)))
    ap.add_argument("--shutters", default=",".join(map(str, Q.SHUTTERS)),
                    help="seconds, comma-separated")
    a = ap.parse_args()
    Q.ISOS = [int(v) for v in a.isos.split(",")]
    Q.SHUTTERS = [float(eval(v)) for v in a.shutters.split(",")]
    OUT = Path(a.out)

    calib = json.loads((ROOT / "qa_calib" / f"{a.layout}.json").read_text())
    static_p = ROOT / "qa_calib" / f"{a.layout}_static.json"
    static = json.loads(static_p.read_text())["targets"] if static_p.exists() else []
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_phone()
    Q.adb("shell", "settings", "put", "global", "stay_on_while_plugged_in", "7", check=False)
    Q.ctrl(mode="photo")
    Q.ctrl(af=False, focus_diopters=round(100.0 / a.focus_cm, 3))

    done = 0
    for idx in range(a.start, a.start + a.n):
        name = f"{a.layout}_s{idx:04d}"
        sdir = OUT / name
        if (sdir / "manifest.json").exists():
            continue
        t0 = time.time()
        try:
            ensure_phone()
            Q.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", check=False)
            sdir.mkdir(parents=True, exist_ok=True)
            img = f"qa_scene_{name}.png"
            gen = G.make_scene_bd if a.mode == "bd" else G.make_scene
            targets = gen(idx, ROOT / img, calib["H_canvas_to_camera"])
            show(img)
            time.sleep(2.0)
            shots, retakes = sweep_scene(sdir)
            manifest = {"scene": name, "layout": a.layout, "seed": idx, "tag": a.tag,
                        "ipad_image": img, "ipad_backlight": "minimum, auto-brightness off",
                        "focus_cm": a.focus_cm, "isos": Q.ISOS, "shutters_s": Q.SHUTTERS,
                        "targets": targets + static, "shots": shots, "retakes": retakes,
                        "duration_s": round(time.time() - t0, 1),
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            (sdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            done += 1
            print(f"{name}: {len(shots)} shots, {retakes} retakes, {manifest['duration_s']}s, "
                  f"{len(targets)} targets  [{done} done this run]", flush=True)
        except Exception:
            print(f"{name}: FAILED\n{traceback.format_exc()}", flush=True)
            time.sleep(20)
    print("COLLECTION_DONE", done, flush=True)


if __name__ == "__main__":
    main()
