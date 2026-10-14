"""
QAE batch collector: generate -> display -> 3-shot pre-check gate -> (if PASS)
15-setting sweep + AE -> sync to the GPU node, where vlm_loop.py scores it.

  python3 qae/collect.py --n 40 --start 0 --layout layoutB
Resumable: scenes with a manifest are skipped; rejected seeds are logged in rejects.json.
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "qae"))
import qa_sweep as Q  # noqa: E402
import qa_collect as C  # noqa: E402
import scenegen_v2 as G  # noqa: E402
from precheck_lib import run_precheck  # noqa: E402

OUT = ROOT / "data" / "qae_batch"
NODE = "chi-gigaio"


def show(name):
    req = urllib.request.Request("http://127.0.0.1:8765/api/cmd", data=json.dumps(
        {"action": "image", "video": name, "color": "#000"}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def adb_reset():
    """adb's server wedges on this Wi-Fi link now and then: shell calls stall for minutes. Reset it."""
    subprocess.run(["adb", "kill-server"], capture_output=True, timeout=30)
    time.sleep(2)
    subprocess.run(["adb", "start-server"], capture_output=True, timeout=60)
    svc = subprocess.run(["adb", "mdns", "services"], capture_output=True, text=True, timeout=30).stdout
    for line in svc.splitlines():
        if "_adb-tls-connect" in line:
            addr = line.split()[-1]
            subprocess.run(["adb", "connect", addr], capture_output=True, text=True, timeout=30)
            Q.SERIAL = addr
            print("adb reset; connected to", addr, flush=True)
            return


def adb_healthy():
    t = time.time()
    try:
        r = subprocess.run(["adb", "-s", Q.SERIAL, "shell", "echo", "ok"], capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and "ok" in r.stdout and time.time() - t < 5
    except subprocess.TimeoutExpired:
        return False


def ensure_app():
    """Bring ManualCam to the foreground if it is not there (phone may have gone home)."""
    Q.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", check=False)
    focus = Q.adb("shell", "dumpsys", "window", check=False)
    if "com.malikh.manualcam" not in focus.split("mCurrentFocus")[-1][:200]:
        Q.adb("shell", "monkey", "-p", "com.malikh.manualcam", "-c", "android.intent.category.LAUNCHER", "1", check=False)
        time.sleep(4)


def sync(sdir):
    for _ in range(3):
        if subprocess.run(["rsync", "-az", "-e", "ssh -o ConnectTimeout=25", str(sdir), f"{NODE}:~/qa/qae_batch/"]).returncode == 0:
            return True
        time.sleep(10)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--layout", default="layoutB"); ap.add_argument("--focus-cm", type=float, default=33.0)
    a = ap.parse_args()
    Q.ISOS = [100, 400, 1600]; Q.SHUTTERS = [1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4]
    calib = json.loads((ROOT / "data" / "qa_calib" / f"{a.layout}.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rej_path = OUT / "rejects.json"
    rejects = json.loads(rej_path.read_text()) if rej_path.exists() else {}
    C.ensure_phone()
    Q.adb("shell", "settings", "put", "global", "stay_on_while_plugged_in", "7", check=False)

    done, seed = 0, a.start
    while done < a.n:
        name = f"qae_{seed:04d}"; seed += 1
        sdir = OUT / name
        if (sdir / "manifest.json").exists():
            done += 1; continue
        if name in rejects:
            continue
        t0 = time.time()
        try:
            if not adb_healthy():
                adb_reset()
            C.ensure_phone()
            ensure_app()
            img = f"qae_scene_{name}.png"
            meta = G.make_scene(int(name[-4:]), ROOT / "data" / img, calib["H_canvas_to_camera"])
            tpath = ROOT / "data" / f"qae_targets_{name}.json"
            tpath.write_text(json.dumps({"scene": name, **meta}, indent=1))
            show(img); time.sleep(2.5)
            ok, verdict, table = run_precheck(name, tpath, ROOT / "data" / "qae_precheck" / name, focus_cm=a.focus_cm)
            print(f"{name}: window {meta['window_frac']:.2f}@{meta['window_pos']} AE 1/{round(1/verdict['ae']['exposure_s'])} ISO{verdict['ae']['iso']} "
                  f"conflict b={verdict['bright_conflict']} d={verdict['dim_conflict']} ctrl={verdict['control_ok']} "
                  f"unrec={verdict['unrecoverable']} -> {'PASS' if ok else 'REJECT'}", flush=True)
            if not ok:
                rejects[name] = {"verdict": verdict, "window_frac": meta["window_frac"]}
                rej_path.write_text(json.dumps(rejects, indent=1))
                continue
            sdir.mkdir(parents=True, exist_ok=True)
            C.ensure_phone(); ensure_app()
            Q.ctrl(af=False, focus_diopters=round(100.0 / a.focus_cm, 3))
            shots, retakes = C.sweep_scene(sdir)
            manifest = {"scene": name, "layout": a.layout, "ipad_image": img, "window_frac": meta["window_frac"],
                        "window_pos": meta["window_pos"], "ipad_backlight": "maximum, auto-brightness off",
                        "focus_cm": a.focus_cm, "isos": Q.ISOS, "shutters_s": Q.SHUTTERS, "precheck": verdict,
                        "targets": meta["targets"], "shots": shots, "retakes": retakes,
                        "duration_s": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            (sdir / "targets.json").write_text(json.dumps({"scene": name, "targets": meta["targets"]}, indent=1))
            (sdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            sync(sdir)
            done += 1
            print(f"{name}: swept {len(shots)} shots, {retakes} retakes, {manifest['duration_s']}s total  [{done}/{a.n}]", flush=True)
        except Exception:
            print(f"{name}: FAILED\n{traceback.format_exc()}", flush=True)
            time.sleep(20)
    print("BATCH_DONE", done, flush=True)


if __name__ == "__main__":
    main()
