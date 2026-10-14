"""
QAE batch, decoupled so the flaky Mac<->node link never blocks the phone.

  python3 qae/batch.py precheck --start 0 --n 60     # phone: generate, display, 3 shots per seed
  python3 qae/batch.py sync                           # loop: push pre-checks/sweeps, pull node scores, write verdicts
  python3 qae/batch.py sweep --n 40                   # phone: 15-setting sweep for scenes with a PASS verdict

Dirs: data/qae_batch/precheck/<scene>/  (ae/dark/bright.jpg, manifest, targets, vlm.json, verdict.json)
      data/qae_batch/sweeps/<scene>/    (shots, manifest, targets, vlm.json)
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
import scenegen_v2  # noqa: E402
import scenegen_v3  # noqa: E402

GENS = {"v2": scenegen_v2, "v3": scenegen_v3}
# pre-check shots per generator: v3 adds a mid setting so the gate can see three-way conflicts
PRECHECK_SHOTS = {"v2": (("dark", (50, 0.001)), ("bright", (3200, 0.25))),
                  "v3": (("dark", (50, 0.001)), ("mid", (100, 1 / 60)), ("bright", (3200, 0.25)))}

NODE = "chi-qae"
BATCH = ROOT / "data" / "qae_batch"
PRE, SWEEPS = BATCH / "precheck", BATCH / "sweeps"
SSH = ["ssh", "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", NODE]
GRID_ISOS, GRID_SHUTTERS = [100, 400, 1600], [1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4]


# ---------------------------------------------------------------- helpers
def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def show(name):
    req = urllib.request.Request("http://127.0.0.1:8765/api/cmd", data=json.dumps(
        {"action": "image", "video": name, "color": "#000"}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def phone_ready(focus_cm):
    """Reset adb if slow, reconnect if dropped, bring ManualCam up, lock focus."""
    try:
        t = time.time()
        r = subprocess.run(["adb", "-s", Q.SERIAL, "shell", "echo", "ok"], capture_output=True, text=True, timeout=15)
        slow = r.returncode != 0 or "ok" not in r.stdout or time.time() - t > 5
    except subprocess.TimeoutExpired:
        slow = True
    if slow:
        subprocess.run(["adb", "kill-server"], capture_output=True, timeout=30); time.sleep(2)
        subprocess.run(["adb", "start-server"], capture_output=True, timeout=60)
        svc = subprocess.run(["adb", "mdns", "services"], capture_output=True, text=True, timeout=30).stdout
        for line in svc.splitlines():
            if "_adb-tls-connect" in line:
                Q.SERIAL = line.split()[-1]
                subprocess.run(["adb", "connect", Q.SERIAL], capture_output=True, text=True, timeout=30)
                log(f"adb reset -> {Q.SERIAL}")
                break
    Q.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP", check=False)
    focus = Q.adb("shell", "dumpsys", "window", check=False)
    if "com.malikh.manualcam" not in focus.split("mCurrentFocus")[-1][:200]:
        Q.adb("shell", "monkey", "-p", "com.malikh.manualcam", "-c", "android.intent.category.LAUNCHER", "1", check=False)
        time.sleep(4)
    Q.ctrl(mode="photo", af=False, focus_diopters=round(100.0 / focus_cm, 3))


def run_quiet(cmd, timeout):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def verdict_v3(d, rows):
    """v3 gate: at least three recoverable positives, negative held everywhere, and at
    least one pair of recoverable targets that no single pre-check shot answers together."""
    labels = [l for l in ("dark", "ae", "mid", "bright") if l in rows]
    v = {"gen": "v3", "negative_ok": True, "unrecoverable": [], "conflict_pairs": [], "control_ok": None}
    table, rec = [], {}
    for t in d["targets"]:
        k = t["id"]; s = {l: rows[l]["scores"][k] for l in labels}
        table.append({"id": k, "band": t["band"], "level": t["level"], **s})
        if t["kind"] == "negative":
            v["negative_ok"] &= all(x >= 1 for x in s.values()); continue
        ok = frozenset(l for l in labels if s[l] >= 1)
        if not ok:
            v["unrecoverable"].append(k)
        else:
            rec[k] = ok
        if t["band"] == "control":
            v["control_ok"] = s["ae"] >= 1
    ks = sorted(rec)
    v["conflict_pairs"] = [[a, b] for i, a in enumerate(ks) for b in ks[i + 1:] if not (rec[a] & rec[b])]
    v["ae_misses"] = [k for k in ks if "ae" not in rec[k]]
    v["recoverable"] = len(rec)
    v["pass"] = v["recoverable"] >= 3 and bool(v["conflict_pairs"]) and v["negative_ok"]
    v["ae"] = {"exposure_s": rows["ae"]["exposure_s"], "iso": rows["ae"]["iso"]}
    v["table"] = table
    return v


def verdict_from(vlm_path):
    d = json.loads(Path(vlm_path).read_text())
    rows = {r["label"]: r for r in d["rows"]}
    if "mid" in rows:
        return verdict_v3(d, rows)
    v = {"bright_conflict": False, "dim_conflict": False, "control_ok": None, "negative_ok": True, "unrecoverable": []}
    table = []
    for t in d["targets"]:
        k = t["id"]; s = {l: rows[l]["scores"][k] for l in ("dark", "ae", "bright")}
        table.append({"id": k, "band": t["band"], "level": t["level"], **s})
        if t["kind"] == "negative":
            v["negative_ok"] &= all(x >= 1 for x in s.values()); continue
        if max(s.values()) < 1:
            v["unrecoverable"].append(k)
        if t["band"] == "bright" and s["dark"] >= 1 and s["ae"] < 1:
            v["bright_conflict"] = True
        if t["band"] == "dim" and s["bright"] >= 1 and s["ae"] < 1:
            v["dim_conflict"] = True
        if t["band"] == "control":
            v["control_ok"] = s["ae"] >= 1
    npos = sum(1 for t in d["targets"] if t["kind"] != "negative")
    v["recoverable"] = npos - len(v["unrecoverable"])
    # unrecoverable targets are tagged and excluded, not fatal: a scene is useful if it holds a
    # conflict and at least two recoverable positive targets
    v["pass"] = v["recoverable"] >= 2 and (v["bright_conflict"] or v["dim_conflict"]) and v["negative_ok"]
    v["ae"] = {"exposure_s": rows["ae"]["exposure_s"], "iso": rows["ae"]["iso"]}
    v["table"] = table
    return v


# ---------------------------------------------------------------- phases
def phase_precheck(a):
    PRE.mkdir(parents=True, exist_ok=True)
    calib = json.loads((ROOT / "data" / "qa_calib" / f"{a.layout}.json").read_text())
    Q.adb("shell", "settings", "put", "global", "stay_on_while_plugged_in", "7", check=False)
    G = GENS[a.gen]
    for seed in range(a.start, a.start + a.n):
        name = f"qae_{seed:04d}" if a.gen == "v2" else f"{a.gen}_{seed:04d}"; out = PRE / name
        if (out / "manifest.json").exists():
            continue
        t0 = time.time()
        try:
            phone_ready(a.focus_cm)
            img = f"qae_scene_{name}.png"
            meta = G.make_scene(seed, ROOT / "data" / img, calib["H_canvas_to_camera"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "targets.json").write_text(json.dumps({"scene": name, **meta}, indent=1))
            show(img); time.sleep(2.5)
            shots = []
            Q.ctrl(ae=True)
            r = Q.shoot(out, "ae", settle_s=3.0); shots.append({"label": "ae", "mode": "ae", **r})
            for label, (iso, sh) in PRECHECK_SHOTS[a.gen]:
                Q.ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
                r = Q.shoot(out, label, settle_s=Q.settle_for(1.0, sh), expect=(iso, sh))
                shots.append({"label": label, "mode": "manual", "iso": iso, "shutter_s": sh, **r})
            Q.ctrl(ae=True)
            (out / "manifest.json").write_text(json.dumps({"scene": name, "layout": a.layout, "gen": a.gen, "ipad_image": img,
                                                           "window_frac": meta["window_frac"], "window_pos": meta["window_pos"],
                                                           "shots": shots}, indent=1))
            log(f"precheck {name}: window {meta['window_frac']:.2f}@{meta['window_pos']}  {time.time()-t0:.0f}s")
        except Exception:
            log(f"precheck {name}: FAILED\n{traceback.format_exc()}")
            for f in out.glob("*"):
                f.unlink()
            time.sleep(15)
    log("PRECHECK_PHASE_DONE")


def phase_sync(a):
    """Forever: push unsynced dirs, pull scores, write verdicts. Every call short and timed."""
    PRE.mkdir(parents=True, exist_ok=True); SWEEPS.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            # push (rsync is incremental; --exclude keeps node-made files)
            for local, remote in ((PRE, "~/qa/qae_batch/precheck/"), (SWEEPS, "~/qa/qae_batch/sweeps/")):
                if any(local.glob("*/manifest.json")):
                    run_quiet(["rsync", "-az", "--exclude", "vlm.json", "--exclude", "verdict.json",
                               "-e", " ".join(SSH[:-1]), str(local) + "/", f"{NODE}:{remote}"], timeout=240)
            # pull vlm.json for dirs that lack one
            need = [d for d in list(PRE.glob("*")) + list(SWEEPS.glob("*"))
                    if (d / "manifest.json").exists() and not (d / "vlm.json").exists()]
            if need:
                r = run_quiet(SSH + ["cd ~/qa/qae_batch && find precheck sweeps -name vlm.json"], timeout=40)
                have = set(r.stdout.split()) if r and r.returncode == 0 else set()
                for d in need:
                    rel = f"{'precheck' if d.parent == PRE else 'sweeps'}/{d.name}/vlm.json"
                    if rel in have:
                        rr = run_quiet(["scp", "-o", "ConnectTimeout=12", "-q", f"{NODE}:~/qa/qae_batch/{rel}", str(d / "vlm.json")], timeout=60)
                        if rr and rr.returncode == 0 and d.parent == PRE and not d.name.startswith("cur_"):
                            v = verdict_from(d / "vlm.json")
                            (d / "verdict.json").write_text(json.dumps(v, indent=1))
                            if v.get("gen") == "v3":
                                log(f"verdict {d.name}: {'PASS' if v['pass'] else 'reject'} pairs={len(v['conflict_pairs'])} rec={v['recoverable']} "
                                    f"ae_misses={v['ae_misses']} unrec={v['unrecoverable']} AE 1/{round(1/v['ae']['exposure_s'])} ISO{v['ae']['iso']}")
                            else:
                                log(f"verdict {d.name}: {'PASS' if v['pass'] else 'reject'} b={v['bright_conflict']} d={v['dim_conflict']} "
                                    f"ctrl={v['control_ok']} unrec={v['unrecoverable']} AE 1/{round(1/v['ae']['exposure_s'])} ISO{v['ae']['iso']}")
                        elif rr and rr.returncode == 0:
                            log(f"pulled sweep scores {d.name}")
        except Exception:
            log(f"sync error\n{traceback.format_exc()}")
        time.sleep(a.interval)


def phase_sweep(a):
    SWEEPS.mkdir(parents=True, exist_ok=True)
    Q.ISOS, Q.SHUTTERS = GRID_ISOS, GRID_SHUTTERS
    done = len([d for d in SWEEPS.glob("*") if (d / "manifest.json").exists()])
    while done < a.n:
        prefix = "qae_" if a.gen == "v2" else a.gen + "_"
        passing = sorted(d for d in PRE.glob(prefix + "*") if (d / "verdict.json").exists()
                         and json.loads((d / "verdict.json").read_text())["pass"]
                         and not (SWEEPS / d.name / "manifest.json").exists())
        if not passing:
            log(f"sweep: waiting for passing verdicts ({done}/{a.n} swept)"); time.sleep(60); continue
        d = passing[0]; name = d.name; sdir = SWEEPS / name; t0 = time.time()
        try:
            pre = json.loads((d / "manifest.json").read_text()); tg = json.loads((d / "targets.json").read_text())
            phone_ready(a.focus_cm)
            show(pre["ipad_image"]); time.sleep(2.5)
            sdir.mkdir(parents=True, exist_ok=True)
            shots, retakes = C.sweep_scene(sdir)
            manifest = {"scene": name, "layout": pre["layout"], "gen": pre.get("gen", "v2"), "ipad_image": pre["ipad_image"],
                        "window_frac": pre["window_frac"], "window_pos": pre["window_pos"],
                        "ipad_backlight": "maximum, auto-brightness off", "focus_cm": a.focus_cm,
                        "isos": Q.ISOS, "shutters_s": Q.SHUTTERS, "precheck": json.loads((d / "verdict.json").read_text()),
                        "targets": tg["targets"], "shots": shots, "retakes": retakes,
                        "duration_s": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            (sdir / "targets.json").write_text(json.dumps({"scene": name, "targets": tg["targets"]}, indent=1))
            (sdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
            done += 1
            log(f"sweep {name}: {len(shots)} shots, {retakes} retakes, {manifest['duration_s']}s  [{done}/{a.n}]")
        except Exception:
            log(f"sweep {name}: FAILED\n{traceback.format_exc()}")
            for f in sdir.glob("*"):
                f.unlink()
            time.sleep(15)
    log("SWEEP_PHASE_DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["precheck", "sync", "sweep"])
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--layout", default="layoutB"); ap.add_argument("--focus-cm", type=float, default=33.0)
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument("--gen", choices=list(GENS), default="v3")
    a = ap.parse_args()
    {"precheck": phase_precheck, "sync": phase_sync, "sweep": phase_sweep}[a.phase](a)
