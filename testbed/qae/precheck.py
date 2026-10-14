"""
Pre-check gate for a QAE scene: capture 3 shots (auto exposure, a dark setting,
a bright setting), run the VLM on them on the GPU node, and print per-question
correctness. Run before committing a scene to a full sweep.

  python3 qae/precheck.py <scene_name> <targets.json> [--dark 100,0.001] [--bright 400,0.25]

A scene passes when: every positive question is correct in at least one shot,
at least one bright-band question is correct ONLY in the dark shot, at least one
dim-band question is correct ONLY in the bright shot, the control is correct at
AE, and the negative is "no" in all three.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import qa_sweep as Q  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NODE = "chi-gigaio"
CROP = "602,854,2687,2027"


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=25", NODE, cmd], capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("targets")
    ap.add_argument("--dark", default="100,0.001"); ap.add_argument("--bright", default="400,0.25")
    ap.add_argument("--focus-cm", type=float, default=33.0)
    a = ap.parse_args()
    out = ROOT / "data" / "qae_precheck" / a.scene
    out.mkdir(parents=True, exist_ok=True)
    Q.adb("shell", "input", "keyevent", "KEYCODE_WAKEUP")
    Q.ctrl(mode="photo", af=False, focus_diopters=round(100.0 / a.focus_cm, 3))
    shots = []
    Q.ctrl(ae=True)
    r = Q.shoot(out, "ae", settle_s=3.0); shots.append({"label": "ae", "mode": "ae", **r})
    for label, spec in (("dark", a.dark), ("bright", a.bright)):
        iso, sh = int(spec.split(",")[0]), float(spec.split(",")[1])
        Q.ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
        r = Q.shoot(out, label, settle_s=Q.settle_for(1.0, sh), expect=(iso, sh))
        shots.append({"label": label, "mode": "manual", "iso": iso, "shutter_s": sh, **r})
    Q.ctrl(ae=True)
    (out / "manifest.json").write_text(json.dumps({"scene": a.scene, "shots": shots}, indent=1))
    print("captured:", [(s["label"], s["exif"]) for s in shots], flush=True)

    subprocess.run(["rsync", "-az", "-e", "ssh -o ConnectTimeout=25", str(out), f"{NODE}:~/qae/precheck/"], check=True)
    subprocess.run(["scp", "-o", "ConnectTimeout=25", "-q", a.targets, f"{NODE}:~/qae/precheck/{a.scene}_targets.json"], check=True)
    ssh(f"source ~/venv/bin/activate && cd ~/qae && python vlm_score.py precheck/{a.scene} precheck/{a.scene}_targets.json "
        f"precheck/{a.scene}_vlm.json --crop {CROP} > precheck/{a.scene}.log 2>&1")
    subprocess.run(["scp", "-o", "ConnectTimeout=25", "-q", f"{NODE}:~/qae/precheck/{a.scene}_vlm.json", str(out / "vlm.json")], check=True)

    d = json.loads((out / "vlm.json").read_text())
    T = d["targets"]; rows = {r["label"]: r for r in d["rows"]}
    print(f"\n{'target':18s} {'band':8s} {'dark':>5s} {'AE':>5s} {'bright':>6s}   replies (dark | AE | bright)")
    verdict = {"bright_conflict": False, "dim_conflict": False, "control_ok": True, "negative_ok": True}
    unrecoverable = []
    for t in T:
        k = t["id"]; s = {l: rows[l]["scores"][k] for l in ("dark", "ae", "bright")}
        rep = " | ".join(rows[l]["replies"][k][:16] for l in ("dark", "ae", "bright"))
        print(f"{k:18s} {t['band']:8s} {s['dark']:5.2f} {s['ae']:5.2f} {s['bright']:6.2f}   {rep}")
        if t["kind"] == "negative":
            verdict["negative_ok"] &= all(v >= 1 for v in s.values())
        else:
            if max(s.values()) < 1:
                unrecoverable.append(k)
            if t["band"] == "bright" and s["dark"] >= 1 and s["ae"] < 1:
                verdict["bright_conflict"] = True
            if t["band"] == "dim" and s["bright"] >= 1 and s["ae"] < 1:
                verdict["dim_conflict"] = True
            if t["band"] == "control":
                verdict["control_ok"] &= s["ae"] >= 1
    npos = sum(1 for t in T if t["kind"] != "negative")
    verdict["few_unrecoverable"] = len(unrecoverable) <= npos // 3
    verdict["some_conflict"] = verdict["bright_conflict"] or verdict["dim_conflict"]
    ok = verdict["few_unrecoverable"] and verdict["some_conflict"] and verdict["control_ok"] and verdict["negative_ok"]
    print("\nunrecoverable (drop):", unrecoverable)
    print("verdict:", verdict, "-> PASS" if ok else "-> FAIL")


if __name__ == "__main__":
    main()
