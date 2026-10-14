"""Three-shot pre-check gate as a function (used by the batch collector)."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import qa_sweep as Q  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NODE = "chi-gigaio"
CROP = "602,854,2687,2027"


def run_precheck(scene, targets_path, out_dir, dark=(50, 0.001), bright=(3200, 0.25), focus_cm=33.0):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Q.ctrl(mode="photo", af=False, focus_diopters=round(100.0 / focus_cm, 3))
    shots = []
    Q.ctrl(ae=True)
    t0 = time.time()
    r = Q.shoot(out, "ae", settle_s=3.0); shots.append({"label": "ae", "mode": "ae", **r})
    print(f"  precheck {scene}: ae shot {time.time()-t0:.0f}s", flush=True)
    for label, (iso, sh) in (("dark", dark), ("bright", bright)):
        Q.ctrl(ae=False, iso=iso, shutter_ns=int(round(sh * 1e9)))
        r = Q.shoot(out, label, settle_s=Q.settle_for(1.0, sh), expect=(iso, sh))
        shots.append({"label": label, "mode": "manual", "iso": iso, "shutter_s": sh, **r})
        print(f"  precheck {scene}: {label} shot {time.time()-t0:.0f}s (attempts {r['attempts']})", flush=True)
    Q.ctrl(ae=True)
    (out / "targets.json").write_text(Path(targets_path).read_text())
    (out / "manifest.json").write_text(json.dumps({"scene": scene, "shots": shots}, indent=1))
    (out / "vlm.json").unlink(missing_ok=True)

    # the node's vlm_loop scores any precheck dir with manifest + targets and no vlm.json
    subprocess.run(["ssh", "-o", "ConnectTimeout=20", NODE, f"rm -f ~/qae/precheck/{scene}/vlm.json"],
                   capture_output=True, timeout=60)
    for _ in range(3):
        if subprocess.run(["rsync", "-az", "--exclude", "vlm.json", "-e", "ssh -o ConnectTimeout=25", str(out),
                           f"{NODE}:~/qae/precheck/"], timeout=180).returncode == 0:
            break
        time.sleep(10)
    deadline = time.time() + 420
    while time.time() < deadline:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", NODE, f"test -f ~/qae/precheck/{scene}/vlm.json && echo yes"],
                           capture_output=True, text=True, timeout=60)
        if "yes" in r.stdout:
            print(f"  precheck {scene}: scored on node after {time.time()-t0:.0f}s", flush=True)
            break
        time.sleep(8)
    else:
        raise RuntimeError(f"{scene}: node did not score the pre-check in time")
    subprocess.run(["scp", "-o", "ConnectTimeout=25", "-q", f"{NODE}:~/qae/precheck/{scene}/vlm.json", str(out / "vlm.json")],
                   check=True, timeout=120)

    d = json.loads((out / "vlm.json").read_text())
    rows = {r["label"]: r for r in d["rows"]}
    verdict = {"bright_conflict": False, "dim_conflict": False, "control_ok": None, "negative_ok": True}
    unrecoverable, table = [], []
    for t in d["targets"]:
        k = t["id"]; s = {l: rows[l]["scores"][k] for l in ("dark", "ae", "bright")}
        table.append({"id": k, "band": t["band"], **s})
        if t["kind"] == "negative":
            verdict["negative_ok"] &= all(v >= 1 for v in s.values())
            continue
        if max(s.values()) < 1:
            unrecoverable.append(k)
        if t["band"] == "bright" and s["dark"] >= 1 and s["ae"] < 1:
            verdict["bright_conflict"] = True
        if t["band"] == "dim" and s["bright"] >= 1 and s["ae"] < 1:
            verdict["dim_conflict"] = True
        if t["band"] == "control":
            verdict["control_ok"] = s["ae"] >= 1
    npos = sum(1 for t in d["targets"] if t["kind"] != "negative")
    verdict["unrecoverable"] = unrecoverable
    verdict["few_unrecoverable"] = len(unrecoverable) <= npos // 3
    verdict["some_conflict"] = verdict["bright_conflict"] or verdict["dim_conflict"]
    ok = verdict["few_unrecoverable"] and verdict["some_conflict"] and verdict["negative_ok"]
    verdict["ae"] = {"exposure_s": rows["ae"]["exposure_s"], "iso": rows["ae"]["iso"]}
    return ok, verdict, table
