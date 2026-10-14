"""Sweep-level analysis of a QAE batch: per-target tags, AE vs best-fixed vs oracle."""
import json, math, sys, collections
from pathlib import Path
import numpy as np
SW = Path(sys.argv[1] if len(sys.argv) > 1 else "data/qae_batch/sweeps")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""   # e.g. "v3_" to analyze one generator only
OUT = SW.parent / ("sweep_targets_analysis" + (f"_{PREFIX.rstrip('_')}" if PREFIX else "") + ".json")
GRID = [(iso, sh) for iso in (100, 400, 1600) for sh in (1/1000, 1/250, 1/60, 1/15, 1/4)]
def near(rows, iso, sh):
    return min((r for r in rows if r["mode"] == "manual"), key=lambda r: abs(math.log2(r["iso"]/iso)) + abs(math.log2(r["exposure_s"]/sh)))
T = []  # one row per positive target
for d in sorted(SW.glob(PREFIX + "*")):
    if not (d / "vlm.json").exists(): continue
    v = json.loads((d / "vlm.json").read_text()); rows = v["rows"]
    ae = np.mean([r["scores"][t["id"]] for r in rows if r["mode"] == "ae"] or [0]) if False else None
    for t in v["targets"]:
        if t["kind"] == "negative": continue
        grid = np.array([near(rows, i, s)["scores"][t["id"]] for i, s in GRID])
        aes = float(np.mean([r["scores"][t["id"]] for r in rows if r["mode"] == "ae"]))
        best = float(grid.max())
        tag = "impossible" if best < 1 else ("no-conflict" if aes >= 0.9 * best else "conflict")
        T.append({"scene": d.name, "id": t["id"], "band": t["band"], "level": t["level"], "ae": aes, "grid": grid, "best": best, "tag": tag})
n_scenes = len({t["scene"] for t in T})
print(f"{n_scenes} scored sweeps, {len(T)} positive targets")
tags = collections.Counter(t["tag"] for t in T)
print("tags:", dict(tags), f"-> recoverable {len(T)-tags['impossible']} ({(len(T)-tags['impossible'])/len(T):.0%}), conflict {tags['conflict']/len(T):.0%}")
print("\nper target type (recoverable only): n, AE, best-fixed, oracle")
R = [t for t in T if t["tag"] != "impossible"]
G = np.stack([t["grid"] for t in R]); fixed_idx = int(G.mean(0).argmax()); fi, fs = GRID[fixed_idx]
print(f"best single setting over all recoverable targets: ISO {fi}, 1/{round(1/fs)}")
def show(name, rs):
    if not rs: return
    g = np.stack([t["grid"] for t in rs])
    print(f"  {name:28s} n={len(rs):3d}  AE={np.mean([t['ae'] for t in rs]):.2f}  fixed={g[:, fixed_idx].mean():.2f}  oracle={np.mean([t['best'] for t in rs]):.2f}")
show("ALL recoverable", R)
BANDS = [b for b in ("bright", "mid", "dim", "control") if any(t["band"] == b for t in R)]
for b in BANDS:
    show(f"  {b} band", [t for t in R if t["band"] == b])
    for k in sorted({t["id"] for t in R if t["band"] == b}): show(f"    {k}", [t for t in R if t["id"] == k])
show("  conflict targets only", [t for t in R if t["tag"] == "conflict"])
# where do oracles sit per band?
for b in BANDS:
    c = collections.Counter(GRID[int(t["grid"].argmax())] for t in R if t["band"] == b and t["tag"] == "conflict")
    print(f"  oracle settings, {b} conflict targets:", [(f"ISO{i} 1/{round(1/s)}", n) for (i, s), n in c.most_common(3)])
json.dump([{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in t.items()} for t in T], open(OUT, "w"))
