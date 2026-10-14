"""
Budgeted multi-shot on scored sweeps: with a budget of B captures, which fixed set of settings
(a bracket) answers the most targets? Greedy set cover on the training scenes (5-fold over scenes),
evaluated on held-out scenes as the fraction of recoverable targets scored 1 by at least one shot
in the set. Also the oracle single shot (1.0 by definition on recoverable targets), auto exposure,
and "AE + B-1 fixed shots".

  python3 qae/multishot.py data/qae_batch/sweeps [--prefix qae_] [--budgets 1,2,3,4]
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

GRID = [(iso, sh) for iso in (100, 400, 1600) for sh in (1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4)]


def load(sweeps, prefix):
    scenes = []
    for d in sorted(Path(sweeps).glob(prefix + "*")):
        if not (d / "vlm.json").exists():
            continue
        v = json.loads((d / "vlm.json").read_text()); rows = v["rows"]
        manual = [r for r in rows if r["mode"] == "manual"]
        def near(iso, sh):
            return min(manual, key=lambda r: abs(math.log2(r["iso"] / iso)) + abs(math.log2(r["exposure_s"] / sh)))
        ae = [r for r in rows if r["mode"] == "ae"]
        T = []
        for t in v["targets"]:
            if t["kind"] == "negative":
                continue
            g = np.array([near(i, s)["scores"][t["id"]] >= 1 for i, s in GRID])
            a = max(r["scores"][t["id"]] for r in ae) >= 1
            if g.any():
                T.append({"band": t["band"], "grid": g, "ae": a})
        scenes.append({"name": d.name, "targets": T})
    return scenes


def greedy(train, B, forced=()):
    chosen = list(forced); items = [t for s in train for t in s["targets"]]
    covered = np.zeros(len(items), bool)
    if "ae" in chosen:
        covered |= np.array([t["ae"] for t in items])
    for c in chosen:
        if c != "ae":
            covered |= np.array([t["grid"][c] for t in items])
    while len(chosen) < B:
        best, bi = -1, None
        for c in range(len(GRID)):
            if c in chosen:
                continue
            gain = int((~covered & np.array([t["grid"][c] for t in items])).sum())
            if gain > best:
                best, bi = gain, c
        chosen.append(bi); covered |= np.array([t["grid"][bi] for t in items])
    return chosen


def coverage(test, chosen):
    items = [t for s in test for t in s["targets"]]
    hit = [(t["ae"] if "ae" in chosen else False) or any(t["grid"][c] for c in chosen if c != "ae") for t in items]
    return float(np.mean(hit)), {b: float(np.mean([h for h, t in zip(hit, items) if t["band"] == b])) for b in sorted({t["band"] for t in items})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps"); ap.add_argument("--prefix", default="qae_"); ap.add_argument("--budgets", default="1,2,3,4")
    a = ap.parse_args()
    scenes = load(a.sweeps, a.prefix)
    n = sum(len(s["targets"]) for s in scenes)
    print(f"{len(scenes)} scenes, {n} recoverable targets; AE alone answers {np.mean([t['ae'] for s in scenes for t in s['targets']]):.2f}")
    rng = random.Random(0); order = list(range(len(scenes))); rng.shuffle(order); folds = [order[i::5] for i in range(5)]
    for B in [int(x) for x in a.budgets.split(",")]:
        res = {"fixed bracket": [], "AE + bracket": []}; sets = {"fixed bracket": [], "AE + bracket": []}
        for f in folds:
            te = [scenes[i] for i in f]; tr = [s for i, s in enumerate(scenes) if i not in f]
            c1 = greedy(tr, B); c2 = greedy(tr, B, forced=("ae",)) if B > 1 else ["ae"]
            res["fixed bracket"].append(coverage(te, c1)); res["AE + bracket"].append(coverage(te, c2))
            sets["fixed bracket"].append(c1); sets["AE + bracket"].append(c2)
        for k in res:
            mean = np.mean([r[0] for r in res[k]]); bands = {b: np.mean([r[1].get(b, np.nan) for r in res[k]]) for b in res[k][0][1]}
            names = sorted({tuple(f"ISO{GRID[c][0]} 1/{round(1/GRID[c][1])}" if c != "ae" else "AE" for c in s) for s in sets[k]})
            print(f"  B={B} {k:14s} answered {mean:.2f}  " + "  ".join(f"{b}={v:.2f}" for b, v in bands.items()) + f"   sets: {names[:2]}")


if __name__ == "__main__":
    main()
