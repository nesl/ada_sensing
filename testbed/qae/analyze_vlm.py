"""
Analyze a VLM-scored sweep: per-question score over the ISO x shutter grid,
tag each target, plot heatmaps.

  python3 analyze_vlm.py qae/vlm_scene01.json qae/vlm_scene01_heat.png
"""
import json
import math
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ISOS = [50, 100, 200, 400, 800, 1600, 3200]
SHS = [1 / 4000, 1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4, 1.0]
SHL = ["1/4000", "1/1000", "1/250", "1/60", "1/15", "1/4", "1s"]


def ev(r):
    return math.log2(r["exposure_s"] * r["iso"])


def main(path, out_png):
    d = json.load(open(path))
    rows = d["rows"]
    man = [r for r in rows if r["mode"] == "manual"]
    ae = [r for r in rows if r["mode"] == "ae"]
    ae_ev = np.mean([ev(r) for r in ae])

    def find(iso, sh):
        return min(man, key=lambda r: abs(math.log2(r["iso"] / iso)) + abs(math.log2(r["exposure_s"] / sh)))

    T = d["targets"]
    M = {t["id"]: np.array([[find(i, s)["scores"][t["id"]] for s in SHS] for i in ISOS]) for t in T}
    tags = {}
    print(f"AE: {[(r['label'], round(r['exposure_s'], 4), r['iso']) for r in ae]}")
    print(f"{'target':18s} {'AE':>5s} {'best':>5s}  best setting     stops vs AE   tag          AE reply")
    for t in T:
        k = t["id"]
        aes = float(np.mean([r["scores"][k] for r in ae]))
        best = max(man, key=lambda r: r["scores"][k])
        bs = best["scores"][k]
        if t["kind"] == "negative":
            tag = "negative"
        elif bs < 0.5:
            tag = "impossible"
        elif aes >= 0.9 * bs:
            tag = "no-conflict"
        else:
            tag = "conflict"
        tags[k] = tag
        sh = best["exposure_s"]
        shl = f"1/{round(1/sh)}" if sh < 1 else "1s"
        rep = ae[0]["replies"][k][:28]
        print(f"{k:18s} {aes:5.2f} {bs:5.2f}  ISO {best['iso']:<5}{shl:>7s}   {ev(best)-ae_ev:+5.1f}        {tag:12s} {rep!r}")
    # shared best single setting over non-negative targets
    keys = [t["id"] for t in T if t["kind"] != "negative"]
    shared = np.mean([M[k] for k in keys], axis=0)
    bi, bj = np.unravel_index(shared.argmax(), shared.shape)
    print(f"best single setting for all: ISO {ISOS[bi]}, {SHL[bj]} -> " +
          ", ".join(f"{k}={M[k][bi, bj]:.2f}" for k in keys))

    n = len(T)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.9))
    for ax, t in zip(np.atleast_1d(axs), T):
        m = M[t["id"]]
        im = ax.imshow(m, origin="lower", cmap="viridis", vmin=0, vmax=1)
        xs = np.linspace(-0.5, 6.5, 100)
        ax.plot(xs, np.log2((2 ** ae_ev / (1 / 4000 * 4 ** xs)) / 50), "r--", lw=1.6)
        a = ae[0]
        ax.plot(math.log(a["exposure_s"] * 4000) / math.log(4), math.log2(a["iso"] / 50), "r*", ms=13, mec="white")
        ax.set_title(f"{t['id']} [{tags[t['id']]}]\nAE {np.mean([r['scores'][t['id']] for r in ae]):.2f} · max {m.max():.2f}", fontsize=9)
        ax.set_xticks(range(7)); ax.set_xticklabels(SHL, fontsize=6, rotation=45)
        ax.set_yticks(range(7)); ax.set_yticklabels(ISOS, fontsize=6)
        ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6.5)
    fig.colorbar(im, ax=axs, fraction=0.012)
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    json.dump(tags, open(path.replace(".json", "_tags.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
