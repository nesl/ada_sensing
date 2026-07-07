"""
Distribution of VisiT's SELECTED confidence in 5% intervals.

For each sample we take the candidate VisiT would pick (max softmax over the 27
candidates) and record that winning confidence. We bin those confidences into
5% intervals [0,5), [5,10), ... [95,100] and report, per bin:
  - share of samples
  - cumulative share
  - accuracy of the selected pick within that bin (is high confidence trustworthy?)

Also prints the same for ALL 162k candidate confidences (not just the winners).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def hist_table(conf_pct, correct, title):
    edges = np.arange(0, 105, 5)  # 0,5,...,100
    b = np.clip((conf_pct // 5).astype(int), 0, 19)  # bin index 0..19
    n = len(conf_pct)
    print(f"\n================ {title} ================")
    print(f"n = {n}")
    print(" interval      | count |  % of n | cumulative |  accuracy")
    cum = 0.0
    rows = []
    for k in range(20):
        m = b == k
        cnt = int(m.sum())
        pct = 100.0 * cnt / max(1, n)
        cum += pct
        acc = 100.0 * correct[m].mean() if cnt else float("nan")
        lo, hi = edges[k], edges[k + 1]
        print(f" [{lo:3d},{hi:3d})%   | {cnt:5d} | {pct:6.2f}% | {cum:7.2f}%   | {acc:7.2f}%")
        rows.append({"lo": int(lo), "hi": int(hi), "count": cnt,
                     "pct": round(pct, 3), "cum_pct": round(cum, 3),
                     "accuracy_pct": None if cnt == 0 else round(acc, 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cache", default=str(here / "downstream_cache_resnet50.npz"))
    ap.add_argument("--out", default=str(here / "confidence_hist_report.json"))
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    conf, correct, gidx = z["conf"], z["correct"], z["gidx"]
    ng = int(gidx.max()) + 1

    # winning candidate per group (argmax confidence)
    sel_conf = np.full(ng, -1.0, np.float32)
    sel_correct = np.zeros(ng, np.bool_)
    for i in range(len(conf)):
        g = gidx[i]
        if conf[i] > sel_conf[g]:
            sel_conf[g] = conf[i]
            sel_correct[g] = correct[i]

    rows_sel = hist_table(sel_conf * 100.0, sel_correct,
                          "VisiT SELECTED-confidence distribution (per sample)")
    rows_all = hist_table(conf * 100.0, correct,
                          "ALL candidate confidences (per image, 162k)")

    json.dump({"selected": rows_sel, "all_candidates": rows_all,
               "num_samples": ng, "num_images": int(len(conf)),
               "selected_mean_conf_pct": round(float(sel_conf.mean()) * 100, 3),
               "selected_acc_pct": round(float(sel_correct.mean()) * 100, 3)},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
