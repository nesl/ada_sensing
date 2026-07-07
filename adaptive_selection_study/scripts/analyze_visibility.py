"""
Test the hypothesis: among VisiT's high-confidence picks, are the WRONG ones
systematically more distorted (too dark / blown-out / low information) than the
RIGHT ones? If so, an exposure/visibility gate can veto them.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

FN = ["lum_mean", "frac_dark", "frac_bright", "contrast", "entropy"]


def visit_select(conf, gidx, ng):
    sel = np.full(ng, -1, np.int64)
    best = np.full(ng, -1.0, np.float32)
    for i in range(len(conf)):
        g = gidx[i]
        if conf[i] > best[g]:
            best[g] = conf[i]; sel[g] = i
    return sel  # index into flat arrays of the selected candidate per group


def summarize(feats_sel, correct_sel, mask, title):
    r = mask & correct_sel
    w = mask & ~correct_sel
    print(f"\n== {title} ==  (n={int(mask.sum())}: right={int(r.sum())}, wrong={int(w.sum())})")
    print(f"{'feature':10s} | {'RIGHT mean':>11s} {'median':>8s} | {'WRONG mean':>11s} {'median':>8s}")
    for k, name in enumerate(FN):
        fr, fw = feats_sel[r, k], feats_sel[w, k]
        if len(fr) and len(fw):
            print(f"{name:10s} | {fr.mean():11.3f} {np.median(fr):8.3f} | "
                  f"{fw.mean():11.3f} {np.median(fw):8.3f}")


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--cache", default=str(here / "combined_cache_resnet50.npz"))
    args = ap.parse_args()
    z = np.load(args.cache, allow_pickle=True)
    conf, correct, gidx, feats = z["conf"], z["correct"], z["gidx"], z["feats"]
    ng = int(gidx.max()) + 1

    sel = visit_select(conf, gidx, ng)
    sel_conf = conf[sel]
    sel_correct = correct[sel]
    feats_sel = feats[sel]

    print(f"VisiT top-1 acc = {100*sel_correct.mean():.2f}%   (n={ng})")

    for thr in (0.90, 0.80, 0.0):
        m = sel_conf > thr
        summarize(feats_sel, sel_correct, m, f"selected picks with conf > {int(thr*100)}%")

    # How separable is right vs wrong among >80% picks, per single feature (ROC-AUC)?
    from itertools import product
    m = sel_conf > 0.80
    y = sel_correct[m].astype(int)
    print("\n-- single-feature separability of RIGHT vs WRONG among >80% picks (AUC; .5=useless) --")
    for k, name in enumerate(FN):
        x = feats_sel[m, k]
        # AUC via rank statistic
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(x) + 1)
        n1 = y.sum(); n0 = len(y) - n1
        auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        # report both directions (some features predict correct when LOW)
        print(f"{name:10s} AUC={auc:.3f}  (|.5-auc|={abs(.5-auc):.3f})")


if __name__ == "__main__":
    main()
