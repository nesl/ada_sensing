"""
Equal-budget comparison from model_text.py results that carry per-candidate rankings (and,
when present, logits). With B captures:
  top-B          each method's B highest-ranked settings for the target
  diversified-B  greedy expected coverage under the head's probabilities p_c = sigmoid(logit_c):
                 pick the setting with the largest marginal gain in 1 - prod(1 - p_c) over the chosen set
  fixed bracket  the query-free competitor: greedy set cover on the other held-out scenes
Coverage = fraction of targets answered (real score 1) by at least one of the B shots.

  python3 qae/budget_eval.py results_seed*.json [--budgets 1,2,3]
"""
import argparse
import json

import numpy as np

METHODS = ["direct", "text", "region", "text_real"]


def greedy_cover(rows, B):
    S = np.array([r["cand_scores"] for r in rows]) >= 1
    chosen, covered = [], np.zeros(len(rows), bool)
    while len(chosen) < B:
        gains = [(~covered & S[:, c]).sum() if c not in chosen else -1 for c in range(S.shape[1])]
        c = int(np.argmax(gains)); chosen.append(c); covered |= S[:, c]
    return chosen


def diversified(logits, B):
    p = 1 / (1 + np.exp(-np.array(logits))); chosen = []; miss = np.ones(())
    for _ in range(B):
        gains = [(miss * p[c]) if c not in chosen else -1 for c in range(len(p))]
        c = int(np.argmax(gains)); chosen.append(c); miss = miss * (1 - p[c])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--budgets", default="1,2,3")
    a = ap.parse_args()
    runs = [json.load(open(f))["rows"] for f in a.files]
    if "ranking" not in runs[0][0]:
        raise SystemExit("results lack rankings; re-run model_text.py with the current script")
    has_logits = "logits" in runs[0][0]
    print(f"{len(runs)} seeds, {len(runs[0])} held-out targets" + ("" if has_logits else " (no logits: diversified policy unavailable)"))
    for B in [int(x) for x in a.budgets.split(",")]:
        top = {m: [] for m in METHODS}; div = {m: [] for m in METHODS}; fixed = []
        for rows in runs:
            S = np.array([r["cand_scores"] for r in rows]) >= 1
            for m in METHODS:
                top[m].append(np.mean([S[i, r["ranking"][m][:B]].any() for i, r in enumerate(rows)]))
                if has_logits:
                    div[m].append(np.mean([S[i, diversified(r["logits"][m], B)].any() for i, r in enumerate(rows)]))
            hit = []
            for sc in sorted({r["scene"] for r in rows}):
                tr = [r for r in rows if r["scene"] != sc]; te = [i for i, r in enumerate(rows) if r["scene"] == sc]
                ch = greedy_cover(tr, B); hit += [S[i, ch].any() for i in te]
            fixed.append(np.mean(hit))
        print(f"  B={B}: fixed bracket {np.mean(fixed):.3f}")
        print("        top-B        " + "  ".join(f"{m} {np.mean(v):.3f}" for m, v in top.items()))
        if has_logits:
            print("        diversified  " + "  ".join(f"{m} {np.mean(v):.3f}" for m, v in div.items()))


if __name__ == "__main__":
    main()
