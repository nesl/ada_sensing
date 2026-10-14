"""
Aggregate several seeds of model_text.py results into mean ± 95% CI per method, plus
success rate (score >= 1), regret (oracle - chosen), and a paired bootstrap of
text (JEPA) minus direct over held-out targets.

  python3 qae/aggregate_seeds.py qae/results_batch1/seeds/model_text_seed*.json
"""
import json
import sys

import numpy as np

METHODS = ["ae", "fixed", "direct", "text", "region", "text_real", "oracle"]


def ci(x):
    from scipy import stats
    x = np.asarray(x, float)
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def main(files):
    runs = [json.load(open(f)) for f in files]
    print(f"{len(runs)} seeds, {len(runs[0]['rows'])} held-out targets per seed")
    for subset, pred in (("all targets", lambda r: True), ("conflict targets", lambda r: r["conflict"]),
                         ("bright band", lambda r: r["band"] == "bright"), ("mid band", lambda r: r["band"] == "mid"),
                         ("dim band", lambda r: r["band"] == "dim"), ("control", lambda r: r["band"] == "control")):
        per_seed = {m: [np.mean([r[m] for r in run["rows"] if pred(r)]) for run in runs if any(pred(r) for r in run["rows"])] for m in METHODS}
        if not per_seed["ae"]:
            continue
        print(f"\n{subset}  (mean score, ± 95% CI over seeds)")
        for m in METHODS:
            mu, h = ci(per_seed[m]); print(f"  {m:10s} {mu:.3f} ± {h:.3f}")
        succ = {m: np.mean([np.mean([r[m] >= 1 for r in run["rows"] if pred(r)]) for run in runs]) for m in ("fixed", "direct", "text", "region")}
        reg = {m: np.mean([np.mean([r["oracle"] - r[m] for r in run["rows"] if pred(r)]) for run in runs]) for m in ("fixed", "direct", "text", "region")}
        print("  success rate (score=1): " + "  ".join(f"{m}={v:.2f}" for m, v in succ.items()))
        print("  regret vs oracle:       " + "  ".join(f"{m}={v:.2f}" for m, v in reg.items()))
        # JEPA minus direct: (a) seeds as the unit (t-interval), (b) scene-cluster bootstrap with a scene's
        # targets and seeds resampled together. Never pool target-evals across seeds as if independent.
        d_seed = np.array([np.mean([r["text"] - r["direct"] for r in run["rows"] if pred(r)]) for run in runs if any(pred(r) for r in run["rows"])])
        m, h = ci(d_seed)
        per_scene = {}
        for run in runs:
            for r in run["rows"]:
                if pred(r):
                    per_scene.setdefault(r["scene"], []).append(r["text"] - r["direct"])
        sc = sorted(per_scene); vals = [np.mean(per_scene[s]) for s in sc]; w = [len(per_scene[s]) for s in sc]
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(sc), len(sc)); bs.append(np.average([vals[i] for i in idx], weights=[w[i] for i in idx]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  JEPA-text minus direct: {m:+.3f}  seed CI [{m-h:+.3f}, {m+h:+.3f}] (n={len(d_seed)} seeds)  "
              f"scene-cluster CI [{lo:+.3f}, {hi:+.3f}] ({len(sc)} scenes; JEPA ahead in {np.mean([v > 0 for v in vals]):.0%})")


if __name__ == "__main__":
    main(sys.argv[1:])
