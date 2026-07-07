"""
Gated VisiT: for each sample, drop 'distorted' candidates, then pick the max-
confidence survivor (fall back to global argmax-conf if all are gated out).
Measures accuracy vs plain VisiT (32.02%) and the Oracle ceiling, and sweeps
single-feature gates + a combined gate.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
z = np.load(here / "combined_cache_resnet50.npz", allow_pickle=True)
conf, correct, gidx, feats = z["conf"], z["correct"], z["gidx"], z["feats"]
FN = list(z["feat_names"])
ng = int(gidx.max()) + 1

# group flat indices by group
by_g = [[] for _ in range(ng)]
for i in range(len(conf)):
    by_g[gidx[i]].append(i)
by_g = [np.array(ix) for ix in by_g]

lum, fdark, fbright, contrast, entropy = (feats[:, k] for k in range(5))

def gated_acc(passmask):
    """passmask: bool per flat candidate (True = keep). Select max-conf survivor
    per group; fall back to global max-conf if none survive."""
    sel_correct = np.zeros(ng, bool)
    changed = 0
    for g in range(ng):
        ix = by_g[g]
        c = conf[ix]
        pm = passmask[ix]
        base = ix[c.argmax()]
        if pm.any():
            cand = ix[pm]
            pick = cand[conf[cand].argmax()]
        else:
            pick = base
        sel_correct[g] = correct[pick]
        if pick != base:
            changed += 1
    return 100 * sel_correct.mean(), changed

base_acc, _ = gated_acc(np.ones(len(conf), bool))
# oracle-S
anyok = np.zeros(ng, bool)
for i in range(len(conf)):
    anyok[gidx[i]] |= correct[i]
print(f"plain VisiT      : {base_acc:.2f}%")
print(f"Oracle-S ceiling : {100*anyok.mean():.2f}%\n")

print("== single-feature gates (keep candidate if condition holds) ==")
sweeps = [
    ("frac_bright <= t", lambda t: fbright <= t, [0.05, 0.1, 0.15, 0.2, 0.3]),
    ("frac_dark   <= t", lambda t: fdark <= t, [0.05, 0.1, 0.15, 0.2, 0.3]),
    ("entropy     >= t", lambda t: entropy >= t, [2.5, 3.0, 3.25, 3.5]),
    ("contrast    >= t", lambda t: contrast >= t, [20, 25, 30, 35]),
    ("lum in [t,255-t]", lambda t: (lum >= t) & (lum <= 255 - t), [20, 40, 60]),
]
for name, fn, ts in sweeps:
    for t in ts:
        acc, ch = gated_acc(fn(t))
        print(f"  {name:18s} t={t:<5} -> acc={acc:5.2f}%  (Δ={acc-base_acc:+.2f}, changed {ch} samples)")

print("\n== combined gate (drop clearly-unusable captures) ==")
for (tb, td, te, tc) in [(0.2, 0.2, 2.5, 20), (0.15, 0.15, 3.0, 25), (0.1, 0.1, 3.25, 30)]:
    pm = (fbright <= tb) & (fdark <= td) & (entropy >= te) & (contrast >= tc)
    acc, ch = gated_acc(pm)
    kept = 100 * pm.mean()
    print(f"  fbright<={tb} fdark<={td} entropy>={te} contrast>={tc}: "
          f"acc={acc:.2f}% (Δ={acc-base_acc:+.2f}), keeps {kept:.1f}% of candidates, changed {ch}")

# Oracle gate: best possible if we could drop only wrong-distorted picks and a correct survivor exists
print("\n== reference: score = confidence + lambda*entropy (rerank, not gate) ==")
for lam in [0.0, 0.05, 0.1, 0.2, 0.4]:
    sc = conf + lam * (entropy / 5.0)
    sel_correct = np.zeros(ng, bool)
    for g in range(ng):
        ix = by_g[g]
        sel_correct[g] = correct[ix[sc[ix].argmax()]]
    print(f"  lambda={lam:<4} -> acc={100*sel_correct.mean():.2f}%")
