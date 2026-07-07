"""
Selector idea: pick the candidate whose luminance is closest to the TRAINING
image luminance (instead of max confidence). Tests:
  1) target = mean luminance of es-train clean images
  2) sweep target T to find the ceiling of luminance-matching
  3) hybrid: keep candidates within +/-band of target, then max-confidence
Compared against plain VisiT (32.02%) and Oracle-S (49.93%).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

here = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def lum_of(pil):
    g = pil.convert("L").resize((128, 128))
    return float(np.asarray(g, np.float32).mean())


def train_target():
    root = here / "es-train"
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    vals = np.array([lum_of(Image.open(p)) for p in imgs], np.float32)
    return imgs, vals


# --- candidates ---
z = np.load(here / "combined_cache_resnet50.npz", allow_pickle=True)
conf, correct, gidx, feats = z["conf"], z["correct"], z["gidx"], z["feats"]
lum = feats[:, 0]
ng = int(gidx.max()) + 1
by_g = [[] for _ in range(ng)]
for i in range(len(conf)):
    by_g[gidx[i]].append(i)
by_g = [np.array(ix) for ix in by_g]

anyok = np.zeros(ng, bool)
for i in range(len(conf)):
    anyok[gidx[i]] |= correct[i]

def visit():
    ok = np.zeros(ng, bool)
    for g in range(ng):
        ix = by_g[g]; ok[g] = correct[ix[conf[ix].argmax()]]
    return 100 * ok.mean()

def lum_match(T):
    ok = np.zeros(ng, bool)
    for g in range(ng):
        ix = by_g[g]; ok[g] = correct[ix[np.abs(lum[ix] - T).argmin()]]
    return 100 * ok.mean()

def hybrid(T, band):
    ok = np.zeros(ng, bool)
    for g in range(ng):
        ix = by_g[g]
        near = ix[np.abs(lum[ix] - T) <= band]
        pick = near[conf[near].argmax()] if len(near) else ix[conf[ix].argmax()]
        ok[g] = correct[pick]
    return 100 * ok.mean()

print("computing es-train luminance target ...")
imgs, tv = train_target()
T_train = float(tv.mean())
print(f"es-train luminance: mean={tv.mean():.1f}  median={np.median(tv):.1f}  std={tv.std():.1f}  (n={len(tv)})")

print(f"\nplain VisiT       : {visit():.2f}%")
print(f"Oracle-S ceiling  : {100*anyok.mean():.2f}%")
print(f"\nluminance-match @ es-train target T={T_train:.1f}: {lum_match(T_train):.2f}%")

print("\n== sweep target T (pure luminance matching) ==")
best = (0, -1)
for T in range(50, 205, 10):
    a = lum_match(T)
    best = max(best, (a, T), key=lambda x: x[0]) if isinstance(best, tuple) else best
    star = ""
    print(f"  T={T:3d} -> {a:5.2f}%")
bestT = max(range(50, 205, 5), key=lum_match)
print(f"  best target T*={bestT} -> {lum_match(bestT):.2f}%  (in-sample optimal)")

print("\n== hybrid: within +/-band of es-train target, then max-conf ==")
for band in [15, 25, 40, 60]:
    print(f"  band=+/-{band} -> {hybrid(T_train, band):.2f}%")

# distribution of picked param under luminance matching (are we just recreating a fixed option / AE-like?)
pidx = z["pidx"]
picks = np.array([by_g[g][np.abs(lum[by_g[g]] - T_train).argmin()] for g in range(ng)])
pp = pidx[picks]
uniq, cnt = np.unique(pp, return_counts=True)
top = sorted(zip(cnt, uniq), reverse=True)[:6]
print("\ntop params chosen by luminance-match:", [f"param_{u+1}:{c}" for c, u in top])
