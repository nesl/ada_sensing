"""
Learned all-27 selector (deployable; sees the 27 captures, NOT the clean image).
Trains a per-candidate quality scorer on resnet50 features to predict either the
feature-distance-to-clean (regression) or correctness (BCE), then selects
argbest score per scene. Compares to VisiT (fixed max-softmax scorer) and the
feature-oracle ceiling. Split: seed-0 3/1/1 by reference image.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

here = Path(__file__).resolve().parent
fe = np.load(here / "feature_cache_resnet50.npz")
fd = np.load(here / "fidelity_cache_resnet50.npz")
assert np.array_equal(fe["gidx"], fd["gidx"]) and np.array_equal(fe["pidx"], fd["pidx"]), "cache misalignment!"
X = fe["feats"].astype(np.float32)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)   # L2-normalize
gidx, pidx = fe["gidx"], fe["pidx"]
feat_cos, correct, conf = fd["feat_cos"], fd["correct"].astype(np.float32), fd["conf"]
meta = json.load(open(here / "group_meta.json"))
ng = len(meta)

# per-group tensors for evaluation
COR = np.zeros((ng, 27), bool); POS = np.full((ng, 27), -1, np.int64); CONF = np.zeros((ng, 27), np.float32)
for i in range(len(gidx)):
    COR[gidx[i], pidx[i]] = correct[i]; POS[gidx[i], pidx[i]] = i; CONF[gidx[i], pidx[i]] = conf[i]

# seed-0 3/1/1 split by reference
from collections import defaultdict
by_class = defaultdict(lambda: defaultdict(list))
for g, m in enumerate(meta): by_class[m["wnid"]][m["ref"]].append(g)
rng = random.Random(0); TR=[]; VA=[]; TE=[]
for wn, refs in sorted(by_class.items()):
    rl = sorted(refs); rng.shuffle(rl)
    for r in rl[:3]: TR += refs[r]
    for r in rl[3:4]: VA += refs[r]
    for r in rl[4:]: TE += refs[r]
TR, VA, TE = map(np.array, (TR, VA, TE))

def sel_acc(groups, score_flat, maximize=True):
    ok = []
    for g in groups:
        pos = POS[g]; s = score_flat[pos]
        j = s.argmax() if maximize else s.argmin()
        ok.append(COR[g, j])
    return 100 * np.mean(ok)

# candidate-index membership per split (a candidate belongs to its group's split)
def cand_idx(groups):
    gs = set(groups.tolist())
    return np.array([i for i in range(len(gidx)) if gidx[i] in gs])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["dist", "correct"], default="dist")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev = torch.device("cuda")
    random.seed(0); torch.manual_seed(0)

    # references
    visit_te = sel_acc(TE, conf, True)
    oracle_te = sel_acc(TE, -feat_cos, True)   # min dist == max (-dist)
    oracleS_te = 100*np.mean([COR[g].any() for g in TE])
    print(f"[refs on TEST] VisiT={visit_te:.2f} | feat-oracle={oracle_te:.2f} | Oracle-S={oracleS_te:.2f}")

    tr_i, va_i, te_i = cand_idx(TR), cand_idx(VA), cand_idx(TE)
    Xt = torch.tensor(X, device=dev)
    if a.target == "dist":
        y = torch.tensor(feat_cos, device=dev)            # regress distance; select MIN
        lossf = lambda p, yy: F.mse_loss(p, yy); maximize = False
    else:
        y = torch.tensor(correct, device=dev)             # predict correctness; select MAX
        lossf = lambda p, yy: F.binary_cross_entropy_with_logits(p, yy); maximize = True

    net = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    tr_t = torch.tensor(tr_i, device=dev)

    best_va, best_te = -1, -1
    for ep in range(1, a.epochs + 1):
        net.train(); perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        for k in range(0, len(perm), a.bs):
            b = perm[k:k+a.bs]
            p = net(Xt[b]).squeeze(1)
            loss = lossf(p, y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            score = net(Xt).squeeze(1).cpu().numpy()
        tr = sel_acc(TR, score, maximize); va = sel_acc(VA, score, maximize); te = sel_acc(TE, score, maximize)
        if va > best_va: best_va, best_te, best_tr = va, te, tr
        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:02d} TRAIN={tr:.2f} val={va:.2f} test={te:.2f}")
    oracle_tr = sel_acc(TR, -feat_cos, True)
    print(f"\nBEST(by val) target={a.target}: TRAIN={best_tr:.2f} val={best_va:.2f}  test={best_te:.2f}")
    print(f"  (train oracle={oracle_tr:.2f}, test VisiT={visit_te:.2f}, test oracle={oracle_te:.2f})")

if __name__ == "__main__":
    main()
