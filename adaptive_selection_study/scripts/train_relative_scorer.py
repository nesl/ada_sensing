"""
RELATIVE (set / joint) selector: sees all 27 candidate features of a scene at
once and scores them against each other via self-attention, trained listwise to
rank by feature-distance-to-clean. Deployable (no clean image at test).

Compared to: VisiT (32), independent per-candidate scorer (~34), feature-oracle (40.8).
Split: seed-0 3/1/1 by reference image. Reports TRAIN/val/TEST.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

here = Path(__file__).resolve().parent
fe = np.load(here / "feature_cache_resnet50.npz")
fd = np.load(here / "fidelity_cache_resnet50.npz")
assert np.array_equal(fe["gidx"], fd["gidx"]) and np.array_equal(fe["pidx"], fd["pidx"])
gidx, pidx = fe["gidx"], fe["pidx"]
meta = json.load(open(here / "group_meta.json"))
ng = len(meta)

# per-scene tensors [ng,27,*] -- vectorized (hoist arrays out of any loop!)
Xn = fe["feats"]; FEAT_COS = fd["feat_cos"]; CORRECT = fd["correct"]; CONFV = fd["conf"]
FEAT = np.zeros((ng, 27, 2048), np.float16)
DIST = np.full((ng, 27), np.nan, np.float32)
COR = np.zeros((ng, 27), bool)
CONF = np.zeros((ng, 27), np.float32)
FEAT[gidx, pidx] = Xn
DIST[gidx, pidx] = FEAT_COS
COR[gidx, pidx] = CORRECT
CONF[gidx, pidx] = CONFV
# L2-normalize features
FEAT = FEAT.astype(np.float32)
FEAT /= (np.linalg.norm(FEAT, axis=2, keepdims=True) + 1e-6)

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

def acc_from_scores(groups, scores):   # scores: [ng,27]  -> top-1 selection acc
    pick = scores[groups].argmax(1)
    return 100 * COR[groups, pick].mean()

def topk_contains(groups, scores, k):  # % scenes where a top-k scored candidate is correct
    order = np.argsort(-scores[groups], axis=1)[:, :k]           # [G,k] indices
    hit = np.take_along_axis(COR[groups], order, axis=1).any(1)  # any correct in top-k
    return 100 * hit.mean()

class RelScorer(nn.Module):
    def __init__(self, d=256, layers=2, heads=4, use_cfg=True, drop=0.2):
        super().__init__()
        self.proj = nn.Linear(2048, d)
        self.use_cfg = use_cfg
        self.cfg = nn.Embedding(27, d)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4*d, dropout=drop,
                                         batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):                      # x: [B,27,2048]
        h = self.proj(x)
        if self.use_cfg:
            h = h + self.cfg.weight.unsqueeze(0)   # config identity per slot
        h = self.enc(h)
        return self.head(h).squeeze(-1)            # [B,27]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--target", choices=["dist", "correct", "combo"], default="dist",
                    help="dist=feature-distance; correct=Oracle-correctness labels; combo=blend")
    ap.add_argument("--alpha", type=float, default=0.5, help="combo weight on correctness")
    ap.add_argument("--loss", choices=["soft_kl", "hard_ce"], default="soft_kl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--use_cfg", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.2)
    a = ap.parse_args()
    dev = torch.device("cuda"); random.seed(0); torch.manual_seed(0)

    visit_te = acc_from_scores(TE, CONF)
    oracle_te = acc_from_scores(TE, -DIST)
    oracle_tr = acc_from_scores(TR, -DIST)
    print(f"[refs] test VisiT={visit_te:.2f} | train oracle={oracle_tr:.2f} | test oracle={oracle_te:.2f} "
          f"| test Oracle-S={100*np.mean([COR[g].any() for g in TE]):.2f}")

    Ft = torch.tensor(FEAT, device=dev)
    d = np.nan_to_num(DIST, nan=np.nanmax(DIST))
    dist_soft = np.exp(-d / a.tau); dist_soft /= dist_soft.sum(1, keepdims=True)
    # correctness target (Oracle-style): among correct configs prefer the closest-to-clean;
    # for all-wrong scenes fall back to the feature-distance target.
    w = np.exp(-d / a.tau) * COR.astype(np.float32)
    has = w.sum(1) > 0
    corr_soft = np.where(has[:, None], w / np.clip(w.sum(1, keepdims=True), 1e-9, None), dist_soft)
    if a.target == "dist":
        tgt = dist_soft
    elif a.target == "correct":
        tgt = corr_soft
    else:  # combo
        tgt = a.alpha * corr_soft + (1 - a.alpha) * dist_soft
        tgt /= tgt.sum(1, keepdims=True)
    SOFT = torch.tensor(tgt).float().to(dev)
    HARD = torch.tensor(tgt.argmax(1)).long().to(dev)
    print(f"target={a.target}  (scenes with a correct config: {100*has.mean():.1f}%)")

    net = RelScorer(a.d, a.layers, use_cfg=bool(a.use_cfg), drop=a.drop).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    TRt = torch.tensor(TR, device=dev)

    best = (-1, -1, -1)
    for ep in range(1, a.epochs + 1):
        net.train(); perm = TRt[torch.randperm(len(TRt), device=dev)]
        for k in range(0, len(perm), a.bs):
            b = perm[k:k+a.bs]
            s = net(Ft[b])
            if a.loss == "soft_kl":
                loss = F.kl_div(F.log_softmax(s, 1), SOFT[b], reduction="batchmean")
            else:
                loss = F.cross_entropy(s, HARD[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            alls = torch.zeros(ng, 27, device=dev)
            for k in range(0, ng, 512):
                alls[k:k+512] = net(Ft[k:k+512])
            sc = alls.cpu().numpy()
        tr = acc_from_scores(TR, sc); va = acc_from_scores(VA, sc); te = acc_from_scores(TE, sc)
        if va > best[1]: best = (tr, va, te); best_sc = sc.copy()
        if ep % 5 == 0 or ep == 1:
            print(f"ep{ep:02d} TRAIN={tr:.2f} val={va:.2f} test={te:.2f}")
    print(f"\nBEST(by val): TRAIN={best[0]:.2f} val={best[1]:.2f} TEST={best[2]:.2f}  "
          f"(VisiT={visit_te:.2f}, indep-scorer~34, oracle={oracle_te:.2f})")

    # top-k "contains a correct capture" on TEST (recall@k of the selector)
    print("\n== TEST top-k contains-correct (%) ==")
    print(f"{'k':>3} | {'relative':>9} | {'VisiT':>7} | {'oracle-rank':>11} | {'random-k':>9}")
    oracleS = 100*np.mean([COR[g].any() for g in TE])
    for k in [1, 3, 5, 10]:
        rel = topk_contains(TE, best_sc, k)
        vis = topk_contains(TE, CONF, k)
        orc = topk_contains(TE, -DIST, k)
        rnd = 100*np.mean([COR[g][:k].any() for g in TE])  # first-k configs (arbitrary order proxy)
        print(f"{k:>3} | {rel:9.2f} | {vis:7.2f} | {orc:11.2f} | {rnd:9.2f}")
    print(f"Oracle-S (any of 27) = {oracleS:.2f}%  (= top-27 ceiling)")

if __name__ == "__main__":
    main()
