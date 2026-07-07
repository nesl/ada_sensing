from pathlib import Path
import numpy as np

here = Path(__file__).resolve().parent
z = np.load(here / "fidelity_cache_resnet50.npz")
conf, correct, feat_cos, struct_l2, gidx = (z["conf"], z["correct"], z["feat_cos"],
                                            z["struct_l2"], z["gidx"])
ng = int(gidx.max()) + 1
by_g = [[] for _ in range(ng)]
for i in range(len(conf)): by_g[gidx[i]].append(i)
by_g = [np.array(ix) for ix in by_g]

anyok = np.zeros(ng, bool)
for i in range(len(conf)): anyok[gidx[i]] |= correct[i]

def sel_acc(score, maximize):
    ok = np.zeros(ng, bool)
    picks = np.zeros(ng, np.int64)
    for g in range(ng):
        ix = by_g[g]; s = score[ix]
        j = ix[s.argmax()] if maximize else ix[s.argmin()]
        ok[g] = correct[j]; picks[g] = j
    return 100 * ok.mean(), picks

visit_acc, visit_pick = sel_acc(conf, True)
fc_acc, fc_pick = sel_acc(feat_cos, False)      # min feature cosine distance
st_acc, st_pick = sel_acc(struct_l2, False)     # min structure L2
print(f"Oracle-S ceiling            : {100*anyok.mean():.2f}%")
print(f"VisiT (max confidence)      : {visit_acc:.2f}%")
print(f"closest-in-FEATURE (min cos): {fc_acc:.2f}%   <-- feature-space label")
print(f"closest-in-STRUCTURE (min L2): {st_acc:.2f}%  <-- pixel/structure label")

# among recoverable scenes, how often does each selector land a correct capture?
rec = anyok
print(f"\n-- restricted to recoverable scenes (n={int(rec.sum())}) --")
for name, pick in [("VisiT", visit_pick), ("feat-cos", fc_pick), ("struct", st_pick)]:
    got = correct[pick][rec].mean()
    print(f"  {name:9s}: {100*got:.2f}% of recoverable scenes recovered  "
          f"(= {100*correct[pick][rec].mean()*rec.mean():.2f}% overall)")

# separability of feat_cos for right vs wrong (per-candidate, all 162k) via AUC
def auc(score, y):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    n1 = y.sum(); n0 = len(y)-n1
    return (ranks[y==1].sum() - n1*(n1+1)/2)/(n1*n0)
print(f"\nper-candidate AUC(feat_cos -> correct): {auc(-feat_cos, correct.astype(int)):.3f}  (higher=better; min-dist predicts correct)")
print(f"per-candidate AUC(confidence -> correct): {auc(conf, correct.astype(int)):.3f}")

# agreement: does min-feat-cos pick the same candidate as max-conf?
print(f"\nfeat-cos vs VisiT pick same candidate: {100*np.mean(fc_pick==visit_pick):.1f}% of scenes")

# blend: rank by conf but break toward low feat_cos; simple z-score blend sweep
cz = (conf - conf.mean())/conf.std()
fz = (feat_cos - feat_cos.mean())/feat_cos.std()
print("\n== blend score = z(conf) - w*z(feat_cos) ==")
for w in [0,0.5,1,2,4]:
    a,_ = sel_acc(cz - w*fz, True)
    print(f"  w={w:<4} -> {a:.2f}%")
