"""
Train the sensor policy on the FEATURE-DISTANCE-TO-CLEAN label.

- Input : the auto-exposure (AE) capture only  -> policy must infer the best
          config from the degraded image, WITHOUT the clean original.
- Target: per-config resnet50 feature cosine distance to the clean source,
          turned into a soft distribution softmax(-dist/tau) over the 27 configs
          (dense supervision), or the argmin config (hard CE).
- Metric: downstream top-1 accuracy of the policy-SELECTED config (cache lookup),
          on a seed-0 3/1/1 split by reference image. All baselines on same split.
"""
from __future__ import annotations
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

here = Path(__file__).resolve().parent
REPO = here.parents[1]           # ada_sensing/
sys.path.insert(0, str(REPO / "policy_network" / "static_pred"))
from policy_model import SensorPolicyNetwork  # noqa

# ---------- data ----------
z = np.load(here / "fidelity_cache_resnet50.npz")
conf_i, correct_i, feat_i, gidx, pidx = (z["conf"], z["correct"], z["feat_cos"], z["gidx"], z["pidx"])
meta = json.load(open(here / "group_meta.json"))
ng = len(meta)

DIST = np.full((ng, 27), np.nan, np.float32)
COR = np.zeros((ng, 27), bool)
CONF = np.zeros((ng, 27), np.float32)
for i in range(len(gidx)):
    DIST[gidx[i], pidx[i]] = feat_i[i]
    COR[gidx[i], pidx[i]] = correct_i[i]
    CONF[gidx[i], pidx[i]] = conf_i[i]
LABELCLS = np.array([m["label"] for m in meta])
AE_PATHS = [m["ae_path"] for m in meta]

def input_paths(spec):
    """spec='ae' -> auto-exposure image; spec='fixed:N' -> param_N capture
    (a constant camera config, which reveals true scene brightness)."""
    if spec == "ae":
        return AE_PATHS
    assert spec.startswith("fixed:")
    N = int(spec.split(":")[1])
    out = []
    for m in meta:
        base = Path(m["ae_path"]).name  # <stem>.JPEG (shared filename)
        out.append(str(here / "es-diverse-test" / "param_control" / m["env"]
                      / f"param_{N}" / m["wnid"] / base))
    return out

# seed-0 3/1/1 split by reference image, per class
def make_split():
    by_class = {}
    for g, m in enumerate(meta):
        by_class.setdefault(m["wnid"], {}).setdefault(m["ref"], []).append(g)
    rng = random.Random(0)
    tr, va, te = [], [], []
    for wnid, refs in sorted(by_class.items()):
        rlist = sorted(refs.keys()); rng.shuffle(rlist)
        for r in rlist[:3]: tr += refs[r]
        for r in rlist[3:4]: va += refs[r]
        for r in rlist[4:]: te += refs[r]
    return np.array(tr), np.array(va), np.array(te)

TR, VA, TE = make_split()

# ---------- baselines (per split, from cache) ----------
def acc_pick(idxs, pick_fn):
    return 100 * np.mean([COR[g, pick_fn(g)] for g in idxs])

def baselines(name, idxs):
    visit = acc_pick(idxs, lambda g: CONF[g].argmax())
    foracle = acc_pick(idxs, lambda g: np.nanargmin(DIST[g]))
    oracleS = 100 * np.mean([COR[g].any() for g in idxs])
    # oracle-F: best fixed config on TRAIN, applied here
    fixed = np.array([COR[g] for g in TR]).mean(0).argmax()
    accF = 100 * np.mean([COR[g, fixed] for g in idxs])
    print(f"  [{name:4s}] VisiT={visit:5.2f} | feat-oracle={foracle:5.2f} | "
          f"Oracle-F(p{fixed+1})={accF:5.2f} | Oracle-S={oracleS:5.2f}")
    return visit, foracle, oracleS, accF

# ---------- policy dataset (AE input) ----------
train_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

INPUT_PATHS = AE_PATHS  # overwritten in main() per --input

class AEDS(Dataset):
    def __init__(self, idxs, tf): self.idxs = idxs; self.tf = tf
    def __len__(self): return len(self.idxs)
    def __getitem__(self, k):
        g = int(self.idxs[k])
        img = self.tf(Image.open(INPUT_PATHS[g]).convert("RGB"))
        return img, g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="mobilenet_v3_small")
    ap.add_argument("--loss", choices=["soft_kl", "hard_ce"], default="soft_kl")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--scope", choices=["head_only", "partial", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--blr", type=float, default=3e-5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--input", default="ae", help="'ae' or 'fixed:N' (e.g. fixed:13)")
    a = ap.parse_args()

    global INPUT_PATHS
    INPUT_PATHS = input_paths(a.input)
    print(f"policy input source: {a.input}")

    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("cuda")
    print(f"split: train={len(TR)} val={len(VA)} test={len(TE)}  (samples)")
    print("baselines:")
    for nm, ix in [("TR", TR), ("VA", VA), ("TE", TE)]:
        baselines(nm, ix)

    # soft targets over 27 configs
    d = np.nan_to_num(DIST, nan=np.nanmax(DIST))
    SOFT = torch.tensor(np.exp(-d / a.tau))
    SOFT = (SOFT / SOFT.sum(1, keepdim=True)).float()
    HARD = torch.tensor(np.nanargmin(DIST, 1)).long()

    model = SensorPolicyNetwork(num_candidates=27, pretrained=True,
                                backbone_name=a.backbone, input_mode="single").to(dev)
    if a.scope == "head_only": model.freeze_backbone()
    elif a.scope == "partial": model.unfreeze_backbone_tail()
    head_params = list(model.policy_head.parameters())
    hp_ids = {id(p) for p in head_params}
    back_params = [p for p in model.parameters() if p.requires_grad and id(p) not in hp_ids]
    opt = torch.optim.AdamW([{"params": head_params, "lr": a.lr},
                             {"params": back_params, "lr": a.blr}], weight_decay=1e-4)

    dl_tr = DataLoader(AEDS(TR, train_tf), batch_size=a.bs, shuffle=True,
                       num_workers=a.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(AEDS(VA, eval_tf), batch_size=256, num_workers=a.workers, pin_memory=True)
    dl_te = DataLoader(AEDS(TE, eval_tf), batch_size=256, num_workers=a.workers, pin_memory=True)

    @torch.no_grad()
    def evaluate(dl):
        model.eval(); dscorr = []; top5 = []; picks = []
        for img, g in dl:
            logits = model(img.to(dev))
            sel = logits.argmax(1).cpu().numpy()
            t5 = logits.topk(5, dim=1).indices.cpu().numpy()
            for gg, s, row in zip(g.numpy(), sel, t5):
                dscorr.append(COR[gg, s]); picks.append(s)
                top5.append(bool(COR[gg, row].any()))
        picks = np.array(picks)
        # (downstream top-1, top-5 contains-correct, argmax histogram)
        return 100 * np.mean(dscorr), 100 * np.mean(top5), np.bincount(picks, minlength=27)

    best_va, best_te, best_ep = -1, -1, -1
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0
        for img, g in dl_tr:
            img = img.to(dev)
            logits = model(img)
            if a.loss == "soft_kl":
                loss = F.kl_div(F.log_softmax(logits, 1), SOFT[g].to(dev), reduction="batchmean")
            else:
                loss = F.cross_entropy(logits, HARD[g].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        va, va5, _ = evaluate(dl_va)
        te, te5, hist = evaluate(dl_te)
        if va > best_va:
            best_va, best_te, best_te5, best_ep = va, te, te5, ep
        print(f"ep{ep:02d} loss={tot/len(dl_tr):.4f} | val_DS={va:5.2f} test_DS={te:5.2f} test_top5={te5:5.2f}")
    print(f"\nBEST (by val) @ep{best_ep}: val_DS={best_va:.2f}  test_DS={best_te:.2f}  test_top5_contains_correct={best_te5:.2f}")

if __name__ == "__main__":
    main()
