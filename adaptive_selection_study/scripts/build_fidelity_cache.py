"""
Label-quality test for the idea: "pick the config whose capture is closest to
the ORIGINAL clean image."

For every candidate we compute distance to its clean source in:
  - feature space  : cosine distance of resnet50 pooled features (2048-d)
  - structure space: L2 of per-image standardized 64x64 grayscale (exposure-invariant)
plus downstream conf/correct (same model). Then we ask: if we SELECT the
min-distance candidate per scene, what downstream accuracy do we get?
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms
from tqdm import tqdm

here = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def struct_vec(pil):
    g = np.asarray(pil.convert("L").resize((64, 64)), np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)   # exposure-invariant structure
    return g.reshape(-1)

class ImgDS(Dataset):
    def __init__(self, items):  # (path, key)
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, key = self.items[i]
        im = Image.open(p).convert("RGB")
        return tf(im), torch.from_numpy(struct_vec(im)), i

@torch.no_grad()
def feats_and_logits(model, x):
    fm = model.forward_features(x)
    pooled = model.forward_head(fm, pre_logits=True)   # [B,2048]
    logits = model.forward_head(fm)                    # [B,1000]
    return pooled, logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(here / "fidelity_cache_resnet50.npz"))
    ap.add_argument("--batch_size", type=int, default=384)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--envs", default="l1,l2,l3,l4,l6,l7")
    a = ap.parse_args()
    envs = a.envs.split(",")
    ci = json.load(open(here / "imagenet_class_index.json"))
    wnid2idx = {v[0]: int(k) for k, v in ci.items()}
    dev = torch.device("cuda")
    model = timm.create_model("resnet50", pretrained=True).eval().to(dev)

    # 1) clean source features + struct, keyed by wnid__stem
    clean_root = here / "es-diverse-test" / "sampled_tin_no_resize2"
    clean_items = []
    for wd in sorted(clean_root.iterdir()):
        if not wd.is_dir() or wd.name not in wnid2idx: continue
        for f in sorted(x for x in wd.iterdir() if x.suffix.lower() in IMG_EXT):
            clean_items.append((str(f), f"{wd.name}__{f.stem}"))
    ckeys = [k for _, k in clean_items]
    kidx = {k: i for i, k in enumerate(ckeys)}
    clean_feat = np.zeros((len(ckeys), 2048), np.float32)
    clean_struct = np.zeros((len(ckeys), 64 * 64), np.float32)
    dl = DataLoader(ImgDS(clean_items), batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True)
    for x, sv, idx in tqdm(dl, desc="clean sources"):
        pooled, _ = feats_and_logits(model, x.to(dev))
        pooled = F.normalize(pooled.float(), dim=-1).cpu().numpy()
        clean_feat[idx.numpy()] = pooled
        clean_struct[idx.numpy()] = sv.numpy()
    print(f"[clean] {len(ckeys)} sources")

    # 2) candidates
    root = here / "es-diverse-test" / "param_control"
    groups, items, gt_list, ck_list = {}, [], [], []
    for env in envs:
        ed = root / env
        if not ed.is_dir(): continue
        for pd in sorted(ed.iterdir()):
            if not pd.name.startswith("param_"): continue
            pidx = int(pd.name.split("_")[1]) - 1
            for wd in sorted(x for x in pd.iterdir() if x.is_dir()):
                if wd.name not in wnid2idx: continue
                gt = wnid2idx[wd.name]
                for img in sorted(x for x in wd.iterdir() if x.suffix.lower() in IMG_EXT):
                    gkey = f"{env}__{wd.name}__{img.stem}"
                    ck = f"{wd.name}__{img.stem}"
                    if gkey not in groups: groups[gkey] = len(groups)
                    items.append((str(img), gkey)); gt_list.append(gt)
                    ck_list.append(kidx[ck]);
    N = len(items)
    gidx = np.array([groups[k] for _, k in items], np.int32)
    pidx = np.array([int(Path(p).parent.parent.name.split('_')[1]) - 1 for p, _ in items], np.int32)
    ck_arr = np.array(ck_list, np.int32)
    gts = np.array(gt_list, np.int64)
    print(f"[cand] images={N} groups={len(groups)}")

    conf = np.zeros(N, np.float32); correct = np.zeros(N, bool)
    feat_cos = np.zeros(N, np.float32); struct_l2 = np.zeros(N, np.float32)
    clean_feat_t = torch.from_numpy(clean_feat).to(dev)
    dl = DataLoader(ImgDS(items), batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=True)
    ptr = 0
    for x, sv, idx in tqdm(dl, desc="candidates"):
        idx = idx.numpy()
        pooled, logits = feats_and_logits(model, x.to(dev))
        pooled = F.normalize(pooled.float(), dim=-1)
        probs = F.softmax(logits.float(), -1); c, pred = probs.max(-1)
        conf[idx] = c.cpu().numpy()
        correct[idx] = (pred.cpu().numpy() == gts[idx])
        cf = clean_feat_t[ck_arr[idx]]                       # [B,2048]
        feat_cos[idx] = (1 - (pooled * cf).sum(-1)).cpu().numpy()
        cs = clean_struct[ck_arr[idx]]                       # [B,4096]
        struct_l2[idx] = np.linalg.norm(sv.numpy() - cs, axis=1)
    np.savez_compressed(a.out, conf=conf, correct=correct, feat_cos=feat_cos,
                        struct_l2=struct_l2, gidx=gidx, pidx=pidx)
    print(f"[done] -> {a.out}")

if __name__ == "__main__":
    main()
