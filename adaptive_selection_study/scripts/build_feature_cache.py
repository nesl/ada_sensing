"""
Extract resnet50 pooled features (2048-d) for all 162k candidates, in the SAME
deterministic enumeration as build_fidelity_cache.py, so features[i] aligns with
fidelity_cache's conf/correct/feat_cos/gidx/pidx[i]. Saves float16 to keep size
manageable (~660MB). Used to train a learned all-27 selector.
"""
from __future__ import annotations
import json
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
ENVS = ["l1", "l2", "l3", "l4", "l6", "l7"]
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

ci = json.load(open(here / "imagenet_class_index.json"))
wnid2idx = {v[0]: int(k) for k, v in ci.items()}
root = here / "es-diverse-test" / "param_control"
groups, items = {}, []
for env in ENVS:
    ed = root / env
    if not ed.is_dir(): continue
    for pd in sorted(ed.iterdir()):
        if not pd.name.startswith("param_"): continue
        pi = int(pd.name.split("_")[1]) - 1
        for wd in sorted(x for x in pd.iterdir() if x.is_dir()):
            if wd.name not in wnid2idx: continue
            for img in sorted(x for x in wd.iterdir() if x.suffix.lower() in IMG_EXT):
                gk = f"{env}__{wd.name}__{img.stem}"
                if gk not in groups: groups[gk] = len(groups)
                items.append((str(img), groups[gk], pi))
N = len(items)
gidx = np.array([g for _, g, _ in items], np.int32)
pidx = np.array([p for _, _, p in items], np.int32)

class DS(Dataset):
    def __init__(s): pass
    def __len__(s): return N
    def __getitem__(s, i): return tf(Image.open(items[i][0]).convert("RGB")), i

dev = torch.device("cuda")
model = timm.create_model("resnet50", pretrained=True).eval().to(dev)
feats = np.zeros((N, 2048), np.float16)
dl = DataLoader(DS(), batch_size=384, num_workers=24, pin_memory=True)
with torch.no_grad():
    for x, idx in tqdm(dl, desc="features"):
        fm = model.forward_features(x.to(dev))
        pooled = model.forward_head(fm, pre_logits=True)  # [B,2048]
        feats[idx.numpy()] = pooled.half().cpu().numpy()
np.savez(here / "feature_cache_resnet50.npz", feats=feats, gidx=gidx, pidx=pidx)
print(f"[done] features {feats.shape} -> feature_cache_resnet50.npz")
