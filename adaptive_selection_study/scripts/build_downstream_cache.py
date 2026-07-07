"""
Run the downstream classifier ONCE over all 162k param_control candidates and
cache per-image (confidence, correct, group, param) to an .npz so every later
analysis (rank, confidence histogram, Oracle-F, per-lighting, ...) is instant.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(sz=224):
    return transforms.Compose([
        transforms.Resize(int(sz * 256 / 224)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def find_param_control_root(base: Path) -> Path:
    for p in base.rglob("param_control"):
        if p.is_dir():
            return p
    raise SystemExit(f"no param_control under {base}")


class Flat(Dataset):
    def __init__(self, items, tf):
        self.items = items  # (path, gt, gidx, pidx)
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, gt, gidx, pidx = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), gt, i


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--base", default=str(here))
    ap.add_argument("--label_map", default=str(here / "imagenet_class_index.json"))
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--envs", default="l1,l2,l3,l4,l6,l7")
    ap.add_argument("--out", default=str(here / "downstream_cache_resnet50.npz"))
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    ci = json.load(open(args.label_map))
    wnid2idx = {v[0]: int(k) for k, v in ci.items()}

    param_root = find_param_control_root(Path(args.base))
    # discover group + param index spaces
    groups = {}   # group_key -> gidx
    group_env = []  # per gidx: env
    items = []
    for env in envs:
        ed = param_root / env
        if not ed.is_dir():
            print(f"[warn] missing {ed}"); continue
        for pd in sorted(ed.iterdir()):
            if not pd.is_dir() or not pd.name.startswith("param_"):
                continue
            pidx = int(pd.name.split("_")[1]) - 1   # param_1 -> 0
            for wd in pd.iterdir():
                if not wd.is_dir() or wd.name not in wnid2idx:
                    continue
                gt = wnid2idx[wd.name]
                for img in wd.iterdir():
                    if img.suffix.lower() not in IMG_EXT:
                        continue
                    gkey = f"{env}__{wd.name}__{img.stem}"
                    if gkey not in groups:
                        groups[gkey] = len(groups)
                        group_env.append(env)
                    items.append((str(img), gt, groups[gkey], pidx))

    N = len(items)
    print(f"[info] images={N} groups={len(groups)} param_root={param_root}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True).eval().to(dev)
    dl = DataLoader(Flat(items, preprocess(args.image_size)), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    conf = np.zeros(N, np.float32)
    correct = np.zeros(N, np.bool_)
    gidx = np.array([it[2] for it in items], np.int32)
    pidx = np.array([it[3] for it in items], np.int32)

    with torch.no_grad():
        for imgs, gts, idxs in tqdm(dl, desc="cache inference"):
            imgs = imgs.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(imgs)
            probs = F.softmax(logits.float(), -1)
            c, pred = probs.max(-1)
            idxs = idxs.numpy()
            conf[idxs] = c.cpu().numpy()
            correct[idxs] = (pred.cpu().numpy() == gts.numpy())

    np.savez_compressed(args.out, conf=conf, correct=correct, gidx=gidx, pidx=pidx,
                        group_env=np.array(group_env))
    json.dump(list(groups.keys()), open(args.out + ".groups.json", "w"))
    print(f"[done] saved cache -> {args.out}  ({N} rows, {len(groups)} groups)")


if __name__ == "__main__":
    main()
