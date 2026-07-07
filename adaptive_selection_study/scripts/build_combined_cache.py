"""
One aligned pass over all 162k param_control candidates that records BOTH:
  - downstream ResNet-50 confidence + correctness
  - image-state / exposure features (to detect too-dark / blown-out captures)

Saves everything index-aligned so we can condition confidence on visibility.
Enumeration is fully sorted for determinism.
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


def model_preprocess(sz=224):
    return transforms.Compose([
        transforms.Resize(int(sz * 256 / 224)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def exposure_feats(pil_img):
    """Compute visibility features from a 128px grayscale version. Returns
    [lum_mean(0-255), frac_dark, frac_bright, contrast_std(0-255), entropy_bits]."""
    g = pil_img.convert("L").resize((128, 128))
    a = np.asarray(g, dtype=np.float32)  # 0..255
    lum_mean = a.mean()
    frac_dark = float((a < 12).mean())     # near-black pixels
    frac_bright = float((a > 243).mean())  # near-white / blown pixels
    contrast = a.std()
    hist, _ = np.histogram(a, bins=32, range=(0, 256))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())  # 0..5 bits
    return np.array([lum_mean, frac_dark, frac_bright, contrast, entropy], np.float32)


def find_param_control_root(base: Path) -> Path:
    for p in base.rglob("param_control"):
        if p.is_dir():
            return p
    raise SystemExit(f"no param_control under {base}")


class Combined(Dataset):
    def __init__(self, items, tf):
        self.items = items  # (path, gt, gidx, pidx)
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, gt, gidx, pidx = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.from_numpy(exposure_feats(img)), gt, i


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--base", default=str(here))
    ap.add_argument("--label_map", default=str(here / "imagenet_class_index.json"))
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=24)
    ap.add_argument("--envs", default="l1,l2,l3,l4,l6,l7")
    ap.add_argument("--out", default=str(here / "combined_cache_resnet50.npz"))
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    ci = json.load(open(args.label_map))
    wnid2idx = {v[0]: int(k) for k, v in ci.items()}

    root = find_param_control_root(Path(args.base))
    groups, group_env, items = {}, [], []
    for env in envs:
        ed = root / env
        if not ed.is_dir():
            continue
        for pd in sorted(ed.iterdir()):
            if not pd.is_dir() or not pd.name.startswith("param_"):
                continue
            pidx = int(pd.name.split("_")[1]) - 1
            for wd in sorted(ed_ for ed_ in pd.iterdir() if ed_.is_dir()):
                if wd.name not in wnid2idx:
                    continue
                gt = wnid2idx[wd.name]
                for img in sorted(f for f in wd.iterdir() if f.suffix.lower() in IMG_EXT):
                    gkey = f"{env}__{wd.name}__{img.stem}"
                    if gkey not in groups:
                        groups[gkey] = len(groups)
                        group_env.append(env)
                    items.append((str(img), gt, groups[gkey], pidx))

    N = len(items)
    print(f"[info] images={N} groups={len(groups)} root={root}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True).eval().to(dev)
    dl = DataLoader(Combined(items, model_preprocess()), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)

    conf = np.zeros(N, np.float32)
    correct = np.zeros(N, np.bool_)
    feats = np.zeros((N, 5), np.float32)
    gidx = np.array([it[2] for it in items], np.int32)
    pidx = np.array([it[3] for it in items], np.int32)
    img_env = np.array([group_env[it[2]] for it in items])

    with torch.no_grad():
        for imgs, fb, gts, idxs in tqdm(dl, desc="combined inference"):
            imgs = imgs.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(imgs)
            probs = F.softmax(logits.float(), -1)
            c, pred = probs.max(-1)
            idxs = idxs.numpy()
            conf[idxs] = c.cpu().numpy()
            correct[idxs] = (pred.cpu().numpy() == gts.numpy())
            feats[idxs] = fb.numpy()

    np.savez_compressed(args.out, conf=conf, correct=correct, gidx=gidx, pidx=pidx,
                        img_env=img_env, feats=feats,
                        feat_names=np.array(["lum_mean", "frac_dark", "frac_bright", "contrast", "entropy"]))
    print(f"[done] -> {args.out}  ({N} rows)")


if __name__ == "__main__":
    main()
