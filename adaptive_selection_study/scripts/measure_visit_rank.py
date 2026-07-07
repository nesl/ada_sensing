"""
Measure the VisiT confidence-rank distribution on ImageNet-ES-Diverse.

For each sample (env, wnid, image-stem) there are 27 param-controlled candidate
images. We run the downstream classifier on every candidate, then order the 27
candidates by VisiT confidence (max softmax) descending. The sample's "rank" is
the 1-based position of the FIRST candidate the classifier gets correct:
  rank 1  -> VisiT's most-confident pick is already correct
  rank 2  -> most-confident pick wrong, 2nd-most-confident correct
  ...
  all-wrong -> no candidate is classified correctly (Oracle-S can't recover it)

Outputs a histogram over ranks (+ all-wrong), plus cumulative coverage
(= accuracy if you could afford the top-k most-confident captures).
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imagenet_preprocess(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def find_param_control_root(base: Path) -> Path:
    # Look for a directory named param_control that contains env subdirs with param_* folders
    for p in base.rglob("param_control"):
        if p.is_dir():
            return p
    raise SystemExit(f"Could not find a param_control/ directory under {base}")


def collect_groups(param_root: Path, envs, wnid2idx):
    """Return list of (image_path, gt_idx, group_key) and #skipped."""
    flat = []
    skipped_cls = 0
    for env in envs:
        env_dir = param_root / env
        if not env_dir.is_dir():
            print(f"[warn] env dir missing: {env_dir}")
            continue
        for param_dir in sorted(env_dir.iterdir()):
            if not param_dir.is_dir():
                continue
            pname = param_dir.name  # param_1 .. param_27
            for wnid_dir in param_dir.iterdir():
                if not wnid_dir.is_dir():
                    continue
                wnid = wnid_dir.name
                if wnid not in wnid2idx:
                    skipped_cls += 1
                    continue
                gt = wnid2idx[wnid]
                for img in wnid_dir.iterdir():
                    if img.suffix.lower() not in IMG_EXT:
                        continue
                    group_key = f"{env}__{wnid}__{img.stem}"
                    flat.append((str(img), gt, group_key, pname))
    return flat, skipped_cls


class FlatImages(Dataset):
    def __init__(self, flat, transform):
        self.flat = flat
        self.transform = transform

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, i):
        path, gt, gkey, pname = self.flat[i]
        img = self.transform(Image.open(path).convert("RGB"))
        return img, gt, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--label_map", default=str(Path(__file__).resolve().parent / "imagenet_class_index.json"))
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "visit_rank_report.json"))
    ap.add_argument("--envs", default="l1,l2,l3,l4,l6,l7")
    args = ap.parse_args()

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]

    # wnid -> imagenet-1k idx
    ci = json.load(open(args.label_map))
    wnid2idx = {v[0]: int(k) for k, v in ci.items()}

    param_root = find_param_control_root(Path(args.base))
    print(f"[info] param_control root: {param_root}")
    flat, skipped = collect_groups(param_root, envs, wnid2idx)
    print(f"[info] images={len(flat)} skipped_unknown_cls={skipped}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True).eval().to(device)

    ds = FlatImages(flat, imagenet_preprocess(args.image_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # per-image results aligned to flat index
    conf = [0.0] * len(flat)
    correct = [False] * len(flat)

    amp = device.type == "cuda"
    with torch.no_grad():
        for imgs, gts, idxs in tqdm(dl, desc="downstream inference"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(imgs)
            probs = F.softmax(logits.float(), dim=-1)
            c, pred = probs.max(dim=-1)
            for j in range(len(idxs)):
                fi = int(idxs[j])
                conf[fi] = float(c[j])
                correct[fi] = bool(int(pred[j]) == int(gts[j]))

    # group candidates
    groups = defaultdict(list)  # gkey -> list of (conf, correct, pname)
    for i, (path, gt, gkey, pname) in enumerate(flat):
        groups[gkey].append((conf[i], correct[i], pname))

    max_rank = 27
    rank_hist = defaultdict(int)   # rank -> count (1..27)
    all_wrong = 0
    ncand_bad = 0
    per_env_rank = defaultdict(lambda: defaultdict(int))

    for gkey, cands in groups.items():
        env = gkey.split("__", 1)[0]
        if len(cands) != max_rank:
            ncand_bad += 1
        # sort by confidence desc; find first correct
        order = sorted(cands, key=lambda t: -t[0])
        rank = None
        for r, (cf, ok, pn) in enumerate(order, start=1):
            if ok:
                rank = r
                break
        if rank is None:
            all_wrong += 1
            per_env_rank[env]["all_wrong"] += 1
        else:
            rank_hist[rank] += 1
            per_env_rank[env][rank] += 1

    total = len(groups)
    recoverable = total - all_wrong

    def pct(x):
        return round(100.0 * x / max(1, total), 3)

    # cumulative coverage over confidence-ranked captures
    cum = 0
    cumulative = {}
    for r in range(1, max_rank + 1):
        cum += rank_hist.get(r, 0)
        cumulative[r] = pct(cum)

    report = {
        "model": args.model,
        "envs": envs,
        "total_samples": total,
        "samples_with_wrong_candidate_count": ncand_bad,
        "visit_top1_acc_pct": pct(rank_hist.get(1, 0)),          # rank==1 fraction
        "oracle_s_recoverable_pct": pct(recoverable),            # any candidate correct
        "all_wrong_pct": pct(all_wrong),
        "rank_hist_count": {str(r): rank_hist.get(r, 0) for r in range(1, max_rank + 1)},
        "rank_hist_pct": {str(r): pct(rank_hist.get(r, 0)) for r in range(1, max_rank + 1)},
        "rank_hist_pct_of_recoverable": {
            str(r): round(100.0 * rank_hist.get(r, 0) / max(1, recoverable), 3)
            for r in range(1, max_rank + 1)
        },
        "cumulative_topk_coverage_pct": cumulative,              # acc if you kept top-k confident captures
        "all_wrong_count": all_wrong,
    }
    json.dump(report, open(args.out, "w"), indent=2)

    # pretty print
    print("\n================ VisiT confidence-rank distribution ================")
    print(f"downstream model      : {args.model}")
    print(f"total samples         : {total}   (each = 27 candidate captures)")
    print(f"VisiT top-1 accuracy  : {report['visit_top1_acc_pct']:.2f}%  (rank==1)")
    print(f"Oracle-S recoverable  : {report['oracle_s_recoverable_pct']:.2f}%  (>=1 candidate correct)")
    print(f"all-wrong (no cand)   : {report['all_wrong_pct']:.2f}%")
    print("\n rank | % of all |  % of recoverable | cumulative top-k acc")
    for r in range(1, max_rank + 1):
        print(f" {r:>4d} | {report['rank_hist_pct'][str(r)]:>7.2f}% | "
              f"{report['rank_hist_pct_of_recoverable'][str(r)]:>15.2f}% | {cumulative[r]:>7.2f}%")
    print(f"\nsaved report -> {args.out}")


if __name__ == "__main__":
    main()
