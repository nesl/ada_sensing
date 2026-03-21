# eval_lens.py
from __future__ import annotations
import argparse
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import *
from lens_core import lens_select_best
from csa import csa1_random, csa2_grid_random, csa3_cost_based

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", type=str, choices=["lens", "random", "oracle_specific"], default="lens")
    p.add_argument("--csa", type=str, choices=["full", "csa1", "csa2", "csa3"], default="full")
    p.add_argument("--k", type=int, default=27)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()

def pick_candidate_ids(sample: Dict[str, Any], args, rng: random.Random) -> List[int]:
    options = sample["candidates"]
    all_ids = [o["option_id"] for o in options]

    if args.csa == "full":
        return all_ids
    if args.csa == "csa1":
        return csa1_random(all_ids, args.k, rng)
    if args.csa == "csa2":
        # expects meta iso/ss/ap to exist (or will still bucketize whatever exists)
        return csa2_grid_random(options, args.k, rng)
    if args.csa == "csa3":
        # expects meta cost to exist
        return csa3_cost_based(options, args.k, rng)
    raise ValueError(args.csa)

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    ds = ManifestLensDataset(args.manifest)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],
    )


    model = load_timm_model(args.model, device=device)
    tfm = imagenet_preprocess(args.image_size)

    correct = 0
    total = 0

    for sample in tqdm(dl, desc=f"Eval {args.mode}/{args.csa}/k={args.k}"):
        label = int(sample["label"])
        candidates = sample["candidates"]  # list[dict], len=27

        chosen_ids = pick_candidate_ids({"candidates": candidates}, args, rng)
        chosen_set = set(chosen_ids)
        chosen = [c for c in candidates if c["option_id"] in chosen_set]

        imgs = [tfm(load_image_rgb(c["path"])) for c in chosen]
        imgs = torch.stack(imgs, dim=0)  # [K,3,H,W]

        if args.mode == "random":
            pick = rng.randrange(len(chosen))
            logits = model(imgs.to(device))
            pred = int(torch.argmax(logits[pick]).item())
        elif args.mode == "lens":
            best_idx, _, best_logits = lens_select_best(model, imgs, device)
            pred = int(torch.argmax(best_logits).item())
        elif args.mode == "oracle_specific":
            # uses GT label: pick setting that yields correct prediction if any; else fallback to lens best
            logits = model(imgs.to(device))
            preds = torch.argmax(logits, dim=-1).tolist()
            if label in preds:
                pred = label
            else:
                conf = torch.softmax(logits, dim=-1).max(dim=-1).values
                best_idx = int(torch.argmax(conf).item())
                pred = preds[best_idx]
        else:
            raise ValueError(args.mode)

        correct += int(pred == label)
        total += 1

        # import pdb; pdb.set_trace()

    acc = 100.0 * correct / max(1, total)
    print(f"Top-1 Acc: {acc:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    main()
