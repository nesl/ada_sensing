"""
Generate manifest_all.json with the format:
{
  "id": "l1__n01443537__ILSVRC2012_val_00000994",
  "env": "l1",
  "label": 1,   # ImageNet-1K index if you pass imagenet_class_index.json
  "candidates": [
    {"option_id": 0, "path": ".../l1/param_1/n01443537/ILSVRC....JPEG", "meta": {"env":"l1","option_name":"param_1"}},
    ...
    {"option_id": 26, "path": ".../l1/param_27/n01443537/ILSVRC....JPEG", "meta": {"env":"l1","option_name":"param_27"}}
  ]
}
"""

import os, json, argparse
from pathlib import Path
from collections import defaultdict

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--param_root", type=str, required=True,
                   help=".../es-diverse-test/param_control")
    p.add_argument("--envs", type=str, default="l1,l2,l3,l4,l6,l7",
                   help="comma-separated env list")
    p.add_argument("--label_map", type=str, required=True,
                   help="Either synset->label json OR imagenet_class_index.json")
    p.add_argument("--out", type=str, default="/mnt/hdd1/yuyang/adaptive_sensing/Lenz/data/manifest_all.json")
    p.add_argument("--expect_k", type=int, default=27,
                   help="expected candidates per sample (default 27)")
    return p.parse_args()

def list_images(root: Path):
    imgs = []
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix in IMG_EXT or p.suffix.lower() in {e.lower() for e in IMG_EXT}):
            imgs.append(p)
    return imgs

def load_label_map(path: str) -> dict:
    """
    Supports two formats:

    1) synset -> int
       {"n01443537": 1, ...}

    2) ImageNet class index format (index -> [synset, name])
       {"0": ["n01440764","tench"], "1": ["n01443537","goldfish"], ...}

    Returns: synset -> int
    """
    data = json.load(open(path, "r"))

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"label_map is empty or not a dict: {path}")

    # Heuristic: if keys look like digits and values look like [synset, name], treat as ImageNet index format
    sample_key = next(iter(data.keys()))
    sample_val = data[sample_key]

    if isinstance(sample_key, str) and sample_key.isdigit() and isinstance(sample_val, (list, tuple)) and len(sample_val) >= 1:
        # Convert index -> [synset, name]  ==>  synset -> index
        synset2idx = {}
        for k, v in data.items():
            if not (isinstance(k, str) and k.isdigit()):
                continue
            if not (isinstance(v, (list, tuple)) and len(v) >= 1):
                continue
            syn = v[0]
            if isinstance(syn, str) and syn.startswith("n"):
                synset2idx[syn] = int(k)
        if len(synset2idx) < 900:
            raise ValueError(
                f"Detected ImageNet-index-like json but synset2idx too small ({len(synset2idx)}). "
                f"Check file: {path}"
            )
        print(f"[label_map] Detected ImageNet class index format. Converted to synset->index (size={len(synset2idx)}).")
        return synset2idx

    # Otherwise assume already synset -> int
    # Quick sanity: keys should look like synsets
    any_syn = any(isinstance(k, str) and k.startswith("n") for k in data.keys())
    if not any_syn:
        print("[label_map] Warning: keys don't look like synsets (n########). You may be passing an unexpected format.")
    print(f"[label_map] Using provided synset->label format (size={len(data)}).")
    return data

def main():
    args = parse_args()
    param_root = Path(args.param_root)
    class2label = load_label_map(args.label_map)

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    manifest = []
    total_bad = 0
    total_opts_seen = set()
    total_skipped_cls = 0
    total_imgs_seen = 0

    for env in envs:
        env_root = param_root / env
        if not env_root.exists():
            raise RuntimeError(f"Env not found: {env_root}")

        all_imgs = list_images(env_root)
        if len(all_imgs) == 0:
            raise RuntimeError(f"No images under {env_root}")

        option_name_to_id = {}
        next_oid = 0
        groups = defaultdict(list)  # key=(cls, stem) -> candidates

        skipped_cls = 0

        for img_path in all_imgs:
            total_imgs_seen += 1

            cls = img_path.parent.name
            opt = img_path.parent.parent.name
            stem = img_path.stem

            if cls not in class2label:
                skipped_cls += 1
                continue

            if opt not in option_name_to_id:
                option_name_to_id[opt] = next_oid
                next_oid += 1

            oid = option_name_to_id[opt]
            total_opts_seen.add(opt)

            groups[(cls, stem)].append({
                "option_id": oid,
                "path": str(img_path),
                "meta": {"env": env, "option_name": opt}
            })

        bad = 0
        for (cls, stem), cand in groups.items():
            cand_sorted = sorted(cand, key=lambda x: x["option_id"])
            if len(cand_sorted) != args.expect_k:
                bad += 1
            manifest.append({
                "id": f"{env}__{cls}__{stem}",
                "env": env,
                "label": int(class2label[cls]),
                "candidates": cand_sorted
            })

        print(f"[{env}] samples={len(groups)} options={len(option_name_to_id)} bad(K!={args.expect_k})={bad} skipped_imgs_due_to_unknown_cls={skipped_cls}")
        total_bad += bad
        total_skipped_cls += skipped_cls

    manifest.sort(key=lambda x: x["id"])
    with open(args.out, "w") as f:
        json.dump(manifest, f)

    print(f"\nWROTE: {args.out}")
    print(f"TOTAL samples={len(manifest)}  TOTAL bad(K!={args.expect_k})={total_bad}")
    print(f"TOTAL imgs seen={total_imgs_seen}  TOTAL skipped due to cls not in label_map={total_skipped_cls}")
    print(f"Option folder names seen (global): {len(total_opts_seen)} (expect 27 like param_1..param_27)")

if __name__ == "__main__":
    main()
