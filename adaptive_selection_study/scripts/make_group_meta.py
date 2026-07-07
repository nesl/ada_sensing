"""
Reproduce the EXACT group enumeration used by build_fidelity_cache.py (pure
filesystem walk, no inference) so we can attach per-group metadata (AE input
path, reference id, class label) aligned to the cache's gidx.
"""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ENVS = ["l1", "l2", "l3", "l4", "l6", "l7"]

ci = json.load(open(here / "imagenet_class_index.json"))
wnid2idx = {v[0]: int(k) for k, v in ci.items()}

pc_root = here / "es-diverse-test" / "param_control"
ae_root = here / "es-diverse-test" / "auto_exposure"

groups = {}   # gkey -> gidx  (first-seen order == cache order)
meta = []     # indexed by gidx
for env in ENVS:
    ed = pc_root / env
    if not ed.is_dir():
        continue
    for pd in sorted(ed.iterdir()):                      # 'param_1' sorts first
        if not pd.name.startswith("param_"):
            continue
        for wd in sorted(x for x in pd.iterdir() if x.is_dir()):
            if wd.name not in wnid2idx:
                continue
            for img in sorted(x for x in wd.iterdir() if x.suffix.lower() in IMG_EXT):
                gkey = f"{env}__{wd.name}__{img.stem}"
                if gkey in groups:
                    continue
                groups[gkey] = len(groups)
                ae_path = ae_root / env / "param_1" / wd.name / img.name
                meta.append({
                    "gidx": groups[gkey],
                    "gkey": gkey,
                    "ref": f"{wd.name}__{img.stem}",       # split unit (shared across envs)
                    "env": env,
                    "wnid": wd.name,
                    "label": wnid2idx[wd.name],
                    "ae_path": str(ae_path),
                    "ae_exists": ae_path.exists(),
                })

missing = [m for m in meta if not m["ae_exists"]]
print(f"groups={len(meta)}  AE-missing={len(missing)}")
if missing[:3]:
    print("example missing:", missing[0]["ae_path"])
json.dump(meta, open(here / "group_meta.json", "w"))
print("saved group_meta.json")
