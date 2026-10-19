"""
Shared pieces for the explicit-adapter study on ImageNet-ES-Diverse.

Question: can the action-conditioned transition (auto-exposure shot + target camera setting ->
what the camera captures at that setting) be an EXPLICIT physical image operation whose parameter
values come from a small MLP, instead of a latent predictor over encoder tokens
(testbed/qa_scoring/jepa_qa.py Predictor)?
"""
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get(
    "ES_ROOT", Path.home() / "camera-sim-assets_ext/imagenet_es_diverse/ImageNet-ES-Diverse/es-diverse-test"))
OUT = Path(__file__).resolve().parent / "_out"
REPO = Path(__file__).resolve().parent.parent
ENVS = ["l1", "l2", "l3", "l4", "l6", "l7"]
ISO_SEQ = [250, 2000, 16000]
SS_SEQ = [1 / 4, 1 / 60, 1 / 1000]
AP_SEQ = [5.0, 9.0, 16.0]
NCFG = 27
AE = NCFG                       # slot of the auto-exposure shot (ae_1) in IMG[:, slot]
H, W = 168, 224                 # 12 x 16 DINOv2 patches (native captures are 278 x 208)
GH, GW = H // 14, W // 14
D = 384
JUDGES = ["resnet50", "openclip"]
BANDS = [(0.0, 0.08, "near-black"), (0.08, 0.3, "dark-mid"), (0.3, 0.6, "well-exposed"), (0.6, 1.01, "bright")]


def param_config(k):
    """param_k (1-indexed) -> (shutter_s, iso, f_number); ISO fastest, then shutter, then aperture.
    Same mapping as exposure_sim/run_es_diverse.py."""
    i = k - 1
    return SS_SEQ[(i // 3) % 3], ISO_SEQ[i % 3], AP_SEQ[i // 9]


CONFIGS = [param_config(k) for k in range(1, NCFG + 1)]
EV = torch.tensor([math.log2(t * iso / n ** 2) for t, iso, n in CONFIGS])      # log2 exposure product
_RAW = torch.tensor([[math.log2(t), math.log2(iso), math.log2(n), math.log2(t * iso / n ** 2)]
                     for t, iso, n in CONFIGS])
SETTING = (_RAW - _RAW.mean(0)) / _RAW.std(0)        # (27, 4) z-scored log2 (t, ISO, N, E)
ADIM = 9                                             # action = [src setting(4), src_is_ae(1), tgt setting(4)]
HOLDOUTS = {
    "iso16000": [k for k, c in enumerate(CONFIGS) if c[1] == 16000],
    "f16": [k for k, c in enumerate(CONFIGS) if c[2] == 16.0],
    "ev_extreme": [int(k) for k in torch.cat([EV.argsort()[:4], EV.argsort()[-4:]])],
}


def action(src, tgt):
    """src, tgt: LongTensor (B,) of capture slots; src == AE is the auto-exposure shot, whose
    setting is unknown (no EXIF in ES-Diverse), so only a flag is given for it."""
    is_ae = src == AE
    s = SETTING[src.clamp(max=NCFG - 1)] * (~is_ae)[:, None].float()
    return torch.cat([s, is_ae[:, None].float(), SETTING[tgt]], 1)


def ev_of(src):
    """log2 exposure of each source slot, NaN for the AE shot."""
    e = EV[src.clamp(max=NCFG - 1)].clone()
    e[src == AE] = float("nan")
    return e


class Index:
    """Scenes (lighting env x reference image), the July study's seed-0 3/1/1 split by reference
    image, and the downstream correctness of every capture (downstream_eval/image_results.jsonl)."""

    def __init__(self):
        res = {}
        for line in open(ROOT / "downstream_eval" / "image_results.jsonl"):
            r = json.loads(line)
            res[r["image_path"]] = [bool(r[f"{j}_top1_correct"]) for j in JUDGES]
        self.scenes = []                                         # (env, wnid, stem)
        for env in ENVS:
            p1 = ROOT / "param_control" / env / "param_1"
            for wd in sorted(x for x in p1.iterdir() if x.is_dir()):
                for img in sorted(wd.glob("*.JPEG")):
                    self.scenes.append((env, wd.name, img.stem))
        S = len(self.scenes)
        self.paths = []
        self.correct = torch.zeros(S, NCFG + 1, len(JUDGES), dtype=torch.bool)
        for s, (env, wn, st) in enumerate(self.scenes):
            rels = [f"param_control/{env}/param_{k + 1}/{wn}/{st}.JPEG" for k in range(NCFG)]
            rels.append(f"auto_exposure/{env}/param_1/{wn}/{st}.JPEG")
            self.paths.append([ROOT / r for r in rels])
            self.correct[s] = torch.tensor([res[r] for r in rels])
        self.env = torch.tensor([ENVS.index(e) for e, _, _ in self.scenes])
        # split: identical to adaptive_selection_study/scripts/train_policy_grid.py
        by_class = defaultdict(lambda: defaultdict(list))
        for s, (env, wn, st) in enumerate(self.scenes):
            by_class[wn][f"{wn}__{st}"].append(s)
        rng = random.Random(0)
        split = {"train": [], "val": [], "test": []}
        self.train_refs = []
        for wn, refs in sorted(by_class.items()):
            rl = sorted(refs)
            rng.shuffle(rl)
            for r in rl[:3]:
                split["train"] += refs[r]
                self.train_refs.append(refs[r])
            for r in rl[3:4]:
                split["val"] += refs[r]
            for r in rl[4:]:
                split["test"] += refs[r]
        self.split = {k: torch.tensor(sorted(v)) for k, v in split.items()}

    def train_subset(self, n_ref, seed=0):
        """Scenes of n_ref training reference images (all 6 envs each), seeded draw."""
        refs = list(self.train_refs)
        random.Random(1000 + seed).shuffle(refs)
        return torch.tensor(sorted(s for r in refs[:n_ref] for s in r))


class _Decode(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        im = Image.open(self.paths[i]).convert("RGB").resize((W, H), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)


def preload(index, workers=16):
    """All 28 captures of every scene as uint8 (S, 28, 3, H, W) in RAM (~19 GB)."""
    flat = [p for row in index.paths for p in row]
    out = torch.empty(len(flat), 3, H, W, dtype=torch.uint8)
    i = 0
    for b in DataLoader(_Decode(flat), batch_size=512, num_workers=workers):
        out[i:i + len(b)] = b
        i += len(b)
    return out.view(len(index.paths), NCFG + 1, 3, H, W)


class Encoder:
    """Frozen DINOv2-small (the encoder of testbed/qae/model_text.py), layer-normed patch tokens.
    Differentiable w.r.t. its input so explicit models can train through it."""

    def __init__(self, device):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, x):                  # x float (B,3,H,W) in [0,1] -> (B, GH*GW, D)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.m(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 1:]
        return F.layer_norm(h.float(), (D,))


def to01(u8, device):
    return u8.to(device, non_blocking=True).float() / 255


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
