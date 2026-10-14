"""
Query-aware exposure selection with an action-conditioned latent predictor (JEPA-style).

Stages (run in order):
  python jepa_qa.py extract  <dataset_dir> <feat_dir>
  python jepa_qa.py train    <feat_dir> <scores.json> <run_dir>
  python jepa_qa.py eval     <feat_dir> <scores.json> <run_dir>

Encoder   : frozen DINOv2-small, patch features on a 64x36 grid (896x504 input).
Predictor : transformer over patch tokens; input = source-shot tokens + embedding of
            (source setting, target setting); output = target-shot tokens. Trained only
            in feature space on real pairs from the sweeps.
Query head: features pooled inside a target's box (+ target kind) -> query score
            (detector / OCR), trained on real features.
Selection : from the auto-exposure shot only, predict tokens at every candidate setting,
            score each target's box with the query head, pick the argmax.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GW, GH = 64, 36                 # feature grid
IMG_W, IMG_H = 4000, 2252       # camera frame
CAND = [(iso, sh) for iso in (100, 400, 1600) for sh in (1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4)]
KINDS = ["ipad_image", "ipad_text", "physical", "distractor"]
DEV = "cuda"


# ---------------------------------------------------------------- features
def crop_rect_from_calib(calib_path, pad=0.06):
    """Bounding box of the iPad in camera px, expanded to the GW:GH aspect."""
    import cv2
    c = json.loads(Path(calib_path).read_text())
    W, H = c["canvas"]
    q = cv2.perspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]])[None],
                                 np.array(c["H_canvas_to_camera"], np.float64))[0]
    x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if w / h < GW / GH:
        w = h * GW / GH
    else:
        h = w * GH / GW
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return [int(max(0, r[0])), int(max(0, r[1])), int(min(IMG_W, r[2])), int(min(IMG_H, r[3]))]


def extract(root, feat_dir, crop=None):
    from PIL import Image
    from transformers import AutoModel
    feat_dir.mkdir(parents=True, exist_ok=True)
    enc = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval().half()
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    for mpath in sorted(root.glob("*/manifest.json")):
        m = json.loads(mpath.read_text())
        out = feat_dir / f"{m['scene']}.pt"
        if out.exists():
            continue
        feats, meta = [], []
        for shot in m["shots"]:
            im = Image.open(mpath.parent / shot["file"]).convert("RGB")
            if crop is not None:
                im = im.crop(tuple(crop))
            im = im.resize((GW * 14, GH * 14))
            x = torch.from_numpy(np.asarray(im)).to(DEV).permute(2, 0, 1)[None].half() / 255
            with torch.no_grad():
                h = enc(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]   # drop CLS
            feats.append(h[0].reshape(GH, GW, -1).cpu())
            meta.append({"label": shot["label"], "mode": shot["mode"],
                         "iso": shot["exif"]["iso"], "exposure_s": shot["exif"]["exposure_s"]})
        torch.save({"feats": torch.stack(feats), "meta": meta, "crop": crop}, out)
        print("features", m["scene"], tuple(torch.stack(feats).shape), flush=True)


# ---------------------------------------------------------------- data
def setting_vec(iso, t):
    return torch.tensor([math.log2(t) / 6.0, math.log2(iso / 400) / 3.0])


def box_mask(box, crop=None):
    cx0, cy0, cx1, cy1 = crop if crop is not None else (0, 0, IMG_W, IMG_H)
    cw, ch = cx1 - cx0, cy1 - cy0
    x0, y0, x1, y1 = box
    fx0, fx1 = (x0 - cx0) / cw * GW, (x1 - cx0) / cw * GW
    fy0, fy1 = (y0 - cy0) / ch * GH, (y1 - cy0) / ch * GH
    r0, r1 = max(0, int(fy0)), min(GH, max(int(fy0) + 1, math.ceil(fy1)))
    c0, c1 = max(0, int(fx0)), min(GW, max(int(fx0) + 1, math.ceil(fx1)))
    m = torch.zeros(GH, GW, dtype=torch.bool)
    m[r0:r1, c0:c1] = True
    return m


def nearest(meta, iso, sh):
    """Index of the manual shot matching a candidate setting (None if absent)."""
    best, bi = 9e9, None
    for i, s in enumerate(meta):
        if s["mode"] != "manual":
            continue
        d = abs(math.log2(s["iso"] / iso)) + abs(math.log2(s["exposure_s"] / sh))
        if d < best:
            best, bi = d, i
    return bi if best < 0.3 else None


def load(feat_dir, scores):
    scenes = []
    for p in sorted(feat_dir.glob("*.pt")):
        name = p.stem
        if name not in scores:
            continue
        d = torch.load(p)
        f = F.layer_norm(d["feats"].float(), (d["feats"].shape[-1],))
        rows = scores[name]["rows"]
        idx = [nearest(d["meta"], iso, sh) for iso, sh in CAND]
        if any(i is None for i in idx):
            continue
        ae = [i for i, s in enumerate(d["meta"]) if s["label"] == "ae_start"][0]
        scenes.append({"name": name, "feats": f, "meta": d["meta"], "cand_idx": idx, "ae": ae,
                       "crop": d.get("crop"), "targets": scores[name]["targets"], "rows": rows})
    return scenes


# ---------------------------------------------------------------- models
class Predictor(nn.Module):
    def __init__(self, dim=384, width=384, depth=4, heads=6):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.zeros(1, GH * GW, width))
        self.act = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, depth)
        self.out = nn.Linear(width, dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, src, a):              # src (B,H,W,D), a (B,4)
        B = src.shape[0]
        x = self.inp(src.reshape(B, GH * GW, -1)) + self.pos + self.act(a)[:, None]
        y = self.out(self.tr(x))
        return (src.reshape(B, GH * GW, -1) + y).reshape(src.shape)   # residual: predict the change


class QueryHead(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2 + len(KINDS), 256), nn.GELU(),
                                 nn.Linear(256, 1))

    def forward(self, feats, mask, kind):   # feats (B,H,W,D), mask (B,H,W), kind (B,K)
        m = mask[..., None].float()
        mean = (feats * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)
        mx = feats.masked_fill(~mask[..., None], -1e4).amax((1, 2))
        return self.net(torch.cat([mean, mx, kind], -1)).squeeze(-1)


def kind_vec(k):
    v = torch.zeros(len(KINDS))
    v[KINDS.index(k)] = 1
    return v


def target_items(sc):
    """(target, mask, kind_vec, scores at the 15 candidates, score at AE)."""
    out = []
    by_label = {r["label"]: r for r in sc["rows"]}
    for t in sc["targets"]:
        tid = t["id"]
        cand_scores = [by_label[sc["meta"][i]["label"]]["scores"][tid] for i in sc["cand_idx"]]
        ae_score = by_label["ae_start"]["scores"][tid]
        out.append((t, box_mask(t["camera_box"], sc["crop"]), kind_vec(t["kind"]),
                    torch.tensor(cand_scores), ae_score))
    return out


# ---------------------------------------------------------------- train
def split(scenes, seed=0):
    rng = random.Random(seed)
    names = sorted(s["name"] for s in scenes)
    rng.shuffle(names)
    n_test = max(2, len(names) // 4)
    test = set(names[:n_test])
    return [s for s in scenes if s["name"] not in test], [s for s in scenes if s["name"] in test]


def train(feat_dir, scores, run_dir, steps=3000, bs=12):
    run_dir.mkdir(parents=True, exist_ok=True)
    scenes = load(feat_dir, scores)
    tr, te = split(scenes)
    print(f"{len(scenes)} scenes: {len(tr)} train / {len(te)} test", flush=True)
    torch.manual_seed(0)
    pred, head = Predictor().to(DEV), QueryHead().to(DEV)
    opt = torch.optim.AdamW(list(pred.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    items = [(s, it) for s in tr for it in target_items(s)]
    for step in range(steps):
        # predictor batch: random (source, target) shot pairs from a training scene
        srcs, tgts, acts = [], [], []
        for _ in range(bs):
            s = random.choice(tr)
            i, j = random.sample(range(len(s["meta"])), 2)
            if random.random() < 0.5:
                i = s["ae"]
            mi, mj = s["meta"][i], s["meta"][j]
            srcs.append(s["feats"][i]); tgts.append(s["feats"][j])
            acts.append(torch.cat([setting_vec(mi["iso"], mi["exposure_s"]),
                                   setting_vec(mj["iso"], mj["exposure_s"])]))
        src, tgt, act = torch.stack(srcs).to(DEV), torch.stack(tgts).to(DEV), torch.stack(acts).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = pred(src, act)
        loss_p = F.smooth_l1_loss(p.float(), tgt)

        # head batch: real features of a random target at a random candidate setting
        fs, ms, ks, ys = [], [], [], []
        for _ in range(64):
            s, (t, mask, kv, cs, _) = random.choice(items)
            c = random.randrange(len(CAND))
            fs.append(s["feats"][s["cand_idx"][c]]); ms.append(mask); ks.append(kv); ys.append(cs[c])
        hp = head(torch.stack(fs).to(DEV), torch.stack(ms).to(DEV), torch.stack(ks).to(DEV))
        loss_h = F.mse_loss(hp, torch.stack(ys).to(DEV))

        loss = loss_p + loss_h
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps - 1:
            print(f"step {step} pred {loss_p.item():.4f} head {loss_h.item():.4f}", flush=True)
    torch.save({"pred": pred.state_dict(), "head": head.state_dict(),
                "train": [s["name"] for s in tr], "test": [s["name"] for s in te]}, run_dir / "model.pt")
    print("TRAIN_DONE", flush=True)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(feat_dir, scores, run_dir):
    scenes = load(feat_dir, scores)
    ck = torch.load(run_dir / "model.pt")
    pred, head = Predictor().to(DEV).eval(), QueryHead().to(DEV).eval()
    pred.load_state_dict(ck["pred"]); head.load_state_dict(ck["head"])
    tr = [s for s in scenes if s["name"] in ck["train"]]
    te = [s for s in scenes if s["name"] in ck["test"]]

    # baseline: best single fixed setting on train targets
    tr_scores = torch.stack([it[3] for s in tr for it in target_items(s)])
    fixed = int(tr_scores.mean(0).argmax())

    # baseline (non-JEPA): direct regressor from AE box features + target setting -> score
    direct = nn.Sequential(nn.Linear(384 * 2 + len(KINDS) + 2, 256), nn.GELU(), nn.Linear(256, 1)).to(DEV)
    with torch.enable_grad():
        opt = torch.optim.AdamW(direct.parameters(), lr=1e-3, weight_decay=0.05)
        items = [(s, it) for s in tr for it in target_items(s)]
        for _ in range(3000):
            X, Y = [], []
            for _ in range(64):
                s, (t, mask, kv, cs, _) = random.choice(items)
                c = random.randrange(len(CAND))
                f = s["feats"][s["ae"]]
                m = mask[..., None].float()
                mean = (f * m).sum((0, 1)) / m.sum().clamp(min=1)
                mx = f.masked_fill(~mask[..., None], -1e4).amax((0, 1))
                X.append(torch.cat([mean, mx, kv, setting_vec(*CAND[c])])); Y.append(cs[c])
            l = F.mse_loss(direct(torch.stack(X).to(DEV)).squeeze(-1), torch.stack(Y).to(DEV))
            opt.zero_grad(); l.backward(); opt.step()

    rows = []
    for s in te:
        ae_f = s["feats"][s["ae"]].to(DEV)
        ma = s["meta"][s["ae"]]
        acts = torch.stack([torch.cat([setting_vec(ma["iso"], ma["exposure_s"]), setting_vec(iso, sh)])
                            for iso, sh in CAND]).to(DEV)
        P = pred(ae_f[None].expand(len(CAND), -1, -1, -1), acts)        # (15,H,W,D)
        real = torch.stack([s["feats"][i] for i in s["cand_idx"]]).to(DEV)
        for t, mask, kv, cs, ae_score in target_items(s):
            M = mask[None].expand(len(CAND), -1, -1).to(DEV)
            K = kv[None].expand(len(CAND), -1).to(DEV)
            jepa = int(head(P, M, K).argmax())
            probe_real = int(head(real, M, K).argmax())
            f = s["feats"][s["ae"]]
            m = mask[..., None].float()
            mean = (f * m).sum((0, 1)) / m.sum().clamp(min=1)
            mx = f.masked_fill(~mask[..., None], -1e4).amax((0, 1))
            X = torch.stack([torch.cat([mean, mx, kv, setting_vec(*c)]) for c in CAND]).to(DEV)
            dsel = int(direct(X).squeeze(-1).argmax())
            rows.append({"scene": s["name"], "target": t["id"], "kind": t["kind"],
                         "label": t.get("query") or t.get("text"),
                         "ae": ae_score, "fixed": float(cs[fixed]), "oracle": float(cs.max()),
                         "jepa": float(cs[jepa]), "direct": float(cs[dsel]),
                         "probe_on_real": float(cs[probe_real]),
                         "jepa_choice": CAND[jepa], "oracle_choice": CAND[int(cs.argmax())]})

    (run_dir / "eval_rows.json").write_text(json.dumps(rows, indent=1))
    keys = ["ae", "fixed", "direct", "jepa", "probe_on_real", "oracle"]
    print(f"\nheld-out: {len(te)} scenes, {len(rows)} targets; fixed setting = ISO {CAND[fixed][0]}, "
          f"{CAND[fixed][1]:.4g}s")
    print("mean real score at the chosen setting:")
    for k in keys:
        print(f"  {k:14s} {np.mean([r[k] for r in rows]):.3f}")
    for kind in KINDS:
        rk = [r for r in rows if r["kind"] == kind]
        if rk:
            print(f"  [{kind}] n={len(rk)} " + "  ".join(f"{k}={np.mean([r[k] for r in rk]):.3f}" for k in keys))
    hard = [r for r in rows if r["oracle"] - r["ae"] > 0.2]
    if hard:
        print(f"  [targets where AE loses >0.2 to oracle] n={len(hard)} "
              + "  ".join(f"{k}={np.mean([r[k] for r in hard]):.3f}" for k in keys))
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "extract":
        crop = crop_rect_from_calib(sys.argv[4]) if len(sys.argv) > 4 else None
        print("crop rect:", crop)
        extract(Path(sys.argv[2]), Path(sys.argv[3]), crop)
    elif stage == "train":
        train(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]))
    elif stage == "eval":
        evaluate(Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]))
