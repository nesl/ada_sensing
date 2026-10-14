"""
Text-conditioned query-aware exposure selection on the QAE batch, with controls.

  python model_text.py extract <sweeps_dir> <feat_dir> <calib.json>
  python model_text.py run <sweeps_dir> <feat_dir> <out.json> [--folds 5 --steps 1500]

Per held-out scene, from the auto-exposure shot only, each method picks one of the
15 captured settings per target; we look up the REAL VLM score there.
Methods: ae, fixed (best single setting on train), region (box+type head on predicted
features), text (CLIP-text query head on predicted features), direct (text head on AE
features + candidate setting, no predictor), text_real (text head on real features of
all settings: upper bound), oracle.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qa_scoring"))
import jepa_qa as J  # noqa: E402  (extract, Predictor, box_mask, setting_vec, crop_rect_from_calib)

GW, GH, D = J.GW, J.GH, 384
CAND = [(iso, sh) for iso in (100, 400, 1600) for sh in (1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4)]
DEV = "cuda"
KINDS = ["vqa_image", "text", "negative"]
HELD = set()


# ---------------------------------------------------------------- data
QDIM = 512          # query embedding size; set by text_embed (512 CLIP, 384 MiniLM)


def text_embed(questions, enc="clip"):
    """Embed every question (and, when given, every paraphrase) once. enc: clip (pooled CLIP text,
    512-d) or minilm (sentence-transformers/all-MiniLM-L6-v2 mean-pooled, 384-d)."""
    global QDIM
    out = {}
    with torch.no_grad():
        if enc == "clip":
            from transformers import CLIPModel, CLIPTokenizer
            tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
            QDIM = 512
            for q in sorted(set(questions)):
                t = tok([q], padding=True, truncation=True, return_tensors="pt").to(DEV)
                po = clip.text_model(**t).pooler_output
                out[q] = F.normalize(clip.text_projection(po), dim=-1)[0].cpu()
        else:
            from transformers import AutoModel, AutoTokenizer
            name = "sentence-transformers/all-MiniLM-L6-v2"
            tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).to(DEV).eval()
            QDIM = 384
            for q in sorted(set(questions)):
                t = tok([q], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
                h = m(**t).last_hidden_state; a = t["attention_mask"][..., None].float()
                out[q] = F.normalize((h * a).sum(1) / a.sum(1), dim=-1)[0].cpu()
    return out


def load(sweeps, feat_dir, prefix=""):
    scenes = []
    for p in sorted(Path(feat_dir).glob(prefix + "*.pt")):
        name = p.stem
        vlm = Path(sweeps) / name / "vlm.json"
        if not vlm.exists():
            continue
        d = torch.load(p)
        feats = F.layer_norm(d["feats"].float(), (D,))
        meta = d["meta"]
        v = json.loads(vlm.read_text())
        by_label = {r["label"]: r for r in v["rows"]}
        idx = []
        for iso, sh in CAND:
            best, bi = 9e9, None
            for i, s in enumerate(meta):
                if s["mode"] != "manual":
                    continue
                dd = abs(math.log2(s["iso"] / iso)) + abs(math.log2(s["exposure_s"] / sh))
                if dd < best:
                    best, bi = dd, i
            idx.append(bi)
        ae = [i for i, s in enumerate(meta) if s["label"] == "ae_start"][0]
        targets = []
        for t in v["targets"]:
            if t["kind"] == "negative":
                continue
            q = t.get("group_question", t["question"])
            scores = torch.tensor([by_label[meta[i]["label"]]["scores"][t["id"]] for i in idx])
            targets.append({"id": t["id"], "band": t["band"], "kind": t["kind"], "q": q,
                            "mask": J.box_mask(t["camera_box"], d.get("crop")), "cand": scores,
                            "ae": float(np.mean([r["scores"][t["id"]] for r in v["rows"] if r["mode"] == "ae"])),
                            "recoverable": bool(scores.max() >= 1)})
        scenes.append({"name": name, "feats": feats, "meta": meta, "cand_idx": idx, "ae": ae, "targets": targets})
    return scenes


# ---------------------------------------------------------------- models
class TextHead(nn.Module):
    """Query (CLIP text, 512-d) cross-attends over patch tokens; optional extra conditioning."""

    def __init__(self, dim=D, width=256, heads=4, layers=2, extra=0):
        super().__init__()
        self.tok = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.zeros(1, GH * GW, width))
        self.q = nn.Linear(QDIM + extra, width)
        self.attn = nn.ModuleList([nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, feats, qvec):               # feats (B,H,W,D), qvec (B,512+extra)
        B = feats.shape[0]
        kv = self.tok(feats.reshape(B, GH * GW, -1)) + self.pos
        q = self.q(qvec)[:, None]
        for a, m in zip(self.attn, self.mlp):
            q = q + a(q, kv, kv)[0]
            q = q + m(q)
        return self.out(q)[:, 0, 0]


class RegionHead(nn.Module):
    def __init__(self, dim=D):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim * 2 + len(KINDS), 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, feats, mask, kind):
        m = mask[..., None].float()
        mean = (feats * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)
        mx = feats.masked_fill(~mask[..., None], -1e4).amax((1, 2))
        return self.net(torch.cat([mean, mx, kind], -1)).squeeze(-1)


def kind_vec(k):
    v = torch.zeros(len(KINDS)); v[KINDS.index(k)] = 1; return v


# ---------------------------------------------------------------- train / eval
def train_fold(tr, temb, steps, bs=8, seed=0, para=None, cands=None):
    torch.manual_seed(seed); random.seed(seed)
    qof = (lambda q: random.choice([q] + para.get(q, []))) if para else (lambda q: q)
    cands = list(cands) if cands else list(range(len(CAND)))   # settings the heads may be trained on
    pred = J.Predictor().to(DEV)
    text, region, direct = TextHead().to(DEV), RegionHead().to(DEV), TextHead(extra=2).to(DEV)
    params = [p for m in (pred, text, region, direct) for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    items = [(s, t) for s in tr for t in s["targets"]]
    for step in range(steps):
        # predictor: pairs of shots within a scene, AE as source half the time
        srcs, tgts, acts = [], [], []
        for _ in range(bs):
            s = random.choice(tr)
            i, j = random.sample(range(len(s["meta"])), 2)
            if random.random() < 0.5:
                i = s["ae"]
            mi, mj = s["meta"][i], s["meta"][j]
            srcs.append(s["feats"][i]); tgts.append(s["feats"][j])
            acts.append(torch.cat([J.setting_vec(mi["iso"], mi["exposure_s"]), J.setting_vec(mj["iso"], mj["exposure_s"])]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = pred(torch.stack(srcs).to(DEV), torch.stack(acts).to(DEV))
        loss_p = F.smooth_l1_loss(p.float(), torch.stack(tgts).to(DEV))
        # heads: real features of a random target at a random candidate; BCE on the VLM score
        fs, qs, ms, ks, ys, fa, qd = [], [], [], [], [], [], []
        for _ in range(32):
            s, t = random.choice(items); c = random.choice(cands)
            qq = qof(t["q"])
            fs.append(s["feats"][s["cand_idx"][c]]); qs.append(temb[qq]); ms.append(t["mask"]); ks.append(kind_vec(t["kind"]))
            ys.append(t["cand"][c].clamp(0, 1)); fa.append(s["feats"][s["ae"]]); qd.append(torch.cat([temb[qq], J.setting_vec(*CAND[c])]))
        fs, qs, ms, ks, ys, fa, qd = (torch.stack(x).to(DEV) for x in (fs, qs, ms, ks, ys, fa, qd))
        loss_h = (F.binary_cross_entropy_with_logits(text(fs, qs), ys)
                  + F.binary_cross_entropy_with_logits(region(fs, ms, ks), ys)
                  + F.binary_cross_entropy_with_logits(direct(fa, qd), ys))
        loss = loss_p + loss_h
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"    step {step} pred {loss_p.item():.4f} heads {loss_h.item():.4f}", flush=True)
    return pred.eval(), text.eval(), region.eval(), direct.eval()


@torch.no_grad()
def eval_fold(te, tr, temb, pred, text, region, direct, para=None, eval_para=False, cands=None):
    # best fixed setting on train (mean score over recoverable train targets), among the trainable settings
    trg = torch.stack([t["cand"] for s in tr for t in s["targets"] if t["recoverable"]])
    allowed = list(cands) if cands else list(range(len(CAND)))
    fixed = allowed[int(trg.mean(0)[allowed].argmax())]
    rows = []
    for s in te:
        ae_f = s["feats"][s["ae"]].to(DEV)
        ma = s["meta"][s["ae"]]
        acts = torch.stack([torch.cat([J.setting_vec(ma["iso"], ma["exposure_s"]), J.setting_vec(*c)]) for c in CAND]).to(DEV)
        P = pred(ae_f[None].expand(len(CAND), -1, -1, -1), acts)
        R = torch.stack([s["feats"][i] for i in s["cand_idx"]]).to(DEV)
        for t in s["targets"]:
            if not t["recoverable"]:
                continue
            wordings = (para.get(t["q"]) or [t["q"]]) if (eval_para and para) else [t["q"]]
            M = t["mask"][None].expand(len(CAND), -1, -1).to(DEV); K = kind_vec(t["kind"])[None].expand(len(CAND), -1).to(DEV)
            cs = t["cand"]; per_w = []
            for w in wordings:   # wording holdout: the head sees a paraphrase it was never trained on
                q = temb[w][None].expand(len(CAND), -1).to(DEV)
                qd = torch.stack([torch.cat([temb[w], J.setting_vec(*c)]) for c in CAND]).to(DEV)
                per_w.append({"text": text(P, q), "region": region(P, M, K),
                              "direct": direct(ae_f[None].expand(len(CAND), -1, -1, -1), qd), "text_real": text(R, q)})
            logits = per_w[0]
            choice = {k: int(v.argmax()) for k, v in logits.items()}; choice["fixed"] = fixed
            score_of = {k: float(np.mean([float(cs[int(lw[k].argmax())]) for lw in per_w])) for k in logits}
            score_of["fixed"] = float(cs[fixed])
            rows.append({"scene": s["name"], "id": t["id"], "band": t["band"], "conflict": t["ae"] < 0.9 * float(cs.max()),
                         "ae": t["ae"], "oracle": float(cs.max()), **score_of,
                         "choices": {k: CAND[v] for k, v in choice.items() if k != "fixed"},
                         # full candidate ranking per method and the real score of every candidate, for equal-budget evaluation
                         "ranking": {k: [int(i) for i in v.argsort(descending=True)] for k, v in logits.items()},
                         "logits": {k: [float(x) for x in v] for k, v in logits.items()},
                         "cand_scores": [float(x) for x in cs]})
    return rows


def run(a):
    scenes = load(a.sweeps, a.feat_dir, a.prefix)
    para = json.loads(Path(a.paraphrases).read_text()) if a.paraphrases else None
    qs_all = [t["q"] for s in scenes for t in s["targets"]]
    if para:
        qs_all += [pp for q in set(qs_all) for pp in para.get(q, [])]
    temb = text_embed(qs_all, a.text_enc)
    print(f"{len(scenes)} scenes, {sum(len(s['targets']) for s in scenes)} targets, "
          f"{sum(t['recoverable'] for s in scenes for t in s['targets'])} recoverable", flush=True)
    global HELD
    HELD = {int(x) for x in a.holdout_settings.split(",")} if a.holdout_settings else set()
    rng = random.Random(a.seed); order = [s["name"] for s in scenes]; rng.shuffle(order)
    folds = [order[i::a.folds] for i in range(a.folds)]
    rows = []
    for k, names in enumerate(folds):
        te = [s for s in scenes if s["name"] in names]; tr = [s for s in scenes if s["name"] not in names]
        if a.holdout:   # held-out question family: train without that kind, test only on it
            tr = [{**s, "targets": [t for t in s["targets"] if t["kind"] != a.holdout]} for s in tr]
            te = [{**s, "targets": [t for t in s["targets"] if t["kind"] == a.holdout]} for s in te]
            tr = [s for s in tr if s["targets"]]; te = [s for s in te if s["targets"]]
        print(f"fold {k}: {len(tr)} train / {len(te)} test", flush=True)
        cands = [int(c) for c in range(len(CAND)) if c not in HELD] if HELD else None
        models = train_fold(tr, temb, a.steps, seed=100 * a.seed + k, para=(para if a.train_para else None), cands=cands)
        rows += eval_fold(te, tr, temb, *models, para=para, eval_para=a.eval_para, cands=cands)
    keys = ["ae", "fixed", "direct", "region", "text", "text_real", "oracle"]

    def line(name, rs):
        return f"  {name:26s} n={len(rs):3d} " + "  ".join(f"{k}={np.mean([r[k] for r in rs]):.2f}" for k in keys)
    print("\nmean REAL score at the chosen setting (held-out, recoverable targets):")
    print(line("all", rows))
    for b in ("bright", "mid", "dim", "control"):
        print(line(f"  {b} band", [r for r in rows if r["band"] == b]))
    print(line("  conflict targets", [r for r in rows if r["conflict"]]))
    print(line("  no-conflict targets", [r for r in rows if not r["conflict"]]))
    Path(a.out).write_text(json.dumps({"rows": rows, "cand": CAND, "seed": a.seed, "prefix": a.prefix, "holdout": a.holdout, "text_enc": a.text_enc, "paraphrases": bool(a.paraphrases), "train_para": a.train_para, "eval_para": a.eval_para, "holdout_settings": a.holdout_settings}, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "run"])
    ap.add_argument("sweeps"); ap.add_argument("feat_dir"); ap.add_argument("out_or_calib")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--prefix", default=""); ap.add_argument("--text_enc", choices=["clip", "minilm"], default="clip"); ap.add_argument("--paraphrases", default="", help="json {question: [paraphrases]}; a random one is used per training sample")
    ap.add_argument("--train_para", action="store_true", help="sample paraphrases during training (needs --paraphrases)")
    ap.add_argument("--eval_para", action="store_true", help="evaluate with the paraphrases of each test question (wording holdout)")
    ap.add_argument("--holdout_settings", default="", help="comma list of CAND indices the heads never see labels for, e.g. 1,5,8,12,14")
    ap.add_argument("--holdout", default="", help="target kind held out of training and used alone for testing, e.g. text or vqa_image")
    a = ap.parse_args()
    if a.stage == "extract":
        J.extract(Path(a.sweeps), Path(a.feat_dir), J.crop_rect_from_calib(a.out_or_calib))
    else:
        a.out = a.out_or_calib
        run(a)
