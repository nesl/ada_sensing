"""
Experiment 1: label efficiency. JEPA-text (predictor on all unlabeled photos) vs
JEPA-text with the predictor restricted to the k labeled scenes vs direct, as a
function of the number of labeled scenes k. Fixed held-out test set.

  python label_curve.py <sweeps_dir> <feat_dir> <out.json> --unlabeled feats_pre feats_hand ... [--ks 4,8,16,32 --seeds 3 --steps 1500]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import model_text as M
import jepa_qa as J

DEV = "cuda"


def load_unlabeled(feat_dir):
    out = []
    for p in sorted(Path(feat_dir).glob("*.pt")):
        d = torch.load(p)
        out.append({"name": p.stem, "feats": F.layer_norm(d["feats"].float(), (M.D,)), "meta": d["meta"]})
    return out


def train(labeled, pred_pool, temb, steps, seed, bs=8):
    torch.manual_seed(seed); random.seed(seed)
    pred, text, direct = J.Predictor().to(DEV), M.TextHead().to(DEV), M.TextHead(extra=2).to(DEV)
    opt = torch.optim.AdamW([p for m in (pred, text, direct) for p in m.parameters()], lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    items = [(s, t) for s in labeled for t in s["targets"]]
    for step in range(steps):
        srcs, tgts, acts = [], [], []
        for _ in range(bs):
            s = random.choice(pred_pool)
            i, j = random.sample(range(len(s["meta"])), 2)
            if random.random() < 0.5 and "ae" in s:
                i = s["ae"]
            mi, mj = s["meta"][i], s["meta"][j]
            srcs.append(s["feats"][i]); tgts.append(s["feats"][j])
            acts.append(torch.cat([J.setting_vec(mi["iso"], mi["exposure_s"]), J.setting_vec(mj["iso"], mj["exposure_s"])]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = pred(torch.stack(srcs).to(DEV), torch.stack(acts).to(DEV))
        loss_p = F.smooth_l1_loss(p.float(), torch.stack(tgts).to(DEV))
        fs, qs, ys, fa, qd = [], [], [], [], []
        for _ in range(32):
            s, t = random.choice(items); c = random.randrange(len(M.CAND))
            fs.append(s["feats"][s["cand_idx"][c]]); qs.append(temb[t["q"]]); ys.append(t["cand"][c].clamp(0, 1))
            fa.append(s["feats"][s["ae"]]); qd.append(torch.cat([temb[t["q"]], J.setting_vec(*M.CAND[c])]))
        fs, qs, ys, fa, qd = (torch.stack(x).to(DEV) for x in (fs, qs, ys, fa, qd))
        loss_h = F.binary_cross_entropy_with_logits(text(fs, qs), ys) + F.binary_cross_entropy_with_logits(direct(fa, qd), ys)
        opt.zero_grad(); (loss_p + loss_h).backward(); opt.step(); sched.step()
    return pred.eval(), text.eval(), direct.eval()


@torch.no_grad()
def evaluate(test, labeled, temb, pred, text, direct):
    trg = torch.stack([t["cand"] for s in labeled for t in s["targets"] if t["recoverable"]])
    fixed = int(trg.mean(0).argmax())
    rows = []
    for s in test:
        ae_f = s["feats"][s["ae"]].to(DEV); ma = s["meta"][s["ae"]]
        acts = torch.stack([torch.cat([J.setting_vec(ma["iso"], ma["exposure_s"]), J.setting_vec(*c)]) for c in M.CAND]).to(DEV)
        P = pred(ae_f[None].expand(len(M.CAND), -1, -1, -1), acts)
        for t in s["targets"]:
            if not t["recoverable"]:
                continue
            q = temb[t["q"]][None].expand(len(M.CAND), -1).to(DEV)
            qd = torch.stack([torch.cat([temb[t["q"]], J.setting_vec(*c)]) for c in M.CAND]).to(DEV)
            cs = t["cand"]
            rows.append({"band": t["band"], "conflict": t["ae"] < 0.9 * float(cs.max()), "ae": t["ae"], "oracle": float(cs.max()),
                         "jepa": float(cs[int(text(P, q).argmax())]), "direct": float(cs[int(direct(ae_f[None].expand(len(M.CAND), -1, -1, -1), qd).argmax())]),
                         "fixed": float(cs[fixed])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps"); ap.add_argument("feat_dir"); ap.add_argument("out")
    ap.add_argument("--unlabeled", nargs="*", default=[]); ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--n_test", type=int, default=8); ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    scenes = M.load(a.sweeps, a.feat_dir, a.prefix)
    temb = M.text_embed([t["q"] for s in scenes for t in s["targets"]])
    unl = [u for d in a.unlabeled for u in load_unlabeled(d)]
    rng = random.Random(0); names = sorted(s["name"] for s in scenes); rng.shuffle(names)
    test = [s for s in scenes if s["name"] in set(names[:a.n_test])]
    pool = [s for s in scenes if s["name"] not in set(names[:a.n_test])]
    print(f"{len(scenes)} labeled scenes ({len(test)} test, {len(pool)} pool), {len(unl)} unlabeled photo sets "
          f"({sum(len(u['meta']) for u in unl)} photos)", flush=True)
    results = []
    for k in [int(x) for x in a.ks.split(",")]:
        for seed in range(a.seeds):
            r = random.Random(100 * k + seed); labeled = r.sample(pool, k)
            others = [s for s in pool if s not in labeled]
            variants = {"jepa_all": labeled + others + unl, "jepa_k": labeled}
            rec = {"k": k, "seed": seed}
            for vname, ppool in variants.items():
                pred, text, direct = train(labeled, ppool, temb, a.steps, seed)
                rows = evaluate(test, labeled, temb, pred, text, direct)
                rec[vname] = float(np.mean([x["jepa"] for x in rows]))
                rec[vname + "_conf"] = float(np.mean([x["jepa"] for x in rows if x["conflict"]]))
                if vname == "jepa_k":   # direct/fixed/ae/oracle are the same in both variants
                    for m in ("direct", "fixed", "ae", "oracle"):
                        rec[m] = float(np.mean([x[m] for x in rows])); rec[m + "_conf"] = float(np.mean([x[m] for x in rows if x["conflict"]]))
            results.append(rec)
            print(f"k={k:2d} seed={seed}: jepa_all={rec['jepa_all']:.2f} jepa_k={rec['jepa_k']:.2f} direct={rec['direct']:.2f} fixed={rec['fixed']:.2f} ae={rec['ae']:.2f}  "
                  f"| conflict: jepa_all={rec['jepa_all_conf']:.2f} jepa_k={rec['jepa_k_conf']:.2f} direct={rec['direct_conf']:.2f}", flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))
    print("\nmean over seeds:")
    for k in sorted({r["k"] for r in results}):
        rs = [r for r in results if r["k"] == k]
        print(f"  k={k:2d}: " + "  ".join(f"{m}={np.mean([r[m] for r in rs]):.2f}±{np.std([r[m] for r in rs]):.2f}" for m in ("jepa_all", "jepa_k", "direct", "fixed")))
    print("CURVE_DONE")


if __name__ == "__main__":
    main()
