"""
Driver for the explicit-adapter study. One process per GPU runs a list of specs in order, sharing the
preloaded captures (~19 GB RAM) and the frozen encoder; specs with a result.json are skipped.

  python run.py specs/round1.json --gpu 1 [--only name1,name2]

Spec kinds:
  heads       selection heads for one seed: quality head on REAL tokens of every setting, and the
              direct head on AE tokens + target setting (no transition model)
  transition  train a transition model (model: latent | explicit), then evaluate fidelity + selection
  static      evaluate a training-free transition (model: copy | physics)
Evaluation always starts from the AE shot, as at deployment.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (AE, BANDS, D, EV, HOLDOUTS, JUDGES, NCFG, OUT, REPO, SETTING, Encoder, Index, action, ev_of,
                    preload, seed_all, to01)
from models import AttnHead, ExplicitAdapter, LatentPredictor, PhysicsOracle


class Ctx:
    def __init__(self, gpu):
        self.dev = torch.device(f"cuda:{gpu}")
        t = time.time()
        self.index = Index()
        self.IMG = preload(self.index)
        self.enc = Encoder(self.dev)
        tr = self.index.split["train"]
        self.fixed = self.index.correct[tr, :NCFG].float().mean(0).argmax(0)    # best single setting on train, per judge
        print(f"[ctx] {len(self.index.scenes)} scenes; " + ", ".join(f"{k}={len(v)}" for k, v in self.index.split.items())
              + f"; fixed setting per judge = param_{[int(k) + 1 for k in self.fixed]}; {time.time() - t:.0f}s", flush=True)

    def img(self, s, slot):
        return to01(self.IMG[s, slot], self.dev)

    @torch.no_grad()
    def prior(self):
        """Scene-agnostic baseline: mean real token map of each setting over the training scenes
        (knows the setting, ignores the AE shot)."""
        if not hasattr(self, "_prior"):
            p = OUT / "prior_tokens.pt"
            if p.exists():
                self._prior = torch.load(p, map_location=self.dev)
            else:
                tr, acc = self.index.split["train"], 0
                for i in range(0, len(tr), 8):
                    sc = tr[i:i + 8]
                    t = self.enc(self.img(sc.repeat_interleave(NCFG), torch.arange(NCFG).repeat(len(sc))))
                    acc = acc + t.view(len(sc), NCFG, -1, D).sum(0)
                self._prior = acc / len(tr)                      # (27, N, D)
                OUT.mkdir(exist_ok=True)
                torch.save(self._prior, p)
        return self._prior


# ---------------------------------------------------------------- heads
def train_heads(spec, ctx):
    seed_all(spec["seed"])
    q, d = AttnHead().to(ctx.dev), AttnHead(adim=4).to(ctx.dev)
    steps, bs = spec.get("steps", 6000), spec.get("bs", 128)
    opt = torch.optim.AdamW(list(q.parameters()) + list(d.parameters()), lr=spec.get("lr", 3e-4), weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    tr = ctx.index.split["train"]
    for step in range(steps):
        s = tr[torch.randint(len(tr), (bs,))]
        k = torch.randint(NCFG, (bs,))
        y = ctx.index.correct[s, k].float().to(ctx.dev)
        with torch.no_grad():
            tk, ta = ctx.enc(ctx.img(s, k)), ctx.enc(ctx.img(s, torch.full_like(k, AE)))
        loss = (F.binary_cross_entropy_with_logits(q(tk), y)
                + F.binary_cross_entropy_with_logits(d(ta, SETTING[k].to(ctx.dev)), y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [heads] step {step} bce {loss.item() / 2:.4f}", flush=True)
    return {"q": q.state_dict(), "d": d.state_dict()}


def load_heads(name, ctx, wait_s=4 * 3600):
    p = OUT / name / "heads.pt"
    t = time.time()
    while not p.exists():                       # the heads spec may still be running on the other GPU
        if time.time() - t > wait_s:
            raise FileNotFoundError(p)
        time.sleep(60)
    ck = torch.load(p, map_location=ctx.dev)
    q, d = AttnHead().to(ctx.dev), AttnHead(adim=4).to(ctx.dev)
    q.load_state_dict(ck["q"])
    d.load_state_dict(ck["d"])
    return q.eval(), d.eval()


# ---------------------------------------------------------------- transitions
def make_transition(spec, ctx):
    m = spec["model"]
    if m == "latent":
        return LatentPredictor(width=spec.get("width", 384), depth=spec.get("depth", 4)).to(ctx.dev)
    if m == "explicit":
        return ExplicitAdapter(gain=spec.get("gain", "phys"), use_stats=spec.get("use_stats", True),
                               grid=tuple(spec.get("grid", [0, 0])), learn_tone=spec.get("learn_tone", True),
                               noise=spec.get("noise", False), hidden=spec.get("hidden", 128)).to(ctx.dev)
    if m == "physics":
        return PhysicsOracle(REPO, ctx.dev)
    if m in ("copy", "prior"):
        return None
    raise ValueError(m)


def predict(spec, model, ctx, s, src, tgt, x_src, t_src, stochastic=None):
    """Predicted target tokens (B, N, D) and, for image-space models, the simulated image."""
    m = spec["model"]
    if m == "copy":
        return t_src, x_src
    if m == "prior":
        return ctx.prior()[tgt.to(ctx.dev)], None
    if m == "latent":
        return model(t_src, action(src, tgt).to(ctx.dev)), None
    if m == "explicit":
        xh = model(x_src, action(src, tgt).to(ctx.dev), ev_of(src).to(ctx.dev), EV[tgt].to(ctx.dev),
                   stochastic=stochastic)
        return ctx.enc(xh), xh
    if m == "physics":
        xh = model(x_src, ctx.index.env[s].to(ctx.dev), tgt.to(ctx.dev))
        return ctx.enc(xh), xh
    raise ValueError(m)


def train_transition(spec, ctx, model):
    seed_all(spec["seed"])
    hold = HOLDOUTS.get(spec.get("holdout"), [])
    allowed = torch.tensor([k for k in range(NCFG) if k not in hold])
    scenes = ctx.index.split["train"] if spec.get("n_ref") is None else ctx.index.train_subset(spec["n_ref"], spec["seed"])
    steps, bs, p_ae = spec["steps"], spec.get("bs", 64), spec.get("p_ae", 0.5)
    lam_pix, lam_feat = spec.get("lam_pix", 1.0), spec.get("lam_feat", 1.0)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"  [{spec['name']}] {sum(p.numel() for p in params)} trainable params, {len(scenes)} train scenes, "
          f"{len(allowed)} target settings", flush=True)
    opt = torch.optim.AdamW(params, lr=spec["lr"], weight_decay=spec.get("wd", 0.05))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    curve, t0 = [], time.time()
    for step in range(steps):
        model.train()
        s = scenes[torch.randint(len(scenes), (bs,))]
        tgt = allowed[torch.randint(len(allowed), (bs,))]
        src = torch.where(torch.rand(bs) < p_ae, torch.full_like(tgt, AE), allowed[torch.randint(len(allowed), (bs,))])
        xs, xt = ctx.img(s, src), ctx.img(s, tgt)
        with torch.no_grad():
            tt = ctx.enc(xt)
            ts = ctx.enc(xs) if spec["model"] == "latent" else None
        if spec["model"] == "latent":
            loss = F.smooth_l1_loss(model(ts, action(src, tgt).to(ctx.dev)), tt)
        else:
            xh = model(xs, action(src, tgt).to(ctx.dev), ev_of(src).to(ctx.dev), EV[tgt].to(ctx.dev))
            loss = lam_pix * F.l1_loss(xh, xt)
            if lam_feat:
                loss = loss + lam_feat * F.smooth_l1_loss(ctx.enc(xh), tt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [{spec['name']}] step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if spec.get("eval_every") and (step + 1) % spec["eval_every"] == 0 and step + 1 < steps:
            val = ctx.index.split["val"]
            sub = val[::max(1, len(val) // spec.get("curve_n", 240))]          # spread over all envs
            f, _ = evaluate(spec, model, ctx, sub)
            b = summarize(f, None, sub, hold, ctx, spec["model"])["fidelity"]["balanced"]
            curve.append({"step": step + 1, "tok_cos": b["tok_cos"], "skill_ln": b["skill_ln"]})
            print(f"  [{spec['name']}] val@{step + 1}: balanced tok_cos {b['tok_cos']:.4f} skill_ln {b['skill_ln']:.4f}",
                  flush=True)
    return curve


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(spec, model, ctx, scenes, heads=None, chunk=8):
    """Fidelity and selection, always from the AE shot. Image-space predictions are layer-normed by the
    encoder; latent regression outputs are not and can shrink toward the conditional mean, which lowers
    MSE without improving direction, so every prediction is also scored after layer norm (mse_ln,
    pred_ln). Noise models are scored over `noise_draws` stochastic draws (metrics and logits averaged)."""
    if hasattr(model, "eval"):
        model.eval()
    draws = spec.get("noise_draws", 4) if spec.get("noise") else 1
    rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state(ctx.dev)   # don't perturb training
    torch.manual_seed(1234)
    try:
        return _evaluate(spec, model, ctx, scenes, heads, chunk, draws)
    finally:
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_gpu, ctx.dev)


def _evaluate(spec, model, ctx, scenes, heads, chunk, draws):
    S = len(scenes)
    f = {k: torch.zeros(S, NCFG) for k in ("tok_cos", "pool_cos", "mse", "mse_ln", "mse_copy", "norm", "psnr", "bright")}
    lg = {k: torch.zeros(S, NCFG, len(JUDGES)) for k in ("pred", "pred_ln", "real", "direct")} if heads else None
    for i in range(0, S, chunk):
        sc = scenes[i:i + chunk]
        n = len(sc)
        s = sc.repeat_interleave(NCFG)
        tgt = torch.arange(NCFG).repeat(n)
        src = torch.full_like(tgt, AE)
        x_real = ctx.img(s, tgt)
        t_real = ctx.enc(x_real)
        x_ae1 = ctx.img(sc, torch.full((n,), AE))
        t_ae1 = ctx.enc(x_ae1)
        x_ae, t_ae = x_ae1.repeat_interleave(NCFG, 0), t_ae1.repeat_interleave(NCFG, 0)
        sl = slice(i, i + n)

        def put(k, v, w=1.0):
            f[k][sl] += w * v.view(n, NCFG).cpu()

        put("mse_copy", ((t_ae - t_real) ** 2).mean((1, 2)))
        put("bright", x_real.mean((1, 2, 3)))
        if heads:
            q, d = heads
            lg["real"][sl] = q(t_real).view(n, NCFG, -1).cpu()
            lg["direct"][sl] = d(t_ae, SETTING[tgt].to(ctx.dev)).view(n, NCFG, -1).cpu()
        for _ in range(draws):
            pred, xh = predict(spec, model, ctx, s, src, tgt, x_ae, t_ae, stochastic=draws > 1)
            pln = F.layer_norm(pred, (D,))
            w = 1.0 / draws
            put("tok_cos", F.cosine_similarity(pred, t_real, dim=-1).mean(1), w)
            put("pool_cos", F.cosine_similarity(pred.mean(1), t_real.mean(1), dim=-1), w)
            put("mse", ((pred - t_real) ** 2).mean((1, 2)), w)
            put("mse_ln", ((pln - t_real) ** 2).mean((1, 2)), w)
            put("norm", pred.norm(dim=-1).mean(1) / D ** 0.5, w)
            if xh is not None:
                put("psnr", 10 * torch.log10(1 / ((xh - x_real) ** 2).mean((1, 2, 3)).clamp(min=1e-10)), w)
            if heads:
                lg["pred"][sl] += w * q(pred).view(n, NCFG, -1).cpu()
                lg["pred_ln"][sl] += w * q(pln).view(n, NCFG, -1).cpu()
        if xh is None:
            f["psnr"][sl] = float("nan")
    return f, lg


def summarize(f, lg, scenes, hold, ctx, model_name):
    held = torch.zeros(NCFG, dtype=torch.bool)
    held[hold] = True

    def agg(mask):
        p, mc = f["psnr"][mask], f["mse_copy"][mask].sum()
        return {"n": int(mask.sum()), "tok_cos": float(f["tok_cos"][mask].mean()),
                "pool_cos": float(f["pool_cos"][mask].mean()),
                "skill": float(1 - f["mse"][mask].sum() / mc), "skill_ln": float(1 - f["mse_ln"][mask].sum() / mc),
                "norm": float(f["norm"][mask].mean()),
                "psnr": None if torch.isnan(p).all() else float(np.nanmean(p.numpy()))}

    every = torch.ones_like(f["tok_cos"], dtype=torch.bool)
    fid = {"all": agg(every)}
    if hold:
        fid["seen"], fid["held"] = agg(every & ~held), agg(every & held)
    for lo, hi, name in BANDS:
        fid[name] = agg((f["bright"] >= lo) & (f["bright"] < hi))
    # headline: mean over brightness bands (near-black targets are ~60% of all and carry ~74% of the
    # copy error, so pooled numbers mostly measure "dark setting -> black tokens")
    bands = [fid[name] for _, _, name in BANDS if fid[name]["n"]]
    fid["balanced"] = {k: float(np.mean([b[k] for b in bands])) for k in ("tok_cos", "pool_cos", "skill", "skill_ln")}
    out = {"fidelity": fid}
    if lg is not None:
        corr = ctx.index.correct[scenes].float()
        sel = {}
        for j, judge in enumerate(JUDGES):
            c = corr[:, :NCFG, j]

            def pick(L):
                return float(c.gather(1, L[..., j].argmax(1, keepdim=True)).mean())

            sel[judge] = {"ae": float(corr[:, AE, j].mean()), "fixed": float(c[:, ctx.fixed[j]].mean()),
                          "direct": pick(lg["direct"]), "q_real": pick(lg["real"]),
                          "q_pred": None if model_name == "copy" else pick(lg["pred"]),
                          "q_pred_ln": None if model_name == "copy" else pick(lg["pred_ln"]),
                          "oracle": float(c.max(1).values.mean())}
        out["selection"] = sel
    return out


# ---------------------------------------------------------------- driver
def run_spec(spec, ctx, reeval=False):
    """Train (if needed) and evaluate one spec. reeval=True re-runs only the evaluation of a finished
    spec from its saved model.pt (keeps the training curve)."""
    d = OUT / spec["name"]
    done = (d / "result.json").exists()
    if reeval and (not done or spec["kind"] == "heads"):
        print(f"[skip reeval] {spec['name']}", flush=True)
        return
    if done and not reeval:
        print(f"[skip] {spec['name']}", flush=True)
        return
    d.mkdir(parents=True, exist_ok=True)
    print(f"[{'reeval' if reeval else 'run'}] {spec['name']}: {json.dumps(spec)}", flush=True)
    t = time.time()
    spec.setdefault("seed", 0)
    res = json.load(open(d / "result.json")) if reeval else {}
    if spec["kind"] == "heads":
        torch.save(train_heads(spec, ctx), d / "heads.pt")
        heads = load_heads(spec["name"], ctx)
        val = ctx.index.split["val"]
        f, lg = evaluate({"model": "copy"}, None, ctx, val, heads)
        res["val"] = summarize(f, lg, val, [], ctx, "copy")["selection"]
    else:
        seed_all(spec["seed"])                      # before construction: same init for every variant
        model = make_transition(spec, ctx)
        hold = HOLDOUTS.get(spec.get("holdout"), [])
        if spec["kind"] == "transition":
            if reeval:
                model.load_state_dict(torch.load(d / "model.pt", map_location=ctx.dev))
            else:
                res["curve"] = train_transition(spec, ctx, model)
                torch.save(model.state_dict(), d / "model.pt")
        heads = load_heads(spec.get("heads", f"heads_s{spec['seed']}"), ctx)
        for split in spec.get("eval", ["val"]):
            scenes = ctx.index.split[split][:spec.get("eval_n")]
            f, lg = evaluate(spec, model, ctx, scenes, heads)
            res[split] = summarize(f, lg, scenes, hold, ctx, spec["model"])
            np.savez_compressed(d / f"metrics_{split}.npz", scenes=scenes.numpy(),
                                **{k: v.numpy() for k, v in f.items()},
                                **{f"logit_{k}": v.numpy() for k, v in lg.items()})
            b, sel = res[split]["fidelity"]["balanced"], res[split]["selection"]
            print(f"  [{spec['name']}] {split}: balanced tok_cos {b['tok_cos']:.4f} skill_ln {b['skill_ln']:.4f} | "
                  + " | ".join(f"{j}: q_pred_ln {sel[j]['q_pred_ln']} direct {sel[j]['direct']:.3f} "
                               f"q_real {sel[j]['q_real']:.3f}" for j in JUDGES), flush=True)
    res["spec"], res["eval_version"] = spec, 2
    res["seconds" if not reeval else "reeval_seconds"] = time.time() - t
    json.dump(res, open(d / "result.json", "w"), indent=1)
    print(f"[done] {spec['name']} in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--reeval", action="store_true", help="re-run only the evaluation of finished specs")
    a = ap.parse_args()
    specs = json.load(open(a.specs))
    if a.only:                                  # run exactly these, in the order given
        by_name = {s["name"]: s for s in specs}
        specs = [by_name[n] for n in a.only.split(",")]
    ctx = Ctx(a.gpu)
    for spec in specs:
        run_spec(spec, ctx, reeval=a.reeval)


if __name__ == "__main__":
    main()
