"""
Figures for the QAE status page, from the result files. Writes SVGs to <out_dir>.

  python3 qae/plots.py <out_dir>
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "results_batch1"
V3 = Path(__file__).resolve().parent.parent / "data" / "qae_batch" / "sweep_targets_analysis_v3.json"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
C = {"ae": "#9aa0a6", "fixed": "#c7a86b", "direct": "#4c78a8", "text": "#e45756", "region": "#54a24b", "jepa_all": "#e45756", "jepa_k": "#f2a0a0"}
NAME = {"ae": "auto exposure", "fixed": "fixed setting", "direct": "direct", "text": "JEPA text", "region": "region (box)"}
import os
if os.environ.get("PNG"):   # also write PNG previews
    import matplotlib.figure as _F
    _orig = _F.Figure.savefig
    _F.Figure.savefig = lambda self, path, *a, **k: (_orig(self, path, *a, **k), _orig(self, str(path).replace(".svg", ".png"), dpi=110))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


def rows_of(pattern):
    return [json.load(open(f))["rows"] for f in sorted(glob.glob(pattern))]


def ci(xs):
    xs = np.asarray(xs, float); return xs.mean(), (1.96 * xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0)


def bands_figure():
    runs = rows_of(str(R / "seeds" / "model_text_seed*.json"))
    subsets = [("all", lambda r: True), ("conflict", lambda r: r["conflict"]), ("bright band", lambda r: r["band"] == "bright"),
               ("dim band", lambda r: r["band"] == "dim"), ("control", lambda r: r["band"] == "control")]
    methods = ["ae", "fixed", "direct", "text", "region"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6)); w = 0.16; x = np.arange(len(subsets))
    for i, m in enumerate(methods):
        mus, hs = [], []
        for _, pred in subsets:
            per = [np.mean([r[m] for r in rows if pred(r)]) for rows in runs]; mu, h = ci(per); mus.append(mu); hs.append(h)
        ax.bar(x + (i - 2) * w, mus, w, yerr=hs, capsize=2, color=C[m], label=NAME[m], error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in subsets]); ax.set_ylim(0, 1.0); ax.set_ylabel("mean VLM score (held-out)")
    ax.set_title("Batch one, repaired: single-shot choice, 5 seeds, ± 95% CI", loc="left")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(OUT / "fig_bands.svg"); plt.close(fig)


def curve_figure():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for f, label, style in ((R / "label_curve_results.json", "JEPA, 171 unlabeled photos", "-"), (R / "label_curve_v3unl.json", "JEPA, 684 unlabeled photos", "--")):
        res = json.load(open(f)); ks = sorted({r["k"] for r in res})
        for key, name, col in (("jepa_all", label, C["text"]), ("direct", "direct", C["direct"]), ("fixed", "fixed setting", C["fixed"])):
            if key != "jepa_all" and style == "--":
                continue
            mu = [np.mean([r[key] for r in res if r["k"] == k]) for k in ks]; sd = [np.std([r[key] for r in res if r["k"] == k]) for k in ks]
            ax.errorbar(ks, mu, yerr=sd, fmt="o" + style, color=col, label=name, capsize=2, lw=1.4, ms=4)
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16, 32]); ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlabel("labeled scenes k"); ax.set_ylabel("mean VLM score (8 test scenes)"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Label efficiency on batch one: flat, with either unlabeled pool", loc="left"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_curve.svg"); plt.close(fig)


def holdout_figure():
    recipes = [("CLIP", "holdout/holdout_{h}_seed*.json"), ("MiniLM", "variant/minilm_holdout_{h}_seed*.json"), ("MiniLM + paraphrases", "variant/para_holdout_{h}_seed*.json")]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    for ax, (h, title) in zip(axes, (("text", "text targets held out\n(trained on image questions)"), ("vqa_image", "image questions held out\n(trained on text targets)"))):
        x = np.arange(len(recipes)); w = 0.25
        for i, m in enumerate(("direct", "text", "region")):
            mus, hs = [], []
            for _, pat in recipes:
                runs = rows_of(str(R / pat.format(h=h)))
                if not runs:
                    mus.append(np.nan); hs.append(0); continue
                mu, hh = ci([np.mean([r[m] for r in rows]) for rows in runs]); mus.append(mu); hs.append(hh)
            ax.bar(x + (i - 1) * w, mus, w, yerr=hs, capsize=2, color=C[m], label=NAME[m], error_kw={"lw": 0.8})
        ae = np.mean([r["ae"] for r in rows_of(str(R / recipes[0][1].format(h=h)))[0]])
        ax.axhline(ae, color=C["ae"], ls=":", lw=1.2); ax.text(len(recipes) - 0.6, ae + 0.015, "auto exposure", fontsize=8, color="#555", ha="right")
        ax.set_xticks(x); ax.set_xticklabels([n for n, _ in recipes], fontsize=8); ax.set_title(title, fontsize=9.5, loc="left"); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("mean VLM score"); axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Held-out question family: no language recipe transfers; the box-grounded model does", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig_holdout.svg"); plt.close(fig)


def settings_figure():
    runs = rows_of(str(R / "holdout2" / "settings_seed*.json"))
    if not runs:
        return
    held = {1, 5, 8, 12, 14}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    methods = ("direct", "text", "region")
    mus, hs = zip(*[ci([np.mean([r[m] for r in rows]) for rows in runs]) for m in methods])
    axes[0].bar(range(3), mus, 0.6, yerr=hs, capsize=3, color=[C[m] for m in methods]); axes[0].set_ylim(0, 1)
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels([NAME[m] for m in methods]); axes[0].set_title("choosing among all 15 settings", loc="left", fontsize=9.5); axes[0].set_ylabel("mean VLM score")
    mus2, hs2 = [], []
    for m in methods:
        per = []
        for rows in runs:
            sel = [r[m] for r in rows if r["ranking"][m][0] in held]; per.append(np.mean(sel) if sel else np.nan)
        mu, h = ci([p for p in per if not np.isnan(p)]); mus2.append(mu); hs2.append(h)
    axes[1].bar(range(3), mus2, 0.6, yerr=hs2, capsize=3, color=[C[m] for m in methods]); axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels([NAME[m] for m in methods]); axes[1].set_title("score when the pick is one of the 5 never-labeled settings", loc="left", fontsize=9.5)
    fig.suptitle(f"Held-out settings on batch one ({len(runs)} seeds): heads trained with labels from 10 of the 15 settings", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_settings.svg"); plt.close(fig)


def budget_figure():
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import budget_eval as BE
    runs = rows_of(str(R / "ranked" / "rank_seed*.json")); Bs = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for m in ("direct", "text", "region"):
        ys = [np.mean([np.mean([(np.array(r["cand_scores"]) >= 1)[r["ranking"][m][:B]].any() for r in rows]) for rows in runs]) for B in Bs]
        ax.plot(Bs, ys, "o-", color=C[m], label=NAME[m] + " top-B")
    fixed = []
    for B in Bs:
        hit = []
        for rows in runs:
            S = np.array([r["cand_scores"] for r in rows]) >= 1
            for sc in sorted({r["scene"] for r in rows}):
                ch = BE.greedy_cover([r for r in rows if r["scene"] != sc], B); hit += [S[i, ch].any() for i, r in enumerate(rows) if r["scene"] == sc]
        fixed.append(np.mean(hit))
    ax.plot(Bs, fixed, "s--", color=C["fixed"], label="fixed bracket (no query)")
    ax.set_xticks(Bs); ax.set_xlabel("captures allowed, B"); ax.set_ylabel("targets answered"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Equal budget on batch one", loc="left"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_budget.svg"); plt.close(fig)


def ground_figure():
    G = json.load(open(R / "ground_b1.json"))
    groups = [("all", lambda r: True), ("bright", lambda r: r["band"] == "bright"), ("dim", lambda r: r["band"] == "dim"), ("text lines", lambda r: r["kind"] == "text"), ("image questions", lambda r: r["kind"] == "vqa_image")]
    fig, ax = plt.subplots(figsize=(6.4, 3.3)); x = np.arange(len(groups)); w = 0.36
    for i, (which, name, col) in enumerate((("ae", "auto-exposure frame", "#9aa0a6"), ("oracle", "best-exposed frame", "#4c78a8"))):
        ax.bar(x + (i - 0.5) * w, [np.mean([r[which]["iou"] >= 0.5 for r in G if p(r)]) for _, p in groups], w, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups]); ax.set_ylabel("targets localized (IoU ≥ 0.5)"); ax.set_ylim(0, 1)
    ax.set_title("Can the VLM locate the query's region? (batch one, 205 targets)", loc="left"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_ground.svg"); plt.close(fig)


def v3_figures():
    if not V3.exists():
        return
    T = json.load(open(V3)); R_ = [t for t in T if t["tag"] != "impossible"]
    isos, shs = (100, 400, 1600), ("1/1000", "1/250", "1/60", "1/15", "1/4")
    bands = ["bright", "mid", "dim", "control"]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.4))
    for ax, b in zip(axes, bands):
        rs = [t for t in R_ if t["band"] == b]; H = np.zeros((3, 5))
        for t in rs:
            H[np.argmax(t["grid"]) // 5, np.argmax(t["grid"]) % 5] += 1
        H /= max(1, len(rs))
        im = ax.imshow(H, cmap="Reds", vmin=0, vmax=0.6); ax.set_title(f"{b} band (n={len(rs)})", fontsize=9.5, loc="left")
        ax.set_xticks(range(5)); ax.set_xticklabels(shs, fontsize=7.5); ax.set_yticks(range(3)); ax.set_yticklabels([f"ISO {i}" for i in isos], fontsize=7.5)
        for i in range(3):
            for j in range(5):
                if H[i, j] >= 0.08:
                    ax.text(j, i, f"{H[i,j]:.0%}", ha="center", va="center", fontsize=7, color="white" if H[i, j] > 0.35 else "black")
    fig.suptitle("v3: where each band's oracle setting sits on the ISO × shutter grid (share of recoverable targets)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_v3_oracle.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(len(bands) + 1); w = 0.26
    G = np.stack([t["grid"] for t in R_]); fixed_idx = int(G.mean(0).argmax())
    for i, (key, name, col) in enumerate((("ae", "auto exposure", C["ae"]), ("fixed", "best fixed (ISO 100, 1/60)", C["fixed"]), ("best", "oracle", "#333"))):
        vals = []
        for b in bands + ["all"]:
            rs = [t for t in R_ if b == "all" or t["band"] == b]
            vals.append(np.mean([t["ae"] for t in rs]) if key == "ae" else np.mean([t["grid"][fixed_idx] for t in rs]) if key == "fixed" else np.mean([t["best"] for t in rs]))
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=name)
    ax.set_xticks(x); ax.set_xticklabels(bands + ["all"]); ax.set_ylim(0, 1.05); ax.set_ylabel("mean VLM score"); ax.set_title("v3 dataset: 2,003 recoverable targets, no learning yet", loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(); fig.savefig(OUT / "fig_v3_bands.svg"); plt.close(fig)


def wording_figure():
    inf = rows_of(str(R / "seeds" / "model_text_seed*.json")); par = rows_of(str(R / "holdout2" / "wording_seed*.json"))
    if not par:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); methods = ("direct", "text", "region"); x = np.arange(len(methods)); w = 0.36
    for i, (runs, name, alpha) in enumerate(((inf, "original wording (in-family)", 1.0), (par, "paraphrased wording (never seen)", 0.55))):
        mus, hs = zip(*[ci([np.mean([r[m] for r in rows]) for rows in runs]) for m in methods])
        ax.bar(x + (i - 0.5) * w, mus, w, yerr=hs, capsize=3, color=[C[m] for m in methods], alpha=alpha, label=name, error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in methods]); ax.set_ylim(0, 1); ax.set_ylabel("mean VLM score")
    ax.set_title("Held-out wording on batch one: trained on original questions, tested on paraphrases", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_wording.svg"); plt.close(fig)


def v3_model_figure():
    runs = rows_of(str(Path(__file__).resolve().parent / "results_v3" / "model_text_seed*.json"))
    if not runs:
        return
    subsets = [("all", lambda r: True), ("conflict", lambda r: r["conflict"]), ("bright", lambda r: r["band"] == "bright"),
               ("mid", lambda r: r["band"] == "mid"), ("dim", lambda r: r["band"] == "dim"), ("control", lambda r: r["band"] == "control")]
    methods = ["ae", "fixed", "direct", "text", "region"]
    fig, ax = plt.subplots(figsize=(9, 3.6)); w = 0.16; x = np.arange(len(subsets))
    for i, m in enumerate(methods):
        mus, hs = zip(*[ci([np.mean([r[m] for r in rows if pred(r)]) for rows in runs]) for _, pred in subsets])
        ax.bar(x + (i - 2) * w, mus, w, yerr=hs, capsize=2, color=C[m], label=NAME[m], error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in subsets]); ax.set_ylim(0, 1.0); ax.set_ylabel("mean VLM score (held-out)")
    ax.set_title(f"v3: single-shot choice, {len(runs)} seeds × 5 folds, 2,003 held-out targets, ± 95% CI", loc="left")
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(OUT / "fig_v3_model.svg"); plt.close(fig)


def v3_curve_figure():
    f = Path(__file__).resolve().parent / "results_v3" / "label_curve.json"
    if not f.exists():
        return
    res = json.load(open(f)); ks = sorted({r["k"] for r in res})
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, name, col, style in (("jepa_all", "JEPA, predictor on all unlabeled photos", C["text"], "-"), ("jepa_k", "JEPA, predictor on the k scenes only", C["jepa_k"], "--"), ("direct", "direct", C["direct"], "-"), ("fixed", "fixed setting", C["fixed"], "-")):
        mu = [np.mean([r[key] for r in res if r["k"] == k]) for k in ks]; sd = [np.std([r[key] for r in res if r["k"] == k]) for k in ks]
        ax.errorbar(ks, mu, yerr=sd, fmt="o" + style, color=col, label=name, capsize=2, lw=1.4, ms=4)
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("labeled scenes k"); ax.set_ylabel("mean VLM score (60 test scenes)"); ax.set_ylim(0.5, 0.95)
    ax.set_title("Label efficiency on v3: JEPA above direct at every k", loc="left"); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_v3_curve.svg"); plt.close(fig)


def settings_all_figure():
    splits = [("batch one, 5 held out\n(edge settings included)", R / "holdout2" / "settings_seed*.json", {1, 5, 8, 12, 14}),
              ("batch one, checkerboard\n(7 held out, all interpolable)", R / "holdout2" / "checker_seed*.json", {1, 3, 5, 7, 9, 11, 13}),
              ("v3, 5 held out\n(edge settings included)", Path(__file__).resolve().parent / "results_v3" / "settings_seed*.json", {1, 5, 8, 12, 14})]
    fig, ax = plt.subplots(figsize=(8.2, 3.5)); x = np.arange(len(splits)); w = 0.26
    for i, m in enumerate(("direct", "text", "region")):
        mus, hs = [], []
        for _, pat, held in splits:
            runs = rows_of(str(pat))
            if not runs:
                mus.append(np.nan); hs.append(0); continue
            per = [np.mean([r[m] for r in rows if r["ranking"][m][0] in held]) for rows in runs]; mu, h = ci(per); mus.append(mu); hs.append(h)
        ax.bar(x + (i - 1) * w, mus, w, yerr=hs, capsize=3, color=C[m], label=NAME[m], error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels([n for n, _, _ in splits], fontsize=8.5); ax.set_ylim(0, 1.05); ax.set_ylabel("score when the pick is a never-labeled setting")
    ax.set_title("Held-out settings: how good are the picks the heads were never supervised on?", loc="left", fontsize=10); ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig_settings_all.svg"); plt.close(fig)


for fn in (bands_figure, curve_figure, holdout_figure, settings_figure, wording_figure, budget_figure, ground_figure, v3_figures, v3_model_figure, v3_curve_figure, settings_all_figure):
    try:
        fn(); print("ok", fn.__name__)
    except Exception as e:
        print("FAILED", fn.__name__, repr(e))
