"""Generate the HTML report for the ParamISP x ImageNet-ES-Diverse run."""
import base64
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
M = json.load(open(RES / "metrics.json"))
metrics = M["metrics"]
ENVS = ["l1", "l2", "l3", "l4", "l6", "l7"]
ISO_SEQ = [250, 2000, 16000]
SS_SEQ = ["1/4", "1/60", "1/1000"]
SS_VAL = [1 / 4, 1 / 60, 1 / 1000]
AP_SEQ = [5.0, 9.0, 16.0]

samples = sorted(d.name for d in RES.iterdir() if d.is_dir())
FULL_GRIDS = [(samples[0], "l1"), (samples[4], "l6")]
STRIP_ENVS = dict(zip(samples, ["l2", "l3", "l4", "l7", "l1", "l2", "l3", "l4", "l6", "l7"]))
STRIP_PICKS = [2, 4, 13, 14, 23]


def b64(p: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def pk(ap_i, ss_i, iso_i):  # param_k from grid indices
    return ap_i * 9 + ss_i * 3 + iso_i + 1


def agg(pred):
    v = [m["psnr_sim"] for m in metrics if pred(m)]
    return float(np.mean(v)) if v else float("nan")


overall_sim = float(np.mean([m["psnr_sim"] for m in metrics]))
overall_ae = float(np.mean([m["psnr_ae_baseline"] for m in metrics]))
env_sim = {e: agg(lambda m, e=e: m["env"] == e) for e in ENVS}
env_ae = {e: float(np.mean([m["psnr_ae_baseline"] for m in metrics if m["env"] == e])) for e in ENVS}

# heatmap: mean psnr per (aperture, ss, iso) across all samples+envs
heat = {}
for ai, ap in enumerate(AP_SEQ):
    for si, ss in enumerate(SS_VAL):
        for ii, iso in enumerate(ISO_SEQ):
            k = pk(ai, si, ii)
            heat[(ai, si, ii)] = agg(lambda m, k=k: m["param"] == k)
hmin, hmax = min(heat.values()), max(heat.values())


def heat_cell(v):
    t = (v - hmin) / (hmax - hmin + 1e-9)
    # sequential single-hue ramp: pale blue -> deep blue (works on both themes)
    c0, c1 = (225, 236, 248), (18, 61, 118)
    r, g, b = (round(a + (bb - a) * t) for a, bb in zip(c0, c1))
    ink = "#ffffff" if t > 0.55 else "#0b2545"
    return f'style="background:rgb({r},{g},{b});color:{ink}"'


def bar_chart():
    """Grouped bars: sim vs AE-copy PSNR per env. Inline SVG."""
    w, h, pad_l, pad_b, pad_t = 640, 240, 44, 28, 14
    ymax = 40
    gw = (w - pad_l - 12) / len(ENVS)
    bw = min(26, gw / 2.6)
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="PSNR by environment: simulation vs AE-copy baseline" style="width:100%;height:auto;max-width:{w}px">']
    for gy in range(0, ymax + 1, 10):
        y = pad_t + (h - pad_t - pad_b) * (1 - gy / ymax)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-6}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" class="tick">{gy}</text>')
    for i, e in enumerate(ENVS):
        cx = pad_l + gw * i + gw / 2
        for j, (val, cls) in enumerate([(env_sim[e], "s1"), (env_ae[e], "s2")]):
            bh = (h - pad_t - pad_b) * val / ymax
            x = cx - bw - 1 + j * (bw + 2)
            parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{h-pad_b-bh:.1f}" width="{bw}" height="{bh:.1f}" rx="3"><title>{e} {"sim" if j==0 else "AE-copy"}: {val:.1f} dB</title></rect>')
            parts.append(f'<text x="{x+bw/2:.1f}" y="{h-pad_b-bh-5:.1f}" text-anchor="middle" class="val">{val:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h-8}" text-anchor="middle" class="tick">{e}</text>')
    parts.append("</svg>")
    return "".join(parts)


def full_grid_html(sample, env):
    d = RES / sample / env
    syn, stem = sample.split("_", 1)[1].rsplit("_ILSVRC", 1)[0], sample
    rows = [f'<div class="gridblock"><div class="gridhead"><img src="{b64(d/"AE.jpg")}" alt="AE input"/><div><span class="lbl">input</span><br/>auto-exposure shot<br/><span class="mono">{env}</span> &middot; <span class="mono">{sample.split("_",1)[1]}</span></div></div>']
    for ai, ap in enumerate(AP_SEQ):
        head = "".join(f'<th class="mono">ISO {iso}<br/>{ss}s</th>' for si, ss in enumerate(SS_SEQ) for iso in ISO_SEQ)
        sim_cells, real_cells = [], []
        for si in range(3):
            for ii in range(3):
                k = pk(ai, si, ii)
                sim_cells.append(f'<td><img src="{b64(d/f"sim_param_{k}.jpg")}" alt="sim param {k}"/></td>')
                real_cells.append(f'<td><img src="{b64(d/f"real_param_{k}.jpg")}" alt="real param {k}"/></td>')
        rows.append(f'''<div class="tablewrap"><table class="imgs"><thead><tr><th class="rowlab">f/{ap:g}</th>{head}</tr></thead>
<tbody><tr><td class="rowlab">sim</td>{"".join(sim_cells)}</tr>
<tr><td class="rowlab">real</td>{"".join(real_cells)}</tr></tbody></table></div>''')
    rows.append("</div>")
    return "".join(rows)


def strip_html(sample, env):
    d = RES / sample / env
    cells = [f'<td><img src="{b64(d/"AE.jpg")}" alt="AE"/><div class="cap">AE input</div></td>']
    for k in STRIP_PICKS:
        i = k - 1
        iso, ss, ap = ISO_SEQ[i % 3], SS_SEQ[(i // 3) % 3], AP_SEQ[i // 9]
        cells.append(f'''<td><img src="{b64(d/f"sim_param_{k}.jpg")}" alt="sim {k}"/><img src="{b64(d/f"real_param_{k}.jpg")}" alt="real {k}"/>
<div class="cap mono">ISO {iso} &middot; {ss} &middot; f/{ap:g}</div></td>''')
    name = sample.split("_", 1)[1]
    return f'<div class="strip"><div class="striplab"><span class="mono">{name}</span> &middot; env <span class="mono">{env}</span> &middot; top sim / bottom real</div><div class="tablewrap"><table class="imgs"><tr>{"".join(cells)}</tr></table></div></div>'


heat_tables = []
for ai, ap in enumerate(AP_SEQ):
    rows = []
    for si, ss in enumerate(SS_SEQ):
        cells = "".join(f'<td {heat_cell(heat[(ai,si,ii)])}>{heat[(ai,si,ii)]:.1f}</td>' for ii in range(3))
        rows.append(f'<tr><td class="rowlab mono">{ss}s</td>{cells}</tr>')
    heat_tables.append(f'''<div class="heat"><div class="heatlab">f/{ap:g}</div><table class="heattab">
<thead><tr><th></th><th class="mono">ISO 250</th><th class="mono">ISO 2k</th><th class="mono">ISO 16k</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>''')

e_ae_rows = "".join(f'<tr><td class="mono">{e}</td><td class="mono">{M["e_ae"][e]:.3f}</td></tr>' for e in ENVS)


def calstrip_html():
    d = RES / samples[0] / "l1"
    picks = [10, 4, 14, 5]
    rows = []
    for tag, pre in [("raw sim", "sim"), ("calibrated", "simcal"), ("real", "real")]:
        cells = "".join(f'<td><img src="{b64(d/f"{pre}_param_{k}.jpg")}" alt="{tag} param {k}"/></td>' for k in picks)
        rows.append(f'<tr><td class="rowlab">{tag}</td>{cells}</tr>')
    head = "".join(f'<th class="mono">{"ISO "+str(ISO_SEQ[(k-1)%3])}<br/>{SS_SEQ[((k-1)//3)%3]}s &middot; f/{AP_SEQ[(k-1)//9]:g}</th>' for k in picks)
    return f'<div class="tablewrap"><table class="imgs"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


calstrip = calstrip_html()


def physstrip_html():
    d = RES / samples[4] / "l6"
    picks = [10, 4, 14, 5, 2]
    rows = []
    for tag, pre in [("ParamISP", "sim"), ("physical v2", "phys"), ("real", "real")]:
        cells = "".join(f'<td><img src="{b64(d/f"{pre}_param_{k}.jpg")}" alt="{tag} param {k}"/></td>' for k in picks)
        rows.append(f'<tr><td class="rowlab">{tag}</td>{cells}</tr>')
    head = "".join(f'<th class="mono">{"ISO "+str(ISO_SEQ[(k-1)%3])}<br/>{SS_SEQ[((k-1)//3)%3]}s &middot; f/{AP_SEQ[(k-1)//9]:g}</th>' for k in picks)
    return f'<div class="tablewrap"><table class="imgs"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


physstrip = physstrip_html()

html = f"""<title>ParamISP on ImageNet-ES-Diverse — exposure simulation vs ground truth</title>
<style>
:root {{
  --bg:#fbfbfa; --ink:#131311; --ink2:#565550; --ink3:#8a897f;
  --card:#f2f1ec; --line:#e0dfd7; --grid:#e7e6de;
  --accent:#2a78d6; --accent2:#eb6834;
}}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{
  --bg:#171716; --ink:#f0efe9; --ink2:#c3c2b7; --ink3:#8a897f;
  --card:#20201e; --line:#33322f; --grid:#2c2b28;
  --accent:#3987e5; --accent2:#d95926;
}} }}
:root[data-theme="dark"] {{
  --bg:#171716; --ink:#f0efe9; --ink2:#c3c2b7; --ink3:#8a897f;
  --card:#20201e; --line:#33322f; --grid:#2c2b28;
  --accent:#3987e5; --accent2:#d95926;
}}
:root[data-theme="light"] {{
  --bg:#fbfbfa; --ink:#131311; --ink2:#565550; --ink3:#8a897f;
  --card:#f2f1ec; --line:#e0dfd7; --grid:#e7e6de;
  --accent:#2a78d6; --accent2:#eb6834;
}}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:15px/1.55 -apple-system,'Segoe UI',system-ui,sans-serif; }}
main {{ max-width:920px; margin:0 auto; padding:40px 20px 80px; }}
h1 {{ font-family:'Iowan Old Style','New York',Georgia,serif; font-size:1.9rem;
  line-height:1.25; font-weight:600; margin:.2em 0 .3em; text-wrap:balance; }}
h2 {{ font-family:'Iowan Old Style','New York',Georgia,serif; font-size:1.25rem;
  font-weight:600; margin:2.2em 0 .5em; }}
p {{ max-width:68ch; color:var(--ink2); margin:.4em 0; }}
p b {{ color:var(--ink); }}
.eyebrow {{ font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--ink3); }}
.mono {{ font-family:ui-monospace,'SF Mono',Menlo,monospace; font-size:.92em;
  font-variant-numeric:tabular-nums; }}
.tiles {{ display:flex; gap:12px; margin:1.4em 0; flex-wrap:wrap; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:14px 18px; min-width:150px; flex:1; }}
.tile .n {{ font-size:1.8rem; font-weight:650; font-variant-numeric:tabular-nums; }}
.tile .n.acc {{ color:var(--accent); }}
.tile .d {{ font-size:.8rem; color:var(--ink3); }}
.legend {{ display:flex; gap:18px; font-size:.82rem; color:var(--ink2); margin:.3em 0 .1em; }}
.legend i {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; }}
svg .s1 {{ fill:var(--accent); }} svg .s2 {{ fill:var(--accent2); }}
svg .tick {{ font:11px ui-monospace,Menlo,monospace; fill:var(--ink3); }}
svg .val {{ font:10px ui-monospace,Menlo,monospace; fill:var(--ink2); }}
.tablewrap {{ overflow-x:auto; }}
table.imgs {{ border-collapse:separate; border-spacing:2px; }}
table.imgs img {{ width:96px; height:auto; display:block; border-radius:3px; }}
table.imgs th {{ font:10px/1.3 ui-monospace,Menlo,monospace; color:var(--ink3); font-weight:500; padding:2px; }}
table.imgs .rowlab {{ font:11px ui-monospace,Menlo,monospace; color:var(--ink2);
  text-align:right; padding-right:8px; white-space:nowrap; vertical-align:middle; }}
.gridblock {{ margin:1.2em 0 2em; }}
.gridhead {{ display:flex; gap:14px; align-items:center; margin-bottom:10px;
  font-size:.85rem; color:var(--ink2); }}
.gridhead img {{ width:110px; border-radius:5px; }}
.lbl {{ font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--ink3); }}
.strip {{ margin:1.4em 0; }}
.striplab {{ font-size:.8rem; color:var(--ink3); margin-bottom:6px; }}
.strip img {{ width:86px; }}
.cap {{ font-size:.68rem; color:var(--ink3); text-align:center; padding-top:2px; }}
.heats {{ display:flex; gap:20px; flex-wrap:wrap; margin:1em 0; }}
.heatlab {{ font-size:.8rem; color:var(--ink2); margin-bottom:4px; font-weight:600; }}
table.heattab {{ border-collapse:separate; border-spacing:2px; }}
table.heattab td, table.heattab th {{ padding:7px 12px; border-radius:4px;
  font:12px ui-monospace,Menlo,monospace; font-variant-numeric:tabular-nums; }}
table.heattab th {{ color:var(--ink3); font-weight:500; }}
table.heattab .rowlab {{ color:var(--ink2); background:none; }}
.note {{ background:var(--card); border:1px solid var(--line); border-radius:8px;
  padding:14px 18px; font-size:.88rem; color:var(--ink2); max-width:none; }}
.note b {{ color:var(--ink); }}
ul {{ color:var(--ink2); max-width:72ch; }} ul b {{ color:var(--ink); }}
table.meta {{ border-collapse:collapse; font-size:.85rem; }}
table.meta td {{ border-bottom:1px solid var(--line); padding:4px 18px 4px 0; color:var(--ink2); }}
</style>
<main>
<div class="eyebrow">ada_sensing &middot; exposure simulation &middot; validation run</div>
<h1>ParamISP simulates the ImageNet-ES-Diverse exposure grid from a single auto-exposure shot</h1>
<p>Pipeline: AE sRGB &rarr; <b>inverse ISP</b> (ParamISP, CVPR 2024, official pre-trained weights) &rarr; pseudo-RAW,
scaled by the physical exposure ratio <span class="mono">t&middot;ISO/N&sup2;</span> &rarr; <b>forward ISP</b> conditioned
on the target parameters &rarr; simulated sRGB. Scored against the dataset's <b>real captures</b> of the same
27 configs (3 apertures &times; 3 shutters &times; 3 ISOs), 10 samples &times; 6 lighting environments = 1,620 comparisons.</p>

<div class="tiles">
<div class="tile"><div class="n acc">{overall_sim:.1f} dB</div><div class="d">simulated vs real, mean PSNR</div></div>
<div class="tile"><div class="n">{overall_ae:.1f} dB</div><div class="d">AE-copy baseline (predict AE image for every config)</div></div>
<div class="tile"><div class="n">1,620</div><div class="d">config comparisons (10 &times; 6 &times; 27)</div></div>
</div>

<h2>PSNR by lighting environment</h2>
<div class="legend"><span><i style="background:var(--accent)"></i>ParamISP simulation</span>
<span><i style="background:var(--accent2)"></i>AE-copy baseline</span></div>
{bar_chart()}
<p>The gap over the baseline is what the simulator actually explains: brightness structure across
the grid, not scene content (the AE copy already has the scene right).</p>

<h2>Where it is accurate — PSNR by config</h2>
<p>Mean over all samples and environments. Darker blue = higher fidelity.</p>
<div class="heats">{"".join(heat_tables)}</div>
<p>The hardest cells are the well-exposed mid-brightness configs (e.g. ISO 2000 at f/5&ndash;f/9), where tone
and color must be exactly right; near-black and blown configs are easy to score on. This is expected
PSNR behaviour &mdash; judge those cells by the grids below.</p>

<h2>Full 27-config grids, sim vs real</h2>
{"".join(full_grid_html(s, e) for s, e in FULL_GRIDS)}

<h2>Quick strips — remaining samples</h2>
<p>AE input, then five configs spanning the range (top sim / bottom real).</p>
{"".join(strip_html(s, STRIP_ENVS[s]) for s in samples if (s, STRIP_ENVS[s]) not in [(a,b) for a,b in FULL_GRIDS])}

<h2>Error decomposition & post-hoc calibration</h2>
<p>Fitting corrections on 400 random sim/real pairs decomposes the error: a global brightness
gain recovers <b>+3.6 dB</b>, a per-channel color gain a further <b>+1.3 dB</b>, and the rest is
structural &mdash; the cross-camera ISP gap (consistent with the &asymp;27.5 dB identity-round-trip ceiling).
Note the headline PSNR is flattered by near-black cells (58% of the grid); well-exposed configs
sit at &asymp;19 dB raw.</p>
<p>A per-environment post-hoc correction (3&times;3 linear color matrix + monotonic tone LUT,
black point pinned, <b>fit only on the 5 calibration images</b>) was applied to all eval sims:</p>
<div class="tablewrap"><table class="heattab">
<thead><tr><th></th><th class="mono">n</th><th class="mono">raw</th><th class="mono">calibrated</th><th class="mono">&Delta;</th></tr></thead><tbody>
<tr><td class="rowlab">overall</td><td class="mono">1620</td><td class="mono">33.9</td><td class="mono">34.7</td><td class="mono">+0.8</td></tr>
<tr><td class="rowlab">near-black</td><td class="mono">960</td><td class="mono">42.1</td><td class="mono">42.1</td><td class="mono">&plusmn;0.0</td></tr>
<tr><td class="rowlab">dark-mid</td><td class="mono">268</td><td class="mono">23.8</td><td class="mono">25.2</td><td class="mono">+1.4</td></tr>
<tr><td class="rowlab">well-exposed</td><td class="mono">153</td><td class="mono">18.6</td><td class="mono">20.5</td><td class="mono">+1.9</td></tr>
<tr><td class="rowlab">bright/blown</td><td class="mono">225</td><td class="mono">22.3</td><td class="mono">24.7</td><td class="mono">+2.4</td></tr>
</tbody></table></div>
{calstrip}
<p><b>Caveat the numbers hide:</b> the least-squares fit earns its PSNR partly by desaturating
(middle row above is visibly grayer than both raw sim and real). Global corrections are at their
ceiling; the remaining gap is config-dependent and spatially varying &mdash; a network-level fix
(per-camera fine-tuning), not a LUT-level one. Calibrated sims saved as
<span class="mono">simcal_param_*.jpg</span> alongside the raw ones.</p>

<h2>The physical simulator wins</h2>
<p>A weight-free alternative run on the identical benchmark: linearize &rarr; scale by
<span class="mono">t&middot;ISO/N&sup2;</span> &rarr; clip &rarr; re-encode, where the per-channel tone curve is a
33-knot monotone LUT <b>fitted to this camera</b> from the same 5 calibration images (3 alternating
iterations of curve fit and per-env AE-exposure refinement; median-based, which avoids the
least-squares desaturation above). v1 assumes the standard sRGB curve; v2 uses the fitted one.</p>
<div class="tablewrap"><table class="heattab">
<thead><tr><th></th><th class="mono">n</th><th class="mono">ParamISP</th><th class="mono">ParamISP+cal</th><th class="mono">phys v1</th><th class="mono">phys v2</th></tr></thead><tbody>
<tr><td class="rowlab">overall</td><td class="mono">1620</td><td class="mono">33.9</td><td class="mono">34.7</td><td class="mono">30.8</td><td class="mono"><b>40.2</b></td></tr>
<tr><td class="rowlab">near-black</td><td class="mono">960</td><td class="mono">42.1</td><td class="mono">42.1</td><td class="mono">37.6</td><td class="mono"><b>48.8</b></td></tr>
<tr><td class="rowlab">dark-mid</td><td class="mono">268</td><td class="mono">23.8</td><td class="mono">25.2</td><td class="mono">23.7</td><td class="mono"><b>29.5</b></td></tr>
<tr><td class="rowlab">well-exposed</td><td class="mono">153</td><td class="mono">18.6</td><td class="mono">20.5</td><td class="mono">20.9</td><td class="mono"><b>23.3</b></td></tr>
<tr><td class="rowlab">bright/blown</td><td class="mono">225</td><td class="mono">22.3</td><td class="mono">24.7</td><td class="mono">17.0</td><td class="mono"><b>26.5</b></td></tr>
</tbody></table></div>
{physstrip}
<p>The fitted physical model beats the pretrained network in every band, numerically and visually
(no desaturation, no warm cast &mdash; middle row vs top row above). Reading: on planar scenes the
exposure problem <i>is</i> a tone curve plus a scalar; what ParamISP adds &mdash; a learned ISP of
<i>other</i> cameras &mdash; costs more than it contributes cross-camera, while the physical model spends
its calibration budget directly on the only unknown that matters. Still missing from both: ISO noise
synthesis and any spatially varying ISP behaviour. Physical sims saved as
<span class="mono">phys_param_*.jpg</span>; the fitted curve at <span class="mono">results/phys_tonecurve.npy</span>;
code in <span class="mono">run_phys_es.py</span> (+ the standalone module <span class="mono">es_sim.py</span> with a
fittable shot/read-noise model).</p>

<h2>Calibration & honest caveats</h2>
<div class="note">
<b>AE exposure estimation.</b> The AE JPEGs carry no EXIF, so the AE exposure product
<span class="mono">E = t&middot;ISO/N&sup2;</span> was estimated per environment: initialized from per-pixel
linear-luminance ratios vs well-exposed reference captures, then refined through the simulator by
brightness matching &mdash; using <b>5 calibration images kept separate from the 10 evaluated samples</b>.
One scalar per environment; no other ground truth enters the simulation.
</div>
<ul>
<li><b>Cross-camera:</b> the ES-Diverse capture camera is not among ParamISP's training cameras; the run
uses the multi-camera pre-training weights with neutral WB/CCM chosen by round-trip self-consistency
(identity round-trip &asymp;27.5 dB). Residual tone/color mismatch is visible in the grids &mdash; sims trend
slightly darker and warmer than real at some settings.</li>
<li><b>No noise module:</b> unlike the original Neural Camera Simulators, ParamISP has no explicit
noise injection; real ISO 16000 grain is not reproduced.</li>
<li><b>Aperture is photometric only:</b> valid here because ES-Studio photographs planar prints.
On real 3D scenes (video), f-number changes depth of field &mdash; this pipeline will not simulate that.</li>
<li><b>Overexposure artifacts:</b> heavily blown configs (1/4s at high ISO) show orange/magenta casts
in simulation where the real capture clips to clean white.</li>
</ul>

<h2>Run metadata</h2>
<table class="meta">
<tr><td>Simulator</td><td class="mono">ParamISP pre_training inverse+forward (official weights)</td></tr>
<tr><td>Device / resolution</td><td class="mono">{M["device"]} &middot; max side 384 px</td></tr>
<tr><td>E_AE per env (t&middot;ISO/N&sup2;)</td><td><table class="meta">{e_ae_rows}</table></td></tr>
<tr><td>Code</td><td class="mono">ada_sensing/exposure_sim/{{paramisp_sim,run_es_diverse}}.py</td></tr>
<tr><td>Full outputs</td><td class="mono">ada_sensing/exposure_sim/results/ (all 1,620 sim/real pairs + metrics.json)</td></tr>
</table>
</main>
"""
out = RES / "report.html"
out.write_text(html)
print("wrote", out, f"{out.stat().st_size/1e6:.1f} MB")
