import base64
from pathlib import Path

SC = Path("/private/tmp/claude-501/-Users-malikh-Desktop/d66e8e68-f0b5-41a6-919a-fb6de7035e29/scratchpad")
A, P = SC / "assets", SC / "plots"


def img(name, cap, wide=False):
    data = base64.b64encode((A / name).read_bytes()).decode()
    return f'<figure class="sample{" wide" if wide else ""}"><img src="data:image/jpeg;base64,{data}" alt=""><figcaption>{cap}</figcaption></figure>'


def fig(name, cap=""):
    t = (P / name).read_text(); t = t[t.index("<svg"):]
    return f'<figure class="chart">{t}{f"<figcaption>{cap}</figcaption>" if cap else ""}</figure>'


def stage(n, when, title, found, body):
    return f'''<section class="stage" id="s{n}">
<div class="rail"><span class="num">{n}</span></div>
<div class="content"><h2>{title}</h2><p class="found"><b>Found</b> {found}</p>{body}</div>
</section>'''


CSS = '''
<title>Query-Aware Exposure Log</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--paper:#f7f5ef;--card:#fffdf8;--ink:#1c1a16;--muted:#6d665a;--rule:#e2dcd0;--accent:#b8641a;--accent-soft:#f6e6d3;--jepa:#d9483f;--direct:#3f6fa3;--region:#4c9a44;--fixed:#b89654;--found:#f3ead9;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#16140f;--card:#1f1c16;--ink:#ebe5d8;--muted:#a69e8e;--rule:#2f2b23;--accent:#e4913f;--accent-soft:#3a2a16;--found:#2b2418;color-scheme:dark}}
:root[data-theme="dark"]{--paper:#16140f;--card:#1f1c16;--ink:#ebe5d8;--muted:#a69e8e;--rule:#2f2b23;--accent:#e4913f;--accent-soft:#3a2a16;--found:#2b2418;color-scheme:dark}
body{background:var(--paper);color:var(--ink);font:16px/1.6 "IBM Plex Sans",system-ui,sans-serif;margin:0;padding-block:32px 64px;padding-inline:clamp(16px,4vw,48px)}
h1,h2,h3{font-family:"Fraunces","Georgia",serif;text-wrap:balance;margin:0}
h1{font-size:clamp(2rem,5vw,3.1rem);font-weight:700;line-height:1.05}
h2{font-size:1.45rem;font-weight:700;margin-bottom:6px}
h3{font-size:1.1rem;font-weight:700;margin:22px 0 6px}
p{max-width:70ch;margin:0 0 12px}
.lede{max-width:70ch;color:var(--muted);font-size:1.05rem;margin-top:10px}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;color:var(--accent);font-weight:600}
.glance{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:28px 0 40px}
.glance a{display:block;text-decoration:none;color:var(--ink);background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;font-size:.85rem;line-height:1.35}
.glance a b{display:block;color:var(--accent);font-family:"IBM Plex Mono",monospace;font-size:.72rem;margin-bottom:4px}
.glance a:hover,.glance a:focus-visible{border-color:var(--accent);outline:none}
.stage{display:grid;grid-template-columns:96px minmax(0,1fr);gap:0 24px;border-top:1px solid var(--rule);padding-block:28px 10px}
.rail{position:sticky;top:12px;align-self:start;display:flex;flex-direction:column;gap:2px}
.rail .num{font-family:"Fraunces",serif;font-size:2rem;font-weight:700;color:var(--accent);line-height:1}
.rail .when{font-family:"IBM Plex Mono",monospace;font-size:.72rem;color:var(--muted);letter-spacing:.04em}
.content{min-width:0}
.found{background:var(--found);border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 6px 6px 0;max-width:76ch}
.found b{color:var(--accent);text-transform:uppercase;letter-spacing:.1em;font-size:.72rem;margin-right:8px}
figure{margin:18px 0}
figure img{display:block;width:100%;max-width:100%;border-radius:6px;border:1px solid var(--rule)}
figure.sample{max-width:900px}
figure.sample.wide{max-width:1100px}
figure.chart{max-width:960px;background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:8px}
figure.chart svg{width:100%;height:auto;max-width:100%}
figcaption{font-size:.85rem;color:var(--muted);margin-top:6px;max-width:80ch}
.wrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.06em}
td.j{color:var(--jepa);font-weight:600} td.d{color:var(--direct)} td.r{color:var(--region)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.aside{border:1px dashed var(--rule);border-radius:8px;padding:10px 14px;color:var(--muted);font-size:.9rem;max-width:76ch}
.ev{list-style:none;padding:0;margin:0;display:grid;gap:8px;max-width:80ch}
.ev li{display:grid;grid-template-columns:150px 1fr;gap:12px;align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:8px}
.ev li span:first-child{font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--accent)}
code{font-family:"IBM Plex Mono",monospace;font-size:.86em}
@media (max-width:640px){.stage{grid-template-columns:1fr;gap:10px}.rail{position:static;flex-direction:row;align-items:baseline;gap:12px}.ev li{grid-template-columns:1fr}}
@media (prefers-reduced-motion:no-preference){.glance a{transition:border-color .15s}}
</style>
'''

HEAD = '''
<header>
<div class="eyebrow">Adaptive sensing · query-aware exposure · Oct 8 to 10, 2026</div>
<h1>From one scene to a verdict on JEPA</h1>
<p class="lede">Everything after the meeting, in the order it happened: the benchmark redesign, two hand-built scenes, batch one and its contamination, the v3 batch of 244 scenes, and the experiments that separate a JEPA-style state predictor from a direct model. Each stage opens with what it established; the samples are real iPad renders and real phone captures, and the charts are drawn from the result files.</p>
<nav class="glance">
<a href="#s1"><b>Oct 8 · morning</b>Benchmark defined</a>
<a href="#s2"><b>Oct 8 · 09:10</b>Scene one: no conflict</a>
<a href="#s3"><b>Oct 8 · noon</b>Scene two: conflict found</a>
<a href="#s4"><b>Oct 8 · 09:00–11:10</b>Batch one captured</a>
<a href="#s5"><b>Oct 8 · afternoon</b>First models</a>
<a href="#s6"><b>Oct 8 · 15:50</b>Curation and v3 design</a>
<a href="#s7"><b>Oct 8 · 17:40</b>Contamination and repair</a>
<a href="#s8"><b>Oct 9 · 01:19–16:38</b>v3 batch captured</a>
<a href="#s9"><b>Oct 9 · 04:30</b>Batch one with CIs</a>
<a href="#s10"><b>Oct 9 · 07:50–12:00</b>Task transfer fails; why</a>
<a href="#s11"><b>Oct 9 · 17:00–22:40</b>Settings and wording tests</a>
<a href="#s12"><b>Oct 9 · 19:30–21:45</b>v3 verdict</a>
<a href="#s13"><b>Oct 10</b>Where it stands</a>
</nav>
</header>
'''


def build():
    s = [CSS]
    s.append(stage(1, "Oct 8<br>morning", "The benchmark, redesigned",
        "the earlier result (a JEPA model beating a fixed setting) came entirely from the light panel being scored as a query; the scene was rebuilt around questions a reviewer would accept.",
        '''<p>The research question is unchanged: given one auto-exposure frame of a static scene and a natural-language question, choose a camera setting (ISO × shutter from a fixed grid) so that the question is answerable from the resulting capture, and test whether a JEPA-style model, which predicts the frame's features at any setting from unlabeled photo pairs, beats a direct model that maps frame and query to a setting with no predictor.</p>
<p>What changed after the meeting is the benchmark. The light panel and toys are gone; the iPad alone is the scene, at maximum backlight in a dark room. Questions and images come from TextVQA and VQAv2 with their own answer keys. A vision-language model (Qwen2.5-VL-7B) answers each question on the whole captured iPad frame, no boxes, and the score is answer containment with an edit-distance floor. Every target is later tagged from its own sweep as impossible (no setting answers it), no-conflict (auto exposure already answers it) or conflict (some setting answers it, auto exposure does not); only recoverable targets are evaluated and the counts are reported. Ground truth exists because every setting of every scene is actually captured.</p>
<p>Rig facts that matter for reading the samples: the phone's crop of the iPad is 2085 × 1173 pixels; the iPad leans back about 4.5°, uncorrected by choice; auto exposure in these mostly black scenes lands near ISO 500 to 3200 at 1/33 s, so anything on a white window clips.</p>'''))

    s.append(stage(2, "Oct 8<br>09:10", "Scene one: six real questions, 49 settings, no conflict",
        "a VLM reads far lower contrast than a detector; at these levels every target was answerable at auto exposure, so the scene held no decision to make.",
        img("scene1_0.jpg", (A / "scene1_0.txt").read_text()) + img("scene1_1.jpg", (A / "scene1_1.txt").read_text())
        + img("scene1_strip_0.jpg", "Each row is one target cut from the real captures, darkest setting on the left, brightest on the right. The dim targets are already faintly visible at auto exposure, which is the problem: the VLM answers them there.", wide=True)
        + img("scene1_vlm_0.jpg", "Per-question VLM score over the ISO × shutter grid for scene one (star = auto exposure, dashed = equal exposure line). Yellow nearly everywhere: no conflict.", wide=True)
        + '''<p>Scene one was hand-built with a bright plate and sign on a white region and a dim plate, dim picture and control on black. The 49-setting sweep (plus auto exposure) was EXIF-verified shot by shot. The VLM answered five of the six questions at auto exposure, so the only thing the scene measured was the model's low-light reading ability. That set the rule for everything after it: a scene only enters the benchmark if a three-shot pre-check (dark, auto, bright) shows at least one target that auto exposure misses and another setting answers.</p>'''))

    s.append(stage(3, "Oct 8<br>noon", "Scene two: five iterations to a conflict",
        "with a small white window (about 12% of the iPad) auto exposure protects the black and clips the window; signs at 75 to 82% gray on white and text at 4 to 6% gray on black are then answerable only at settings eight stops apart.",
        img("scene2_0.jpg", "Scene 2e, the version that passed: per-question score over the grid. The bright sign (left) is answered only in the top-left, short exposures; the dim text (right) only at long exposures; the control everywhere. The star marks auto exposure, which misses both.", wide=True)
        + '''<p>Four versions failed the pre-check for opposite reasons: signs at 85 to 95% gray were still read at auto exposure when the white area was large, and signs at 88 to 94% vanished in the dark shot once the window was small; dim text at 3.5% was unreadable at every setting. Version 2e fixed the window at about 12% of the screen, the sign at 80%, the dim text at 4 to 6%, and the control picture at 60%. Its full sweep confirmed a query-dependent conflict under the VLM metric: the per-target oracle settings sit about eight stops apart and the best single fixed setting fails one band or the other. The working levels became the generator's defaults.</p>'''))

    s.append(stage(4, "Oct 8<br>09:00–11:10", "Batch one: 57 pre-checks, 41 pass, 40 swept",
        "199 targets, of which 120 recoverable and 108 conflicts; auto exposure answers 27%, the best fixed setting 52% and sacrifices a whole band (bright 0.89 against dim 0.05, or the reverse).",
        img("b1_example.jpg", "A batch-one scene (qae_0048) at three settings, with the VLM's verdict for its sign, its dim text and its control picture. The sign needs the short exposure, the dim text the long one; no single setting gets both.", wide=True)
        + '''<p>The generator (v2) draws a white window of 12, 20, 30 or 45% of the screen at a random edge, puts a TextVQA image and a 72 to 88% gray sign inside it, and places a dim TextVQA image (5 to 9%), one or two dark strings (4 to 6%), a VQAv2 control (45 to 65%) and a negative question ("is there a dog") on the black. Text in one region shares a grouped read-all-text question scored by containment, so wording never localizes. The pipeline is three decoupled phases because the link to the GPU node drops for minutes at a time: the phone captures pre-checks, a sync loop pushes them and pulls verdicts, and passing scenes get the 15-setting sweep (ISO 100/400/1600 × 1/1000 to 1/4, plus two auto-exposure shots, about 90 s a scene).</p>
<div class="aside">What this batch was later found to contain: 17 of its 40 sweeps had shots of the wrong scene (stage 7). The numbers above are from the contaminated scoring and are kept here because they are what the first models were trained on; the repaired numbers are in stage 7.</div>'''))

    s.append(stage(5, "Oct 8<br>afternoon", "First models and a flat label curve",
        "a text-conditioned JEPA model worked on held-out scenes (0.68 vs direct 0.65 vs fixed 0.49), but a label-efficiency test could not separate JEPA from direct at any number of labeled scenes.",
        '''<p>Features: frozen DINOv2-small patch tokens (64 × 36 × 384) on the iPad crop. Predictor: an action-conditioned transformer that maps the auto-exposure features plus a (from, to) setting pair to the features at the target setting, trained on all photo pairs within a scene; this is the JEPA part. Query heads on top: <em>text</em> (a CLIP text embedding cross-attends over the predicted patch tokens), <em>region</em> (the target box is given instead of language; the upper bound for localization), and <em>direct</em> (the same cross-attention head over the auto-exposure features with the candidate setting appended to the query, no predictor). Each picks one of the 15 settings per target from the auto-exposure shot alone; the real VLM score at that setting is the result. Five-fold cross-validation over scenes.</p>
<div class="wrap"><table><tr><th>batch one, contaminated scoring</th><th>auto exp.</th><th>fixed</th><th>direct</th><th>JEPA text</th><th>region</th><th>oracle</th></tr>
<tr><td>held-out recoverable targets (120)</td><td>0.27</td><td>0.49</td><td class="d">0.65</td><td class="j">0.68</td><td class="r">0.80</td><td>1.00</td></tr>
<tr><td>conflict targets (108)</td><td>0.19</td><td>0.50</td><td class="d">0.62</td><td class="j">0.66</td><td class="r">0.78</td><td>1.00</td></tr></table></div>
<p>The label-efficiency experiment then trained with k = 4, 8, 16 or 32 labeled scenes (three draws each, eight fixed test scenes), comparing JEPA with its predictor on every unlabeled photo, JEPA restricted to the k scenes, and direct. All three were within noise at every k and none rose with k. Diagnosis at the time: the decision is nearly a two-way band rule that four scenes already teach, the test set is too small, and the scenes are too easy. That diagnosis drove the next two stages.</p>'''))

    s.append(stage(6, "Oct 8<br>15:50–17:35", "Curating the questions, designing v3",
        "188 of 234 eligible pool items are answerable by the VLM when shown alone and well exposed; the rest are out. v3 scenes carry three luminance bands and 11 to 15 targets each.",
        '''<p>The pool grew to 360 items (200 TextVQA, 160 VQAv2). Each eligible item was displayed alone at full brightness and photographed twice (auto exposure, and ISO 100 at 1/60) and kept only if the VLM answered it correctly in at least one shot; 33 of the keepers were answerable only at the mid setting, because a lone bright image on black overexposes under auto exposure. Typical drops: a clock face the model misreads, a name tag it cannot resolve.</p>
<p>Generator v3 splits the usable iPad area into a 4 × 2 cell grid. The white window covers one to four contiguous cells. Inside it: a TextVQA image and two signs of different contrast (72 to 80% and 84 to 90% gray). On the black: a mid band (images at 22 to 35%, texts at 24 to 35% gray), a dim band (images at 5 to 9%, texts at 4 to 6% and one rung at 3%), a VQAv2 control at 45 to 65%, extra cells filled with more dim or mid images, and a negative. Dim items sit in cells adjacent to the window. The pre-check grew to four shots (dark, auto, mid, bright) and the gate now requires at least three recoverable targets, the negative holding everywhere, and at least one pair of targets that no single pre-check shot answers together.</p>'''
        + '<div class="two">' + img("v3prev_0.jpg", (A / "v3prev_0.txt").read_text()) + img("v3prev_1.jpg", (A / "v3prev_1.txt").read_text()) + '</div>'))

    s.append(stage(7, "Oct 8<br>17:40–20:15", "A forgotten process, and the repair",
        "a test collector started at 08:40 had been switching the iPad to its own scenes and firing the phone every two minutes all day; 17 of 40 batch-one sweeps held wrong-scene photos. Re-captured, re-scored, re-trained the same evening.",
        img("contamination_pair.jpg", "How it was caught. Two consecutive shots of the same batch-one sweep: the expected scene with its window top-right, then, one minute later, a different scene entirely. A per-shot check (brightness inside the expected window against the rest of the iPad) found 183 of 680 batch-one photos like this.", wide=True)
        + '''<p>The first v3 pre-check showed two different scenes across its four shots, which led to the stray <code>collect.py</code>. The process check before each capture had grepped for the batch and curation scripts by name and missed it. The 17 contaminated sweeps were deleted locally and on the node and re-captured on the clean display with the same scene images and targets (plus one more passing scene), checked again (zero wrong-scene shots), re-scored, and both experiments re-run. The curation pass had only suffered false drops; re-capturing those recovered 14 items (174 to 188). Two rules came out of it: list every Python process before any capture, and run the display check on every batch before scoring.</p>
<div class="wrap"><table><tr><th>batch one</th><th>recoverable</th><th>conflicts</th><th>auto exp.</th><th>fixed</th><th>direct</th><th>JEPA text</th><th>region</th></tr>
<tr><td>contaminated (40 scenes)</td><td>120 / 199</td><td>108</td><td>0.27</td><td>0.50</td><td class="d">0.65</td><td class="j">0.68</td><td class="r">0.80</td></tr>
<tr><td>repaired (41 scenes)</td><td>151 / 205</td><td>124</td><td>0.33</td><td>0.53</td><td class="d">0.78</td><td class="j">0.83</td><td class="r">0.86</td></tr></table></div>
<p>A third of the earlier "impossible" tags had been wrong-scene photos. Every method rose by 0.1 to 0.3 on clean labels, and an anomaly disappeared with them: the head on real features had scored below the head on predicted features (0.55 vs 0.68); on clean labels it is 0.84 vs 0.83, as it should be. The label curve, re-run, stayed flat.</p>'''))

    s.append(stage(8, "Oct 9<br>01:19–16:38", "v3 captured: 301 pre-checks, 244 sweeps, 3,098 targets",
        "the three-way structure is real: bright targets want 1/1000, dim targets 1/4, and the mid band sits between and is mostly answerable at auto exposure, which gives the benchmark the stay-near-auto class it lacked.",
        img("v3_scene_render.jpg", "A v3 scene as rendered for the iPad (seed 0; gamma-boosted here so the dim band is visible). Window of three cells with the plate image and two signs; mid-level books and texts; dim bin picture and strings; the ice-cream truck is the control.")
        + img("v3_precheck_grid.jpg", "The same scene under the four pre-check shots, as the phone sees it. Auto exposure clips the window and leaves the dim strings black; the dark shot reads the signs; the bright shot reads the dim strings and nothing else.", wide=True)
        + img("v3_ladder.jpg", "The sweep at ISO 100 across the five shutters, with the VLM's verdict for three targets of the scene. The bright sign is read at 1/1000 and 1/250, the mid text from 1/250 on, the dim text only at 1/4: no single setting answers all three.", wide=True)
        + '''<p>Capture ran unattended: pre-checks from 01:19 to 06:27 (about 80 s a scene, zero failures), the sweep phase started itself and ran to 16:38 (about 2.5 minutes a scene, zero retakes), and a chain script stopped the loop, ran the display check (4,148 photos, no wrong-scene shot) and launched the model runs. Scoring happened on a 48-hour lease on a single A100 (liqid03 at CHI@TACC; every gigaio node was booked or flagged unreservable) and kept pace with the phone throughout. 245 of 301 scenes passed the gate (81%).</p>'''
        + fig("fig_v3_bands.svg", "Auto exposure, the best fixed setting and the oracle per band, before any learning. The best fixed setting (ISO 100, 1/60) answers the bright band and abandons the dim one.")
        + fig("fig_v3_oracle.svg", "Where each band's oracle setting sits on the grid. Bright at the shortest exposure, dim at the longest, mid spread between, controls short.")
        + '''<p>Of 3,098 positive targets, 2,003 are recoverable (65%), 1,205 are conflicts (39%) and 798 are no-conflict, mostly mid-band texts that auto exposure already reads. The impossible third is the 3% text rung (by design) and the small dim and mid images, which the scorer cannot read at any setting; that is the known limit of the benchmark, it leans toward text.</p>'''))

    s.append(stage(9, "Oct 9<br>04:30–05:45", "Batch one with confidence intervals",
        "JEPA beats direct by +0.052 (95% CI +0.032 to +0.073 over seeds; +0.021 to +0.088 clustered by scene, 40 scenes); the gain is all in the dim band; four times more unlabeled photos do not move the label curve.",
        fig("fig_bands.svg", "Repaired batch one, five seeds. JEPA text above direct everywhere except the control band, where a fixed setting beats both learned models (19 control targets are too few to learn 'stay near auto').")
        + fig("fig_curve.svg", "Label efficiency on batch one with the original unlabeled pool (171 photos) and with the v3 pre-check photos added (684). Flat either way: on batch one the task saturates, the predictor is not short of data.")
        + '''<p>Success rate (target fully answered): fixed 0.50, direct 0.73, JEPA 0.81, region 0.84. With only 40 scenes the scene-clustered interval is wide and JEPA is ahead in just 55% of scenes; batch one shows the direction, v3 carries the weight. On the bright band the two tie (+0.016), because both already choose the short exposure; the edge (+0.098) is where the choice is among long exposures.</p>'''))

    s.append(stage(10, "Oct 9<br>07:50–14:40", "Task transfer fails, for both; the reason, and a side track",
        "neither model transfers to a question family it never saw, with any text encoder or paraphrase augmentation; the query cannot be grounded in the auto-exposure frame where it matters (4% of bright targets); a query-free two-shot bracket answers 92% on batch one.",
        fig("fig_holdout.svg", "Train on one target family, test on the other. Both language models drop to the auto-exposure level with text held out; with image questions held out direct keeps 0.54 and JEPA 0.41 to 0.45. The region model, grounded by a box instead of words, holds 0.66 to 0.74 in both directions, under all three recipes.")
        + '''<p>This was the test closest to the hypothesis (the task changes, the state model stays), and it came back uninformative. The held-out families are not new wordings but new task types with no training example of what they need, so paraphrasing cannot teach them; the sentence encoder and an encoder-only ablation landed in the same place. What the region model shows is that the predictor and the features carry over; the language side does not.</p>'''
        + fig("fig_ground.svg", "Asked for a box per question, the VLM finds 19% of targets in the auto-exposure frame and 4% in the bright band; in the best-exposed frame it finds 68% of text lines. A clipped region has nothing to locate.")
        + '''<p>That measurement says why: the information needed to localize the query is not in the one frame the camera has, precisely for the targets where the setting decision matters. The two ways around it are to predict the frame at other settings and ground in the prediction (the JEPA route), or to spend a second capture.</p>'''
        + fig("fig_budget.svg", "The query-free competitor at equal budget: with two shots and no query, a fixed bracket (ISO 100 1/60 + ISO 400 1/4) answers 92% of batch-one targets; the learned models' top-2 trail it because their second pick is a neighbour of the first.")
        + '''<div class="aside">Side track, kept short: the multi-shot and equal-budget numbers answer a reviewer's question about baselines, not the JEPA-vs-direct one. They fix the framing (query-aware single-shot exposure matters where a second capture is unavailable or costly) and were parked after that; a "diversified" top-B policy under independence turned out to equal top-B exactly.</div>'''))

    s.append(stage(11, "Oct 9<br>17:00–22:40", "The tests that separate state prediction from the task",
        "JEPA's picks of settings it was never supervised on are right 0.93 (batch one) and 0.99 (v3) of the time against direct's 0.79 and 0.66, when the held-out settings lie at the edge of the labeled range; under reworded questions JEPA holds 0.78 vs direct 0.64 (batch one) and 0.73 vs 0.62 (v3).",
        '''<p>After the task-type test, the question was reframed: JEPA predicts the state, the task is read from the query, so the state model should generalize over the <em>action space</em> and the head should be robust to <em>how the task is phrased</em>, without asking for zero-shot task types. Two tests, both JEPA vs direct with the same cross-attention head and the same encoder.</p>'''
        + fig("fig_settings_all.svg", "Held-out settings: the heads never see labels for some settings but may choose them. When a model picks one of those never-labeled settings, how often is it right? First split (five held out, two at the grid's edge): JEPA 0.93, direct 0.79. Checkerboard (seven held out, each between labeled neighbours): no difference. v3, first split, 2,003 targets per seed: JEPA 0.99, direct 0.66.")
        + '''<p>The checkerboard null locates the mechanism. When a held-out setting sits between labeled neighbours, direct's log-scale setting vector interpolates and both manage. When it lies beyond the labeled range, JEPA's action-conditioned predictor still knows what that setting does to the frame and its rare picks of it are essentially always right, while direct's are wrong a third of the time. Overall scores move less (JEPA 0.83 vs direct 0.80 on batch one, 0.85 vs 0.77 on v3) because the never-labeled settings are the oracle for only 15 to 23% of targets.</p>'''
        + fig("fig_wording.svg", "Held-out wording on batch one: trained on the original questions, tested on eight unseen paraphrases of each, score averaged. Direct loses 0.16, JEPA 0.05; on v3 the drops are 0.19 and 0.13, with direct falling below the fixed setting.")
        + '''<p>Both heads see the same query embedding; the difference is what the query attends over. Direct attends over the auto-exposure frame with the setting appended to the query, so when the wording moves the decision moves with it. JEPA's head attends over the predicted state of the candidate setting and its decision is anchored there. Under the strict measure (right under every paraphrase) JEPA is at 0.52 and direct at 0.25 on batch one, 0.44 and 0.19 on v3.</p>'''))

    s.append(stage(12, "Oct 9<br>19:30–21:45", "v3 verdict: five seeds and the label curve",
        "JEPA beats direct by +0.057 on 2,003 held-out targets (95% CI +0.031 to +0.083 across seeds, +0.046 to +0.069 clustering by scene; JEPA ahead in 72% of scenes); clear in the bright, mid and dim bands, marginal on controls; JEPA with 8 labeled scenes matches direct with 64.",
        fig("fig_v3_model.svg", "v3, five seeds × five folds. JEPA text within 0.02 of the model that is given the box; direct five times less stable across seeds, and in some seeds failing to learn the short exposure for the bright band.")
        + '''<div class="wrap"><table><tr><th>v3, 2,003 held-out targets per seed</th><th>auto exp.</th><th>fixed</th><th>direct</th><th>JEPA text</th><th>region</th><th>JEPA minus direct</th></tr>
<tr><td>all</td><td>0.46</td><td>0.67</td><td class="d">0.81 ± 0.02</td><td class="j">0.86 ± 0.00</td><td class="r">0.88</td><td>+0.057 [+0.046, +0.069] by scene; [+0.031, +0.083] by seed</td></tr>
<tr><td>conflict (1,205)</td><td>0.10</td><td>0.56</td><td class="d">0.76 ± 0.04</td><td class="j">0.83</td><td class="r">0.84</td><td>+0.071</td></tr>
<tr><td>bright (606)</td><td>0.10</td><td>0.95</td><td class="d">0.84 ± 0.08</td><td class="j">0.91</td><td class="r">0.94</td><td>+0.076 [+0.060, +0.092]</td></tr>
<tr><td>mid (695)</td><td>0.85</td><td>0.81</td><td class="d">0.79</td><td class="j">0.84</td><td class="r">0.87</td><td>+0.047 [+0.027, +0.069]</td></tr>
<tr><td>dim (501)</td><td>0.27</td><td>0.10</td><td class="d">0.80 ± 0.05</td><td class="j">0.87</td><td class="r">0.86</td><td>+0.061 [+0.036, +0.087]</td></tr>
<tr><td>control (201)</td><td>0.67</td><td>0.80</td><td class="d">0.76</td><td class="j">0.79</td><td class="r">0.78</td><td>+0.029 [+0.002, +0.057]</td></tr></table></div>
<p>On the intervals: the five seeds are the honest unit (each is a full five-fold run with its own split and initialization) and give +0.057 with a t-interval of +0.031 to +0.083; a bootstrap that resamples whole scenes, keeping a scene's targets and seeds together, gives +0.046 to +0.069; JEPA is ahead on average in 72% of the 244 scenes. The per-band intervals in the table are the scene-clustered ones: bright, mid and dim exclude zero comfortably, the control band only just. An earlier version of this page quoted +0.051 to +0.064 from a bootstrap over target evaluations pooled across seeds, which treats five copies of the same target as independent and is too narrow. Success rate: fixed 0.65, direct 0.77, JEPA 0.84, region 0.85. Regret against the oracle: 0.33, 0.19, 0.14, 0.12. At equal budget, JEPA's top-2 (0.90) ties the query-free two-shot bracket (0.91) on v3, where on batch one it trailed it.</p>'''
        + fig("fig_v3_curve.svg", "Label efficiency on v3, 60 test scenes, three seeds. JEPA above direct at every k; the two JEPA variants are level, so the gain comes from the predictor's self-supervised objective on the labeled scenes' own sweeps rather than from extra unlabeled scenes.")
        + '''<p>JEPA at k = 8 (0.81 to 0.82) matches direct at k = 64 (0.83), about an eightfold label saving at equal accuracy. Both curves are still shallow over a 32× range of labels, so the benchmark remains easier than its size suggests, but the ordering is stable and significant at every point.</p>'''))

    s.append(stage(13, "Oct 10", "Where it stands",
        "four replicated results for JEPA over direct, one open problem shared by both, and a benchmark that reads text far better than images.",
        '''<h3>Evidence, JEPA vs direct</h3>
<ul class="ev">
<li><span>in-family, v3</span><span>+0.057 on 2,003 targets (95% CI +0.031 to +0.083 over seeds; +0.046 to +0.069 clustered by scene; ahead in 72% of scenes); clear in bright, mid and dim, marginal on controls; JEPA within 0.02 of the box-given model.</span></li>
<li><span>label efficiency, v3</span><span>JEPA above direct at every k; 8 labeled scenes ≈ direct with 64. Source: the predictor's objective on the labeled scenes' own sweeps.</span></li>
<li><span>held-out settings</span><span>picks of never-labeled, edge-of-range settings right 0.93 (batch one) and 0.99 (v3) of the time vs 0.79 and 0.66 for direct; no gap when held-out settings are interpolable.</span></li>
<li><span>held-out wording</span><span>+0.14 (batch one) and +0.11 (v3) under unseen paraphrases; direct falls below the fixed setting on v3.</span></li>
<li><span>not solved</span><span>zero-shot transfer to a task type never seen: both fail, with any encoder or augmentation; the query cannot be localized in the auto-exposure frame where it matters.</span></li>
</ul>
<h3>Open problems</h3>
<p>The scorer reads text reliably and images poorly (dim and most mid-level images are impossible at every setting), so the benchmark leans toward text and the VQAv2 controls contribute little. The learned models still lose to a fixed setting on controls. A query-free two-shot bracket answers over 90% on both batches, so the single-capture regime has to be argued from the application, or the multi-shot policy made genuinely diversified. The task-type gap needs either training across many task types or a query grounded in a predicted frame rather than in language alone.</p>
'''))

    out = "\n".join(s)
    (SC / "qae_status.html").write_text(out)
    print(len(out) // 1024, "KB")


build()
