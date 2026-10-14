"""
Curation pass: can the VLM answer each pool item at all, when the item is shown
alone, large, and well exposed? Keeps the benchmark free of questions that fail
for reasons unrelated to exposure.

  python3 qae/curate.py capture          # phone: display each item alone, 2 shots (AE + ISO100 1/60)
  python3 qae/curate.py verdict          # after the node has scored: write data/vqa_pool/curated.json

Capture dirs go to data/qae_batch/precheck/cur_<item>/ so the existing sync loop
and node vlm_loop score them (manifest.json + targets.json -> vlm.json).
"""
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "qae"))
import qa_sweep as Q  # noqa: E402
from batch import phone_ready, PRE  # noqa: E402
from PIL import Image  # noqa: E402

POOL = ROOT / "data" / "vqa_pool"
CANVAS = (2388, 1668)


def norm_answers(answers):
    c = Counter(a.strip().lower() for a in answers if a.strip())
    top, n = c.most_common(1)[0]
    keep = [a for a, k in c.most_common(4) if k >= 2 or a == top]
    return keep, n / max(1, len(answers))


def show(name):
    req = urllib.request.Request("http://127.0.0.1:8765/api/cmd", data=json.dumps(
        {"action": "image", "video": name, "color": "#000"}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=5).read()


def capture():
    meta = json.loads((POOL / "pool.json").read_text())
    phone_ready(33.0)
    for m in meta:
        key = m["file"].replace(".jpg", "")
        out = PRE / f"cur_{key}"
        if (out / "manifest.json").exists():
            continue
        answers, agree = norm_answers(m["answers"])
        if agree < 0.5 or answers[0] in ("yes", "no", "unanswerable"):
            continue
        # render the item alone, large, at full brightness on black
        im = Image.open(POOL / m["file"]).convert("RGB")
        w = 1500; h = int(im.size[1] * w / im.size[0])
        if h > 1300:
            h = 1300; w = int(im.size[0] * h / im.size[1])
        canvas = Image.new("RGB", CANVAS, "black"); canvas.paste(im.resize((w, h), Image.LANCZOS), ((CANVAS[0] - w) // 2, 250 + (1300 - h) // 2))
        png = f"qae_cur_{key}.png"; canvas.save(ROOT / "data" / png)
        out.mkdir(parents=True, exist_ok=True)
        t = {"id": "item", "band": "curation", "kind": "vqa_image", "level": 1.0, "question": m["question"], "answers": answers, "canvas_box": None, "camera_box": None}
        (out / "targets.json").write_text(json.dumps({"scene": out.name, "targets": [t]}, indent=1))
        try:
            show(png); time.sleep(2.0)
            shots = []
            Q.ctrl(ae=True); r = Q.shoot(out, "ae", settle_s=2.5); shots.append({"label": "ae", "mode": "ae", **r})
            mid = lambda: Q.ctrl(ae=False, iso=100, shutter_ns=int(1e9 / 60))
            mid(); r = Q.shoot(out, "mid", settle_s=2.5, expect=(100, 1 / 60), resend=mid)
            shots.append({"label": "mid", "mode": "manual", "iso": 100, "shutter_s": 1 / 60, **r})
            Q.ctrl(ae=True)
            (out / "manifest.json").write_text(json.dumps({"scene": out.name, "item": m["file"], "shots": shots}, indent=1))
            print(f"captured {key}: {m['question'][:50]!r} -> {answers[:2]}", flush=True)
        except Exception as e:
            print(f"{key}: FAILED {e}", flush=True)
            for f in out.glob("*"):
                f.unlink()
            phone_ready(33.0)
    print("CURATION_CAPTURE_DONE", flush=True)


def verdict():
    meta = {m["file"]: m for m in json.loads((POOL / "pool.json").read_text())}
    kept, dropped = [], []
    for d in sorted(PRE.glob("cur_*")):
        if not (d / "vlm.json").exists():
            continue
        v = json.loads((d / "vlm.json").read_text()); rows = {r["label"]: r for r in v["rows"]}
        s = {l: rows[l]["scores"]["item"] for l in rows}
        key = d.name[4:] + ".jpg"; m = meta[key]
        rec = {**m, "answers_norm": v["targets"][0]["answers"], "cur_scores": s,
               "cur_reply": {l: rows[l]["replies"]["item"] for l in rows}}
        (kept if max(s.values()) >= 1 else dropped).append(rec)
    json.dump(kept, open(POOL / "curated.json", "w"), indent=1)
    print(f"curated: kept {len(kept)}, dropped {len(dropped)}  (of {len(kept)+len(dropped)} scored)")
    print("  kept by source:", Counter(k["source"] for k in kept))


if __name__ == "__main__":
    {"capture": capture, "verdict": verdict}[sys.argv[1]]()
