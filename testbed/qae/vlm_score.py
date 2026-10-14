"""
Score a sweep with a vision-language model answering each target's question
from the WHOLE captured frame (cropped to the iPad region; no target boxes).

  python vlm_score.py <sweep_dir> <targets.json> <out.json> [--crop x0,y0,x1,y1]

score: text/vqa -> 1 if a known answer is contained in the normalized reply,
       else 1 - normalized edit distance to the closest answer (ANLS-style, floor 0.5 -> 0)
       negative -> 1 if reply starts with "no"
"""
import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def edit(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def score(reply, t):
    r = norm(reply)
    if t["kind"] == "negative":
        return 1.0 if r.startswith("no") else 0.0
    # candidate strings: the whole reply plus each listed line/item
    cands = [r] + [norm(x.strip("-*• ")) for x in re.split(r"[\n,;]", reply) if x.strip("-*• ").strip()]
    best = 0.0
    for a in t["answers"]:
        a = norm(a)
        if not a:
            continue
        if a in r:
            return 1.0
        for cnd in cands:
            d = 1 - edit(a, cnd) / max(len(a), len(cnd), 1)
            best = max(best, d if d >= 0.5 else 0.0)
    return round(best, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep"); ap.add_argument("targets"); ap.add_argument("out")
    ap.add_argument("--crop", default=None)
    a = ap.parse_args()
    sweep = Path(a.sweep)
    manifest = json.loads((sweep / "manifest.json").read_text())
    targets = json.loads(Path(a.targets).read_text())["targets"]
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for shot in manifest["shots"]:
        im = Image.open(sweep / shot["file"]).convert("RGB")
        if crop:
            im = im.crop(crop)
        im.thumbnail((1280, 1280))
        rec = {"label": shot["label"], "mode": shot["mode"], "iso": shot["exif"]["iso"],
               "exposure_s": shot["exif"]["exposure_s"], "scores": {}, "replies": {}}
        cache = {}
        for t in targets:
            q = t.get("group_question", t["question"])
            if q in cache:
                reply = cache[q]
                rec["replies"][t["id"]] = reply
                rec["scores"][t["id"]] = score(reply, t)
                continue
            msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                     {"type": "text", "text": q + (" List every piece of text you can read." if "group_question" in t else " Answer with a few words only.")}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=96, do_sample=False)
            reply = proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            cache[q] = reply
            rec["replies"][t["id"]] = reply
            rec["scores"][t["id"]] = score(reply, t)
        rows.append(rec)
        print(shot["label"], {k: (round(v, 2), rec["replies"][k][:18]) for k, v in rec["scores"].items()}, flush=True)
    Path(a.out).write_text(json.dumps({"sweep": sweep.name, "model": MODEL, "crop": crop,
                                       "targets": targets, "rows": rows}, indent=1))
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
