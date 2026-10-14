"""
Can a VLM locate the region a question refers to, in the auto-exposure frame (and in the
oracle-exposed frame for comparison)? For every positive target of every scored sweep,
ask Qwen2.5-VL for a bounding box and report IoU against the known camera box.

  python ground_eval.py <sweeps_dir> <out.json> [--prefix qae_] [--crop x0,y0,x1,y1] [--limit N]
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CAND = [(iso, sh) for iso in (100, 400, 1600) for sh in (1 / 1000, 1 / 250, 1 / 60, 1 / 15, 1 / 4)]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ask_box(proc, model, img, question):
    prompt = (f"The question below refers to one object, text, or region in this image. Locate that region and output "
              f"its bounding box as JSON: {{\"bbox_2d\": [x1, y1, x2, y2]}}. Output only the JSON.\n\nQuestion: {question}")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=64, do_sample=False)
    reply = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", reply)
    if not m:
        return None, reply
    box = [int(v) for v in m.groups()]
    # Qwen2.5-VL coordinates are in the processor's resized image space
    thw = inp["image_grid_thw"][0].tolist(); rh, rw = thw[1] * 14, thw[2] * 14
    sx, sy = img.size[0] / rw, img.size[1] / rh
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps"); ap.add_argument("out"); ap.add_argument("--prefix", default="qae_")
    ap.add_argument("--crop", default="602,854,2687,2027"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cx0, cy0, cx1, cy1 = (int(v) for v in a.crop.split(","))
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    rows = []
    dirs = sorted(Path(a.sweeps).glob(a.prefix + "*"))
    if a.limit:
        dirs = dirs[: a.limit]
    for d in dirs:
        if not (d / "vlm.json").exists():
            continue
        v = json.loads((d / "vlm.json").read_text()); byl = {r["label"]: r for r in v["rows"]}
        shots = {r["label"]: r for r in v["rows"] if r["mode"] == "manual"}
        for t in v["targets"]:
            if t["kind"] == "negative" or not t.get("camera_box"):
                continue
            gt = [t["camera_box"][0] - cx0, t["camera_box"][1] - cy0, t["camera_box"][2] - cx0, t["camera_box"][3] - cy0]
            best_label, best_s = None, -1
            for lab, r in shots.items():
                if r["scores"][t["id"]] > best_s:
                    best_s, best_label = r["scores"][t["id"]], lab
            rec = {"scene": d.name, "id": t["id"], "band": t["band"], "kind": t["kind"], "recoverable": best_s >= 1}
            for which, lab in (("ae", "ae_start"), ("oracle", best_label)):
                img = Image.open(d / f"{lab}.jpg").convert("RGB").crop((cx0, cy0, cx1, cy1))
                q = t.get("group_question", t["question"]) if t["kind"] == "text" else t["question"]
                if t["kind"] == "text":   # grouped questions name the region, not the line: ask for the line itself
                    q = f"the text \"{t['answers'][0]}\""
                box, reply = ask_box(proc, model, img, q)
                rec[which] = {"box": box, "iou": iou(box, gt) if box else 0.0, "reply": reply[:80]}
            rows.append(rec)
        print(f"{d.name}: {len(rows)} targets so far; mean IoU ae {sum(r['ae']['iou'] for r in rows)/len(rows):.2f} "
              f"oracle {sum(r['oracle']['iou'] for r in rows)/len(rows):.2f}", flush=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
    print("GROUND_DONE", flush=True)


if __name__ == "__main__":
    main()
