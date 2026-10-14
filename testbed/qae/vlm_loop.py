"""
Node-side scoring loop: load the VLM once, then score every sweep directory under
<root> that has manifest.json + targets.json and no vlm.json yet. Polls forever.

  python vlm_loop.py ~/qa/qae_batch --crop 602,854,2687,2027
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vlm_score import MODEL, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+"); ap.add_argument("--crop", default=None)
    a = ap.parse_args()
    crop = tuple(int(v) for v in a.crop.split(",")) if a.crop else None
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print("model ready", flush=True)

    def ask(im, q, group):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                 {"type": "text", "text": q + (" List every piece of text you can read." if group else " Answer with a few words only.")}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=96, do_sample=False)
        return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    while True:
        todo = [p for root in a.roots for p in sorted(Path(root).glob("*/manifest.json"))
                if (p.parent / "targets.json").exists() and not (p.parent / "vlm.json").exists()]
        if not todo:
            time.sleep(15)
            continue
        for mp in todo:
            sweep = mp.parent
            manifest = json.loads(mp.read_text())
            targets = json.loads((sweep / "targets.json").read_text())["targets"]
            if any(not (sweep / s["file"]).exists() for s in manifest["shots"]):
                continue  # still syncing
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
                    if q not in cache:
                        cache[q] = ask(im, q, "group_question" in t)
                    rec["replies"][t["id"]] = cache[q]
                    rec["scores"][t["id"]] = score(cache[q], t)
                rows.append(rec)
            (sweep / "vlm.json").write_text(json.dumps({"sweep": sweep.name, "model": MODEL, "crop": crop,
                                                        "targets": targets, "rows": rows}, indent=1))
            print("scored", sweep.name, len(rows), "shots", flush=True)


if __name__ == "__main__":
    main()
