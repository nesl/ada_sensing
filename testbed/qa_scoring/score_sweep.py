"""
Score every shot of a sweep against per-target queries.

  python score_sweep.py <sweep_dir> <targets.json> <out.json>

Per target, per shot:
  ipad_car / physical : OWLv2 max confidence for the target's text query, run on
                        a padded crop around the target box
  ipad_text           : EasyOCR on the crop; score = 1 - normalized edit distance
                        between the best read and the true text (0 if nothing read)
  distractor (panel)  : fraction of pixels in the box below clipping (texture visible)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

PAD = 0.30


def padded_crop(im, box):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    W, H = im.size
    return im.crop((max(0, int(x0 - PAD * w)), max(0, int(y0 - PAD * h)),
                    min(W, int(x1 + PAD * w)), min(H, int(y1 + PAD * h))))


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def norm(s):
    return "".join(ch for ch in s.upper() if ch.isalnum())


def main():
    sweep, targets_path, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    manifest = json.loads((sweep / "manifest.json").read_text())
    targets = json.loads(Path(targets_path).read_text())["targets"]

    dev = "cuda"
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    rows = []
    for shot in manifest["shots"]:
        im = Image.open(sweep / shot["file"]).convert("RGB")
        arr = np.asarray(im)
        rec = {"label": shot["label"], "mode": shot["mode"],
               "iso": shot["exif"]["iso"], "exposure_s": shot["exif"]["exposure_s"], "scores": {}}
        det = [t for t in targets if t["kind"] in ("ipad_car", "ipad_image", "physical")]
        if det:
            crops = [padded_crop(im, t["box"]) for t in det]
            queries = [[t["query"]] for t in det]
            with torch.no_grad():
                inp = proc(text=queries, images=crops, return_tensors="pt").to(dev)
                logits = owl(**inp).logits  # (n, patches, 1)
            for t, lg in zip(det, logits):
                rec["scores"][t["id"]] = float(torch.sigmoid(lg).max())
        for t in targets:
            if t["kind"] == "ipad_text":
                crop = np.asarray(padded_crop(im, t["box"]))
                reads = reader.readtext(crop, detail=1)
                truth = norm(t["text"])
                best, best_read = 0.0, ""
                for _, txt, conf in reads:
                    s = 1 - edit_distance(norm(txt), truth) / max(len(truth), 1)
                    if s > best:
                        best, best_read = max(0.0, s), txt
                full = norm(" ".join(r[1] for r in reads))
                if full:
                    s = 1 - edit_distance(full, truth) / max(len(truth), len(full))
                    if s > best:
                        best, best_read = max(0.0, s), " ".join(r[1] for r in reads)
                rec["scores"][t["id"]] = round(best, 4)
                rec.setdefault("ocr", {})[t["id"]] = best_read
            elif t["kind"] == "distractor":
                x0, y0, x1, y1 = t["box"]
                region = arr[y0:y1, x0:x1].max(axis=2)
                rec["scores"][t["id"]] = round(float((region < 250).mean()), 4)
        rows.append(rec)
        print(rec["label"], {k: round(v, 3) for k, v in rec["scores"].items()}, flush=True)

    Path(out_path).write_text(json.dumps({"sweep": sweep.name, "targets": targets, "rows": rows}, indent=1))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
