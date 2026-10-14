"""
Score every shot of every scene in the query-aware dataset.

  python score_dataset.py <dataset_dir> <out.json>

Targets come from each scene's manifest (camera_box). Per target, per shot:
  ipad_image / physical : OWLv2 max confidence for the target's query, on a padded crop
  ipad_text             : EasyOCR match to the true text (1 - normalized edit distance)
  distractor            : fraction of the box not clipped (texture visible)
Scenes already present in <out.json> are skipped, so it can be re-run as data arrives.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from score_sweep import edit_distance, norm, padded_crop


def text_score(reader, crop, truth):
    reads = reader.readtext(np.asarray(crop), detail=1)
    truth = norm(truth)
    best = 0.0
    for _, txt, _ in reads:
        best = max(best, 1 - edit_distance(norm(txt), truth) / max(len(truth), 1))
    full = norm(" ".join(r[1] for r in reads))
    if full:
        best = max(best, 1 - edit_distance(full, truth) / max(len(truth), len(full)))
    return max(0.0, best)


def main():
    root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval()
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    for mpath in sorted(root.glob("*/manifest.json")):
        m = json.loads(mpath.read_text())
        if m["scene"] in results:
            continue
        targets = m["targets"]
        for i, t in enumerate(targets):
            t.setdefault("id", f"t{i}")
        rows = []
        for shot in m["shots"]:
            im = Image.open(mpath.parent / shot["file"]).convert("RGB")
            arr = np.asarray(im)
            sc = {}
            det = [t for t in targets if t["kind"] in ("ipad_image", "physical")]
            if det:
                crops = [padded_crop(im, t["camera_box"]) for t in det]
                with torch.no_grad():
                    inp = proc(text=[[t["query"]] for t in det], images=crops, return_tensors="pt").to("cuda")
                    logits = owl(**inp).logits
                for t, lg in zip(det, logits):
                    sc[t["id"]] = float(torch.sigmoid(lg).max())
            for t in targets:
                if t["kind"] == "ipad_text":
                    sc[t["id"]] = round(text_score(reader, padded_crop(im, t["camera_box"]), t["text"]), 4)
                elif t["kind"] == "distractor":
                    x0, y0, x1, y1 = t["camera_box"]
                    sc[t["id"]] = round(float((arr[y0:y1, x0:x1].max(axis=2) < 250).mean()), 4)
            rows.append({"label": shot["label"], "mode": shot["mode"], "iso": shot["exif"]["iso"],
                         "exposure_s": shot["exif"]["exposure_s"], "scores": sc})
        results[m["scene"]] = {"targets": targets, "rows": rows}
        out_path.write_text(json.dumps(results))
        print(m["scene"], len(rows), "shots,", len(targets), "targets", flush=True)
    print("SCORING_DONE", len(results))


if __name__ == "__main__":
    main()
