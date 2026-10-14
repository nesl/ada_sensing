"""
QAE scene generator v2: bright-window layout with VLM-calibrated levels.

Factors per scene (seeded):
  window_frac   fraction of the iPad covered by the white window: {0.12, 0.20, 0.30, 0.45}
  window_pos    which corner / edge the window sits at
  bright band   one TextVQA image at 100% + one sign at 72-88% gray (inside the window)
  dim band      one TextVQA image at 5-9% + one or two dark strings at 3.8-6% gray
  control       one VQAv2 image at 45-65% (should be answerable at auto exposure)
  negative      "is there a <absent object>" (answer: no)
Text targets inside one region share a "read all the text" question and are
scored by containment, so the VLM is never asked to localize by wording.
"""
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = (2388, 1668)
AREA = (60, 250, 2328, 1300)            # usable area below the status bar
POOL = Path(__file__).resolve().parent.parent / "data" / "vqa_pool"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
WORDS = ["EXIT", "GATE", "YIELD", "LANE", "STOP", "DOCK", "ZONE", "BAY", "PIER", "ROW"]
NEGATIVES = ["a dog", "a cat", "a bicycle", "an umbrella", "a clock", "a boat", "a guitar", "a horse"]


def _pool():
    meta = json.loads((POOL / "pool.json").read_text())
    def agreed(m, thr):
        from collections import Counter
        c = Counter(a.strip().lower() for a in m["answers"] if a.strip())
        a, n = c.most_common(1)[0]
        return (a, n / len(m["answers"])) if c else (None, 0)
    text_items, obj_items = [], []
    for m in meta:
        a, ag = agreed(m, 0.7)
        if not a or ag < 0.7 or len(a) > 20 or a in ("yes", "no", "unanswerable"):
            continue
        answers = sorted({x.strip().lower() for x in m["answers"] if x.strip()}, key=lambda x: -m["answers"].count(x))[:3]
        item = {"file": m["file"], "question": m["question"], "answers": answers, "top": a}
        (text_items if "textvqa" in m["source"] else obj_items).append(item)
    return text_items, obj_items


def _rand_text(rng):
    if rng.random() < 0.5:
        return f"{rng.randint(1, 9)}{''.join(rng.choices(string.ascii_uppercase, k=3))} {rng.randint(100, 999)}"
    return f"{rng.choice(WORDS)} {rng.choice(string.ascii_uppercase)}{rng.randint(1, 9)}"


def make_scene(seed, out_png, H_canvas_to_cam):
    rng = random.Random(seed)
    text_items, obj_items = _pool()
    W, H = CANVAS
    ax0, ay0, ax1, ay1 = AREA
    aw, ah = ax1 - ax0, ay1 - ay0
    canvas = Image.new("RGB", CANVAS, "black")
    d = ImageDraw.Draw(canvas)
    T = []

    # --- window
    frac = rng.choice([0.12, 0.20, 0.30, 0.45])
    ww = int(np.sqrt(frac * aw * ah * 1.1))     # slightly wider than tall
    wh = int(frac * aw * ah / ww)
    ww, wh = min(ww, aw - 40), min(wh, ah - 40)
    pos = rng.choice(["tl", "tr", "bl", "br", "l", "r"])
    wx = ax0 + 20 if "l" in pos else ax1 - 20 - ww
    wy = ay0 + 20 if "t" in pos else (ay1 - 20 - wh if "b" in pos else ay0 + (ah - wh) // 2)
    win = (wx, wy, wx + ww, wy + wh)
    d.rectangle(win, fill=(255, 255, 255))

    def paste(item, box_w, x, y, level):
        im = Image.open(POOL / item["file"]).convert("RGB")
        h = int(im.size[1] * box_w / im.size[0])
        im = im.resize((box_w, h), Image.LANCZOS).point(lambda v, l=level: int(v * l))
        canvas.paste(im, (x, y))
        return [x, y, x + box_w, y + h]

    # bright band: image + sign inside the window
    bi = rng.choice(text_items)
    iw = min(ww - 40, int(ww * 0.9))
    ih = int(Image.open(POOL / bi["file"]).size[1] * iw / Image.open(POOL / bi["file"]).size[0])
    if ih > wh - 140:
        iw = int(iw * (wh - 140) / ih); ih = wh - 140
    box = paste(bi, iw, wx + 20, wy + 20, 1.0)
    T.append({"id": "bright_img", "band": "bright", "kind": "vqa_image", "level": 1.0, "source": bi["file"],
              "question": bi["question"], "answers": bi["answers"], "canvas_box": box})
    g = rng.randint(184, 224)
    s = _rand_text(rng)
    fs = max(40, min(72, (ww - 60) // max(6, len(s)) * 2))
    f = ImageFont.truetype(FONT, fs)
    tx, ty = wx + 24, box[3] + 20
    d.text((tx, ty), s, font=f, fill=(g, g, g)); l, t, r, b = d.textbbox((tx, ty), s, font=f)
    GQB = "Read all the text printed inside the white rectangle."
    T.append({"id": "bright_sign", "band": "bright", "kind": "text", "level": round(g / 255, 3),
              "question": GQB, "group_question": GQB, "answers": [s.lower(), s.lower().replace(" ", "")],
              "canvas_box": [l - 8, t - 8, r + 8, b + 8]})

    # dark region = usable area minus the window: place dim items in the largest free strip
    free = []
    if wx - ax0 > 600: free.append((ax0, ay0, wx - 30, ay1))
    if ax1 - (wx + ww) > 600: free.append((wx + ww + 30, ay0, ax1, ay1))
    if wy - ay0 > 420: free.append((ax0, ay0, ax1, wy - 30))
    if ay1 - (wy + wh) > 420: free.append((ax0, wy + wh + 30, ax1, ay1))
    free.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    fx0, fy0, fx1, fy1 = free[0]
    cols = max(1, (fx1 - fx0) // 700)
    slots = [(fx0 + 20 + i * ((fx1 - fx0 - 40) // cols), fy0 + 20, (fx1 - fx0 - 40) // cols - 20) for i in range(cols)]
    rng.shuffle(slots)
    # dim image
    di = rng.choice([x for x in text_items if x["file"] != bi["file"]])
    sx, sy, sw = slots[0]
    lvl = round(rng.uniform(0.05, 0.09), 3)
    box = paste(di, min(sw, 660), sx, sy, lvl)
    T.append({"id": "dim_img", "band": "dim", "kind": "vqa_image", "level": lvl, "source": di["file"],
              "question": di["question"], "answers": di["answers"], "canvas_box": box})
    # control
    ci = rng.choice(obj_items)
    if len(slots) > 1:
        sx2, sy2, sw2 = slots[1]
        lvl = round(rng.uniform(0.45, 0.65), 3)
        cb = paste(ci, min(sw2, 600), sx2, sy2, lvl)
        T.append({"id": "control_img", "band": "control", "kind": "vqa_image", "level": lvl, "source": ci["file"],
                  "question": ci["question"], "answers": ci["answers"], "canvas_box": cb})
    # dim texts under the dim image
    GQD = "Read all the dark text written on the black background."
    ty = box[3] + 40
    for k in range(rng.randint(1, 2)):
        g = rng.randint(10, 15)
        s = _rand_text(rng)
        f = ImageFont.truetype(FONT, 72)
        if ty + 90 > fy1:
            break
        d.text((sx, ty), s, font=f, fill=(g, g, g)); l, t, r, b = d.textbbox((sx, ty), s, font=f)
        T.append({"id": f"dim_text{k}", "band": "dim", "kind": "text", "level": round(g / 255, 3),
                  "question": GQD, "group_question": GQD, "answers": [s.lower(), s.lower().replace(" ", "")],
                  "canvas_box": [l - 8, t - 8, r + 8, b + 8]})
        ty = b + 36

    used = " ".join(x["question"] + " " + " ".join(x["answers"]) for x in (bi, di, ci)).lower()
    neg = rng.choice([n for n in NEGATIVES if n.split()[-1] not in used])
    T.append({"id": "negative", "band": "none", "kind": "negative", "level": None,
              "question": f"Is there {neg} anywhere in the image? Answer yes or no.", "answers": ["no"], "canvas_box": None})

    canvas.save(out_png)
    Hm = np.array(H_canvas_to_cam, np.float64)
    for t in T:
        if t["canvas_box"]:
            x0, y0, x1, y1 = t["canvas_box"]
            p = cv2.perspectiveTransform(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])[None], Hm)[0]
            t["camera_box"] = [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())]
    return {"window_frac": frac, "window_pos": pos, "window_canvas": list(win), "targets": T}
