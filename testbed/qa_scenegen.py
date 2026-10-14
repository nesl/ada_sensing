"""
Random iPad scenes with known targets, for the query-aware exposure dataset.

Each scene: 1-4 ImageNet images and 1-3 text strings, each at its own random
brightness, placed without overlap in the unoccluded part of the iPad. Target
boxes are recorded in canvas pixels and mapped to camera pixels with the layout's
homography (qa_calib.py).
"""
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = (2388, 1668)
# usable area: below the status bar, above the strip hidden by the toy car / figurine
REGION = (80, 250, 2308, 1250)
POOL = Path("/Users/malikh/Desktop/ada_sensing/data/ImageNet-ES-Diverse/es-train/tin_no_resize_sample_removed")
CLASSES = Path("/Users/malikh/Desktop/ada_sensing/data/imagenet_class_index.json")
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"]
WORDS = ["STOP", "EXIT", "GATE", "YIELD", "PARKING", "NO ENTRY", "SCHOOL", "BUS", "TAXI",
         "OPEN", "CLOSED", "WARNING", "DANGER", "LANE", "ONE WAY", "DETOUR", "SLOW"]


def _names():
    d = json.loads(CLASSES.read_text())
    return {v[0]: v[1].replace("_", " ") for v in d.values()}


def random_text(rng):
    kind = rng.random()
    if kind < 0.4:   # license-plate-like
        return (f"{rng.randint(1, 9)}{''.join(rng.choices(string.ascii_uppercase, k=3))} "
                f"{rng.randint(100, 999)}")
    if kind < 0.7:   # word + number
        return f"{rng.choice(WORDS)} {rng.randint(1, 99)}"
    return rng.choice(WORDS)


def _place(rng, w, h, taken, tries=200):
    x0, y0, x1, y1 = REGION
    for _ in range(tries):
        x = rng.randint(x0, x1 - w)
        y = rng.randint(y0, y1 - h)
        box = (x, y, x + w, y + h)
        if all(box[2] + 40 < b[0] or b[2] + 40 < box[0] or box[3] + 40 < b[1] or b[3] + 40 < box[1]
               for b in taken):
            return box
    return None


def make_scene(seed, out_png, H_canvas_to_cam):
    rng = random.Random(seed)
    names = _names()
    wnids = sorted(p.name for p in POOL.iterdir() if p.is_dir())
    canvas = Image.new("RGB", CANVAS, "black")
    draw = ImageDraw.Draw(canvas)
    taken, targets = [], []

    for _ in range(rng.randint(1, 4)):
        wnid = rng.choice(wnids)
        src = rng.choice(sorted((POOL / wnid).glob("*.JPEG")))
        im = Image.open(src).convert("RGB")
        w = rng.randint(380, 760)
        h = int(im.size[1] * w / im.size[0])
        max_h = (REGION[3] - REGION[1]) - 20
        if h > max_h:   # tall images: scale down to fit the usable band
            w, h = int(w * max_h / h), max_h
        box = _place(rng, w, h, taken)
        if box is None:
            continue
        level = round(rng.uniform(0.15, 1.0), 3)
        canvas.paste(im.resize((w, h), Image.LANCZOS).point(lambda v, l=level: int(v * l)), box[:2])
        taken.append(box)
        targets.append({"kind": "ipad_image", "wnid": wnid, "query": ("an " if names.get(wnid, wnid)[0] in "aeiou" else "a ") + names.get(wnid, wnid),
                        "source": src.name, "level": level, "canvas_box": list(box)})

    for _ in range(rng.randint(1, 3)):
        txt = random_text(rng)
        size = rng.randint(70, 150)
        font = ImageFont.truetype(rng.choice(FONTS), size)
        l, t, r, b = draw.textbbox((0, 0), txt, font=font)
        box = _place(rng, r - l + 20, b - t + 20, taken)
        if box is None:
            continue
        level = round(rng.uniform(0.15, 1.0), 3)
        g = int(255 * level)
        draw.text((box[0] + 10 - l, box[1] + 10 - t), txt, font=font, fill=(g, g, g))
        taken.append(box)
        targets.append({"kind": "ipad_text", "text": txt, "level": level, "canvas_box": list(box)})

    canvas.save(out_png)
    Hm = np.array(H_canvas_to_cam, np.float64)
    for t in targets:
        x0, y0, x1, y1 = t["canvas_box"]
        pts = cv2.perspectiveTransform(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])[None], Hm)[0]
        t["camera_box"] = [int(pts[:, 0].min()), int(pts[:, 1].min()),
                           int(pts[:, 0].max()), int(pts[:, 1].max())]
    return targets


def make_scene_bd(seed, out_png, H_canvas_to_cam):
    """Two-band scene: one half a white block with bright content, the other half black
    with dim content. Both halves hold legitimate queries that want opposite exposures."""
    rng = random.Random(seed)
    names = _names()
    wnids = sorted(p.name for p in POOL.iterdir() if p.is_dir())
    canvas = Image.new("RGB", CANVAS, "black")
    draw = ImageDraw.Draw(canvas)
    W, H = CANVAS
    bright_left = rng.random() < 0.5
    bands = {"bright": (60, 250, W // 2 - 40, 1300), "dim": (W // 2 + 40, 250, W - 60, 1300)}
    if not bright_left:
        bands = {"bright": bands["dim"], "dim": bands["bright"]}
    draw.rectangle(bands["bright"], fill=(255, 255, 255))
    targets = []
    for band, (x0, y0, x1, y1) in bands.items():
        taken = []
        n_img = rng.randint(1, 2)
        for _ in range(n_img):
            wnid = rng.choice(wnids)
            src = rng.choice(sorted((POOL / wnid).glob("*.JPEG")))
            im = Image.open(src).convert("RGB")
            w = rng.randint(420, 620)
            h = int(im.size[1] * w / im.size[0])
            if h > 620:
                w, h = int(w * 620 / h), 620
            box = None
            for _ in range(200):
                x = rng.randint(x0 + 20, x1 - w - 20); y = rng.randint(y0 + 20, y1 - 220 - h)
                b = (x, y, x + w, y + h)
                if all(b[2] + 30 < t[0] or t[2] + 30 < b[0] or b[3] + 30 < t[1] or t[3] + 30 < b[1] for t in taken):
                    box = b; break
            if box is None:
                continue
            level = round(rng.uniform(0.7, 1.0), 3) if band == "bright" else round(rng.uniform(0.06, 0.12), 3)
            canvas.paste(im.resize((w, h), Image.LANCZOS).point(lambda v, l=level: int(v * l)), box[:2])
            taken.append(box)
            targets.append({"kind": "ipad_image", "band": band, "wnid": wnid,
                            "query": ("an " if names.get(wnid, wnid)[0] in "aeiou" else "a ") + names.get(wnid, wnid),
                            "source": src.name, "level": level, "canvas_box": list(box)})
        txt = random_text(rng)
        font = ImageFont.truetype(rng.choice(FONTS), rng.randint(80, 130))
        l, t, r, b = draw.textbbox((0, 0), txt, font=font)
        tw, th = r - l + 20, b - t + 20
        if tw > (x1 - x0 - 40):
            continue
        x = rng.randint(x0 + 20, x1 - tw - 20); y = rng.randint(y1 - 210, y1 - th)
        if band == "bright":
            g = rng.randint(90, 150); level = round(g / 255, 3)
        else:
            g = rng.randint(11, 24); level = round(g / 255, 3)
        draw.text((x + 10 - l, y + 10 - t), txt, font=font, fill=(g, g, g))
        targets.append({"kind": "ipad_text", "band": band, "text": txt, "level": level,
                        "canvas_box": [x, y, x + tw, y + th]})
    canvas.save(out_png)
    Hm = np.array(H_canvas_to_cam, np.float64)
    for tg in targets:
        x0, y0, x1, y1 = tg["canvas_box"]
        pts = cv2.perspectiveTransform(np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])[None], Hm)[0]
        tg["camera_box"] = [int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())]
    return targets
