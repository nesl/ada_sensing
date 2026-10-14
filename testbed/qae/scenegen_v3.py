"""
QAE scene generator v3: three luminance bands on a cell grid, several targets per
band at different contrast, embedded placement (dim items next to the window).

Grid: the usable iPad area is split into 4 x 2 cells. The white window covers a
contiguous block of 1..4 cells (frac 0.125 .. 0.5); every other cell is black.

Targets (all images come from the curated pool when data/vqa_pool/curated.json exists):
  bright  TextVQA image at 100% inside the window
          two signs on white: A at 72-80% gray, B at 84-90% gray (B is the harder one)
  mid     TextVQA image at 22-35%, one text at 24-35% gray, in a cell adjacent to the window
  dim     TextVQA image at 5-9%, text A at 4-6% gray, text B at 2.8-3.4% gray
  control VQAv2 image at 45-65% (answerable at auto exposure)
  negative "is there <absent object>" (answer: no)
Texts on the black background share one grouped read-all question; signs inside
the window share another. Scoring is by containment, so wording never localizes.
"""
import json
import random
import string
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS = (2388, 1668)
AREA = (60, 250, 2328, 1300)
COLS, ROWS = 4, 2
POOL = Path(__file__).resolve().parent.parent / "data" / "vqa_pool"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
WORDS = ["EXIT", "GATE", "YIELD", "LANE", "STOP", "DOCK", "ZONE", "BAY", "PIER", "ROW", "HALL", "DESK"]
NEGATIVES = ["a dog", "a cat", "a bicycle", "an umbrella", "a clock", "a boat", "a guitar", "a horse", "a banana", "a kite"]
GQ_WINDOW = "Read all the text printed inside the white rectangle."
GQ_BLACK = "Read all the text written directly on the black background, outside the white rectangle."
WINDOW_SHAPES = [(1, 1), (2, 1), (1, 2), (3, 1), (2, 2)]     # (cols, rows) -> frac 0.125 .. 0.5


def _pool():
    cur = POOL / "curated.json"
    if cur.exists():
        meta = json.loads(cur.read_text())
        items = [{"file": m["file"], "question": m["question"], "answers": m["answers_norm"], "source": m["source"]} for m in meta]
    else:   # fallback: agreement filter on the raw pool
        items = []
        for m in json.loads((POOL / "pool.json").read_text()):
            c = Counter(a.strip().lower() for a in m["answers"] if a.strip())
            if not c:
                continue
            top, n = c.most_common(1)[0]
            if n / len(m["answers"]) < 0.7 or len(top) > 20 or top in ("yes", "no", "unanswerable"):
                continue
            items.append({"file": m["file"], "question": m["question"], "answers": [a for a, _ in c.most_common(3)], "source": m["source"]})
    text_items = [x for x in items if "textvqa" in x["source"]]
    obj_items = [x for x in items if "textvqa" not in x["source"]]
    return text_items, obj_items


def _rand_text(rng):
    if rng.random() < 0.5:
        return f"{rng.randint(1, 9)}{''.join(rng.choices(string.ascii_uppercase, k=3))} {rng.randint(100, 999)}"
    return f"{rng.choice(WORDS)} {rng.choice(string.ascii_uppercase)}{rng.randint(1, 9)}"


def make_scene(seed, out_png, H_canvas_to_cam):
    rng = random.Random(seed)
    text_items, obj_items = _pool()
    ax0, ay0, ax1, ay1 = AREA
    cw, ch = (ax1 - ax0) // COLS, (ay1 - ay0) // ROWS
    canvas = Image.new("RGB", CANVAS, "black")
    d = ImageDraw.Draw(canvas)
    T = []

    def cell_rect(c, r):
        return (ax0 + c * cw, ay0 + r * ch, ax0 + (c + 1) * cw, ay0 + (r + 1) * ch)

    # --- window: contiguous block of cells
    wc, wr = rng.choice(WINDOW_SHAPES)
    c0, r0 = rng.randint(0, COLS - wc), rng.randint(0, ROWS - wr)
    wcells = {(c, r) for c in range(c0, c0 + wc) for r in range(r0, r0 + wr)}
    wx0, wy0 = cell_rect(c0, r0)[:2]; wx1, wy1 = cell_rect(c0 + wc - 1, r0 + wr - 1)[2:]
    win = (wx0 + 12, wy0 + 12, wx1 - 12, wy1 - 12)
    d.rectangle(win, fill=(255, 255, 255))
    frac = round(wc * wr / (COLS * ROWS), 3)

    def paste(item, x, y, max_w, max_h, level):
        im = Image.open(POOL / item["file"]).convert("RGB")
        s = min(max_w / im.size[0], max_h / im.size[1])
        w, h = int(im.size[0] * s), int(im.size[1] * s)
        im = im.resize((w, h), Image.LANCZOS).point(lambda v, l=level: int(v * l))
        canvas.paste(im, (x, y))
        return [x, y, x + w, y + h]

    def text(x, y, s, g, fs, max_w):
        fs = max(36, min(fs, int(fs * max_w / max(1, d.textlength(s, font=ImageFont.truetype(FONT, fs))))))
        f = ImageFont.truetype(FONT, fs)
        d.text((x, y), s, font=f, fill=(g, g, g))
        l, t, r, b = d.textbbox((x, y), s, font=f)
        return [l - 8, t - 8, r + 8, b + 8]

    used_files = set()

    def pick(items):
        it = rng.choice([x for x in items if x["file"] not in used_files]); used_files.add(it["file"]); return it

    # --- bright band inside the window: image + two signs of different contrast
    ww, wh = win[2] - win[0], win[3] - win[1]
    bi = pick(text_items)
    wide = ww > 1.6 * wh           # signs beside the image in a wide window, below it otherwise
    if wide:
        box = paste(bi, win[0] + 24, win[1] + 24, ww // 2 - 48, wh - 48, 1.0)
        sx, ty, smax = box[2] + 40, win[1] + 40, win[2] - box[2] - 64
    else:
        box = paste(bi, win[0] + 24, win[1] + 24, ww - 48, wh - 48 - 2 * 76, 1.0)
        sx, ty, smax = win[0] + 24, box[3] + 16, ww - 48
    T.append({"id": "bright_img", "band": "bright", "kind": "vqa_image", "level": 1.0, "source": bi["file"],
              "question": bi["question"], "answers": bi["answers"], "canvas_box": box})
    for k, (lo, hi) in enumerate(((0.72, 0.80), (0.84, 0.90))):
        g = int(255 * rng.uniform(lo, hi)); s = _rand_text(rng)
        bb = text(sx, ty, s, g, 60, smax)
        T.append({"id": f"bright_sign{k}", "band": "bright", "kind": "text", "level": round(g / 255, 3),
                  "question": GQ_WINDOW, "group_question": GQ_WINDOW, "answers": [s.lower(), s.lower().replace(" ", "")], "canvas_box": bb})
        ty = bb[3] + 12

    # --- black cells: those sharing an edge with the window first (embedded placement)
    others = [(c, r) for c in range(COLS) for r in range(ROWS) if (c, r) not in wcells]
    adj = [cr for cr in others if any(abs(cr[0] - w[0]) + abs(cr[1] - w[1]) == 1 for w in wcells)]
    far = [cr for cr in others if cr not in adj]
    rng.shuffle(adj); rng.shuffle(far)
    order = adj + far                   # dim first, so the dim band sits against the window
    plan = ["dim", "mid", "control", "texts"] + ["extra"] * (len(order) - 4)

    def cell_image(cr, item, level, tid, band):
        x0, y0, x1, y1 = cell_rect(*cr)
        bx = paste(item, x0 + 20, y0 + 20, x1 - x0 - 40, y1 - y0 - 40 - 100, level)
        T.append({"id": tid, "band": band, "kind": "vqa_image", "level": level, "source": item["file"],
                  "question": item["question"], "answers": item["answers"], "canvas_box": bx})
        return bx

    def black_text(x, y, g, tid, band, max_w):
        s = _rand_text(rng)
        bb = text(x, y, s, g, 68, max_w)
        T.append({"id": tid, "band": band, "kind": "text", "level": round(g / 255, 3),
                  "question": GQ_BLACK, "group_question": GQ_BLACK, "answers": [s.lower(), s.lower().replace(" ", "")], "canvas_box": bb})
        return bb

    for cr, role in zip(order, plan):
        x0, y0, x1, y1 = cell_rect(*cr)
        if role == "dim":
            lvl = round(rng.uniform(0.05, 0.09), 3)
            bx = cell_image(cr, pick(text_items), lvl, "dim_img", "dim")
            black_text(x0 + 20, bx[3] + 16, rng.randint(10, 15), "dim_text0", "dim", x1 - x0 - 40)
        elif role == "mid":
            lvl = round(rng.uniform(0.22, 0.35), 3)
            bx = cell_image(cr, pick(text_items), lvl, "mid_img", "mid")
            black_text(x0 + 20, bx[3] + 16, rng.randint(60, 90), "mid_text0", "mid", x1 - x0 - 40)
        elif role == "control":
            lvl = round(rng.uniform(0.45, 0.65), 3)
            cell_image(cr, pick(obj_items), lvl, "control_img", "control")
        elif role == "texts":
            ty = y0 + 40
            specs = [("dim_text1", "dim", rng.randint(10, 15)), ("dim_text2", "dim", rng.randint(7, 9)), ("mid_text1", "mid", rng.randint(60, 90))]
            rng.shuffle(specs)
            for tid, band, g in specs:
                bb = black_text(x0 + 20, ty, g, tid, band, x1 - x0 - 40)
                ty = bb[3] + 40
        elif role == "extra" and text_items:
            # extra cells hold one more image in a random band: more targets per frame
            band = rng.choice(["dim", "mid"])
            lvl = round(rng.uniform(0.05, 0.09) if band == "dim" else rng.uniform(0.22, 0.35), 3)
            n = sum(1 for t in T if t["id"].startswith(band + "_img"))
            cell_image(cr, pick(text_items), lvl, f"{band}_img{n}", band)

    used = " ".join(x["question"] + " " + " ".join(x["answers"]) for x in text_items + obj_items if x["file"] in used_files).lower()
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
    return {"gen": "v3", "window_frac": frac, "window_pos": f"{c0},{r0}", "window_shape": [wc, wr], "window_canvas": list(win), "targets": T}


if __name__ == "__main__":
    import sys
    calib = json.loads((POOL.parent / "qa_calib" / "layoutB.json").read_text())
    for seed in range(int(sys.argv[1]), int(sys.argv[2])):
        m = make_scene(seed, POOL.parent / f"qae_v3_{seed:04d}.png", calib["H_canvas_to_camera"])
        print(seed, m["window_shape"], m["window_pos"], [(t["id"], t["level"]) for t in m["targets"]])
