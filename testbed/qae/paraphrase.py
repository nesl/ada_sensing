"""
Generate paraphrases for every question in the QAE targets with the node's Qwen2.5-VL model
used as a text-only LLM. Output: {question: [paraphrase, ...]} in <out.json>.

  python paraphrase.py <out.json> <targets_root> [<targets_root> ...] [--n 8]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("roots", nargs="+"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    qs = set()
    for root in a.roots:
        for t in Path(root).glob("*/targets.json"):
            for x in json.loads(t.read_text())["targets"]:
                qs.add(x.get("group_question", x["question"]))
    qs = sorted(qs)
    print(f"{len(qs)} unique questions", flush=True)
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    for i, q in enumerate(qs):
        if q in out and len(out[q]) >= a.n:
            continue
        prompt = (f"Rewrite the following question about a photo in {a.n} different ways. Keep the exact meaning and the "
                  f"same target (the same object, text, or region), vary the wording and sentence structure, and keep each "
                  f"rewrite short. Output one rewrite per line, numbered.\n\nQuestion: {q}")
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=320, do_sample=True, temperature=0.9, top_p=0.95)
        reply = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        paras = []
        for line in reply.split("\n"):
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip().strip('"')
            if line and line.lower() != q.lower() and line not in paras and len(line) < 200:
                paras.append(line)
        out[q] = paras[: a.n]
        if i % 20 == 0:
            print(f"  {i}/{len(qs)}: {q[:50]!r} -> {out[q][:2]}", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("PARAPHRASE_DONE", sum(len(v) for v in out.values()), "paraphrases for", len(out), "questions", flush=True)


if __name__ == "__main__":
    main()
