"""
Run policy-network configs A-G (from scripts/train_policy.sh) locally, with
downstream top-1 accuracy on the seed-0 3/1/1 test split (comparable to all our
other numbers). Single AE input, MobileNetV3-Small.

A Lens/full/hard  B Lens/head/hard  C Oracle/head/hard  D Oracle/head/soft
E Oracle/partial/soft(resume D)  F Oracle/full/hard  G Oracle/full/soft
"""
from __future__ import annotations
import json, random, sys, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

here = Path(__file__).resolve().parent
REPO = here.parents[1]
sys.path.insert(0, str(REPO / "policy_network" / "static_pred"))
from policy_model import SensorPolicyNetwork

# ---- per-scene signals ----
fd = np.load(here / "fidelity_cache_resnet50.npz")
gp = np.load(here / "gtprob_cache_resnet50.npz")
assert np.array_equal(fd["gidx"], gp["gidx"]) and np.array_equal(fd["pidx"], gp["pidx"])
gidx, pidx = fd["gidx"], fd["pidx"]
meta = json.load(open(here / "group_meta.json")); ng = len(meta)
CONF = np.zeros((ng, 27), np.float32); COR = np.zeros((ng, 27), bool); GTP = np.zeros((ng, 27), np.float32)
CONF[gidx, pidx] = fd["conf"]; COR[gidx, pidx] = fd["correct"]; GTP[gidx, pidx] = gp["gt_prob"]

# ---- labels ----
lens_hard = CONF.argmax(1)
oracle_hard = np.zeros(ng, np.int64); oracle_soft = np.zeros((ng, 27), np.float32)
for g in range(ng):
    cor = COR[g]
    if cor.any():
        idxs = np.where(cor)[0]; w = GTP[g, idxs]
        oracle_hard[g] = idxs[w.argmax()]
        oracle_soft[g, idxs] = w / w.sum()
    else:
        bi = GTP[g].argmax(); oracle_hard[g] = bi; oracle_soft[g, bi] = 1.0

# ---- split ----
from collections import defaultdict
by_class = defaultdict(lambda: defaultdict(list))
for g, m in enumerate(meta): by_class[m["wnid"]][m["ref"]].append(g)
rng = random.Random(0); TR=[]; VA=[]; TE=[]
for wn, refs in sorted(by_class.items()):
    rl = sorted(refs); rng.shuffle(rl)
    for r in rl[:3]: TR += refs[r]
    for r in rl[3:4]: VA += refs[r]
    for r in rl[4:]: TE += refs[r]
TR, VA, TE = map(np.array, (TR, VA, TE))

# ---- preload AE images (uint8, resize256/crop224) ----
pre = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
AE = torch.zeros(ng, 3, 224, 224, dtype=torch.uint8)
class AEload(Dataset):
    def __len__(s): return ng
    def __getitem__(s, g):
        im = pre(Image.open(meta[g]["ae_path"]).convert("RGB"))
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1), g
for x, g in tqdm(DataLoader(AEload(), batch_size=256, num_workers=24), desc="preload AE"):
    AE[g] = x
print("AE images preloaded:", tuple(AE.shape))

dev = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def norm(u8):  # uint8 [B,3,224,224] -> normalized float
    return (u8.to(dev).float() / 255.0 - MEAN) / STD

soft_t = torch.tensor(oracle_soft, device=dev)
lens_t = torch.tensor(lens_hard, device=dev); orac_t = torch.tensor(oracle_hard, device=dev)
COR_t = COR

CONFIGS = {
 "A": dict(label="lens",   scope="full",    loss="hard", lr=2e-5, blr=1e-6),
 "B": dict(label="lens",   scope="head",    loss="hard", lr=1e-4, blr=5e-6),
 "C": dict(label="oracle", scope="head",    loss="hard", lr=1e-4, blr=5e-6),
 "D": dict(label="oracle", scope="head",    loss="soft", lr=1e-4, blr=5e-6),
 "E": dict(label="oracle", scope="partial", loss="soft", lr=5e-5, blr=2e-6, resume="D"),
 "F": dict(label="oracle", scope="full",    loss="hard", lr=2e-5, blr=1e-6),
 "G": dict(label="oracle", scope="full",    loss="soft", lr=2e-5, blr=1e-6),
}
EPOCHS = 50; BS = 16

def hard_label(cfg):
    return lens_t if cfg["label"] == "lens" else orac_t

def run(name, cfg, init_state=None):
    random.seed(0); torch.manual_seed(0)
    net = SensorPolicyNetwork(27, pretrained=True, backbone_name="mobilenet_v3_small", input_mode="single").to(dev)
    if init_state is not None:
        net.load_state_dict(init_state)
    if cfg["scope"] == "head": net.freeze_backbone()
    elif cfg["scope"] == "partial": net.unfreeze_backbone_tail()
    head = list(net.policy_head.parameters()); hid = {id(p) for p in head}
    back = [p for p in net.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.AdamW([{"params": head, "lr": cfg["lr"]},
                             {"params": back, "lr": cfg["blr"]}], weight_decay=5e-4)
    HL = hard_label(cfg)

    @torch.no_grad()
    def ds_acc(idxs):
        net.eval(); ok = []
        for k in range(0, len(idxs), 256):
            b = idxs[k:k+256]
            sel = net(norm(AE[b])).argmax(1).cpu().numpy()
            ok.extend(COR_t[b, sel])
        return 100 * np.mean(ok)

    best_va, best_te, best_state = -1, -1, None
    TRt = TR.copy()
    for ep in range(1, EPOCHS + 1):
        net.train(); np.random.shuffle(TRt)
        for k in range(0, len(TRt), BS):
            b = TRt[k:k+BS]
            logits = net(norm(AE[b]))
            if cfg["loss"] == "hard":
                loss = F.cross_entropy(logits, HL[b])
            else:
                loss = F.kl_div(F.log_softmax(logits, 1), soft_t[b], reduction="batchmean")
            opt.zero_grad(); loss.backward(); opt.step()
        va = ds_acc(VA); te = ds_acc(TE)
        if va > best_va: best_va, best_te = va, te; best_state = copy.deepcopy(net.state_dict())
    return best_va, best_te, best_state

results = {}; states = {}
for name in ["A", "B", "C", "D", "E", "F", "G"]:
    cfg = CONFIGS[name]
    init = states.get(cfg.get("resume")) if "resume" in cfg else None
    va, te, st = run(name, cfg, init)
    states[name] = st; results[name] = (va, te)
    print(f"[{name}] {cfg['label']}/{cfg['scope']}/{cfg['loss']:>4}  val_DS={va:.2f}  TEST_DS={te:.2f}")

print("\n==== POLICY A-G (downstream test acc, AE input, mobilenet) ====")
for name in ["A","B","C","D","E","F","G"]:
    c = CONFIGS[name]; va, te = results[name]
    print(f"  {name}: {c['label']:>6}/{c['scope']:>7}/{c['loss']:>4}  test={te:.2f}%")
print(f"  refs: VisiT={100*np.mean([COR[g][CONF[g].argmax()] for g in TE]):.2f}  "
      f"fixed={100*np.array([COR[g] for g in TR]).mean(0).max():.2f}(train)  "
      f"OracleS={100*np.mean([COR[g].any() for g in TE]):.2f}")
