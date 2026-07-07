"""Per-candidate GT-class softmax probability (needed to build Oracle labels),
aligned to the same enumeration as the other caches."""
import json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms
from tqdm import tqdm

here = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}; ENVS = ["l1","l2","l3","l4","l6","l7"]
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
ci = json.load(open(here/"imagenet_class_index.json")); wnid2idx = {v[0]:int(k) for k,v in ci.items()}
root = here/"es-diverse-test"/"param_control"
groups, items, gts = {}, [], []
for env in ENVS:
    ed = root/env
    if not ed.is_dir(): continue
    for pd in sorted(ed.iterdir()):
        if not pd.name.startswith("param_"): continue
        pi = int(pd.name.split("_")[1])-1
        for wd in sorted(x for x in pd.iterdir() if x.is_dir()):
            if wd.name not in wnid2idx: continue
            g = wnid2idx[wd.name]
            for img in sorted(x for x in wd.iterdir() if x.suffix.lower() in IMG_EXT):
                gk=f"{env}__{wd.name}__{img.stem}"
                if gk not in groups: groups[gk]=len(groups)
                items.append((str(img),groups[gk],pi)); gts.append(g)
N=len(items); gidx=np.array([g for _,g,_ in items],np.int32); pidx=np.array([p for _,_,p in items],np.int32)
gts=np.array(gts,np.int64)
class DS(Dataset):
    def __len__(s): return N
    def __getitem__(s,i): return tf(Image.open(items[i][0]).convert("RGB")), i
dev=torch.device("cuda"); model=timm.create_model("resnet50",pretrained=True).eval().to(dev)
gtp=np.zeros(N,np.float32)
dl=DataLoader(DS(),batch_size=384,num_workers=24,pin_memory=True)
with torch.no_grad():
    for x,idx in tqdm(dl,desc="gt_prob"):
        p=F.softmax(model(x.to(dev)).float(),-1).cpu().numpy()
        idx=idx.numpy(); gtp[idx]=p[np.arange(len(idx)), gts[idx]]
np.savez(here/"gtprob_cache_resnet50.npz", gt_prob=gtp, gidx=gidx, pidx=pidx)
print(f"[done] gt_prob {gtp.shape} -> gtprob_cache_resnet50.npz")
