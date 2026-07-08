"""
Accuracy / recall vs capture-budget K on the seed-0 test split.

Non-adaptive capture strategies (no training):
  - spread  : K configs evenly spaced along the exposure (brightness) ladder
  - random  : K random configs (averaged over many draws)
Final pick among the K captured: max-confidence (VisiT-among-K) -- fully deployable.

Reports, per K:
  recall@K   = P(a correct config is among the K captured)   [ceiling for any picker]
  acc@K      = downstream top-1 after the confidence pick     [deployable]
Reference points: fixed-config 34.0, VisiT(all 27) 32.2, relative scorer 36.4, Oracle-S 51.
"""
import numpy as np, json, random
from collections import defaultdict

h='/home/ubuntu/ada_sensing/data/ImageNet-ES-Diverse/'
fd=np.load(h+'fidelity_cache_resnet50.npz'); cb=np.load(h+'combined_cache_resnet50.npz')
gidx,pidx=fd['gidx'],fd['pidx']; ng=int(gidx.max())+1
CONF=np.zeros((ng,27),np.float32); COR=np.zeros((ng,27),bool); LUM=np.zeros((ng,27),np.float32)
CONF[gidx,pidx]=fd['conf']; COR[gidx,pidx]=fd['correct']
cg,cp=cb['gidx'],cb['pidx']; LUM[cg,cp]=cb['feats'][:,0]
meta=json.load(open(h+'group_meta.json'))
by=defaultdict(lambda:defaultdict(list))
for g,m in enumerate(meta): by[m['wnid']][m['ref']].append(g)
rng=random.Random(0); TE=[]
for wn,refs in sorted(by.items()):
    rl=sorted(refs); rng.shuffle(rl)
    for r in rl[4:]: TE+=refs[r]
TE=np.array(TE)

# config order by mean brightness across all scenes (the exposure ladder)
order=np.argsort(LUM.mean(0))

def eval_subset(S):
    S=np.array(S)
    sub_conf=CONF[np.ix_(TE,S)]; sub_cor=COR[np.ix_(TE,S)]
    pick=sub_conf.argmax(1)
    acc=100*sub_cor[np.arange(len(TE)),pick].mean()
    recall=100*sub_cor.any(1).mean()
    return acc, recall

print(f"test scenes={len(TE)}")
print(" K | spread_acc spread_recall | rand_acc rand_recall")
rr=np.random.RandomState(0)
for K in [1,2,3,4,5,6,8,10,13,16,20,27]:
    # spread: K configs evenly spaced along brightness order
    idx=np.round(np.linspace(0,26,K)).astype(int)
    S=sorted(set(order[idx].tolist()))
    # if dedup shrank it, top up along order
    j=0
    while len(S)<K:
        if order[j] not in S: S.append(int(order[j]))
        j+=1
    sa,sr=eval_subset(S)
    # random: average over draws
    ra=[]; rc=[]
    for _ in range(200):
        Sr=rr.choice(27,K,replace=False)
        a,c=eval_subset(Sr); ra.append(a); rc.append(c)
    print(f"{K:3d} |   {sa:5.2f}     {sr:6.2f}      |  {np.mean(ra):5.2f}    {np.mean(rc):6.2f}")

print("\nref: fixed-config=34.0  VisiT(all27,conf-pick)=32.2  relative-scorer(all27)=36.4  Oracle-S=51.0")
