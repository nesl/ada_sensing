"""
Combined method: budgeted adaptive capture + our feature-distance signal.

A set-transformer over the configs captured SO FAR produces:
  - next-capture head : which un-captured config to shoot next (trained to imitate
                        the oracle order = increasing feature-distance-to-clean)
  - quality head      : a relative score over captured configs for the final pick
                        (trained on correctness; attention makes it relative)

Deploys with NO clean image (feature-distance is only the training signal).
Greedy rollout from a fixed smart first shot; reports downstream accuracy vs K.
"""
import numpy as np, json, random, sys
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0

h='/home/ubuntu/ada_sensing/data/ImageNet-ES-Diverse/'
fe=np.load(h+'feature_cache_resnet50.npz'); fd=np.load(h+'fidelity_cache_resnet50.npz')
assert np.array_equal(fe['gidx'],fd['gidx']) and np.array_equal(fe['pidx'],fd['pidx'])
gidx,pidx=fe['gidx'],fd['pidx']; ng=int(gidx.max())+1
FEAT=np.zeros((ng,27,2048),np.float32); CONF=np.zeros((ng,27),np.float32)
COR=np.zeros((ng,27),bool); DIST=np.full((ng,27),9.9,np.float32)
FEAT[gidx,pidx]=fe['feats']; CONF[gidx,pidx]=fd['conf']; COR[gidx,pidx]=fd['correct']; DIST[gidx,pidx]=fd['feat_cos']
FEAT/=np.linalg.norm(FEAT,axis=2,keepdims=True)+1e-6
meta=json.load(open(h+'group_meta.json'))
by=defaultdict(lambda:defaultdict(list))
for g,m in enumerate(meta): by[m['wnid']][m['ref']].append(g)
rng=random.Random(0); TR=[];VA=[];TE=[]
for wn,refs in sorted(by.items()):
    rl=sorted(refs); rng.shuffle(rl)
    for r in rl[:3]:TR+=refs[r]
    for r in rl[3:4]:VA+=refs[r]
    for r in rl[4:]:TE+=refs[r]
TR,VA,TE=map(np.array,(TR,VA,TE))
dev=torch.device("cuda"); torch.manual_seed(SEED); np.random.seed(SEED)
FEATt=torch.tensor(FEAT,device=dev); CONFt=torch.tensor(CONF,device=dev)
ORACLE=np.argsort(DIST,axis=1)            # per scene: configs by increasing distance (best first)
ORACLEt=torch.tensor(ORACLE,device=dev)
COR_np=COR
c0=int(np.array([COR[g] for g in TR]).mean(0).argmax())   # smart first shot = best fixed config on train
print(f"first-shot config c0 = param_{c0+1}")

D=128
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.pe=nn.Embedding(27,D); s.fp=nn.Linear(2048,D); s.cp=nn.Linear(1,D)
        enc=nn.TransformerEncoderLayer(D,4,4*D,dropout=0.1,batch_first=True,activation='gelu')
        s.enc=nn.TransformerEncoder(enc,2)
        s.nxt=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,27))
        s.qual=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,1))
    def tokens(s,feat,conf,pid):    # feat[B,L,2048] conf[B,L] pid[B,L]
        return s.fp(feat)+s.cp(conf.unsqueeze(-1))+s.pe(pid)
    def forward(s,feat,conf,pid):
        H=s.enc(s.tokens(feat,conf,pid))     # [B,L,D]
        pooled=H.mean(1)
        return s.nxt(pooled), s.qual(H).squeeze(-1), H

net=Net().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
DISTt=torch.tensor(DIST,device=dev)

def gather(scenes, cfgs):   # scenes[B], cfgs[B,L] -> feat[B,L,2048],conf[B,L]
    b=torch.tensor(scenes,device=dev).unsqueeze(1).expand(-1,cfgs.shape[1])
    return FEATt[b,cfgs], CONFt[b,cfgs]

# ---- train (teacher-forced on oracle prefixes) ----
for ep in range(1,61):
    net.train(); perm=np.random.permutation(TR); tot=0
    for k in range(0,len(perm),256):
        sc=perm[k:k+256]; B=len(sc); sc_t=torch.tensor(sc,device=dev)
        # (1) capture head: teacher-forced on oracle-best-first prefix -> predict next best
        L=np.random.randint(1,27)
        pre=ORACLEt[sc_t][:,:L]
        feat,conf=gather(sc,pre)
        nxt_logits,_,_=net(feat,conf,pre)
        loss_next=F.cross_entropy(nxt_logits,ORACLEt[sc_t][:,L])
        # (2) selector head: RANDOM subset, listwise feature-distance (subset-robust relative selector)
        M=np.random.randint(2,28)
        rc=torch.argsort(torch.rand(B,27,device=dev),1)[:,:M]      # [B,M] random configs
        featr,confr=gather(sc,rc)
        _,qualr,_=net(featr,confr,rc)
        dr=DISTt[sc_t.unsqueeze(1).expand(-1,M),rc]
        soft=torch.softmax(-dr/0.05,1)
        loss_q=F.kl_div(F.log_softmax(qualr,1),soft,reduction='batchmean')
        loss=loss_next+loss_q
        opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
    if ep%10==0: print(f"ep{ep} loss={tot/(len(perm)//256):.3f}")

# ---- greedy rollout on a split; accuracy vs K ----
@torch.no_grad()
def pick_on(scenes, cfgs):  # cfgs [N,M] -> picked config per scene (selector argmax)
    ct=torch.tensor(cfgs,device=dev)
    feat,conf=gather(scenes,ct)
    _,qual,_=net(feat,conf,ct)
    j=qual.argmax(1).cpu().numpy()
    return cfgs[np.arange(len(scenes)),j]

@torch.no_grad()
def rollout(scenes, Ks):
    net.eval(); N=len(scenes)
    captured=torch.full((N,1),c0,dtype=torch.long,device=dev)
    out={}; picks={}; caps={}
    for t in range(1,28):
        feat,conf=gather(scenes,captured)
        nxt,qual,_=net(feat,conf,captured)
        if t in Ks:
            cap=captured.cpu().numpy()
            cc=np.take_along_axis(COR_np[scenes],cap,1); cf=np.take_along_axis(CONF[scenes],cap,1)
            pick=cap[np.arange(N),qual.argmax(1).cpu().numpy()]
            out[t]=(100*COR_np[scenes,pick].mean(),100*cc.any(1).mean(),
                    100*(cc&(cf>=0.5)).any(1).mean(),100*cc[np.arange(N),cf.argmax(1)].mean())
            picks[t]=pick; caps[t]=cap
        if t==27: break
        mask=torch.zeros(N,27,device=dev); mask.scatter_(1,captured,-1e9)
        captured=torch.cat([captured,(nxt+mask).argmax(1,keepdim=True)],1)
    return out,picks,caps

Ks=[1,2,3,4,5,6,8,10,13,16,20,27]
te,picks,caps=rollout(TE,Ks)
print("\n K | acc(qual) | conf_pick | recall(any) | reliable_recall")
for K in Ks:
    a,r,rel,cp=te[K]; print(f"{K:3d} |   {a:5.2f}   |   {cp:5.2f}   |   {r:6.2f}    |   {rel:6.2f}")

# ---- WHY does fewer beat all-27? ----
print("\n=== mechanism analysis (test) ===")
rr=np.random.RandomState(0)
for K in [4,10]:
    accs=[100*COR_np[TE,pick_on(TE,np.stack([rr.choice(27,K,replace=False) for _ in TE]))].mean() for _ in range(15)]
    print(f"selector acc @K={K}:  adaptive={te[K][0]:.2f}   random={np.mean(accs):.2f}   (all-27={te[27][0]:.2f})")

p10,p27=picks[10],picks[27]; cor10=COR_np[TE,p10]; cor27=COR_np[TE,p27]
gain=(cor10==1)&(cor27==0); loss=(cor10==0)&(cor27==1)
print(f"\nK=10 vs K=27:  gain(10 right,27 wrong)={gain.sum()}  loss(10 wrong,27 right)={loss.sum()}  net=+{gain.sum()-loss.sum()}")
# on gain scenes: what did all-27 pick, and was it avoided by the K=10 capture?
gi=np.where(gain)[0]
conf27_gain=CONF[TE[gi],p27[gi]]
avoided=np.array([p27[i] not in caps[10][i] for i in gi])
print(f"on GAIN scenes: all-27's wrong pick had mean confidence {100*conf27_gain.mean():.1f}%; "
      f"{100*avoided.mean():.0f}% of those picks were NOT in the K=10 captured set (avoided distractors)")
print(f"overall: all-27 wrong picks mean confidence = {100*CONF[TE[cor27==0],p27[cor27==0]].mean():.1f}% (confident-but-wrong distractors)")
print("\nref: fixed=34.0  VisiT=32.2  relative-scorer(all27)=36.4  reliably-recoverable~34.5  Oracle-S=51.0")

for K in Ks:
    print(f"RES {SEED} {K} {te[K][0]:.3f} {te[K][1]:.3f}")
