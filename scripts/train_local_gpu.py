#!/usr/bin/env python3
"""All-in-one local GPU training with physics-based synthetic generation.

Usage:
    python scripts/train_local_gpu.py
    python scripts/train_local_gpu.py --samples 2000 --epochs 100
"""
from __future__ import annotations
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

MIRROR_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

class PhysicsGenerator:
    def __init__(s, tf=64, seed=42): s.tf=tf; s.rng=np.random.default_rng(seed); s.real={}
    def load(s, tid, seqs): s.real[tid]=seqs
    def generate(s, tid, n=500): return [s._aug(s.real[tid][s.rng.integers(len(s.real[tid]))].copy()) for _ in range(n)]
    def _aug(s, d):
        d=s._rsz(d,s.tf)
        if s.rng.random()<.8: d=s._speed(d,s.rng.uniform(.75,1.3))
        if s.rng.random()<.7:
            r=d.copy(); com_y=np.mean(d[1,:,[5,6,11,12]],axis=1); base=com_y[0]; f=s.rng.uniform(.8,1.25)
            for v in range(17): r[1,:,v]=base+(d[1,:,v]-base)*f
            d=np.clip(r,0,1)
        if s.rng.random()<.6:
            r=d.copy(); a=math.radians(s.rng.uniform(-15,15)); co,si=math.cos(a),math.sin(a)
            cx=d[0].mean(1,keepdims=True); cy=d[1].mean(1,keepdims=True); x,y=d[0]-cx,d[1]-cy
            r[0]=x*co-y*si+cx; r[1]=x*si+y*co+cy; d=np.clip(r,0,1)
        if s.rng.random()<.5:
            r=d.copy(); f=s.rng.uniform(.9,1.1)
            cx=np.mean(d[0,:,[5,6,11,12]],axis=1,keepdims=True); cy=np.mean(d[1,:,[5,6,11,12]],axis=1,keepdims=True)
            r[0]=cx+(d[0]-cx)*f; r[1]=cy+(d[1]-cy)*f; d=np.clip(r,0,1)
        if s.rng.random()<.7: r=d.copy(); r[0]=np.clip(d[0]+s.rng.uniform(-.1,.1),0,1); r[1]=np.clip(d[1]+s.rng.uniform(-.1,.1),0,1); d=r
        if s.rng.random()<.7: r=d.copy(); r[:2]=np.clip(d[:2]+s.rng.normal(0,.004,size=(2,d.shape[1],1))+s.rng.normal(0,.003,size=(2,d.shape[1],17)),0,1); d=r
        if s.rng.random()<.3:
            r=d.copy(); r[0]=1.-d[0]
            for l,ri in MIRROR_PAIRS: r[:,:,l],r[:,:,ri]=r[:,:,ri].copy(),r[:,:,l].copy()
            d=r
        if s.rng.random()<.5: d[2]*=s.rng.uniform(.75,1.,size=(1,d.shape[1],1))
        return s._rsz(d,s.tf)
    def _speed(s,d,f):
        C,T,V=d.shape; nT=max(8,int(T/f)); idx=np.linspace(0,T-1,nT); r=np.zeros((C,nT,V),dtype=d.dtype)
        for c in range(C):
            for v in range(V): r[c,:,v]=np.interp(idx,np.arange(T),d[c,:,v])
        return s._rsz(r,s.tf)
    def _rsz(s,d,t):
        C,T,V=d.shape
        if T==t: return d
        idx=np.linspace(0,T-1,t); r=np.zeros((C,t,V),dtype=d.dtype)
        for c in range(C):
            for v in range(V): r[c,:,v]=np.interp(idx,np.arange(T),d[c,:,v])
        return r

def build_adj(n=17):
    edges=[(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    a=np.eye(n,dtype=np.float32)
    for i,j in edges: a[i,j]=a[j,i]=1.
    d=np.sum(a,1); di=np.where(d>0,np.power(d,-.5),0.); return np.diag(di)@a@np.diag(di)

class SGC(nn.Module):
    def __init__(s,ic,oc,adj): super().__init__(); s.register_buffer('adj',adj); s.c=nn.Conv2d(ic,oc,1); s.bn=nn.BatchNorm2d(oc); s.r=nn.ReLU(True)
    def forward(s,x):
        B,C,T,V=x.shape; xf=x.permute(0,2,1,3).reshape(B*T,C,V)
        return s.r(s.bn(s.c(torch.matmul(xf,s.adj).reshape(B,T,C,V).permute(0,2,1,3))))
class TC(nn.Module):
    def __init__(s,ic,oc,k=9): super().__init__(); s.c=nn.Conv2d(ic,oc,(k,1),padding=(k//2,0)); s.bn=nn.BatchNorm2d(oc); s.r=nn.ReLU(True)
    def forward(s,x): return s.r(s.bn(s.c(x)))
class Block(nn.Module):
    def __init__(s,ic,oc,adj): super().__init__(); s.s=SGC(ic,oc,adj); s.t=TC(oc,oc); s.res=nn.Sequential(nn.Conv2d(ic,oc,1),nn.BatchNorm2d(oc)) if ic!=oc else nn.Identity(); s.r=nn.ReLU(True)
    def forward(s,x): return s.r(s.t(s.s(x))+s.res(x))
class STGCN(nn.Module):
    def __init__(s, nc, ic=3, nj=17, h=None):
        super().__init__()
        if h is None: h=[64,64,128,128,256]
        adj=torch.FloatTensor(build_adj(nj)); s.ibn=nn.BatchNorm1d(ic*nj)
        ls=[]; p=ic
        for hi in h: ls.append(Block(p,hi,adj)); p=hi
        s.blocks=nn.ModuleList(ls); s.gap=nn.AdaptiveAvgPool2d(1); s.drop=nn.Dropout(.3); s.fc=nn.Linear(h[-1],nc)
    def forward(s,x):
        B,C,T,V=x.shape; x=s.ibn(x.permute(0,2,1,3).reshape(B*T,C*V)).reshape(B,T,C,V).permute(0,2,1,3)
        for b in s.blocks: x=b(x)
        return s.fc(s.drop(s.gap(x).squeeze(-1).squeeze(-1)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoints-dir", default="data/clips/keypoints")
    parser.add_argument("--relabels", default="data/training/relabels.json")
    parser.add_argument("--output", default="data/models/stgcn_physics.pt")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  PkVision — Local GPU Training\n  {'='*45}")
    print(f"  Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type=="cuda" else ""))

    kp_dir = Path(args.keypoints_dir)
    relabels = json.load(open(args.relabels))
    trick_kps = {}
    for fn, tid in relabels.items():
        if tid == "not_a_flip": continue
        stem = Path(fn).stem; kp = kp_dir / f"{stem}.npy"
        if kp.exists():
            d = np.load(kp).astype(np.float32)
            if d.shape[0]==3 and d.shape[2]==17: trick_kps.setdefault(tid,[]).append(d)

    print(f"  Real: {sum(len(v) for v in trick_kps.values())} clips, {len(trick_kps)} tricks\n")
    gen = PhysicsGenerator(tf=64, seed=42)
    for tid, seqs in trick_kps.items(): gen.load(tid, seqs)

    all_data, all_labels = [], []
    classes = sorted(trick_kps.keys())
    classes.append("no_trick")
    c2i = {c:i for i,c in enumerate(classes)}

    # Generate no_trick class
    for _ in range(args.samples):
        noise = np.random.rand(3,64,17).astype(np.float32)*0.3+0.3; noise[2]=0.8
        all_data.append(noise); all_labels.append(c2i["no_trick"])

    for tid in classes[:-1]:
        if tid not in gen.real: continue
        n = min(args.samples, max(300, len(trick_kps[tid])*80))
        print(f"    {tid:25s} {len(trick_kps[tid]):3d} real -> {n:5d} synth")
        for s in gen.generate(tid, n=n): all_data.append(s); all_labels.append(c2i[tid])
        for seq in trick_kps[tid]: all_data.append(gen._rsz(seq,64)); all_labels.append(c2i[tid])

    X=torch.FloatTensor(np.stack(all_data)); y=torch.LongTensor(all_labels)
    print(f"\n  Dataset: {X.shape[0]} samples, {len(classes)} classes")

    ds=TensorDataset(X,y); vs=max(1,int(len(ds)*.15))
    tds,vds=random_split(ds,[len(ds)-vs,vs],generator=torch.Generator().manual_seed(42))
    tl=DataLoader(tds,batch_size=args.batch_size,shuffle=True,pin_memory=device.type=="cuda")
    vl=DataLoader(vds,batch_size=args.batch_size,pin_memory=device.type=="cuda")

    model=STGCN(nc=len(classes)).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params\n")

    crit=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=0.
    for ep in range(args.epochs):
        t0=time.time(); model.train(); tls,tc,tt=0.,0,0
        for bx,by in tl:
            bx,by=bx.to(device),by.to(device); opt.zero_grad(); o=model(bx); l=crit(o,by); l.backward(); opt.step()
            tls+=l.item()*bx.size(0); tc+=(o.argmax(1)==by).sum().item(); tt+=bx.size(0)
        sch.step(); model.eval(); vls,vc,vt=0.,0,0
        with torch.no_grad():
            for bx,by in vl:
                bx,by=bx.to(device),by.to(device); o=model(bx); l=crit(o,by)
                vls+=l.item()*bx.size(0); vc+=(o.argmax(1)==by).sum().item(); vt+=bx.size(0)
        ta,va=tc/max(tt,1),vc/max(vt,1)
        if (ep+1)%10==0 or ep==0: print(f"  Epoch {ep+1:3d}/{args.epochs} | train {tls/max(tt,1):.4f} {ta:.0%} | val {vls/max(vt,1):.4f} {va:.0%} | {time.time()-t0:.1f}s")
        if va>=best:
            best=va; Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state_dict":model.state_dict(),"classes":classes,"epoch":ep,"val_acc":va,
                         "config":{"num_classes":len(classes),"in_channels":3,"num_joints":17}},args.output)
    print(f"\n  Best: {best:.0%} | Model: {args.output} | Classes: {len(classes)}")

if __name__ == "__main__":
    main()
