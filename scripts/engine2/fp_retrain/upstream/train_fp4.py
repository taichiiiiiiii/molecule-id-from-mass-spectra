"""Spectrum->fingerprint training with BCE + hard-negative contrastive ranking loss.
   Ranking score is f_cand . z (exact Bayes log-likelihood up to a candidate-independent constant)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, pickle, os, math, argparse
import fpmodel

def dev_of():
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

class Data:
    def __init__(self, spec='ds/spec.npz', fpdir='up_train', pooldir='.', device='cpu', pool_on_gpu=True):
        z=np.load(spec)
        self.off=z['off']; self.mz=z['mz']; self.it=z['it']; self.prec=z['prec']
        self.ad=z['ad'].astype(np.int64); self.ins=z['ins'].astype(np.int64)
        self.ce=z['ce']; self.mode=z['mode']; self.six=z['six']; self.is_val=z['is_val']
        m=pickle.load(open(f'{fpdir}/fp_meta.pkl','rb')); self.nbits=m['nbits']; self.skeys=m['keys']
        pm=pickle.load(open(f'{pooldir}/pool_meta.pkl','rb'))
        self.pmass=np.load(f'{pooldir}/pool_mass.npy')
        PFP=np.unpackbits(np.load(f'{pooldir}/pool_fp.npy'),axis=1)[:, :self.nbits]
        self.pkeys=pm['keys']
        k2p={k:i for i,k in enumerate(self.pkeys)}
        self.s2p=np.array([k2p.get(k,-1) for k in self.skeys], dtype=np.int64)
        self.device=device
        if pool_on_gpu and device=='cuda':
            self.PFP=torch.from_numpy(np.ascontiguousarray(PFP)).to(device)
        else:
            self.PFP=torch.from_numpy(np.ascontiguousarray(PFP))
        # group spectra by structure so training inputs can be MERGED the way test molecules are:
        # the hidden test gives 1-16 spectra per molecule (median 3), but training on single
        # spectra creates a train/test mismatch.
        order=np.argsort(self.six, kind='mergesort')
        s_sorted=self.six[order]
        bounds=np.searchsorted(s_sorted, np.arange(s_sorted.max()+2))
        self._grp_order=order; self._grp_bounds=bounds
        self.tr=np.where(~self.is_val)[0]; self.va=np.where(self.is_val)[0]
        keep = self.s2p[self.six[self.tr]]>=0
        self.tr = self.tr[keep]
        print(f'nbits={self.nbits} pool={len(self.pmass)} train={len(self.tr)} val={len(self.va)}',flush=True)

    def sample_negs(self, pos_pidx, K, ppm=10.0, rng=None):
        m=self.pmass[pos_pidx]; tol=m*ppm/1e6
        lo=np.searchsorted(self.pmass, m-tol,'left'); hi=np.searchsorted(self.pmass, m+tol,'right')
        out=np.empty((len(pos_pidx),K), dtype=np.int64)
        for r,(a,b,p) in enumerate(zip(lo,hi,pos_pidx)):
            n=b-a
            if n<=1: out[r]=rng.integers(0,len(self.pmass),K)   # no isomers -> random decoys
            else:
                c=rng.integers(a,b,K)
                bad=(c==p)
                if bad.any(): c[bad]=np.where(c[bad]+1<b, c[bad]+1, a)
                out[r]=c
        return out

    def peers(self, i, rng, kmax=4):
        """Other spectra of the same structure (for merge augmentation)."""
        s=self.six[i]
        a,b=self._grp_bounds[s], self._grp_bounds[s+1]
        if b-a<=1: return [i]
        cand=self._grp_order[a:b]
        k=int(rng.integers(2, min(kmax, b-a)+1))
        pick=rng.choice(cand, size=min(k,len(cand)), replace=False)
        if i not in pick: pick=np.concatenate([[i], pick[:-1]])
        return list(pick)

    def _merged(self, idxs, maxlen):
        mzs=[]; its=[]
        for j in idxs:
            a,b=self.off[j], min(self.off[j]+maxlen, self.off[j+1])
            mzs.append(self.mz[a:b]); its.append(self.it[a:b])
        mz=np.concatenate(mzs); it=np.concatenate(its)
        o=np.argsort(mz); mz,it=mz[o],it[o]
        keep=np.ones(len(mz),bool)
        for j in range(1,len(mz)):
            if mz[j]-mz[j-1] < 0.005:
                if it[j]>=it[j-1]: keep[j-1]=False
                else: keep[j]=False
        mz,it=mz[keep],it[keep]
        if len(mz)>maxlen:
            top=np.argsort(-it)[:maxlen]; top.sort(); mz,it=mz[top],it[top]
        return mz,it

    def batch(self, ids, K=0, rng=None, ppm=10.0, aug=False, merge_p=0.0):
        B=len(ids); L=fpmodel.MAX_PEAKS
        lens=np.minimum(self.off[ids+1]-self.off[ids], L); N=max(int(lens.max()),1)
        mz=np.zeros((B,N),np.float32); it=np.zeros((B,N),np.float32); pad=np.ones((B,N),bool)
        for r,(i,l) in enumerate(zip(ids,lens)):
            a=self.off[i]; mz[r,:l]=self.mz[a:a+l]; it[r,:l]=self.it[a:a+l]; pad[r,:l]=False
        if merge_p>0 and rng is not None:
            for r,i in enumerate(ids):
                if rng.random()>=merge_p: continue
                pk=self.peers(i,rng)
                if len(pk)<2: continue
                m2,i2=self._merged(pk, N)
                n2=len(m2)
                mz[r,:]=0; it[r,:]=0; pad[r,:]=True
                mz[r,:n2]=m2; it[r,:n2]=i2; pad[r,:n2]=False
        if aug and rng is not None:
            # spectra are noisy measurements; jitter them so the model cannot memorise
            # exact peak patterns of the 276k training structures (the earlier run
            # overfit hard: val hardneg-top1 peaked at 0.46 then decayed to 0.13).
            keep = rng.random((B,N)) > rng.uniform(0.0, 0.30, size=(B,1))   # peak dropout
            pad = pad | (~keep)
            it = it * np.exp(rng.normal(0.0, 0.25, size=(B,N))).astype(np.float32)  # intensity jitter
            mz = mz * (1.0 + rng.normal(0.0, 5e-6, size=(B,N))).astype(np.float32)  # m/z jitter (~5 ppm)
            allpad = pad.all(1)
            if allpad.any(): pad[allpad,0]=False
        d=self.device; T=lambda x: torch.as_tensor(x,device=d)
        ce=np.where(self.ce[ids]<0,25.0,self.ce[ids]).astype(np.float32)
        inp=(T(mz),T(it),T(pad),T(self.prec[ids]),T(self.ad[ids]),T(self.ins[ids]),T(ce),T(self.mode[ids]))
        pos=self.s2p[self.six[ids]]
        cand=None
        if K>0:
            negs=self.sample_negs(pos,K,ppm,rng)
            cand=np.concatenate([pos[:,None],negs],1)         # (B,K+1), col 0 = positive
            cand=torch.as_tensor(cand, device=self.PFP.device)
        ypos=self.PFP[torch.as_tensor(pos, device=self.PFP.device)].to(d).float()
        return inp, ypos, cand

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=80000); ap.add_argument('--bs',type=int,default=256)
    ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--d',type=int,default=512)
    ap.add_argument('--layers',type=int,default=6); ap.add_argument('--K',type=int,default=63)
    ap.add_argument('--lam',type=float,default=1.0); ap.add_argument('--warm',type=int,default=2000)
    ap.add_argument('--out',default='fp_model.pt'); ap.add_argument('--resume',default='')
    ap.add_argument('--val_every',type=int,default=2000); ap.add_argument('--max_minutes',type=float,default=1e9)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--merge_p',type=float,default=0.6)
    ap.add_argument('--spec',default='ds/spec.npz'); ap.add_argument('--fpdir',default='up_train'); ap.add_argument('--pooldir',default='.')
    a=ap.parse_args()
    dev=dev_of(); print('device',dev,flush=True)
    D=Data(a.spec,a.fpdir,a.pooldir,device=dev)
    model=fpmodel.FPNet(D.nbits,d=a.d,layers=a.layers).to(dev)
    print('params %.1fM'%(sum(p.numel() for p in model.parameters())/1e6),flush=True)
    torch.manual_seed(a.seed)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01,betas=(0.9,0.98))
    scaler=torch.amp.GradScaler('cuda') if dev=='cuda' else None
    start=0
    if a.resume and os.path.exists(a.resume):
        ck=torch.load(a.resume,map_location=dev); model.load_state_dict(ck['model'])
        try: opt.load_state_dict(ck['opt'])
        except Exception: pass
        start=ck['step']; print('resume',start,flush=True)
    def lr_at(s):
        if s<a.warm: return a.lr*s/max(1,a.warm)
        p=(s-a.warm)/max(1,a.steps-a.warm); return a.lr*(0.02+0.98*0.5*(1+math.cos(math.pi*min(p,1.0))))
    rng=np.random.default_rng(a.seed)
    t0=time.time(); rb=rc=racc=0.0; nr=0
    best_val=-1.0   # keep the checkpoint that generalises best, not the last one
    def save(step):
        torch.save({'model':model.state_dict(),'opt':opt.state_dict(),'step':step,
                    'nbits':D.nbits,'d':a.d,'layers':a.layers}, a.out)
    for step in range(start,a.steps):
        for g in opt.param_groups: g['lr']=lr_at(step)
        ids=rng.choice(D.tr,size=a.bs,replace=False)
        inp,ypos,cand = D.batch(ids,K=a.K,rng=rng,aug=True,merge_p=a.merge_p)
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast('cuda',dtype=torch.float16) if dev=='cuda' else torch.autocast('mps',dtype=torch.float16) if dev=='mps' else torch.autocast('cpu',enabled=False)
        with ctx:
            z=model(*inp)
        z=z.float()
        lb=F.binary_cross_entropy_with_logits(z,ypos)
        FPc=D.PFP[cand].to(z.device).float()                     # (B,K+1,nbits)
        raw=torch.bmm(FPc, z.unsqueeze(-1)).squeeze(-1)          # (B,K+1) = f.z  (exact Bayes LL up to const)
        # centre within the example: subtracting (mean_c f_c).z is a per-example constant,
        # so the ranking is untouched, but the magnitudes stay small enough for a stable softmax.
        sc=(raw - raw.mean(dim=1, keepdim=True)) / math.sqrt(D.nbits)
        tgt=torch.zeros(len(ids),dtype=torch.long,device=z.device)
        lc=F.cross_entropy(sc,tgt)
        loss=lb+a.lam*lc
        if not torch.isfinite(loss):
            print(f'  non-finite loss at step {step}; skipping', flush=True)
            opt.zero_grad(set_to_none=True); continue
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        rb+=lb.item(); rc+=lc.item(); racc+=(sc.argmax(1)==0).float().mean().item(); nr+=1
        if step%200==0:
            el=time.time()-t0
            print(f'step {step} bce {rb/max(nr,1):.4f} ctr {rc/max(nr,1):.4f} top1 {racc/max(nr,1):.3f} lr {lr_at(step):.2e} {el:.0f}s {(step-start+1)*a.bs/max(el,1):.0f} spec/s',flush=True)
            rb=rc=racc=0.0; nr=0
        if (step+1)%a.val_every==0:
            model.eval(); vb=[];vacc=[]
            with torch.no_grad():
                vids=rng.choice(D.va,size=min(2048,len(D.va)),replace=False)
                for s in range(0,len(vids),128):
                    ii=vids[s:s+128]
                    pos=D.s2p[D.six[ii]]
                    if (pos<0).any(): ii=ii[pos>=0]
                    if len(ii)==0: continue
                    inp,ypos,cand=D.batch(ii,K=a.K,rng=rng,merge_p=a.merge_p)
                    z=model(*inp).float()
                    vb.append(F.binary_cross_entropy_with_logits(z,ypos).item())
                    FPc=D.PFP[cand].to(z.device).float()
                    raw=torch.bmm(FPc,z.unsqueeze(-1)).squeeze(-1)
                    sc=(raw-raw.mean(dim=1,keepdim=True))/math.sqrt(D.nbits)
                    vacc.append((sc.argmax(1)==0).float().mean().item())
            va=float(np.mean(vacc))
            print(f'  VAL step {step} bce {np.mean(vb):.4f} hardneg-top1 {va:.3f}'
                  + ('  <- best' if va>best_val else ''),flush=True)
            model.train()
            if va>best_val:
                best_val=va; save(step+1)          # only overwrite when validation improves
        if (time.time()-t0)/60>a.max_minutes:
            print('time budget reached; keeping best-val checkpoint',flush=True); break
    print('done. best hardneg-top1 = %.3f'%best_val,flush=True)

if __name__=='__main__': main()
