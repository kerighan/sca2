"""
SCA2 vs Transformer — token-level TinyPython benchmark.

Dataset: BertilBraun/TinyPython (streamed)
Tokenizer: tiktoken cl100k_base
Task: causal LM over task_description + Python function
Models: 1-layer SCA2 (C+D) vs 1-layer causal Transformer

Install:
    pip install torch datasets tiktoken

Run:
    python bench_tinypython.py --steps 3000 --examples 20000 --block 128 --batch 16

The script caches tokenized data locally after the first run.
"""
import argparse, math, random, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare(args):
    import tiktoken
    from datasets import load_dataset
    enc=tiktoken.get_encoding("cl100k_base")
    eos=enc.eot_token
    cache=f"tinypython_cl100k_{args.examples}.pt"
    if os.path.exists(cache):
        z=torch.load(cache); return z["train"],z["val"],enc.n_vocab
    ds=load_dataset("BertilBraun/TinyPython","small",split="train",streaming=True)
    toks=[]
    for i,e in enumerate(ds):
        if i>=args.examples: break
        # Include instruction because code generation, not merely code continuation.
        text="Write a Python function for the following task:\n\n"+e["task_description"]+"\n\n"+e["code"]
        toks.extend(enc.encode(text,allowed_special=set()))
        toks.append(eos)
    ids=torch.tensor(toks,dtype=torch.long)
    n=int(.9*len(ids)); tr,va=ids[:n],ids[n:]
    torch.save({"train":tr,"val":va},cache)
    print("tokens",len(ids),"train",len(tr),"val",len(va),"vocab",enc.n_vocab)
    return tr,va,enc.n_vocab

def get_batch(data,B,T,g,device):
    ix=torch.randint(len(data)-T-1,(B,),generator=g)
    x=torch.stack([data[i:i+T] for i in ix]).to(device)
    y=torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x,y

class CHead(nn.Module):
    def __init__(self,d,M):
        super().__init__(); dv=d//2
        self.K=nn.Linear(d,M,False); self.V=nn.Linear(d,dv,False)
        self.theta=nn.Parameter(torch.zeros(M)) # structured Fourier init
        self.register_buffer("omega",2*math.pi*torch.arange(M)/M)
        self.wr=nn.Parameter(torch.ones(M)); self.wi=nn.Parameter(torch.zeros(M))
    def forward(self,z,h):
        T=z.size(1); p=torch.arange(T,device=z.device,dtype=z.dtype)
        pw=self.K(h)*self.theta+p[:,None]*self.omega
        pq=self.K(z)*self.theta+p[:,None]*self.omega
        v=self.V(z); zr=v[:,:,None]*pw.cos()[:,:,:,None]; zi=v[:,:,None]*pw.sin()[:,:,:,None]
        sr,si=zr.cumsum(1),zi.cumsum(1); qr,qi=pq.cos()[:,:,:,None],-pq.sin()[:,:,:,None]
        rr,ii=sr*qr-si*qi,sr*qi+si*qr
        u=torch.cat([(rr*self.wr[None,None,:,None]-ii*self.wi[None,None,:,None]).mean(2),
                     (rr*self.wi[None,None,:,None]+ii*self.wr[None,None,:,None]).mean(2)],-1)
        return u*torch.rsqrt(u.square().mean(-1,keepdim=True)+1e-6)

class DHead(nn.Module):
    """Simple grouped complex recurrent D-head."""
    def __init__(self,d,M=16,G=8):
        super().__init__(); self.dv=d//2; self.M=M; self.G=G; assert self.dv%G==0
        self.V=nn.Linear(d,self.dv,False); self.gr=nn.Linear(d,M*G); self.gi=nn.Linear(d,M*G)
        self.qr=nn.Linear(d,M*self.dv,False); self.qi=nn.Linear(d,M*self.dv,False)
    def forward(self,z,h):
        B,T,_=z.shape; gs=self.dv//self.G; v=self.V(z)
        gr=torch.tanh(self.gr(h)).view(B,T,self.M,self.G)/math.sqrt(2)
        gi=torch.tanh(self.gi(h)).view(B,T,self.M,self.G)/math.sqrt(2)
        sr=torch.zeros(B,self.M,self.dv,device=z.device,dtype=z.dtype); si=torch.zeros_like(sr); out=[]
        for t in range(T):
            rg,ig=sr.view(B,self.M,self.G,gs),si.view(B,self.M,self.G,gs)
            ar,ai=gr[:,t,:,:,None],gi[:,t,:,:,None]
            sr=(ar*rg-ai*ig).reshape(B,self.M,self.dv)+v[:,t,None,:]
            si=(ar*ig+ai*rg).reshape(B,self.M,self.dv)
            qr=self.qr(z[:,t]).view(B,self.M,self.dv); qi=self.qi(z[:,t]).view(B,self.M,self.dv)
            qn=torch.sqrt(qr.square()+qi.square()+1e-6); qr,qi=qr/qn,qi/qn
            out.append(torch.cat([(sr*qr+si*qi).mean(1),(si*qr-sr*qi).mean(1)],-1))
        u=torch.stack(out,1)
        return u*torch.rsqrt(u.square().mean(-1,keepdim=True)+1e-6)

class SCA2(nn.Module):
    def __init__(self,V,d=128,Mc=64,Md=16,G=8,ff=256):
        super().__init__(); self.e=nn.Embedding(V,d); self.n=nn.LayerNorm(d)
        self.c=CHead(d,Mc); self.dh=DHead(d,Md,G); self.mix=nn.Linear(2*d,d)
        self.fn=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d))
        self.on=nn.LayerNorm(d); self.o=nn.Linear(d,V)
    def forward(self,t):
        x=self.e(t); z=self.n(x); h=torch.zeros_like(z); h[:,1:]=z[:,:-1]
        x=x+self.mix(torch.cat([self.c(z,h),self.dh(z,h)],-1))
        x=x+self.ff(self.fn(x)); return self.o(self.on(x))

class Transformer(nn.Module):
    def __init__(self,V,d=128,heads=4,ff=256,maxlen=512):
        super().__init__(); self.e=nn.Embedding(V,d); self.p=nn.Embedding(maxlen,d)
        self.b=nn.TransformerEncoderLayer(d,heads,ff,dropout=0,batch_first=True,norm_first=True,activation="gelu")
        self.n=nn.LayerNorm(d); self.o=nn.Linear(d,V)
    def forward(self,t):
        T=t.size(1); x=self.e(t)+self.p(torch.arange(T,device=t.device))[None]
        mask=torch.triu(torch.ones(T,T,device=t.device,dtype=torch.bool),1)
        return self.o(self.n(self.b(x,src_mask=mask)))

@torch.no_grad()
def evaluate(m,val,args,device):
    m.eval(); g=torch.Generator().manual_seed(999); ls=[]
    for _ in range(20):
        x,y=get_batch(val,args.batch,args.block,g,device)
        ls.append(F.cross_entropy(m(x).flatten(0,1),y.flatten()).item())
    m.train(); return sum(ls)/len(ls)

def train(name,m,tr,va,args,device):
    m.to(device); opt=torch.optim.AdamW(m.parameters(),lr=args.lr); g=torch.Generator().manual_seed(123)
    print(name,"params",sum(p.numel() for p in m.parameters()))
    t0=time.perf_counter()
    for st in range(1,args.steps+1):
        x,y=get_batch(tr,args.batch,args.block,g,device); opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(x).flatten(0,1),y.flatten()); loss.backward(); opt.step()
        if st in {100,250,500,1000,2000,args.steps}:
            vl=evaluate(m,va,args,device)
            print(name,st,"train",round(loss.item(),4),"val",round(vl,4),
                  "tok/s",round(st*args.batch*args.block/(time.perf_counter()-t0)))
    return m

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--examples",type=int,default=20000); p.add_argument("--steps",type=int,default=3000)
    p.add_argument("--block",type=int,default=128); p.add_argument("--batch",type=int,default=16)
    p.add_argument("--d",type=int,default=128); p.add_argument("--Mc",type=int,default=64)
    p.add_argument("--Md",type=int,default=16); p.add_argument("--G",type=int,default=8)
    p.add_argument("--ff",type=int,default=256); p.add_argument("--lr",type=float,default=3e-4)
    a=p.parse_args(); device="cuda" if torch.cuda.is_available() else "cpu"
    tr,va,V=prepare(a)
    torch.manual_seed(0); train("SCA2",SCA2(V,a.d,a.Mc,a.Md,a.G,a.ff),tr,va,a,device)
    torch.manual_seed(0); train("Transformer",Transformer(V,a.d,4,a.ff,a.block),tr,va,a,device)
if __name__=="__main__": main()
