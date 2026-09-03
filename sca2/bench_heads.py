"""Per-head timing, so a win in one head isn't masked by the other."""
import argparse, time, sys, torch
from .ref import CHeadRef, DHeadRef
from .registry import VARIANTS

def _t(fn, dev, iters=10, warmup=3, trials=3):
    for _ in range(warmup): fn()
    best=float('inf')
    for _ in range(trials):
        if dev.startswith('cuda'): torch.cuda.synchronize()
        t0=time.perf_counter()
        for _ in range(iters): fn()
        if dev.startswith('cuda'): torch.cuda.synchronize()
        best=min(best,(time.perf_counter()-t0)/iters)
    return best*1e3

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('-B',type=int,default=8); p.add_argument('-T',type=int,default=128)
    p.add_argument('--d',type=int,default=128); p.add_argument('--Mc',type=int,default=64)
    p.add_argument('--Md',type=int,default=16); p.add_argument('--G',type=int,default=8)
    p.add_argument('--dtype',default='float32')
    a=p.parse_args(argv); dev=a.device; dt=getattr(torch,a.dtype)
    z=torch.randn(a.B,a.T,a.d,device=dev,dtype=dt,requires_grad=True)
    h=torch.randn(a.B,a.T,a.d,device=dev,dtype=dt)
    zt=torch.randn(a.B,a.d,device=dev,dtype=dt); ht=torch.randn(a.B,a.d,device=dev,dtype=dt)

    seen={}
    for name,v in VARIANTS.items():
        for slot,cls,args in (('C',v['c'],(a.d,a.Mc)),('D',v['d'],(a.d,a.Md,a.G))):
            if cls in seen: continue
            seen[cls]=1
            m=cls(*args).to(device=dev,dtype=dt)
            torch.cuda.reset_peak_memory_stats() if dev.startswith('cuda') else None
            with torch.no_grad(): fwd=_t(lambda: m.prefill(z,h),dev,5)
            mem=torch.cuda.max_memory_allocated()/1e6 if dev.startswith('cuda') else 0
            def fb():
                m.zero_grad(set_to_none=True)
                if z.grad is not None: z.grad=None
                m.prefill(z,h)[0].square().mean().backward()
            torch.cuda.reset_peak_memory_stats() if dev.startswith('cuda') else None
            tr=_t(fb,dev,5); memt=torch.cuda.max_memory_allocated()/1e6 if dev.startswith('cuda') else 0
            with torch.no_grad():
                st=m.init_state(a.B,z.device,dt)
                def dec():
                    s=st
                    for _ in range(32): _,s=m.step(zt,ht,s)
                d=_t(dec,dev,1,warmup=2)/32*1e3
            print(f"{slot} {cls.__name__:<14s} fwd {fwd:8.2f} ms ({mem:6.0f} MB)  "
                  f"fwd+bwd {tr:8.2f} ms ({memt:6.0f} MB)  decode {d:8.1f} us/tok")
            del m
            if dev.startswith('cuda'): torch.cuda.empty_cache()
    return 0

if __name__=='__main__': sys.exit(main())
