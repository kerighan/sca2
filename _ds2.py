import torch, time
from sca2.ref import LayerCfg
from sca2.registry import build
B,T,d = 16,256,128
x = torch.randn(B,T,d,device='cuda',requires_grad=True)
def bench(name,var,iters=3):
    cfg = LayerCfg(d,128,16,8,256,freq='rope',theta_scale=0.0,max_len=T)
    t0=time.perf_counter(); m = build(var,cfg,device='cuda')
    def f():
        m.zero_grad(set_to_none=True)
        if x.grad is not None: x.grad = None
        m.prefill(x)[0].square().mean().backward()
    f(); comp=time.perf_counter()-t0
    f()
    torch.cuda.synchronize(); t1=time.perf_counter()
    for _ in range(iters): f()
    torch.cuda.synchronize(); dt=(time.perf_counter()-t1)/iters
    print(f'{name:<26s} {dt*1e3:9.1f} ms/step  {B*T/dt:9.0f} tok/s   (first call {comp:.0f}s)', flush=True)
bench('delta eager',     'polar_delta')
bench('delta compiled',  'polar_delta_cc')
