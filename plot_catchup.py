"""plot/A_vs_gdn.png: val loss and gap to GDN for the catch-up arms. Re-run any time."""
import json, collections, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=collections.defaultdict(list)
import sys
LOG=sys.argv[1] if len(sys.argv)>1 else 'runs/catchup.jsonl'
for l in open(LOG):
    r=json.loads(l); C[r['model']].append((r['tokens'],r['val']))
def ip(c,x):
    for i in range(1,len(c)):
        if c[i][0]>=x>=c[0][0]:
            (x0,y0),(x1,y1)=c[i-1],c[i]; return y0+(y1-y0)*(x-x0)/max(x1-x0,1)
arms5={'A: damped long head + short dft head':('l5_A_s0','#c0392b',2.4),
      'B: A + key-verification gate':('l5_B_s0','#16a085',2.4),
      'GDN (Gated DeltaNet)':('l5_gdn_s0','#2c3e50',2.4)}
arms={'A: damped long head + short dft head':('catch_shortdamp_s0','#c0392b',2.4),
      'B: A + key-verification gate':('catch_shortdampkv_s0','#16a085',2.4),
      'GDN (Gated DeltaNet)':('catch_gdn_s0','#2c3e50',2.4),
      'gen3 (cdelta)':('catch_gen3_s0','#7f8c8d',1.0),
      'short head only':('catch_short_s0','#e67e22',1.0),
      'short + gate (combo)':('catch_shortkv_s0','#8e44ad',1.0)}
if 'long5h' in LOG: arms=arms5
OUT='plot/long5h.png' if 'long5h' in LOG else 'plot/A_vs_gdn.png'
GDK=[k for k,_,_ in arms.values() if 'gdn' in k][0]
for k,_,_ in arms.values():
    if k not in C: print("missing", k)
GD=C[GDK]
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,9),sharex=True,gridspec_kw={'height_ratios':[3,2]})
for lab,(k,col,lw) in arms.items():
    c=C.get(k)
    if not c: continue
    x=np.array([t for t,_ in c])/1e6; y=np.array([v for _,v in c])
    done = c[-1][0] >= (1.3e9 if 'long5h' in LOG else 500e6)
    ax1.plot(x[1:],y[1:],color=col,lw=lw,label=lab+('' if done else f'  (running, {c[-1][0]/1e6:.0f}M)'),alpha=1 if lw>1.5 else 0.7)
ax1.set_ylim(2.25 if 'long5h' in LOG else 2.55,3.6); ax1.set_ylabel('val loss (nats)'); ax1.grid(alpha=.3); ax1.legend()
ax1.set_title(('pycode XL corpus (1.57B), 5 h per arm' if 'long5h' in LOG else 'pycode big corpus')+', 4 layers d=128, matched params (~186k/layer), seed 0, 60 val batches / eval')
for lab,(k,col,lw) in arms.items():
    c=C.get(k)
    if not c or k==GDK: continue
    hi=min(c[-1][0],GD[-1][0])
    if hi<80e6: continue
    xs=np.arange(40e6,hi+1,10e6); g=np.array([ip(c,x)-ip(GD,x) for x in xs])
    ax2.plot(xs/1e6,g,color=col,lw=0.7,alpha=.3)
    if len(g)>=5: ax2.plot(xs[2:-2]/1e6,np.convolve(g,np.ones(5)/5,'valid'),color=col,lw=lw,label=lab+' (5-pt smooth)')
    if k in ('catch_shortdamp_s0','catch_shortdampkv_s0','l5_A_s0','l5_B_s0') and hi>300e6:
        xs2=np.arange(200e6,hi+1,10e6); g2=np.array([ip(c,x)-ip(GD,x) for x in xs2]); s,c0=np.polyfit(np.log(xs2),g2,1)
        ax2.plot(xs2/1e6,s*np.log(xs2)+c0,color=col,lw=1,ls='--',label=f'{lab.split(":")[0]} trend beyond 200M: {s:+.2f}/ln(tokens)')
ax2.axhline(0,color='k',lw=0.8); ax2.set_ylim(-0.5,0.15); ax2.set_xlabel('tokens (M)'); ax2.set_ylabel('gap to GDN (nats)'); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT,dpi=130); print('saved',OUT)
