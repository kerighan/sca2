# lapa.benchmarks

Clean, scalable benchmarks of Laplace Attention against Gated DeltaNet (fla's
reference implementation) and a causal-attention ceiling. Every script logs
JSONL (one record per eval, with params/layer and state size) and has a `plot`
subcommand that reads the log back.

## Copy capacity (`copy.py`)

Reproduce a random string just read: `[BOS] s_1..s_L [SEP] s_1..s_L`, scored on
the second copy. No prior can help, so it measures state capacity and addressing
alone. Lengths are cycled during training and reported separately, giving one
accuracy-vs-step curve per length per arm.

Local sanity (d=128, minutes):
```bash
python -m lapa.benchmarks.copy run --arm lapa --label "LapA M=190" --lengths 16,32,64,128 --steps 3000 --log runs/copy_d128.jsonl
python -m lapa.benchmarks.copy run --arm gdn  --label "GDN 3x60"   --lengths 16,32,64,128 --steps 3000 --ff 260 --log runs/copy_d128.jsonl
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
```

At scale (d=1024, Spark; needs >> 8 GB). Two questions: does `M` have to grow
with `d` (if `M=256` still copies L=512 at d=1024, the LapA mixer stays linear in
d and its speed advantage grows with scale), and where does each arm's cliff sit
at matched state.
```bash
C="--d 1024 --layers 2 --ff 2048 --lengths 128,256,512,1024 --symbols 256 --batch 16 --steps 6000 --lr 5e-4 --dtype bf16 --compile --log runs/copy_d1024.jsonl"
for M in 128 256 512 1024; do python -m lapa.benchmarks.copy run --arm lapa --label "LapA M=$M" --M $M --dv 256 $C; done
python -m lapa.benchmarks.copy run --arm gdn --label "GDN 8x128" --gdn-heads 8 --gdn-head-k 128 $C     # state 140k ~ LapA M=256 (136k)
python -m lapa.benchmarks.copy run --arm gdn --label "GDN 4x256" --gdn-heads 4 --gdn-head-k 256 $C     # state 271k ~ LapA M=512
python -m lapa.benchmarks.copy plot runs/copy_d1024.jsonl --out plot/copy_d1024.png
```
Params are reported, not matched: GDN's q/k/v/gate projections scale with d²
(9.5M/layer at d=1024 vs LapA's 6.1M); the honest axis is accuracy vs state.

## Speed (`speed.py`, `vs_gdn.py`)

```bash
python -m lapa.benchmarks.speed                                  # compiled/eager x bf16/fp32
python -m lapa.benchmarks.speed --sections                       # long / short / FFN split
python -m lapa.benchmarks.speed --chunks 64,128,256,512 --paths batched,chunk
python -m lapa.benchmarks.vs_gdn                                 # LapA vs GDN, 3 columns
python -m lapa.benchmarks.vs_gdn --gdn-kernel naive              # the pre-Spark comparison
```

Both use a blocked design (every arm inside every round, only within-round ratios
kept). `vs_gdn` defaults to fla's **Triton** kernels, which run on Blackwell but not
on sm_75 — they are worth 1.66x over fla's naive reference, so every speed number in
this repo taken before the Spark is measured against an arm that is 1.66x too slow.
`--gdn-kernel naive` reproduces it for comparison. See `../../SPARK.md` §9.

## Rules that carried over from the campaign (see `../../WINNERS.md`)

- Time only in blocked designs (`python -m lapa.benchmarks.speed`, `python -m
  sca2.autotune`); a tok/s printed while another job shares the GPU is not a
  measurement.
- A seed changes a run's level far more than its shape; convergence runs (one seed,
  long) beat extra seeds at short budget for reading whether a gap closes.
- Nothing is a verdict before ~90% of the budget; eval spikes come from duplicated
  documents, check the train loss at the spike.
