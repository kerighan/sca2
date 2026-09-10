#!/usr/bin/env bash
# COPY CAPACITY AT d=1024 -- the Spark protocol (needs >> 8 GB; do not run on the 2070).
#
# Question 1 (speed argument): does Mc have to grow with d? At d=128 the copy cliff
# tracked Mc; if at d=1024 Mc=256 still copies L=512 verbatim, the LapA mixer stays
# LINEAR in d and the speed ratio to GDN grows with scale. Mc sweep at fixed dv.
# Question 2 (capacity): per-length accuracy CURVES over training, one curve per L,
# LapA vs GDN at matched state (Mc=256,dv=256 -> 136k floats ~ GDN 8x128 -> 140k).
#
# Lengths 128..1024 -> Tmax = 2*1024+2 = 2050. 2 layers so the mixer is what is
# measured. GDN's params/layer are larger at equal state (9.5M vs 6.1M: its q/k/v/gate
# projections scale with d); the honest axis here is accuracy vs STATE, as in
# bench_copy.py's header -- params are reported, not matched.
#
#   bash copy_d1024.sh            # ~4 arms x (steps) ; adjust --steps/--batch to the GPU
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
COMMON="--d 1024 --layers 2 --ff 2048 --dv 256 --Ls 16 --lengths 128,256,512,1024 --symbols 256
        --batch 16 --steps 6000 --eval-every 250 --eval-batches 4 --eval-batch 32 --lr 5e-4
        --log runs/copy_d1024.jsonl"
python -u bench_copy.py $COMMON --arms "cshort_damph_cc/Mc=128,cshort_damph_cc/Mc=256,cshort_damph_cc/Mc=512,cshort_damph_cc/Mc=1024"
python -u bench_copy.py $COMMON --arms "gdn_cc/gdn_heads=8:gdn_head_k=128,gdn_cc/gdn_heads=4:gdn_head_k=256"
echo "##### COPY D1024 DONE -- plot: per-length accuracy vs step, one panel per L, LapA(Mc) vs GDN(state)"
