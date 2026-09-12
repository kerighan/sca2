#!/usr/bin/env bash
# SPARK.md question (a): does M have to grow with d?
#
# Now urgent rather than merely first in the list. The d=1024 LM run has LapA v1 at
# M=256 sitting +0.17 nats BEHIND GDN across the whole 50-380M token range, a ~0.20
# reversal of the -0.03 it held at d=128, and the loss is worst exactly on tokens
# repeated within the window -- in-context retrieval, the thing the long head's
# addressing is for. The natural reading is that M=256 is not enough addressing at
# d=1024. This is the cheap, direct test of that, and the one SPARK.md §3(a) says to
# run before anything else.
#
# Read it as: if M=256 still copies L=512 at d=1024, M is set by CAPACITY and the
# LM gap is about something else. If the cliff tracks M/d, M must scale with width --
# which removes the linear-in-d mixer and with it the speed argument.
#
# rope-base 2048, not 1000: the rule is 2*base ~ context and these runs use
# T = 2*1024+2 ~ 2050 (SPARK.md §3(a)).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "LONG5H_D1024 DONE" runs/long5h_d1024.log 2>/dev/null; do sleep 120; done
echo "##### $(date +%H:%M) LM queue finished, starting the M sweep"

LOG=runs/copy_d1024.jsonl
C="--d 1024 --layers 2 --ff 4096 --lengths 128,256,512,1024 --symbols 256
   --batch 16 --steps 6000 --lr 5e-4 --dtype bf16 --compile --log $LOG
   --rope-base 2048 --slow-frac 0.25 --L 64"

for M in 128 256 512 1024; do
  echo "##### $(date +%H:%M) LapA M=$M"
  python -u -m lapa.benchmarks.copy run --arm lapa --label "LapA M=$M" --M "$M" --dv 256 $C
done
echo "##### $(date +%H:%M) GDN 8x128"
python -u -m lapa.benchmarks.copy run --arm gdn  --label "GDN 8x128"  --gdn-heads 8 --gdn-head-k 128 $C
echo "##### $(date +%H:%M) GDN2 8x128"
python -u -m lapa.benchmarks.copy run --arm gdn2 --label "GDN2 8x128" --gdn-heads 8 --gdn-head-k 128 $C
python -m lapa.benchmarks.copy plot "$LOG" --out plot/copy_d1024.png
echo "##### COPY_D1024 DONE"
