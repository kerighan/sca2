# Running the Zyda-2 experiment on vast.ai

## What this is for

Every architecture we have tried lands on the same wall-clock curve on
codeparrot: at 14 h of training, four models spanning 111M–157M parameters and
22k–35k tok/s sit within 0.071 nats of each other, with the optimum in the
*middle* of the range rather than at an extreme. That is the compute-optimal
frontier being flat, not the architectures being equivalent.

The open question is whether the small, fast models stay on that frontier or
eventually fall off it. Fitted slopes say the gap grows from ~0.00 to ~0.025
nats between 1.5B and 10B tokens — right at our 0.02 noise floor, so a single
deeper run on the same corpus would not settle it. Two things have to change
together: a real pretraining corpus (Zyda-2, so the result transfers), and
about 4× the compute at equal wall-clock per arm.

## The pipeline

```
prep_zyda.py   Zyda-2 sample-100BT -> 32k BPE -> uint16 .bin + .json meta
pretrain.py    memory-maps the .bin, so corpus size is bounded by disk not RAM
vast/          provision -> setup -> submit -> logs/collect -> teardown
```

The tokenizer is fitted **locally, once**, and uploaded. A corpus built with a
different BPE is a different experiment, and refitting on the host would
silently produce one.

## Sequence

```bash
# 0. once, locally: the tokenizer (~2 min) and a small corpus to smoke-test with
python prep_zyda.py --tokens 2000000000 --out zyda32k_2B --bpe zyda_bpe32k

# 1. preview offers and the real hourly cost, rent nothing
python -m vast.provision --dry-run

# 2. rent, wait for ssh, prove a bf16 matmul on a real GPU
python -m vast.provision --max-hourly 0.90

# 3. upload code + tokenizer, install deps, smoke-test the layer,
#    then build the corpus ON the host (~8 h for 15B tokens)
python -m vast.setup --tokens 15000000000

# 4. follow the build
python -m vast.logs --name prep_zyda

# 5. one arm at a time — they share a GPU, so concurrency would destroy
#    the wall-clock comparison that is the whole point
python -m vast.submit --name z_dv256 --arm dv256 --hours 64

# 6. start the collector immediately, in the foreground, in its own shell.
#    Never detach it: a detached watcher cannot notify anyone when it ends,
#    so the run finishes in silence and the hardware keeps billing.
python -m vast.collect --log zyda --watch 900

# 7. when every arm has been collected AND validated
python -m vast.teardown --yes      # exits non-zero if anything still bills
```

Arms: `dv128` (111M, fastest), `dv256` (121M), `dv384` (130M, the current
optimum), `gdn` (157M, the reference). All share `--init-v2 --v-silu
--gdn-gate --lam-free --layer-scale --Ls 128`; only `Mc`/`dv` differ, so the
comparison isolates mixer size.

## Costs

At $0.60/h, one 64 h arm is about $38. The minimal decisive set is three arms
(`dv128`, `dv384`, `gdn`) = ~$115 plus ~8 h of corpus build. Two seeds per arm
would halve the noise floor to ~0.014 and roughly double that.

An idle instance bills at the same rate as a busy one, and a *stopped*
instance still bills its disk. `vast.teardown` reports both.

## Things that will bite

- **`pip install torch` anywhere.** The image ships a CUDA-enabled build; a
  requirement naming torch pulls the CPU wheel over it and training fails with
  no GPU. `vast/setup.py` installs `--no-deps` and never names torch.
- **The Triton kernel is tuned for the GB10 (sm 12.1).** On a 4090 (sm 8.9) or
  5090 (sm 12.0) the tile sizes in `lapa/triton_scan.py` are not optimal and
  may need a retune; `TRITON.md` documents the constants. `vast/setup.py`
  smoke-tests a forward and backward, so a host where the kernel does not
  build fails before any long job.
- **Corpus size vs epochs.** At 64 h the fastest arm sees roughly 8–15B tokens
  depending on the host. Below ~1 epoch there is no repetition; repeated data
  favours the larger model, which biases exactly the effect being measured.
  Size `--tokens` above what the fastest arm will consume.
- **PID, not `pgrep -f`.** The pattern matches the grep itself, so a watcher
  built on it reports a dead run as alive. Every module here reads the PID
  written at submission.
