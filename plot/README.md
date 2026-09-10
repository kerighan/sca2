# Where the SCA2 vs Gated DeltaNet comparison actually stands

Regenerate everything with:

```bash
python plot/copy_figs.py            # fig1-fig5 + RESULTS.md, from runs/copy*.jsonl
python plot/lm_figs.py runs/pub3.jsonl   # fig6
```

No figure re-runs a model. Every number is read out of a JSONL in `runs/`, so a
plot can only show what was measured.

## The one-line summary

**SCA2 wins the copy task decisively. GDN wins language modelling. The copy
advantage does not currently transfer.** Both halves of that sentence are
measured, and the second half is the open problem.

## What is solid

**Copy, at matched recurrent state (fig2, fig3).** Pairing state sizes to
within 0.5% (`Mc` for SCA2, `gdn_head_k` for GDN), exact-match on a 128-symbol
string at step 12k:

| state (floats) | SCA2 | GDN |
|---:|---|---|
| ~9.4k | **0.77** | 0.14 |
| ~34k | **0.94** | 0.50 |
| ~67k | **0.98** | 0.73 |

Three points, same direction, gaps of 0.25-0.63. SCA2 also gets there for less
compute: at ~67k state it solved in 4500 steps / 131s against GDN's >12000
steps / 636s.

**Depth is what unlocks long copy (fig4).** At `Mc`=128, 1 and 2 layers were
still under 0.8 at step 8k; 4 layers solved at step 2000. This holds with the
state held constant too (`Mc` = 256/128/64 for 1/2/4 layers), so it is depth
itself and not the extra state that buys the capability. GDN does not benefit:
4 layers left it at 0.34.

**`Mc` is the capacity axis, not state size in general.** The `Md`=32 control
carries *more* state than `Mc`=128 (41218 vs 34050 floats), costs 75 ms/step
against 30, and copies worse. Widening the D head buys nothing here.

**Width `d` is the wrong place to spend (dim sweep in RESULTS.md).** At fixed
state, `d`=64 with `Mc`=256 solved in 4500 steps with 62k params/layer;
`d`=256 with `Mc`=64 needed 672k params/layer and never solved. GDN is flat in
`d` (0.43 / 0.40 / 0.42 for d = 64 / 128 / 256) -- expected, since its state
`heads*head_k^2` does not depend on `d`.

**A protocol trap, worth its own figure (fig5).** Training on a length mix that
includes an out-of-capacity length collapses the *solvable* lengths: same
layer, same padding, same budget, L128 exact-match goes 0.97 -> 0.33 and even
L32 drops from 1.00 to 0.95. This is what made the first pass of this study
conclude the opposite of the truth.

## What is not solid

- **Single seed everywhere.** Exact-match moves by ~0.2 between consecutive
  evals below `Mc`=128. Individual points in fig1 are not readable; only the
  trend is. The `Mc`=32 vs 64 ordering in particular is noise.
- **Nothing is trained to convergence.** Most arms were still climbing at the
  cut, so the copy numbers are lower bounds, not plateaus.
- **State-matching unbalances parameters** (GDN `head_k`=101 carries 44% more
  params than SCA2 `Mc`=256). No configuration matches both at once; that is a
  property of the two architectures, not of the protocol.

## The open problem: language modelling (fig6)

TinyPython, 1 epoch over 16.2M tokens, 2 layers, d=128, parameter-matched
(1.78M vs 1.83M), equal tokens:

| | val loss | tok/s | training time |
|---|---|---|---|
| SCA2 `v3polarflat_cc` | 0.877 | 208k | 78s |
| **GDN** | **0.797** | 157k | 104s |

GDN is better per token *and* per second, despite SCA2 running 1.3x faster.
The generated samples say the same thing more bluntly: GDN emits valid Python,
SCA2 emits code with syntax errors (`len(best | None)`, `if strings =
strings`). See the end of `runs/pub3.log`.

So the copy advantage is real and large, and it does not show up as better
language modelling. Candidate explanations, none tested:

1. **Depth.** The copy win needed 4 layers; this LM run used 2, which is
   exactly where copy was weakest.
2. **`Mc` is small here.** `Mc`=128 is the configuration that copies *worst* in
   fig1. The LM run never tried `Mc`=256, which is where copy became free.
3. **Copy may not be the bottleneck on TinyPython.** 1 epoch of templated
   short functions may simply not require long-range retrieval, in which case
   the benchmark cannot show the advantage and a retrieval-heavy corpus is
   needed.
4. **The D head / read-out may be the weak part** for next-token prediction,
   independent of the C head's addressing.

Distinguishing (1) and (2) is cheap -- rerun fig6 at 4 layers and `Mc`=256 --
and should come before any further architecture work.

## Earlier mistakes, recorded so they are not repeated

- Reported **token accuracy** instead of exact-match. On a 128-token string
  0.99 token accuracy still fails the copy ~3 times in 4.
- Compared at a **fixed step budget** and called an undertrained arm
  capacity-limited. `Mc`=128 read 0.57 at step 6k and 0.91 at 12k.
- Read GDN's state as 24840 floats by counting only the top level of its state
  dict, which made the task look rigged in SCA2's favour.
- Trained **132 epochs** on a 20k-example corpus, ranking the arms on
  memorisation with val loss well past its minimum. Hence `--epochs`.
