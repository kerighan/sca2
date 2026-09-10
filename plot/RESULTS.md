# Copy task: measured results

Exact-match = the entire string reproduced after the separator.
`solve` = first step with exact-match >= 0.95 on L=128, twice in a row.

## copy_state -- SCA2 vs GDN at matched state size

| arm | state | solve | L16 | L32 | L64 | L128 |
|---|---:|---:|---|---|---|---|
| `v3polarflat_cc/Mc=32` | 9474 | >12000 | 1.00 | 0.98 | 0.96 | 0.77 |
| `gdn_cc/gdn_head_k=35` | 9240 | >12000 | 1.00 | 1.00 | 0.88 | 0.14 |
| `v3polarflat_cc` | 34050 | >12000 | 1.00 | 0.99 | 0.99 | 0.94 |
| `gdn_cc/gdn_head_k=71` | 34080 | >12000 | 1.00 | 1.00 | 0.96 | 0.50 |
| `v3polarflat_cc/Mc=256` | 66818 | 4500 | 1.00 | 1.00 | 1.00 | 0.98 |
| `gdn_cc/gdn_head_k=101` | 66660 | >12000 | 1.00 | 1.00 | 0.99 | 0.73 |

## copy_mc -- SCA2 C-head width sweep (+ Md control)

| arm | state | solve | L16 | L32 | L64 | L128 |
|---|---:|---:|---|---|---|---|
| `v3polarflat_cc/Mc=32` | 9474 | >12000 | 1.00 | 0.97 | 0.96 | 0.76 |
| `v3polarflat_cc/Mc=64` | 17666 | >12000 | 1.00 | 0.99 | 0.91 | 0.69 |
| `v3polarflat_cc` | 34050 | >12000 | 1.00 | 1.00 | 0.99 | 0.91 |
| `v3polarflat_cc/Mc=256` | 66818 | 4000 | 1.00 | 1.00 | 1.00 | 1.00 |
| `v3polarflat_cc/Md=32` | 41218 | >6000 | 1.00 | 0.98 | 0.92 | 0.66 |

## copy_dim -- model width at matched state

| arm | state | solve | L16 | L32 | L64 | L128 |
|---|---:|---:|---|---|---|---|
| `v3polarflat_cc/d=64:Mc=256:ff=182` | 33410 | 4500 | 1.00 | 1.00 | 0.97 | 0.96 |
| `gdn_cc/d=64:gdn_head_k=71:ff=182` | 34080 | >8000 | 1.00 | 1.00 | 0.95 | 0.43 |
| `v3polarflat_cc` | 34050 | >8000 | 1.00 | 0.97 | 0.96 | 0.74 |
| `gdn_cc/gdn_head_k=71` | 34080 | >8000 | 1.00 | 0.98 | 0.95 | 0.40 |
| `v3polarflat_cc/d=256:Mc=64:ff=728` | 35330 | >8000 | 1.00 | 0.98 | 0.98 | 0.88 |
| `gdn_cc/d=256:gdn_head_k=71:ff=728` | 34080 | >8000 | 1.00 | 1.00 | 0.97 | 0.42 |

## copy_depth -- depth

| arm | state | solve | L16 | L32 | L64 | L128 |
|---|---:|---:|---|---|---|---|
| `v3polarflat_cc/layers=1` | 17025 | >8500 | 0.99 | 0.93 | 0.77 | 0.41 |
| `v3polarflat_cc/layers=1:Mc=256` | 33409 | >8000 | 1.00 | 0.99 | 0.98 | 0.93 |
| `v3polarflat_cc/layers=4:Mc=64` | 35332 | 2500 | 1.00 | 1.00 | 1.00 | 0.99 |
| `v3polarflat_cc/layers=4` | 68100 | 2000 | 1.00 | 1.00 | 1.00 | 1.00 |
| `gdn_cc/layers=4:gdn_head_k=50` | 35400 | >8000 | 1.00 | 0.98 | 0.95 | 0.34 |
