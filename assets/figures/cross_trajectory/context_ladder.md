# context_ladder

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/ftpfn_xdecpfn/context_ladder.pdf` (fig:sim:context-sweep)

Files: `context_ladder.png`

## What is shown
- One panel per problem (PT2 dead-time, Cartpole, Furuta), XDecPFN and FT-PFN, every rung drawn at equal spacing, a subset of rungs labeled.
- The query is fixed (last trial of each `oracle181` run, the first 50 non-crashed in seed-major order), the context is the first n trials of the same run. The top Cartpole/Furuta rung is labeled with its stored value 180 (the thesis axis printed 179).

## Metric
- y: conditioned LL, mean over the five cutoffs per query, then the mean over the queries (not normalized, a mean rather than the medians used elsewhere).

## Key numbers
- PT2 dead-time XDecPFN: n=0 -3.370, n=1 -3.086, n=2 -2.813, n=4 -2.716, n=8 -2.361, n=16 -2.236, n=33 -2.185, n=67 -2.199, n=134 -2.255 (50-50 queries per rung)
- PT2 dead-time FT-PFN: n=0 -4.099, n=1 -3.925, n=2 -3.774, n=4 -3.368, n=8 -2.982, n=16 -2.720, n=33 -2.860, n=67 -2.917, n=134 -3.142 (50-50 queries per rung)
- cartpole XDecPFN: n=0 -7.289, n=1 -6.772, n=2 -6.543, n=5 -6.331, n=11 -6.201, n=22 -6.202, n=45 -6.215, n=90 -6.207, n=180 -6.191 (50-50 queries per rung)
- cartpole FT-PFN: n=0 -7.247, n=1 -7.072, n=2 -7.014, n=5 -6.954, n=11 -6.787, n=22 -6.767, n=45 -6.792, n=90 -6.801, n=180 -6.779 (50-50 queries per rung)
- furuta XDecPFN: n=0 -4.294, n=1 -3.701, n=2 -3.550, n=5 -3.111, n=11 -3.009, n=22 -2.923, n=45 -2.894, n=90 -2.891, n=180 -2.843 (50-50 queries per rung)
- furuta FT-PFN: n=0 -4.203, n=1 -3.625, n=2 -3.461, n=5 -3.136, n=11 -3.134, n=22 -2.929, n=45 -2.842, n=90 -2.890, n=180 -3.049 (50-50 queries per rung)

## Data
- XDecPFN on PT2 dead-time is the thesis cell `xdecpfn/pt2_deadtime/thesis/default` when read from the thesis import (the `/lnjw128` twin is not drawn).

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `prediction_ladder`: 6 units (thesis), hashes 087c55d2cd, f26f0b6ce8, 55bf06c724, c31a3b019f, d7f1c4c1c1, d91ff879d8
