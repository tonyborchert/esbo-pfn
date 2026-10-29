# ftpfn_xdecpfn_stop_sheet

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/ftpfn_xdecpfn/ftpfn_xdecpfn_stop_sheet.pdf` (fig:sim:ftpfn-xdecpfn-stop-sheet)

Files: `ftpfn_xdecpfn_stop_sheet.png`

## What is shown
- Three panels (PT2 dead-time, Cartpole, Furuta), XDecPFN and FT-PFN, one curve per context size n in [0, 2, 4, 8, 16, 32, 64], alpha rising linearly from 0.30 (n = 0) to 0.95 (n = 64).

## Metric
- Stop at the first predictable step with crash probability >= tau, tau swept over every emitted probability. x: share of non-crashed queries stopped (shown up to 0.1, y fitted to that window), y: mean steps saved on the 50 crashed queries per context size. Each panel has its own y range.

## Key numbers
- PT2 dead-time XDecPFN steps saved at false-stop <= 0.1: n=0 82.8, n=2 71.2, n=4 94.7, n=8 90.3, n=16 77.6, n=32 56.7, n=64 70.5
- PT2 dead-time FT-PFN steps saved at false-stop <= 0.1: n=0 60.1, n=2 36.1, n=4 57.2, n=8 55.3, n=16 47.0, n=32 31.7, n=64 41.1
- cartpole XDecPFN steps saved at false-stop <= 0.1: n=0 274.7, n=2 292.8, n=4 386.4, n=8 412.9, n=16 372.1, n=32 395.4, n=64 396.3
- cartpole FT-PFN steps saved at false-stop <= 0.1: n=0 226.6, n=2 200.6, n=4 277.6, n=8 307.0, n=16 281.0, n=32 290.9, n=64 314.8
- furuta XDecPFN steps saved at false-stop <= 0.1: n=0 76.8, n=2 114.5, n=4 126.0, n=8 161.3, n=16 174.3, n=32 202.9, n=64 204.2
- furuta FT-PFN steps saved at false-stop <= 0.1: n=0 19.3, n=2 47.5, n=4 75.5, n=8 103.8, n=16 116.9, n=32 137.9, n=64 120.7

## Data
- curves from: thesis import
- cartpole n=0: 50 crashed / 204 ok queries
- cartpole n=2: 50 crashed / 203 ok queries
- cartpole n=4: 50 crashed / 279 ok queries
- cartpole n=8: 50 crashed / 309 ok queries
- cartpole n=16: 50 crashed / 314 ok queries
- cartpole n=32: 50 crashed / 317 ok queries
- cartpole n=64: 50 crashed / 335 ok queries
- furuta n=0: 50 crashed / 348 ok queries
- furuta n=2: 50 crashed / 346 ok queries
- furuta n=4: 50 crashed / 357 ok queries
- furuta n=8: 50 crashed / 381 ok queries
- furuta n=16: 50 crashed / 387 ok queries
- furuta n=32: 50 crashed / 417 ok queries
- furuta n=64: 50 crashed / 425 ok queries
- pt2_deadtime n=0: 50 crashed / 187 ok queries
- pt2_deadtime n=2: 50 crashed / 198 ok queries
- pt2_deadtime n=4: 50 crashed / 255 ok queries
- pt2_deadtime n=8: 50 crashed / 278 ok queries
- pt2_deadtime n=16: 50 crashed / 251 ok queries
- pt2_deadtime n=32: 50 crashed / 209 ok queries
- pt2_deadtime n=64: 50 crashed / 214 ok queries

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 6 units (thesis), hashes 1ad9037ecf, 8e8797c3f6, 96a9260a20, b29e8f110f, ddb25cbecb, ec7e9d428b
