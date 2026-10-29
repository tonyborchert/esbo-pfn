# landscape_furuta

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/ftpfn_xdecpfn/landscape_furuta_BEST_NO_DELETE.pdf` (fig:sim:landscape-furuta)

Files: `landscape_furuta.png`

## What is shown
- 2 x 5 sheet on one 2-D oblique slice of one Furuta plant: XDecPFN (top row) and FT-PFN (bottom row) at context sizes n = 1, 4, 16, 64, the real landscape top right, the color key bottom right.
- Each pixel is one slice controller. Blue: per-panel rank of the mean total cost (brighter = cheaper), red: crash probability, blended additively (black = expensive and safe).

## Metric
- Real panel: mean J_T over the non-crashed repeats and the crash share of the six repeats. Model panels: the predictive mean of J_T without crash (value part of the bar distribution) and the predicted crash probability for the unrun controller (`predict_unrun`).

## Key numbers
- XDecPFN n=1: J in [239, 411], mean crash probability 62%
- XDecPFN n=4: J in [242, 437], mean crash probability 45%
- XDecPFN n=16: J in [276, 428], mean crash probability 57%
- XDecPFN n=64: J in [277, 432], mean crash probability 70%
- FT-PFN n=1: J in [250, 401], mean crash probability 62%
- FT-PFN n=4: J in [284, 429], mean crash probability 58%
- FT-PFN n=16: J in [245, 435], mean crash probability 62%
- FT-PFN n=64: J in [252, 446], mean crash probability 72%
- real: J in [242, 413], mean crash probability 62%

## Data
- Plant: instance 1 of the thesis-imported Furuta instance list (first of instances 0-9 with a true crash share in [0.25, 0.85] (0.62)). Slice: `landscapes` unit slice seed 1, 24 x 24 grid, 6 repeats per point, extent 0.2729.
- XDecPFN context: prefixes of one `xdecpfn-pi` run (seed 0, 64 trials, 6 crashed) on the same plant with `xdecpfn/furuta/thesis/default` (checkpoint 1526703a195b9062), cached under `_cache/examples/`.
- FT-PFN context: prefixes of one `ftpfn-ei` run (seed 0, 64 trials, 16 crashed) on the same plant with `ftpfn/furuta/thesis/default` (checkpoint 31d80ad8f4bb7557), cached under `_cache/examples/`.
- Live results (simulated runs, true slice, model predictions) are cached as npz under `_cache/examples/` next to this figure, keyed by plant, slice, model checkpoint hash and context. Delete that directory to recompute them.

## Sources
- file `_cache/examples/slice_furuta_thesis_i0001_s1_g24_r6/landscape.npz`
