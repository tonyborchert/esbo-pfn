# fig3_cross

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/at_the_end/fig3_cross.pdf` (fig:sim:fig3-cross)

Files: `fig3_cross.png`

## What is shown
- one glyph per (model, context size) on the Pareto front over the context sizes (staircase line, steps-post), context-blind models (DecPFN, DecPFN-no-St hollow, LC-PFN) as one point pooled over every context size
- series: DecPFN = `decpfn/{p}/binned/1M`, DecPFN-no-St = `decpfn-nostates/{p}/binned/1M`, LC-PFN = `lcpfn/{p}/binned/1M`, FT-PFN = `ftpfn/{p}/default/default`, XDecPFN = `xdecpfn/{p}/default/default`, Trajectory-TabPFN = `tabpfn/50`, Segment-TabPFN = `segment-tabpfn/split`, Segment-GP = `segment-gp`
- y ranges: all problems [-1, 2]
- glyphs hidden behind a higher-context glyph (line still drawn): FT-PFN n=32, FT-PFN n=16, FT-PFN n=8, FT-PFN n=4, Trajectory-TabPFN n=4, FT-PFN n=2, XDecPFN n=0, FT-PFN n=0

## Metric
- y: median over non-crashed queries of the per-trial conditioned LL (mean over the five cutoffs 10..90 % of T) minus the per-problem median med_D of the participants, pooled across problems on the pooled panel (higher is better)
- x: median inference seconds per prediction (log), the pooled panel takes the median across problems, context-blind models use their n = 0 time (lower is better)

## Key numbers
- all problems, DecPFN (decpfn/{p}/binned/1M): pooled 0.990 @ 0.002653 s
- all problems, LC-PFN (lcpfn/{p}/binned/1M): pooled 0.353 @ 0.003979 s
- all problems, DecPFN-no-St (decpfn-nostates/{p}/binned/1M): pooled 0.602 @ 0.002693 s
- all problems, XDecPFN (xdecpfn/{p}/default/default): n=0 0.611 @ 0.003356 s, n=2 1.086 @ 0.005846 s, n=4 1.207 @ 0.008054 s, n=8 1.284 @ 0.01265 s, n=16 1.349 @ 0.02207 s, n=32 1.438 @ 0.04013 s, n=64 1.463 @ 0.07506 s
- all problems, FT-PFN (ftpfn/{p}/default/default): n=0 0.173 @ 0.003336 s, n=2 1.086 @ 0.003391 s, n=4 1.133 @ 0.003496 s, n=8 1.375 @ 0.003553 s, n=16 1.458 @ 0.003715 s, n=32 1.504 @ 0.003871 s, n=64 1.590 @ 0.004515 s
- all problems, Trajectory-TabPFN (tabpfn/50): n=0 -17.943 @ 0.3428 s, n=2 -0.742 @ 0.3844 s (not drawn), n=4 -0.326 @ 0.366 s, n=8 0.282 @ 0.3727 s, n=16 0.584 @ 0.3821 s (not drawn), n=32 0.677 @ 0.3767 s, n=64 0.772 @ 0.6067 s
- all problems, Segment-TabPFN (segment-tabpfn/split): n=2 -15.353 @ 0.0001176 s, n=4 -1.193 @ 0.3468 s, n=8 -0.273 @ 0.3537 s, n=16 0.265 @ 0.6839 s, n=32 0.376 @ 2.438 s, n=64 0.601 @ 4.197 s
- all problems, Segment-GP (segment-gp): n=2 -15.353 @ 0.0001155 s, n=4 -2.047 @ 0.1082 s (not drawn), n=8 -0.142 @ 0.1067 s, n=16 0.372 @ 0.2072 s, n=32 0.242 @ 0.7717 s (not drawn), n=64 0.552 @ 1.315 s
- med_D: boiler -4.6723, cartpole -7.6728, furuta -4.5805, pt2_deadtime -5.0832, threetank_pi 9.9019

## Data
- source: thesis
- thesis import, variant `thesis` (the numbers sim_results.tex quotes): pooled values are `fig3_cross` as imported (already normalized), per-problem values are `fig3_per_problem` minus the imported `fig3_med_D` (median over the thesis's full participant set of 38 old models, not the spec's)
- no timing record on the problem, the cross-problem median time of the same model and context size is used: furuta/DecPFN-no-St

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `prediction_timing`: 382 units (thesis), hashes 00da9f7ed2, adc956edb4, ad8326c8f6, ad0813c702, abcc3ec088, abc4b5e92b, ...
- `thesis_series`: 161 units (thesis), hashes 01761f98af, 0369cea4c0, 0c4c093618, 122023a33e, 34fa8890fd, 51ba1c8d96, ...
