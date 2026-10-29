# fig3_sheet

Thesis asset: `writing/thesis/assets_generated/simulation/prediction/at_the_end/fig3_sheet.pdf` (fig:app:fig3-sheet)

Files: `fig3_sheet.png`

## What is shown
- one glyph per (model, context size) on the Pareto front over the context sizes (staircase line, steps-post), context-blind models (DecPFN, DecPFN-no-St hollow, LC-PFN) as one point pooled over every context size
- series: DecPFN = `decpfn/{p}/binned/1M`, DecPFN-no-St = `decpfn-nostates/{p}/binned/1M`, LC-PFN = `lcpfn/{p}/binned/1M`, FT-PFN = `ftpfn/{p}/default/default`, XDecPFN = `xdecpfn/{p}/default/default`, Trajectory-TabPFN = `tabpfn/50`, Segment-TabPFN = `segment-tabpfn/split`, Segment-GP = `segment-gp`
- y ranges: all problems [-1, 2], boiler [-1, 2], three-tank [-1, 2], PT2 dead-time [-2, 4], cartpole [-1, 2], furuta [-1, 3]

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
- boiler, DecPFN (decpfn/{p}/binned/1M): pooled 1.031 @ 0.002653 s
- boiler, LC-PFN (lcpfn/{p}/binned/1M): pooled 0.580 @ 0.003979 s
- boiler, DecPFN-no-St (decpfn-nostates/{p}/binned/1M): pooled 0.841 @ 0.002702 s
- boiler, XDecPFN (xdecpfn/{p}/default/default): n=0 0.692 @ 0.003503 s, n=2 1.080 @ 0.006016 s, n=4 0.850 @ 0.008341 s (not drawn), n=8 1.026 @ 0.01265 s (not drawn), n=16 1.083 @ 0.02207 s, n=32 1.103 @ 0.04013 s, n=64 0.939 @ 0.07506 s (not drawn)
- boiler, FT-PFN (ftpfn/{p}/default/default): n=0 -0.938 @ 0.003407 s, n=2 1.324 @ 0.003442 s, n=4 0.144 @ 0.003664 s (not drawn), n=8 0.758 @ 0.003646 s (not drawn), n=16 1.265 @ 0.003773 s (not drawn), n=32 1.330 @ 0.003954 s, n=64 1.450 @ 0.005031 s
- boiler, Trajectory-TabPFN (tabpfn/50): n=0 -18.354 @ 0.3428 s, n=2 -0.248 @ 0.3802 s (not drawn), n=4 -1.387 @ 0.3611 s, n=8 0.679 @ 0.368 s, n=16 1.057 @ 0.3788 s (not drawn), n=32 1.090 @ 0.3742 s, n=64 0.123 @ 0.588 s (not drawn)
- boiler, Segment-TabPFN (segment-tabpfn/split): n=2 -1.226 @ 0.3449 s, n=4 -1.892 @ 0.3443 s, n=8 -0.446 @ 1.72 s, n=16 0.275 @ 3.092 s, n=32 0.479 @ 5.493 s, n=64 0.516 @ 10.92 s
- boiler, Segment-GP (segment-gp): n=2 -1.064 @ 0.1089 s, n=4 -3.766 @ 0.1082 s, n=8 0.192 @ 0.5474 s, n=16 0.942 @ 0.9954 s, n=32 0.617 @ 1.807 s (not drawn), n=64 0.526 @ 3.829 s (not drawn)
- three-tank, DecPFN (decpfn/{p}/binned/1M): pooled 0.823 @ 0.002613 s
- three-tank, LC-PFN (lcpfn/{p}/binned/1M): pooled 0.035 @ 0.00392 s
- three-tank, DecPFN-no-St (decpfn-nostates/{p}/binned/1M): pooled 0.118 @ 0.002629 s
- three-tank, XDecPFN (xdecpfn/{p}/default/default): n=0 -0.925 @ 0.003336 s, n=2 0.911 @ 0.005743 s, n=4 0.766 @ 0.008054 s (not drawn), n=8 0.892 @ 0.01259 s (not drawn), n=16 0.738 @ 0.02101 s (not drawn), n=32 1.048 @ 0.03757 s, n=64 0.990 @ 0.07133 s (not drawn)
- three-tank, FT-PFN (ftpfn/{p}/default/default): n=0 -0.150 @ 0.003264 s, n=2 1.144 @ 0.003336 s, n=4 1.206 @ 0.003496 s, n=8 1.203 @ 0.003475 s, n=16 1.156 @ 0.003506 s (not drawn), n=32 1.376 @ 0.003731 s, n=64 1.210 @ 0.004348 s (not drawn)
- three-tank, Trajectory-TabPFN (tabpfn/50): n=0 -32.928 @ 0.3365 s, n=2 0.254 @ 0.3728 s (not drawn), n=4 0.381 @ 0.3518 s, n=8 0.304 @ 0.3439 s, n=16 -0.126 @ 0.3563 s (not drawn), n=32 0.613 @ 0.3702 s, n=64 0.667 @ 0.6067 s
- three-tank, Segment-TabPFN (segment-tabpfn/split): n=2 -0.615 @ 0.3483 s, n=4 0.371 @ 0.6681 s, n=8 0.457 @ 0.6833 s, n=16 0.779 @ 0.6839 s, n=32 0.946 @ 1.368 s, n=64 1.079 @ 3.093 s
- three-tank, Segment-GP (segment-gp): n=2 -1.291 @ 1.036 s (not drawn), n=4 -0.580 @ 0.2096 s (not drawn), n=8 0.443 @ 0.2085 s (not drawn), n=16 0.651 @ 0.2072 s, n=32 0.898 @ 0.4152 s, n=64 1.314 @ 0.9451 s
- PT2 dead-time, DecPFN (decpfn/{p}/binned/1M): pooled 1.756 @ 0.002779 s
- PT2 dead-time, LC-PFN (lcpfn/{p}/binned/1M): pooled 1.283 @ 0.004169 s
- PT2 dead-time, DecPFN-no-St (decpfn-nostates/{p}/binned/1M): pooled 1.612 @ 0.004013 s
- PT2 dead-time, XDecPFN (xdecpfn/{p}/default/default): n=0 1.120 @ 0.003363 s, n=2 1.301 @ 0.003472 s, n=4 2.860 @ 0.008048 s (not drawn), n=8 2.975 @ 0.01226 s (not drawn), n=16 2.810 @ 0.0229 s (not drawn), n=32 2.142 @ 0.004829 s, n=64 3.209 @ 0.00506 s
- PT2 dead-time, FT-PFN (ftpfn/{p}/default/default): n=0 0.671 @ 0.003479 s (not drawn), n=2 0.725 @ 0.003352 s, n=4 2.344 @ 0.003564 s (not drawn), n=8 2.970 @ 0.003538 s, n=16 2.790 @ 0.003715 s (not drawn), n=32 1.565 @ 0.00385 s (not drawn), n=64 2.851 @ 0.004137 s (not drawn)
- PT2 dead-time, Trajectory-TabPFN (tabpfn/50): n=0 -17.943 @ 0.362 s, n=2 -1.945 @ 0.3974 s (not drawn), n=4 0.230 @ 0.3704 s, n=8 0.714 @ 0.3727 s, n=16 0.551 @ 0.3821 s (not drawn), n=32 -0.928 @ 0.3767 s (not drawn), n=64 0.829 @ 0.3741 s
- PT2 dead-time, Segment-TabPFN (segment-tabpfn/split): n=2 -17.943 @ 0.0001169 s, n=4 -3.963 @ 0.0001888 s, n=8 -1.197 @ 0.0002114 s, n=16 -1.470 @ 0.3713 s (not drawn), n=32 -2.196 @ 0.3708 s (not drawn), n=64 -0.808 @ 0.0004001 s
- PT2 dead-time, Segment-GP (segment-gp): n=2 -17.943 @ 0.0001121 s, n=4 -4.654 @ 0.0002087 s, n=8 -2.013 @ 0.0002292 s, n=16 -2.590 @ 0.1115 s (not drawn), n=32 -2.670 @ 0.1108 s (not drawn), n=64 -1.380 @ 0.0003564 s
- cartpole, DecPFN (decpfn/{p}/binned/1M): pooled 0.854 @ 0.002631 s
- cartpole, LC-PFN (lcpfn/{p}/binned/1M): pooled 0.242 @ 0.003947 s
- cartpole, DecPFN-no-St (decpfn-nostates/{p}/binned/1M): pooled 0.515 @ 0.002685 s
- cartpole, XDecPFN (xdecpfn/{p}/default/default): n=0 1.269 @ 0.003356 s, n=2 1.312 @ 0.005941 s (not drawn), n=4 1.613 @ 0.003648 s, n=8 1.516 @ 0.0139 s (not drawn), n=16 1.330 @ 0.02326 s (not drawn), n=32 1.561 @ 0.04312 s (not drawn), n=64 1.483 @ 0.08071 s (not drawn)
- cartpole, FT-PFN (ftpfn/{p}/default/default): n=0 1.095 @ 0.003146 s, n=2 1.182 @ 0.003391 s (not drawn), n=4 1.387 @ 0.003254 s, n=8 1.373 @ 0.003553 s (not drawn), n=16 1.232 @ 0.003667 s (not drawn), n=32 1.458 @ 0.003942 s, n=64 1.409 @ 0.004671 s (not drawn)
- cartpole, Trajectory-TabPFN (tabpfn/50): n=0 -15.353 @ 0.34 s, n=2 -2.210 @ 0.3844 s (not drawn), n=4 -0.166 @ 0.366 s, n=8 -0.034 @ 0.3742 s, n=16 0.170 @ 0.3824 s (not drawn), n=32 0.390 @ 0.3809 s, n=64 0.525 @ 0.6413 s
- cartpole, Segment-TabPFN (segment-tabpfn/split): n=2 -15.353 @ 7.795e-05 s, n=4 -0.129 @ 0.6994 s (not drawn), n=8 0.035 @ 0.3537 s, n=16 0.323 @ 2.114 s, n=32 0.384 @ 3.541 s, n=64 0.495 @ 9.546 s
- cartpole, Segment-GP (segment-gp): n=2 -15.353 @ 8.042e-05 s, n=4 -0.522 @ 0.2104 s (not drawn), n=8 0.335 @ 0.1047 s, n=16 0.574 @ 0.6322 s, n=32 0.491 @ 1.065 s (not drawn), n=64 0.676 @ 3.004 s
- furuta, DecPFN (decpfn/{p}/binned/1M): pooled 0.841 @ 0.002728 s
- furuta, LC-PFN (lcpfn/{p}/binned/1M): pooled 0.116 @ 0.004092 s
- furuta, DecPFN-no-St (decpfn-nostates/{p}/binned/1M): pooled 0.252 @ 0.002693 s
- furuta, XDecPFN (xdecpfn/{p}/default/default): n=0 0.317 @ 0.003331 s, n=2 1.105 @ 0.005846 s, n=4 1.165 @ 0.008484 s, n=8 1.583 @ 0.01334 s, n=16 2.206 @ 0.02207 s, n=32 2.155 @ 0.04038 s (not drawn), n=64 2.229 @ 0.07532 s
- furuta, FT-PFN (ftpfn/{p}/default/default): n=0 -0.226 @ 0.003336 s, n=2 0.666 @ 0.00341 s, n=4 0.844 @ 0.003456 s, n=8 1.323 @ 0.003661 s, n=16 1.822 @ 0.003715 s, n=32 1.859 @ 0.003871 s, n=64 2.086 @ 0.004515 s
- furuta, Trajectory-TabPFN (tabpfn/50): n=0 -18.445 @ 0.3456 s, n=2 -1.045 @ 0.4048 s (not drawn), n=4 -0.764 @ 0.3721 s, n=8 -0.061 @ 0.378 s, n=16 1.267 @ 0.3915 s, n=32 1.050 @ 0.3889 s, n=64 1.440 @ 0.7301 s
- furuta, Segment-TabPFN (segment-tabpfn/split): n=2 -18.445 @ 0.0001176 s, n=4 -2.220 @ 0.3468 s, n=8 -0.747 @ 0.3476 s, n=16 0.216 @ 0.349 s, n=32 0.740 @ 2.438 s, n=64 1.238 @ 4.197 s
- furuta, Segment-GP (segment-gp): n=2 -18.445 @ 0.0001155 s, n=4 -2.283 @ 0.1082 s (not drawn), n=8 -0.862 @ 0.1067 s, n=16 -0.136 @ 0.1104 s, n=32 0.084 @ 0.7717 s, n=64 0.367 @ 1.315 s
- med_D: boiler -4.6723, cartpole -7.6728, furuta -4.5805, pt2_deadtime -5.0832, threetank_pi 9.9019

## Data
- source: thesis
- thesis import, variant `thesis` (the numbers sim_results.tex quotes): pooled values are `fig3_cross` as imported (already normalized), per-problem values are `fig3_per_problem` minus the imported `fig3_med_D` (median over the thesis's full participant set of 38 old models, not the spec's)
- no timing record on the problem, the cross-problem median time of the same model and context size is used: furuta/DecPFN-no-St

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `prediction_timing`: 382 units (thesis), hashes 00da9f7ed2, adc956edb4, ad8326c8f6, ad0813c702, abcc3ec088, abc4b5e92b, ...
- `thesis_series`: 161 units (thesis), hashes 01761f98af, 0369cea4c0, 0c4c093618, 122023a33e, 34fa8890fd, 51ba1c8d96, ...
