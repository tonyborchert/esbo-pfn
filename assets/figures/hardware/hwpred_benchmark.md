# hwpred_benchmark

## What is shown
- one panel per rig, one circle per (predictor, context size n) with n written inside, lines are Pareto staircases over n (dominated rungs dropped), medians below -7 omitted, overlapping circles keep the higher n
- context-blind predictors ['decpfn'] pooled to one smaller unnumbered point (every query counted once, x = median of the per-rung timings)

## Metric
- y = `median_ll_all`: median over queries of the per-query mean conditioned LL over the cutoffs (crashed queries included for `_all`)
- x = seconds per prediction (median per context size)

## Data
- without_weight segment-gp: `hardware_prediction` unit 31f2d8fb0fbd (model `segment-gp (gp_cold_refit_growth 0.0)`, provenance `thesis`, logs `thesis`)
- without_weight decpfn: `hardware_prediction` unit e6ef4752c2ae (model `rigreset_deadtime-strat_kick0.6_binned`, provenance `thesis`, logs `thesis`)
- without_weight xdecpfn: `hardware_prediction` unit bcfa5df91f6d (model `c1r2_furuta_lnJ/f_s20260931_seg6_final.pt`, provenance `thesis`, logs `thesis`)
- with_weight segment-gp: `hardware_prediction` unit 13168a26d5f8 (model `segment-gp (gp_cold_refit_growth 0.0)`, provenance `thesis`, logs `thesis`)
- with_weight decpfn: `hardware_prediction` unit aae5863e9b07 (model `rigreset_deadtime-strat_kick0.6_binned`, provenance `thesis`, logs `thesis`)
- with_weight xdecpfn: `hardware_prediction` unit 02a16b362ade (model `c1r2_furuta_lnJ/f_s20260931_seg6_final.pt`, provenance `thesis`, logs `thesis`)

## Numbers
- without_weight segment-gp (median_ll_all per n) 1: -23.026, 2: -23.026, 5: -16.019, 11: -7.220, 22: -6.784, 45: -8.555, 90: -23.026
- without_weight segment-gp seconds per prediction per n: 90: 6.337, 45: 2.641, 22: 1.326, 11: 0.5872, 5: 0.2426, 2: 0.08169, 1: 6.419e-05
- without_weight decpfn (median_ll_all per n) 0: -3.693, 1: -3.693, 2: -3.678, 5: -3.653, 11: -3.667, 22: -3.641, 45: -3.540, 90: -1.581
- without_weight decpfn seconds per prediction per n: 90: 0.003024, 45: 0.00292, 22: 0.002915, 11: 0.002956, 5: 0.002879, 2: 0.002725, 1: 0.002681, 0: 0.002839
- without_weight decpfn pooled (every query once): median LL -3.693 over 743 queries at 0.002897 s
- without_weight xdecpfn (median_ll_all per n) 0: -4.808, 1: -3.376, 2: -2.998, 5: -2.786, 11: -2.626, 22: -2.497, 45: -2.366, 90: -0.028
- without_weight xdecpfn seconds per prediction per n: 90: 0.04922, 45: 0.01401, 22: 0.01142, 11: 0.009737, 5: 0.008685, 2: 0.008268, 1: 0.007929, 0: 0.007412
- without_weight segment-gp drawn front: n=22 (-6.784, 1.326 s)
- without_weight decpfn drawn front: n=pooled (-3.693, 0.002897 s)
- without_weight xdecpfn drawn front: n=0 (-4.808, 0.007412 s), n=1 (-3.376, 0.007929 s), n=2 (-2.998, 0.008268 s), n=5 (-2.786, 0.008685 s), n=11 (-2.626, 0.009737 s), n=22 (-2.497, 0.01142 s), n=45 (-2.366, 0.01401 s), n=90 (-0.028, 0.04922 s)
- without_weight glyphs hidden by overlap (line still drawn): xdecpfn n=1, xdecpfn n=2, xdecpfn n=11, xdecpfn n=22
- with_weight segment-gp (median_ll_all per n) 1: -23.026, 2: -23.026, 5: -6.297, 11: -4.273, 22: -4.187, 45: -4.539, 90: -23.026
- with_weight segment-gp seconds per prediction per n: 90: 5.098, 45: 2.821, 22: 1.41, 11: 0.6802, 5: 0.2271, 2: 0.0749, 1: 5.169e-05
- with_weight decpfn (median_ll_all per n) 0: -4.010, 1: -4.010, 2: -3.971, 5: -3.992, 11: -3.951, 22: -3.902, 45: -3.798, 90: -2.029
- with_weight decpfn seconds per prediction per n: 90: 0.004396, 45: 0.003779, 22: 0.003732, 11: 0.003729, 5: 0.003949, 2: 0.004418, 1: 0.004131, 0: 0.003762
- with_weight decpfn pooled (every query once): median LL -4.010 over 694 queries at 0.003864 s
- with_weight xdecpfn (median_ll_all per n) 0: -5.095, 1: -3.625, 2: -3.312, 5: -3.117, 11: -2.937, 22: -2.985, 45: -2.796, 90: -0.809
- with_weight xdecpfn seconds per prediction per n: 90: 0.05757, 45: 0.0138, 22: 0.01131, 11: 0.009812, 5: 0.009093, 2: 0.007879, 1: 0.008113, 0: 0.007704
- with_weight segment-gp drawn front: n=5 (-6.297, 0.2271 s), n=11 (-4.273, 0.6802 s), n=22 (-4.187, 1.41 s)
- with_weight decpfn drawn front: n=pooled (-4.010, 0.003864 s)
- with_weight xdecpfn drawn front: n=0 (-5.095, 0.007704 s), n=2 (-3.312, 0.007879 s), n=5 (-3.117, 0.009093 s), n=11 (-2.937, 0.009812 s), n=45 (-2.796, 0.0138 s), n=90 (-0.809, 0.05757 s)
- with_weight glyphs hidden by overlap (line still drawn): xdecpfn n=5, xdecpfn n=11
