# hwplay_optimization__without_weight

## What is shown
- rig `without_weight` (without weight), methods ['esbo-gp', 'bo', 'esrs', 'decpfn-h', 'decpfn', 'xdecpfn-pi']
- (a) median over seeds of the unscaled log10 regret against reset-normalized wall-clock, 25/75 % seed band
- (b) the same over the first 60 trials
- (c) mean over seeds of the cumulative crashed-trial count against wall-clock, 25/75 % band
- legend above the panels

## Metric
- regret = log10(J^- - J_best), J^- the best successful cost so far, J_best the best successful cost on the rig over the shown methods
- wall-clock = cumulative trial time with every reset replaced by the mean reset of the reference method (`hardware.metrics.wall_axis`)
- a seed without a success yet is left out of the median at that x

## Data
- rig `without_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- J_best = 256.541 (method `esrs`, seed 4), reset normalization 9.280 s (mean reset of `decpfn-h`)
- esbo-gp: seeds [0, 1, 2, 3, 4], final log10 regret per seed [0.749, 0.603, 0.254, 0.850, 0.737], median 0.737, crashed trials per seed [6, 9, 13, 17, 8] (mean 10.6), trials per seed [104, 116, 121, 129, 109], run end [1943, 2090, 2182, 2404, 2093] s
- esbo-gp: median log10 regret after 60 trials 0.790
- bo: seeds [0, 1, 2, 3, 4], final log10 regret per seed [0.941, 0.354, 0.692, 0.835, 0.341], median 0.692, crashed trials per seed [14, 30, 25, 17, 22] (mean 21.6), trials per seed [69, 81, 78, 73, 76], run end [1229, 1436, 1392, 1288, 1336] s
- bo: median log10 regret after 60 trials 0.736
- esrs: seeds [0, 1, 2, 3, 4], final log10 regret per seed [0.736, 0.586, 0.810, 0.730, -12.000], median 0.730, crashed trials per seed [7, 16, 15, 12, 14] (mean 12.8), trials per seed [180, 180, 180, 180, 180], run end [2504, 2496, 2561, 2502, 2495] s
- esrs: median log10 regret after 60 trials 0.953
- decpfn-h: seeds [0, 1, 2, 3, 4], final log10 regret per seed [0.788, 1.001, 0.528, 0.755, 0.389], median 0.755, crashed trials per seed [4, 8, 3, 2, 4] (mean 4.2), trials per seed [143, 155, 142, 152, 124], run end [2794, 2769, 2532, 2687, 2346] s
- decpfn-h: median log10 regret after 60 trials 0.788
- decpfn: seeds [0, 1, 2, 3, 4], final log10 regret per seed [0.243, 0.951, 0.832, 0.517, 1.096], median 0.832, crashed trials per seed [5, 5, 5, 3, 2] (mean 4.0), trials per seed [126, 97, 112, 161, 180], run end [1967, 1583, 1759, 2416, 2528] s
- decpfn: median log10 regret after 60 trials 1.057
- xdecpfn-pi: seeds [0, 1, 2, 3, 4], final log10 regret per seed [-0.275, -0.212, 0.190, 0.008, 0.063], median 0.008, crashed trials per seed [3, 4, 3, 4, 3] (mean 3.4), trials per seed [160, 89, 87, 126, 127], run end [2310, 1449, 1418, 1903, 1905] s
- xdecpfn-pi: median log10 regret after 60 trials 0.190
