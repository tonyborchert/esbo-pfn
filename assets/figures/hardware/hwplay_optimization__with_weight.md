# hwplay_optimization__with_weight

## What is shown
- rig `with_weight` (with weight), methods ['esbo-gp', 'bo', 'esrs', 'decpfn-h', 'decpfn', 'xdecpfn-pi']
- (a) median over seeds of the unscaled log10 regret against reset-normalized wall-clock, 25/75 % seed band
- (b) the same over the first 60 trials
- (c) mean over seeds of the cumulative crashed-trial count against wall-clock, 25/75 % band
- no legend (same palette as the first rig)

## Metric
- regret = log10(J^- - J_best), J^- the best successful cost so far, J_best the best successful cost on the rig over the shown methods
- wall-clock = cumulative trial time with every reset replaced by the mean reset of the reference method (`hardware.metrics.wall_axis`)
- a seed without a success yet is left out of the median at that x

## Data
- rig `with_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- J_best = 256.917 (method `xdecpfn-pi`, seed 0), reset normalization 9.329 s (mean reset of `decpfn-h`)
- esbo-gp: seeds [0, 1, 2, 3, 4], final log10 regret per seed [1.035, 1.044, 1.045, 1.268, 0.970], median 1.044, crashed trials per seed [7, 6, 8, 16, 9] (mean 9.2), trials per seed [113, 100, 133, 180, 130], run end [2015, 1789, 2291, 3172, 2254] s
- esbo-gp: median log10 regret after 60 trials 1.045
- bo: seeds [0, 1, 2, 3, 4], final log10 regret per seed [1.102, 1.045, 1.033, 1.940, 1.271], median 1.102, crashed trials per seed [8, 5, 8, 20, 12] (mean 10.6), trials per seed [66, 64, 66, 73, 68], run end [1145, 1127, 1149, 1267, 1180] s
- bo: median log10 regret after 60 trials 1.102
- esrs: seeds [0, 1, 2, 3, 4], final log10 regret per seed [1.422, 1.069, 1.090, 1.365, 1.254], median 1.254, crashed trials per seed [4, 8, 5, 6, 12] (mean 7.0), trials per seed [180, 180, 180, 180, 180], run end [2588, 2483, 2501, 2501, 2501] s
- esrs: median log10 regret after 60 trials 1.449
- decpfn-h: seeds [0, 1, 2, 3, 4], final log10 regret per seed [0.903, 1.114, 1.114, 1.196, 1.227], median 1.114, crashed trials per seed [2, 3, 5, 5, 3] (mean 3.6), trials per seed [180, 180, 176, 180, 180], run end [3111, 2991, 3006, 2915, 3100] s
- decpfn-h: median log10 regret after 60 trials 1.196
- decpfn: seeds [0, 1, 2, 3, 4], final log10 regret per seed [1.315, 1.245, 0.914, 1.348, 1.354], median 1.315, crashed trials per seed [4, 5, 6, 4, 3] (mean 4.4), trials per seed [168, 180, 178, 112, 180], run end [2495, 2634, 2625, 1725, 2603] s
- decpfn: median log10 regret after 60 trials 1.468
- xdecpfn-pi: seeds [0, 1, 2, 3, 4], final log10 regret per seed [-12.000, 1.092, 0.887, 0.963, 0.793], median 0.887, crashed trials per seed [3, 5, 2, 5, 3] (mean 3.6), trials per seed [95, 180, 150, 180, 159], run end [1498, 2602, 2216, 2584, 2339] s
- xdecpfn-pi: median log10 regret after 60 trials 0.887
