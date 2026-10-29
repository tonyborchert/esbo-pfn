# hwplay_regret_violin

## What is shown
- one panel per rig ['without_weight', 'with_weight'], per method two violins: left 50%, right 100% of the wall-clock budget (darker), seeds as dots, the line is the median
- violin body = smoothed mirrored histogram on one bin grid for the whole figure

## Metric
- per-seed unscaled log10 regret of the incumbent at the fraction of the longest reset-normalized run on the rig
- the seed whose incumbent defines J_best sits on the regret floor and is left out of its violin

## Data
- rig `without_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]
- rig `with_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- without_weight: J_best = 256.541
- without_weight bo at 50% (x = 1393 s): seeds [0, 1, 2, 3, 4] values [0.941, 0.354, 0.692, 0.835, 0.341] median 0.692
- without_weight bo at 100% (x = 2794 s): seeds [0, 1, 2, 3, 4] values [0.941, 0.354, 0.692, 0.835, 0.341] median 0.692
- without_weight decpfn at 50% (x = 1393 s): seeds [0, 1, 2, 3, 4] values [0.243, 0.951, 1.133, 0.762, 1.121] median 0.951
- without_weight decpfn at 100% (x = 2794 s): seeds [0, 1, 2, 3, 4] values [0.243, 0.951, 0.832, 0.517, 1.096] median 0.832
- without_weight decpfn-h at 50% (x = 1393 s): seeds [0, 1, 2, 3, 4] values [0.788, 1.001, 0.589, 0.755, 0.834] median 0.788
- without_weight decpfn-h at 100% (x = 2794 s): seeds [0, 1, 2, 3, 4] values [0.788, 1.001, 0.528, 0.755, 0.389] median 0.755
- without_weight esbo-gp at 50% (x = 1393 s): seeds [0, 1, 2, 3, 4] values [0.790, 0.607, 0.693, 0.850, 0.824] median 0.790
- without_weight esbo-gp at 100% (x = 2794 s): seeds [0, 1, 2, 3, 4] values [0.749, 0.603, 0.254, 0.850, 0.737] median 0.737
- without_weight esrs at 50% (x = 1393 s): seeds [0, 1, 2, 3, 4] values [0.736, 0.586, 0.928, 0.730, 0.953] median 0.736
- without_weight esrs at 100% (x = 2794 s): seeds [0, 1, 2, 3] values [0.736, 0.586, 0.810, 0.730] median 0.733, excluded (defines J_best) [4]
- without_weight xdecpfn-pi at 50% (x = 1393 s): seeds [0, 1, 2, 3, 4] values [-0.275, -0.212, 0.190, 0.625, 0.063] median 0.063
- without_weight xdecpfn-pi at 100% (x = 2794 s): seeds [0, 1, 2, 3, 4] values [-0.275, -0.212, 0.190, 0.008, 0.063] median 0.008
- with_weight: J_best = 256.917
- with_weight bo at 50% (x = 1582 s): seeds [0, 1, 2, 3, 4] values [1.102, 1.045, 1.033, 1.940, 1.271] median 1.102
- with_weight bo at 100% (x = 3172 s): seeds [0, 1, 2, 3, 4] values [1.102, 1.045, 1.033, 1.940, 1.271] median 1.102
- with_weight decpfn at 50% (x = 1582 s): seeds [0, 1, 2, 3, 4] values [1.476, 1.468, 0.914, 1.424, 1.354] median 1.424
- with_weight decpfn at 100% (x = 3172 s): seeds [0, 1, 2, 3, 4] values [1.315, 1.245, 0.914, 1.348, 1.354] median 1.315
- with_weight decpfn-h at 50% (x = 1582 s): seeds [0, 1, 2, 3, 4] values [0.903, 1.114, 1.201, 1.196, 1.227] median 1.196
- with_weight decpfn-h at 100% (x = 3172 s): seeds [0, 1, 2, 3, 4] values [0.903, 1.114, 1.114, 1.196, 1.227] median 1.114
- with_weight esbo-gp at 50% (x = 1582 s): seeds [0, 1, 2, 3, 4] values [1.035, 1.044, 1.045, 1.528, 0.985] median 1.044
- with_weight esbo-gp at 100% (x = 3172 s): seeds [0, 1, 2, 3, 4] values [1.035, 1.044, 1.045, 1.268, 0.970] median 1.044
- with_weight esrs at 50% (x = 1582 s): seeds [0, 1, 2, 3, 4] values [1.955, 1.069, 1.157, 1.449, 1.434] median 1.434
- with_weight esrs at 100% (x = 3172 s): seeds [0, 1, 2, 3, 4] values [1.422, 1.069, 1.090, 1.365, 1.254] median 1.254
- with_weight xdecpfn-pi at 50% (x = 1582 s): seeds [1, 2, 3, 4] values [1.092, 0.887, 1.232, 0.793] median 0.989, excluded (defines J_best) [0]
- with_weight xdecpfn-pi at 100% (x = 3172 s): seeds [1, 2, 3, 4] values [1.092, 0.887, 0.963, 0.793] median 0.925, excluded (defines J_best) [0]
