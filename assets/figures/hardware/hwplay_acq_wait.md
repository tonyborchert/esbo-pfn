# hwplay_acq_wait

## What is shown
- one panel per rig ['without_weight', 'with_weight'], bar = mean over seeds of the seed's mean wait per trial, dots = seeds, methods ['esbo-gp', 'bo', 'decpfn-h', 'decpfn', 'xdecpfn-pi'] (ESRS has no acquisition and is left out)

## Metric
- wait = propose time per trial (refit and acquisition optimization, `timings.propose_s`), the early stopping decisions during a trial are not included, resets are not counted

## Data
- rig `without_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]
- rig `with_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- without_weight bo: mean 0.197 s per trial, per seed 0: 0.255, 1: 0.179, 2: 0.183, 3: 0.187, 4: 0.182
- without_weight decpfn: mean 0.144 s per trial, per seed 0: 0.152, 1: 0.135, 2: 0.147, 3: 0.133, 4: 0.155
- without_weight decpfn-h: mean 3.136 s per trial, per seed 0: 3.923, 1: 3.223, 2: 2.732, 3: 2.451, 4: 3.351
- without_weight esbo-gp: mean 2.769 s per trial, per seed 0: 2.785, 1: 2.323, 2: 2.422, 3: 2.920, 4: 3.397
- without_weight xdecpfn-pi: mean 0.066 s per trial, per seed 0: 0.067, 1: 0.062, 2: 0.065, 3: 0.067, 4: 0.067
- with_weight bo: mean 0.199 s per trial, per seed 0: 0.200, 1: 0.204, 2: 0.216, 3: 0.186, 4: 0.189
- with_weight decpfn: mean 0.136 s per trial, per seed 0: 0.137, 1: 0.139, 2: 0.136, 3: 0.134, 4: 0.131
- with_weight decpfn-h: mean 2.565 s per trial, per seed 0: 3.012, 1: 2.515, 2: 2.335, 3: 1.963, 4: 3.000
- with_weight esbo-gp: mean 2.430 s per trial, per seed 0: 2.430, 1: 2.159, 2: 2.180, 3: 3.121, 4: 2.260
- with_weight xdecpfn-pi: mean 0.069 s per trial, per seed 0: 0.066, 1: 0.069, 2: 0.068, 3: 0.070, 4: 0.069
