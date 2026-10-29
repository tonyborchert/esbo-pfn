# hwplay_best_controller__500s__with_weight

## What is shown
- rig `with_weight` (with weight), methods ['xdecpfn-pi', 'esbo-gp']
- left: arm angle, right: pendulum angle (dashed, unwrapped, 0 deg upright, the trace shifted by a multiple of 360 deg so that it ends upright), degrees over time within the trial (120 Hz)

## Metric
- controller = the incumbent (best successful trial so far) at 500 s of reset-normalized wall-clock of the seed with the median incumbent cost (index (n - 1) // 2 of the ranked seeds), one rig execution, traces from the stored x channels of that trial

## Data
- rig `with_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- xdecpfn-pi: median seed 2, incumbent trial index 12, J = 265.7, kick sign -1, theta [90.000, 1.115, 11.629, 4.507, 1.099, 1.186, 1.287, 3.366], incumbent cost per seed at 500 s (ranked) seed 0: 257.8, seed 4: 265.6, seed 2: 265.7, seed 3: 274.0, seed 1: 277.3
- esbo-gp: median seed 3, incumbent trial index 21, J = 310.2, kick sign +1, theta [84.891, 1.120, 12.069, 5.069, 0.935, 1.220, 1.390, 2.109], incumbent cost per seed at 500 s (ranked) seed 2: 287.3, seed 4: 296.2, seed 3: 310.2, seed 1: 346.2, seed 0: 350.0
