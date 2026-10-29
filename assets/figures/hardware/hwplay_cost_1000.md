# hwplay_cost_1000

## What is shown
- one panel per rig ['without_weight', 'with_weight'], median over seeds of the best successful raw cost so far over the first 1000 s of reset-normalized wall-clock, 25/75 % seed band, one y range for all panels (lowest band edge to highest median line, band spikes of the first seconds clipped)
- numbers: median at 100 s checkpoints, seeds with a success in brackets

## Metric
- raw cost J_T of the best successful trial so far (not regret), a seed without a success is left out at that x

## Data
- rig `without_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]
- rig `with_weight`: 30 `hardware_optimization` units (provenance `thesis`) in `$PFN_HOME/benchmarks/db.sqlite`, methods and seeds bo [0, 1, 2, 3, 4], decpfn [0, 1, 2, 3, 4], decpfn-h [0, 1, 2, 3, 4], esbo-gp [0, 1, 2, 3, 4], esrs [0, 1, 2, 3, 4], xdecpfn-pi [0, 1, 2, 3, 4]

## Numbers
- without_weight: reset normalization 9.280 s
- without_weight t = 100 s: bo 279.0 (5), decpfn 276.0 (5), decpfn-h 274.9 (5), esbo-gp 274.2 (5), esrs 273.4 (5), xdecpfn-pi 278.2 (5)
- without_weight t = 200 s: bo 273.4 (5), decpfn 270.7 (5), decpfn-h 271.1 (5), esbo-gp 272.7 (5), esrs 272.8 (5), xdecpfn-pi 271.0 (5)
- without_weight t = 300 s: bo 265.7 (5), decpfn 270.6 (5), decpfn-h 268.9 (5), esbo-gp 268.1 (5), esrs 272.8 (5), xdecpfn-pi 260.8 (5)
- without_weight t = 400 s: bo 265.4 (5), decpfn 270.6 (5), decpfn-h 268.9 (5), esbo-gp 264.6 (5), esrs 267.8 (5), xdecpfn-pi 260.8 (5)
- without_weight t = 500 s: bo 262.0 (5), decpfn 270.6 (5), decpfn-h 266.6 (5), esbo-gp 263.6 (5), esrs 265.5 (5), xdecpfn-pi 258.1 (5)
- without_weight t = 600 s: bo 262.0 (5), decpfn 269.8 (5), decpfn-h 266.6 (5), esbo-gp 262.7 (5), esrs 265.5 (5), xdecpfn-pi 258.1 (5)
- without_weight t = 700 s: bo 262.0 (5), decpfn 269.8 (5), decpfn-h 266.6 (5), esbo-gp 262.7 (5), esrs 265.5 (5), xdecpfn-pi 258.1 (5)
- without_weight t = 800 s: bo 262.0 (5), decpfn 269.8 (5), decpfn-h 263.1 (5), esbo-gp 262.7 (5), esrs 265.5 (5), xdecpfn-pi 258.1 (5)
- without_weight t = 900 s: bo 262.0 (5), decpfn 268.8 (5), decpfn-h 262.7 (5), esbo-gp 262.7 (5), esrs 265.5 (5), xdecpfn-pi 258.1 (5)
- without_weight t = 1000 s: bo 262.0 (5), decpfn 268.0 (5), decpfn-h 262.7 (5), esbo-gp 262.7 (5), esrs 265.5 (5), xdecpfn-pi 258.1 (5)
- with_weight: reset normalization 9.329 s
- with_weight t = 100 s: bo 369.9 (5), decpfn 363.5 (5), decpfn-h 361.6 (5), esbo-gp 362.3 (5), esrs 358.1 (5), xdecpfn-pi 355.0 (5)
- with_weight t = 200 s: bo 362.3 (5), decpfn 358.5 (5), decpfn-h 353.3 (5), esbo-gp 360.9 (5), esrs 358.0 (5), xdecpfn-pi 330.9 (5)
- with_weight t = 300 s: bo 357.4 (5), decpfn 341.4 (5), decpfn-h 352.9 (5), esbo-gp 350.0 (5), esrs 344.1 (5), xdecpfn-pi 274.0 (5)
- with_weight t = 400 s: bo 357.4 (5), decpfn 341.4 (5), decpfn-h 296.8 (5), esbo-gp 310.2 (5), esrs 344.1 (5), xdecpfn-pi 265.7 (5)
- with_weight t = 500 s: bo 293.9 (5), decpfn 341.4 (5), decpfn-h 274.7 (5), esbo-gp 310.2 (5), esrs 285.0 (5), xdecpfn-pi 265.7 (5)
- with_weight t = 600 s: bo 288.5 (5), decpfn 335.1 (5), decpfn-h 272.6 (5), esbo-gp 310.2 (5), esrs 285.0 (5), xdecpfn-pi 265.7 (5)
- with_weight t = 700 s: bo 284.3 (5), decpfn 301.5 (5), decpfn-h 272.6 (5), esbo-gp 310.2 (5), esrs 285.0 (5), xdecpfn-pi 265.6 (5)
- with_weight t = 800 s: bo 272.6 (5), decpfn 286.3 (5), decpfn-h 272.6 (5), esbo-gp 284.2 (5), esrs 285.0 (5), xdecpfn-pi 265.6 (5)
- with_weight t = 900 s: bo 269.6 (5), decpfn 286.3 (5), decpfn-h 272.6 (5), esbo-gp 276.0 (5), esrs 285.0 (5), xdecpfn-pi 265.6 (5)
- with_weight t = 1000 s: bo 269.6 (5), decpfn 286.3 (5), decpfn-h 272.6 (5), esbo-gp 268.0 (5), esrs 285.0 (5), xdecpfn-pi 264.6 (5)
