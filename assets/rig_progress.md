# rig_progress

## What is shown
- median over seeds of the log10 regret of the best successful trial so far against reset-normalized wall-clock time, one panel per rig configuration

## Metric
- log10(J- - J_best), J_best the best cost recorded on that rig

## Data
- `hardware_optimization` units (the figure group `hardware`)

## Numbers
- without_weight bo: final median 0.692
- without_weight decpfn: final median 0.832
- without_weight decpfn-h: final median 0.755
- without_weight esbo-gp: final median 0.737
- without_weight esrs: final median 0.730
- without_weight xdecpfn-pi: final median 0.008
- with_weight bo: final median 1.102
- with_weight decpfn: final median 1.315
- with_weight decpfn-h: final median 1.114
- with_weight esbo-gp: final median 1.044
- with_weight esrs: final median 1.254
- with_weight xdecpfn-pi: final median 0.887
