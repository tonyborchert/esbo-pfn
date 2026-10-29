# hwpred_ablation_bars

## What is shown
- one panel per rig, four DecPFN variants: dead-time in the prior or not (dark or light blue) x state channels or not (filled or outlined)
- the y axis does not start at 0, every bar grows from the shared axis floor

## Metric
- y = `median_ll`: median over the non-crashed queries of the per-query mean conditioned LL over the cutoffs, every query with its full run history as context

## Data
- without_weight decpfn: `hardware_prediction` unit caba03ffe959 (model `rigreset_deadtime-unif_kick0.6_binned`, provenance `thesis`)
- without_weight decpfn-nostates: `hardware_prediction` unit 27920ebab3e7 (model `rigreset_deadtime-unif_kick0.6_nostates`, provenance `thesis`)
- without_weight decpfn-nodeadtime: `hardware_prediction` unit d1131b48a95c (model `rigreset_nodeadtime_kick0.6_binned`, provenance `thesis`)
- without_weight decpfn-nodeadtime-nostates: `hardware_prediction` unit 756b62fdc516 (model `rigreset_nodeadtime_kick0.6_nostates`, provenance `thesis`)
- with_weight decpfn: `hardware_prediction` unit f460c1f7b1c0 (model `rigreset_deadtime-unif_kick0.6_binned`, provenance `thesis`)
- with_weight decpfn-nostates: `hardware_prediction` unit 99e2b0a17abc (model `rigreset_deadtime-unif_kick0.6_nostates`, provenance `thesis`)
- with_weight decpfn-nodeadtime: `hardware_prediction` unit c273c3d17281 (model `rigreset_nodeadtime_kick0.6_binned`, provenance `thesis`)
- with_weight decpfn-nodeadtime-nostates: `hardware_prediction` unit cf154416b276 (model `rigreset_nodeadtime_kick0.6_nostates`, provenance `thesis`)

## Numbers
- y range [-4.379, -3.774] (not from 0)
- without_weight: dead-time, states -3.951 (n=469) | dead-time, no states -3.861 (n=469) | no dead-time, states -3.909 (n=469) | no dead-time, no states -4.005 (n=469)
- with_weight: dead-time, states -4.279 (n=508) | dead-time, no states -4.129 (n=508) | no dead-time, states -4.112 (n=508) | no dead-time, no states -4.293 (n=508)
