# results_cards

## What is shown
- four cards: pooled prediction score, simulation optimization regret, hardware regret on the first rig, hardware crashes (mean over rigs)

## Metric
- prediction: pooled median ll_norm at the context size (context-blind models at their single value), best variant per family
- optimization: log10 of the mean over problems of the per-problem median scaled regret at 100 % of the simulated-time budget
- hardware: median over all seeds of the unscaled log10 regret at the end of the wall-clock budget (as the thesis text quotes it)

## Data
- `thesis_series` (figures fig3_cross, optplay_finals, optplay_reference, variant `thesis`) and `hardware_optimization` units

## Numbers
- prediction FT-PFN (`ftpfn/{p}/thesis/default`): +1.590
- prediction XDecPFN (`xdecpfn/{p}/thesis/default`): +1.463
- prediction DecPFN (`decpfn/{p}/binned/1M`): +0.990
- prediction Trajectory-TabPFN (`tabpfn/100`): +0.833
- prediction Segment-TabPFN (`segment-tabpfn/split`): +0.601
- prediction Segment-GP (`segment-gp`): +0.552
- prediction LC-PFN (`lcpfn/{p}/binned/1M`): +0.353
- optimization xdecpfn-pi: -0.735 (per-problem median scaled regret boiler 0.177, cartpole 0.087, furuta 0.191, pt2_deadtime 0.212, threetank_pi 0.254)
- optimization ftpfn-ei: -0.712 (per-problem median scaled regret boiler 0.174, cartpole 0.147, furuta 0.304, pt2_deadtime 0.163, threetank_pi 0.182)
- optimization decpfn: -0.311 (per-problem median scaled regret boiler 0.290, cartpole 0.346, furuta 0.900, pt2_deadtime 0.559, threetank_pi 0.348)
- optimization decpfn-h: -0.279 (per-problem median scaled regret boiler 0.820, cartpole 0.401, furuta 0.866, pt2_deadtime 0.333, threetank_pi 0.208)
- optimization esbo-gp: -0.273 (per-problem median scaled regret boiler 0.553, cartpole 0.370, furuta 0.820, pt2_deadtime 0.376, threetank_pi 0.549)
- optimization esbo-oracle: -0.245 (per-problem median scaled regret boiler 0.536, cartpole 0.331, furuta 0.872, pt2_deadtime 0.414, threetank_pi 0.693)
- optimization bo: -0.169 (per-problem median scaled regret boiler 0.536, cartpole 0.364, furuta 1.001, pt2_deadtime 0.788, threetank_pi 0.701)
- optimization esrs: +0.000 (per-problem median scaled regret boiler 1.000, cartpole 1.000, furuta 1.000, pt2_deadtime 1.000, threetank_pi 1.000)
- hardware without_weight esbo-gp: median log10 regret 0.737, mean crashes 10.6
- hardware without_weight bo: median log10 regret 0.692, mean crashes 21.6
- hardware without_weight esrs: median log10 regret 0.730, mean crashes 12.8
- hardware without_weight decpfn-h: median log10 regret 0.755, mean crashes 4.2
- hardware without_weight decpfn: median log10 regret 0.832, mean crashes 4.0
- hardware without_weight xdecpfn-pi: median log10 regret 0.008, mean crashes 3.4
- hardware with_weight esbo-gp: median log10 regret 1.044, mean crashes 9.2
- hardware with_weight bo: median log10 regret 1.102, mean crashes 10.6
- hardware with_weight esrs: median log10 regret 1.254, mean crashes 7.0
- hardware with_weight decpfn-h: median log10 regret 1.114, mean crashes 3.6
- hardware with_weight decpfn: median log10 regret 1.315, mean crashes 4.4
- hardware with_weight xdecpfn-pi: median log10 regret 0.887, mean crashes 3.6
- not in the database (not drawn): optimization esbo-t-tabpfn
