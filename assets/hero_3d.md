# hero_3d

## What is shown
- axes t (seconds), J_t (raw cost), theta (one controller coordinate); blue lines = real rig trials, orange fills = predicted total cost mu +- sigma and mu +- 2 sigma at t = T, star = next controller (argmax PI over the slice), red graph above the wall = predicted crash probability, dotted drop lines and red rings = trials shown stopped early

## Metric
- mean and standard deviation of the predicted bar distribution without the crash atom, PI = P(J_T < incumbent J) with the crash atom counted as no improvement

## Data
- `hardware_optimization` units of the rig and the model checkpoint named below

## Numbers
- model `xdecpfn/furuta/thesis/default` (`predict_unrun` over 41 slice points), rig `without_weight`, method `bo`, provenance `thesis`
- slice dimension `mu` (index 0), range of the predicted mean per dimension: mu 386.8, E_bar 294.4, q1 0.5, q2 0.2, q3 0.8, q4 2.8, r 22.6, K_P_barrier 9.9
- incumbent J = 258.73, next controller at mu = 90.000 (PI 0.348, mu 261.4, p_crash 0.003)
- trial A: index 25, mu = 88.780, shown to step 600 (complete), J at the end of the shown part 258.7, recorded J_T 258.7
- trial B: index 12, mu = 71.805, shown to step 208 (stopped), J at the end of the shown part 368.9, recorded J_T 407.5
- trial C: index 3, mu = 70.769, shown to step 138 (stopped), J at the end of the shown part 289.8, recorded J_T 417.6
