# problems_strip

## What is shown
- 12 random controllers per problem on the nominal plant (with measurement noise), cumulative cost over t / T, red crosses mark crashes

## Metric
- raw cumulative cost J_t (up to the crash for a crashed trial), y axis per panel up to 1.25 x the 85 % quantile of the last values

## Data
- live rollouts, seed 20261029

## Numbers
- boiler: 0 of 12 crashed
- threetank_pi: 0 of 12 crashed
- pt2_deadtime: 3 of 12 crashed
- cartpole: 10 of 12 crashed
- furuta: 12 of 12 crashed
