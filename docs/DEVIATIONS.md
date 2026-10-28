# Deviations from the thesis text

The thesis text is the specification of this code. The items below are the only places where the
code makes a choice that the thesis does not state or states differently.

1. **GP refit approximation.** The objective GP and the Segment-GPs of ESBO-GP refit their
   hyperparameters from scratch only when the training set has grown by `cold_refit_growth`
   relative to the last full fit, and warm start from the previous hyperparameters otherwise.
   Optimization runs use `cold_refit_growth = 0.05`. The thesis does not mention this setting.
   `0.0` makes every fit a full fit.
2. **Training set sizes.** The 100k and 1M models train on exactly 100,000 and 1,000,000 distinct
   trajectories, one pass each, so the number of optimizer steps is `floor(N / 128)` at batch
   size 128.
3. **Choices where the thesis is silent.**
   - Seeds of data generation, training, instances and runs, all derived through
     `esbopfn.util.seeds.substream`.
   - Sizes of the validation sets.
   - The rejection rates of the trajectory priors (unsuccessful share, crash share) are estimated
     by pilot sampling.
   - The initial design is iid uniform on the search box.
   - The candidate set of the GP acquisition (BO, ESBO-GP, ESBO-Oracle, ESBO-T-TabPFN, DecPFN,
     DecPFN-H) is 2000 points drawn uniformly from the search box.
