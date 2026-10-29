# Documentation

Start with [REPRODUCE.md](REPRODUCE.md) to run the pipeline end to end.

| Doc | What it covers |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | the full walkthrough from an empty machine to every figure, locally and on Slurm, with compute estimates |
| [DATA.md](DATA.md) | the C++ generators, building them, dataset ids, pool and context-shard formats, seeding, the priors as implemented, verification |
| [TRAINING.md](TRAINING.md) | LC-PFN and DecPFN with their variants, the shared training core, model ids and the model tree, hyperparameter files and search |
| [CROSS_TRAJECTORY.md](CROSS_TRAJECTORY.md) | XDecPFN and FT-PFN: architecture, data, training, inference and the import of the thesis checkpoints |
| [LEGACY_MODELS.md](LEGACY_MODELS.md) | exporting and importing the thesis DecPFN, DecPFN-no-St and LC-PFN checkpoints and the legacy classes that serve them |
| [PREDICTORS.md](PREDICTORS.md) | the prediction types and the predictor interface, the exact GP, Segment-GP and the TabPFN baselines with their checkpoint |
| [STRATEGIES.md](STRATEGIES.md) | the ESBO loop, budgets, stop rules, every optimization strategy and their parameters |
| [BENCHMARKS.md](BENCHMARKS.md) | benchmark kinds, the database, task tables, metrics, plot-group specs, running on Slurm, importing the thesis results |
| [HARDWARE.md](HARDWARE.md) | the QUBE-Servo 2 rig: workstation setup, protocol, reset, rig field, prediction benchmark, importing the rig logs |
| [FIGURES.md](FIGURES.md) | every plot group and figure, what renders from the thesis import, the README art and `assets/figures/` |
| [CLUSTER.md](CLUSTER.md) | `PFN_HOME` and `PFN_PROFILE`, cluster profiles, `pfn submit`, adding a cluster |
| [CONTRIBUTING.md](CONTRIBUTING.md) | setup, code conventions, extension points, tests |

The problems and the universal prior are documented next to their code in
[`src/esbopfn/problems/README.md`](../src/esbopfn/problems/README.md).

## Names used throughout

| Name | Meaning |
|---|---|
| `PFN_HOME` | output root, default `./out` |
| problem ids | `boiler`, `threetank_pi`, `pt2_deadtime`, `cartpole`, `furuta` |
| model ids | `<model>/<problem>/<variant>/<size>[/<run>]`, e.g. `decpfn/cartpole/binned/1M`, `xdecpfn/furuta/default/default`, `xdecpfn/furuta/thesis/default` for imported thesis weights |
| predictor ids | model ids plus `segment-gp`, `tabpfn/<L>`, `segment-tabpfn/<split\|rate>` |
| strategy ids | `esrs`, `bo`, `esbo-gp`, `esbo-oracle`, `esbo-t-tabpfn`, `decpfn`, `decpfn-h`, `xdecpfn-pi`, `xdecpfn-ei`, `ftpfn-ei`, `ftpfn-pi` |
| dataset ids | `traj/<p>/1M`, `traj/<p>/100k`, `sets/<p>/xdecpfn`, `contexts/<p>/ftpfn` and their validation sets |
| rig ids | `without_weight`, `with_weight` (`src/esbopfn/configs/rigs.yaml`) |
| display names | ESRS, BO, ESBO-GP, ESBO-Oracle, ESBO-T-TabPFN, DecPFN, DecPFN-H, DecPFN-no-St, LC-PFN, XDecPFN-PI, FT-PFN-EI, Segment-GP, Trajectory-TabPFN, Segment-TabPFN |
| `provenance` | `thesis` for imported units and checkpoints, `native` for units computed with this code |
