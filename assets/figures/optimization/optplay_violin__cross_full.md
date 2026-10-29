# optplay_violin__cross_full

Thesis asset: `writing/thesis/assets_generated/simulation/optimization/at_the_end/optplay_violin__cross_full.pdf` (fig:sim:optplay-violin-cross-full)

Files: `optplay_violin__cross_full.png`

## What is shown
- one violin per method and budget fraction (100 % of the simulated time budget 15 T min(d, 4)), body = mirrored histogram drawn smooth, line = median
- methods: ESRS, BO, ESBO-GP, ESBO-T-TabPFN, DecPFN, DecPFN-H, XDecPFN-PI, FT-PFN-EI, ESBO-Oracle

## Metric
- scaled regret (J - J_best) / (J_ESRS - J_best) per run, log10 with display floor -4 (values below are drawn at the floor), lower is better
- J_best: best final cost of the reference strategies on the instance, J_ESRS: the final incumbent of ESRS

## Key numbers
- all problems: median log10 scaled regret: ESRS 100% 0.000 (n=100, 0 at floor); BO 100% -0.200 (n=100, 2 at floor); ESBO-GP 100% -0.320 (n=100, 1 at floor); DecPFN 100% -0.253 (n=100, 6 at floor); DecPFN-H 100% -0.347 (n=100, 2 at floor); XDecPFN-PI 100% -0.824 (n=100, 14 at floor); FT-PFN-EI 100% -0.758 (n=100, 12 at floor); ESBO-Oracle 100% -0.313 (n=100, 4 at floor)
- 50 % budget not shown: the per-run incumbent traces (native `optimization` units, `pfn bench run optimization`) are not in the database, the thesis import holds final incumbents only

## Data
- thesis import (`optplay_finals`, final incumbents at the time budget) on boiler, threetank_pi, pt2_deadtime, cartpole, furuta. `J_best` = `optplay_reference` of the import (best successful final cost over every thesis cell on disk, the thesis definition)

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 165 units (thesis), hashes 07b1755f3a, 0d30333e94, 109851d786, 174e983a4d, 3d3b157215, 3e140b898d, ...
