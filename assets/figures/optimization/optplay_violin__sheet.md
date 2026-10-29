# optplay_violin__sheet

Thesis asset: `writing/thesis/assets_generated/simulation/optimization/at_the_end/optplay_violin__sheet.pdf` (fig:app:optplay-violin-sheet)

Files: `optplay_violin__sheet.png`

## What is shown
- one violin per method and budget fraction (100 % of the simulated time budget 15 T min(d, 4)), body = mirrored histogram drawn smooth, line = median
- methods: ESRS, BO, ESBO-GP, ESBO-T-TabPFN, DecPFN, DecPFN-H, XDecPFN-PI, FT-PFN-EI, ESBO-Oracle

## Metric
- scaled regret (J - J_best) / (J_ESRS - J_best) per run, log10 with display floor -4 (values below are drawn at the floor), lower is better
- J_best: best final cost of the reference strategies on the instance, J_ESRS: the final incumbent of ESRS

## Key numbers
- all problems: median log10 scaled regret: ESRS 100% 0.000 (n=100, 0 at floor); BO 100% -0.200 (n=100, 2 at floor); ESBO-GP 100% -0.320 (n=100, 1 at floor); DecPFN 100% -0.253 (n=100, 6 at floor); DecPFN-H 100% -0.347 (n=100, 2 at floor); XDecPFN-PI 100% -0.824 (n=100, 14 at floor); FT-PFN-EI 100% -0.758 (n=100, 12 at floor); ESBO-Oracle 100% -0.313 (n=100, 4 at floor)
- boiler: median log10 scaled regret: ESRS 100% 0.000 (n=20, 0 at floor); BO 100% -0.272 (n=20, 2 at floor); ESBO-GP 100% -0.259 (n=20, 0 at floor); DecPFN 100% -0.565 (n=20, 3 at floor); DecPFN-H 100% -0.087 (n=20, 0 at floor); XDecPFN-PI 100% -0.753 (n=20, 5 at floor); FT-PFN-EI 100% -0.760 (n=20, 3 at floor); ESBO-Oracle 100% -0.272 (n=20, 2 at floor)
- three-tank: median log10 scaled regret: ESRS 100% 0.000 (n=20, 0 at floor); BO 100% -0.172 (n=20, 0 at floor); ESBO-GP 100% -0.270 (n=20, 1 at floor); DecPFN 100% -0.466 (n=20, 3 at floor); DecPFN-H 100% -0.690 (n=20, 2 at floor); XDecPFN-PI 100% -0.628 (n=20, 3 at floor); FT-PFN-EI 100% -0.741 (n=20, 0 at floor); ESBO-Oracle 100% -0.179 (n=20, 0 at floor)
- PT2 dead-time: median log10 scaled regret: ESRS 100% 0.000 (n=20, 0 at floor); BO 100% -0.104 (n=20, 0 at floor); ESBO-GP 100% -0.425 (n=20, 0 at floor); DecPFN 100% -0.253 (n=20, 0 at floor); DecPFN-H 100% -0.479 (n=20, 0 at floor); XDecPFN-PI 100% -0.675 (n=20, 2 at floor); FT-PFN-EI 100% -0.789 (n=20, 4 at floor); ESBO-Oracle 100% -0.383 (n=20, 2 at floor)
- cartpole: median log10 scaled regret: ESRS 100% 0.000 (n=20, 0 at floor); BO 100% -0.440 (n=20, 0 at floor); ESBO-GP 100% -0.432 (n=20, 0 at floor); DecPFN 100% -0.462 (n=20, 0 at floor); DecPFN-H 100% -0.396 (n=20, 0 at floor); XDecPFN-PI 100% -1.061 (n=20, 3 at floor); FT-PFN-EI 100% -0.834 (n=20, 2 at floor); ESBO-Oracle 100% -0.480 (n=20, 0 at floor)
- furuta: median log10 scaled regret: ESRS 100% 0.000 (n=20, 0 at floor); BO 100% 0.001 (n=20, 0 at floor); ESBO-GP 100% -0.086 (n=20, 0 at floor); DecPFN 100% -0.046 (n=20, 0 at floor); DecPFN-H 100% -0.063 (n=20, 0 at floor); XDecPFN-PI 100% -0.721 (n=20, 1 at floor); FT-PFN-EI 100% -0.519 (n=20, 3 at floor); ESBO-Oracle 100% -0.060 (n=20, 0 at floor)
- 50 % budget not shown: the per-run incumbent traces (native `optimization` units, `pfn bench run optimization`) are not in the database, the thesis import holds final incumbents only

## Data
- thesis import (`optplay_finals`, final incumbents at the time budget) on boiler, threetank_pi, pt2_deadtime, cartpole, furuta. `J_best` = `optplay_reference` of the import (best successful final cost over every thesis cell on disk, the thesis definition)

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 165 units (thesis), hashes 07b1755f3a, 0d30333e94, 109851d786, 174e983a4d, 3d3b157215, 3e140b898d, ...
