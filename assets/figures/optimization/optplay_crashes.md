# optplay_crashes

Thesis asset: `writing/thesis/assets_generated/simulation/optimization/at_the_end/optplay_crashes.pdf` (fig:sim:optplay-crashes)

Files: `optplay_crashes.png`

## What is shown
- mean number of crashed trials among the trials finished so far, over % of the simulated time budget, averaged over the runs on all instances
- methods: ESRS, BO, ESBO-GP, ESBO-T-TabPFN, DecPFN, DecPFN-H, XDecPFN-PI, FT-PFN-EI, ESBO-Oracle

## Metric
- crash count (lower is better)

## Key numbers
- PT2 dead-time at 100% (thesis): ESRS 11.90, BO 8.25, ESBO-GP 4.25, DecPFN 2.30, DecPFN-H 2.65, XDecPFN-PI 1.80, FT-PFN-EI 2.14, ESBO-Oracle 6.50
- cartpole at 100% (thesis): ESRS 53.80, BO 15.45, ESBO-GP 14.60, DecPFN 4.55, DecPFN-H 4.40, XDecPFN-PI 3.82, FT-PFN-EI 8.94, ESBO-Oracle 16.45
- furuta at 100% (thesis): ESRS 15.70, BO 14.00, ESBO-GP 7.60, DecPFN 7.15, DecPFN-H 5.70, XDecPFN-PI 2.33, FT-PFN-EI 10.73, ESBO-Oracle 5.90

## Data
- pt2_deadtime: thesis
- cartpole: thesis
- furuta: thesis
- ESBO-T-TabPFN is not in the thesis crash aggregates

## Sources
- **source: thesis import** (units imported with `pfn import thesis-results`, `provenance: thesis`)
- `thesis_series`: 99 units (thesis), hashes 01e95d2c64, 102814594d, 23e00eae4e, 24f00dfe80, 41537389d5, 572dc91413, ...
