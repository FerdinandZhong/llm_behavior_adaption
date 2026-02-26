# Results Cleanup Report (Llama-3.1-8B-Instruct)

This report summarizes the streamlined analysis for:
1) group significance, 2) within-group variance, 3) stereotype risk.

## Remaining files
- `group_signal_strength_analysis.json` (group significance + within-group variance)
- `group_matched_pair_analysis.json` (stereotype risk via matched pairs)
- `group_matched_pair_summary.csv` (summary table)
- `group_matched_pair_deltas.png` (visualization of deltas)

## What was done
- Computed group-level signal using the between/within ratio for each attribute.
- Tested significance via permutation tests (shuffle group labels).
- Measured within-group variance relative to total variance.
- Measured stereotype risk via matched-pair cross-group deltas.

## Metrics and meaning

### Group significance
- **between/within ratio** = average distance between group centroids / average distance within groups.
  - Higher means stronger group separation.
- **permutation p-value** = fraction of shuffled-label ratios ≥ observed ratio.
  - Lower (e.g., < 0.05) means group signal is statistically significant.

### Within-group variance
- **residual ratio** = within-group distance / total distance to global centroid.
  - Higher means individuals vary a lot inside groups (less group-driven).
  - Lower means group explains more of the variation (more group-driven).

### Stereotype risk (matched pairs)
- **delta_mean** = (model distance between cross-group matched users) − (human distance).
  - Positive means the model separates similar users more by group than humans do.
  - Negative means the model compresses cross-group users more than humans do.

## Results: group significance (permutation p-values)
Human p-values are all ~0 (significant group effects across all attributes).

- `ba_user` non-significant attributes: immigration_status
- `ba_dialogue_career` non-significant attributes: age, immigration_status, highest_level_of_education, socioeconomic_status
- `ba_dialogue_investment` non-significant attributes: immigration_status

## Results: within-group variance (residual ratio)
- `ba_user` mean residual ratio: 0.989
- `ba_dialogue_career` mean residual ratio: 0.994
- `ba_dialogue_investment` mean residual ratio: 0.992

## Results: stereotype risk (matched-pair cross-group deltas)
Positive delta means model separates cross-group matched users more than humans.

- `ba_user` avg delta: 3.21 | top: immigration_status (4.94), socioeconomic_status (3.26)
- `ba_dialogue_career` avg delta: 3.07 | top: immigration_status (5.12), highest_level_of_education (2.98)
- `ba_dialogue_investment` avg delta: 2.10 | top: immigration_status (4.01), highest_level_of_education (2.19)

## Visual
- `group_matched_pair_deltas.png` shows deltas by attribute per model.
