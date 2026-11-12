# 世界模型 · 综述与基准统计 Survey & benchmark statistics  `1147-1152`

6 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 1147 | `fig_wm_survey_papers_by_year.py` | (10.4, 6.4) | Papers per year 2016-2026 stacked into six subareas (six blue rungs), cumulative total on a warm right axis, three call-outs (steepest jump, half-corpus year, peak year) all computed from the plotted arrays |
| 1148 | `fig_wm_survey_venue_split.py` | (10.4, 6.4) | Venue split: left, 14 venues ranked by total with count + share at the bar end and a C/J badge column; right, the same rows as a venue x year count grid on one shared blue scale; bars are the grid row sums, warm outline marks the busiest cell |
| 1149 | `fig_wm_survey_capability_matrix.py` | (9.0, 6.8) | 12 benchmarks x 9 capabilities dot matrix (solid / half / open = full / partial / no support), stacked full+partial coverage bar per row, per-column full+partial tally strip, warm outline on the capability with the fewest support points |
| 1150 | `fig_wm_survey_family_tree.py` | (10.4, 6.4) | Method family tree: 18 methods placed at their publication year in three banded lineages, parent-child S-curves in the family's blue rung, marker area = citations with a three-value size key, one dashed cross-family link, warm ring on the `argmax` citation node |
| 1151 | `fig_wm_survey_leaderboard.py` | (9.0, 6.8) | Leaderboard table: 12 methods x 5 metrics shaded by per-column score (0 = worst, 1 = best) with raw values printed, warm outline on each column's best placed by argmin / argmax under that column's direction, mean-rank bar column, direction strip under the table |
| 1152 | `fig_wm_survey_citation_bump.py` | (10.4, 6.4) | Bump chart of citation rank for 8 methods over 2021-2026; ranks are the argsort of the citation matrix, deep blue = climbed, pale blue = slipped, warm = finishes first, endpoint labels carry the final citation count |
