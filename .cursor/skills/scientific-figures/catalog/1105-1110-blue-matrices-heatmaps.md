# 世界模型 · 蓝色矩阵与热图 Blue matrices & heatmaps  `1105-1110`

6 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 1105 | `fig_wm_blue_similarity_matrix.py` | (7.4, 6.8) | 10x10 benchmark similarity matrix, three-decimal cell labels with automatic white/ink flip, 45-degree column names, narrow quarter-height "Sim." colourbar |
| 1106 | `fig_wm_blue_error_grid.py` | (7.0, 9.6) | 8 scenes x 5 methods rollout error maps on a dark navy base, one slim colourbar per row, scene names set vertically at the left edge, single shared scale for all 40 panels |
| 1107 | `fig_wm_blue_confusion_pair.py` | (9.0, 4.6) | Before/after adaptation confusion matrices on one shared scale and one colourbar, diagonal cells outlined, footer comparing overall accuracy with the warm accent on the gain |
| 1108 | `fig_wm_blue_corr_dendrogram.py` | (7.6, 6.8) | Clustered latent-factor correlation heatmap with hairline navy dendrograms on top and left, colourbar in the free top-right corner, 20 labels at 6.4 pt without collisions |
| 1109 | `fig_wm_blue_occupancy_density.py` | (10.4, 3.6) | Occupancy-grid triptych sharpening over three exploration stages, muted blue unknown space, black leader lines to bold "Robot position (A/B/C)" call-outs, corner letters (a)(b)(c) |
| 1110 | `fig_wm_blue_attention_matrix.py` | (10.0, 7.0) | 4 layers x 6 heads of causal attention maps on one shared capped scale and one colourbar, per-panel `L{i}H{j}` tags, a different attention regime per layer |
