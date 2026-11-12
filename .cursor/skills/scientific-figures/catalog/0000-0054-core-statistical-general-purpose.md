# 基础统计与通用图表 · Core statistical & general-purpose  `0-54`

55 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 0 | `fig_00_style_showcase.py` | Design-system poster for `pubstyle.py`: every palette and journal theme, helper-function demos, colour-vision safety check and the "what makes a figure look expensive" checklist | 上手前先通读一遍配色与版式规范, "样式总览, 配色板, 设计系统" |
| 1 | `fig_threshold_regression.py` | Piecewise/threshold regression grid: dual-slope R² + vertical threshold + LOWESS trend + CI band | driver-vs-response with a breakpoint; "阈值/分段回归" |
| 2 | `fig_marginal_violin.py` | Joint scatter with top/right marginal histograms, threshold split, fit + CI, inset violin | "边缘直方图 + 小提琴", before/after threshold |
| 3 | `fig_ml_prediction_grid.py` | Predicted-vs-actual grid across many ML models, 1:1 line, fit line, marginal KDE, R²/RMSE/MAE box | "模型对比", regression benchmark panel |
| 4 | `fig_pred_importance.py` | Fit scatter (fitting vs standard curve) + feature importance bar ranking, two periods | random-forest %IncMSE / importance ranking |
| 5 | `fig_corrplot.py` | corrplot: circle-sized+colored correlation matrix + high-\ | r\ |
| 6 | `fig_pairplot.py` | Pair grid: upper=shaded r values, diag=KDE, lower=scatter+fit, one hue per panel | seaborn `pairplot`, multi-hue correlation |
| 7 | `fig_constraint_boundary.py` | Constraint-boundary scatter: grey cloud + upper/lower quantile envelope fits + R² | "约束线/边界分析" ecology response |
| 8 | `fig_coupling_zones.py` | Coupling coordination degree time series over labelled zone bands + proportion pies | "耦合协调度", zone-classified trend + pie |
| 9 | `fig_circular_bars.py` | Radial grouped barplot grid, rotated outward labels, concentric series | "环形柱状图 / circular barplot" |
| 10 | `fig_ecoservice_lines.py` | Inside/outside-reserve service-change line charts with reference bars, multi-county series | ecosystem service change over periods |
| 11 | `fig_shap_pdp.py` | SHAP beeswarm summary + column of PDP trend panels with threshold + count strip | model interpretation, "SHAP + 部分依赖" |
| 12 | `fig_raincloud.py` | Raincloud: half-violin + box + jittered strip per group | distribution comparison, "云雨图" |
| 13 | `fig_ridgeline.py` | Ridgeline/joyplot: overlapping density ridges over ordered categories | distribution shift over time, "山脊图" |
| 14 | `fig_clustermap.py` | Clustered heatmap with row+column dendrograms + annotated cells | correlation clustering, "聚类热图" |
| 15 | `fig_radar.py` | Radar/spider chart, filled multi-series polygons | multi-metric profile comparison, "雷达图" |
| 16 | `fig_streamgraph.py` | Streamgraph: wiggle-baseline stacked area | composition flow over time, "河流图" |
| 17 | `fig_chord.py` | Chord diagram: bezier-ribbon flows between categories | transition/flow matrix, "和弦图" |
| 18 | `fig_hexbin.py` | Hexbin density scatter + marginal histograms + fit | big point clouds, overplotting, "六边形密度" |
| 19 | `fig_bump.py` | Bump chart: smooth ranking-change lines over time | rank dynamics, "排名变化图" |
| 20 | `fig_waterfall.py` | Waterfall: cumulative +/- contribution decomposition | attribution/decomposition, "瀑布图" |
| 21 | `fig_taylor.py` | Taylor diagram: model skill (correlation + std + RMSE arcs) | model evaluation, "泰勒图" |
| 22 | `fig_sankey.py` | Sankey: multi-stage bezier-ribbon flow diagram | source→sector→fate flows, "桑基图" |
| 23 | `fig_treemap.py` | Treemap: squarified value-sized nested rectangles | hierarchical share, "矩形树图" |
| 24 | `fig_contour_density.py` | 2D KDE filled contour density + scatter + marginals | bivariate distribution, "二维密度/等高线" |
| 25 | `fig_calendar_heatmap.py` | Calendar heatmap: daily values as week×weekday grid | daily time series, "日历热图" |
| 26 | `fig_upset.py` | UpSet plot: set-intersection sizes + dot matrix | many-set overlaps (Venn alt), "UpSet图" |
| 27 | `fig_spatial_grid.py` | 3×2 spatial raster maps, shared diverging colorbar, year badges, N-arrow, scale bar | multi-year gridded maps, "空间栅格/地图网格" |
| 28 | `fig_learning_curves.py` | 3×2 multi-trial reward curves + variance bands + success/fail zones | RL/training curves, "学习曲线/多trial" |
| 29 | `fig_trend_ci.py` | 3×2 time trends: scatter + OLS + 95% CI + slope/R²/stars | temporal trends, "趋势+置信带" |
| 30 | `fig_violin_grid.py` | 3×2 violins + inner box + mean dot + significance bracket | group distribution comparison, "小提琴对比" |
| 31 | `fig_hist_kde_grid.py` | 3×2 histogram + KDE + mean/median + stats box | distribution shapes, "直方+KDE" |
| 32 | `fig_barerr_grid.py` | 3×2 grouped bars + SEM error bars + significance stars | paired category comparison, "误差棒分组柱" |
| 33 | `fig_shap_donut.py` | SHAP importance bars + nested category donut + beeswarm (2 rows) | detailed model attribution, "SHAP甜甜圈组合" |
| 34 | `fig_roc_pr_grid.py` | 3×2 ROC + PR curves with AUC/AP annotations | model classification evaluation, "ROC/PR" |
| 35 | `fig_residual_grid.py` | 3×2 regression diagnostics: residuals, Q-Q, scale-location | model validation, "残差诊断" |
| 36 | `fig_dualaxis_grid.py` | 3×2 dual y-axis time series (line + bar/line pair) | driver-response co-evolution, "双轴时序" |
| 37 | `fig_stacked_area_grid.py` | 3×2 100% stacked area composition trends | share over time, "堆叠面积占比" |
| 38 | `fig_volcano.py` | Volcano plot: log2FC vs −log10(p), up/down/non-sig | differential analysis, "火山图" |
| 39 | `fig_dumbbell.py` | Dumbbell before/after paired comparison | policy impact, "哑铃图/前后对比" |
| 40 | `fig_lollipop.py` | Lollipop ranking chart with value labels | feature/importance ranking, "棒棒糖图" |
| 41 | `fig_parallel_coords.py` | Parallel coordinates by group | multi-metric profiles, "平行坐标" |
| 42 | `fig_heatmap_month_grid.py` | 3×2 month×year anomaly heatmaps, shared colorbar | seasonal patterns, "月×年热图" |
| 43 | `fig_scatter_marginal_grid.py` | 3×2 scatter + regression + CI + marginal KDE | bivariate with marginals, "散点+边缘密度" |
| 44 | `fig_dotplot.py` | Dot plot with CI error bars (Cleveland plot) | group comparison, "点图/CI对比" |
| 45 | `fig_manhattan.py` | Manhattan plot: −log10(p) by chromosome, alternating colors | GWAS-style, "曼哈顿图" |
| 46 | `fig_qq_grid.py` | 3×2 Q-Q normality panels with confidence envelope | distribution check, "Q-Q图" |
| 47 | `fig_box_jitter_grid.py` | 3×2 box + jitter + mean diamond + significance | clean group compare, "箱线+抖动" |
| 48 | `fig_corr_network.py` | Correlation network: spring layout, signed edge colors | variable relationships, "相关网络" |
| 49 | `fig_slope_chart.py` | Slope chart: start-end paired lines, direction colors | before/after ranking, "斜率图" |
| 50 | `fig_funnel.py` | Funnel chart: centered rounded trapezoids with % labels | pipeline stages, "漏斗图" |
| 51 | `fig_gantt.py` | Gantt timeline with rounded bars + today marker | project schedule, "甘特图" |
| 52 | `fig_polar_rose.py` | Polar rose / Nightingale chart with gradient greens | cyclic seasonal data, "玫瑰图" |
| 53 | `fig_bubble_matrix.py` | Bubble matrix: color=correlation, size=weight | matrix with dual encoding, "气泡矩阵" |
| 54 | `fig_gradient_area_grid.py` | 3×2 gradient-filled area time series + peak labels | infographic-style trends, "渐变面积" |
