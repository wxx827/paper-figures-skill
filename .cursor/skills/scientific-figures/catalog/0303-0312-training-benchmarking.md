# 深度学习训练与评测 · Training & benchmarking  `303-312`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 303 | `fig_training_curves_grid.py` | Six-dataset 2x3 grid of train/validation accuracy curves for four methods, hue per method and dash for the train split, +-1 sigma seed ribbons, best-epoch rule and one shared y range across all panels | Training dynamics across benchmarks, "训练曲线" "多数据集对比" |
| 304 | `fig_scaling_laws.py` | Log-log loss against parameters, tokens and compute for three model families, least-squares power-law fits with the exponent printed per family and a dashed extrapolation into a shaded region | Neural scaling laws and extrapolation, "缩放律" "幂律拟合" |
| 305 | `fig_ablation_table_viz.py` | Ablation table redrawn as graphics: tick/cross component matrix, one data bar plus tint per metric cell and an aggregate relative-gain bar, with the full model highlighted | Component ablation summary, "消融实验" "表格可视化" |
| 306 | `fig_lr_schedule_compare.py` | Five learning-rate schedules over a shared epoch axis with the validation accuracy they produce below, transition points carried through both panels by dashed rules and a sorted final-accuracy read-out | Learning-rate schedule study, "学习率调度" "双面板对齐" |
| 307 | `fig_loss_landscape.py` | 3-D loss surface with a contour projection on the floor beside a top view carrying two numerically integrated optimiser paths, arrow heads, epoch stamps and a local-minimum trap | Optimisation landscape and escape behaviour, "损失地形" "优化轨迹" |
| 308 | `fig_efficiency_pareto.py` | Accuracy versus log-scale inference latency for eighteen models, bubble area for parameter count, hue for model family, a computed non-dominated staircase and a "better" direction arrow | Accuracy-efficiency trade-off, "帕累托前沿" "效率对比" |
| 309 | `fig_robustness_radar.py` | Radar of accuracy under six corruption families for four methods, paired with a tinted card of half-size severity-sweep small multiples that share one y range | Robustness and corruption benchmarks, "鲁棒性雷达" "扰动衰减" |
| 310 | `fig_benchmark_leaderboard.py` | Ranked horizontal bars whose segments decompose the aggregate score into five benchmarks, a previous-SOTA rule and a companion rank table with number-plus-tint cells | Leaderboard and SOTA comparison, "排行榜" "排名热图" |
| 311 | `fig_convergence_seeds.py` | Three methods, one panel each: ten translucent per-seed traces, a heavy seed mean, a +-1 sigma ribbon and a mean +- std / coefficient-of-variation call-out, all on identical axes | Seed variability and reproducibility, "多种子收敛" "稳定性" |
| 312 | `fig_transfer_matrix.py` | Source-by-target transfer heat map on a diverging scale with a greyed same-task diagonal, hierarchical-clustering order shared by rows and columns, annotated cells and slim marginal mean bars | Transfer learning and task affinity, "迁移矩阵" "任务相关性" |
