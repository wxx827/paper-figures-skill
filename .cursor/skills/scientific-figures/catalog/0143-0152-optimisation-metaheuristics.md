# 优化与元启发式算法 · Optimisation & metaheuristics  `143-152`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 143 | `fig_benchmark_convergence.py` | 4x2 board of eight benchmark blocks; each pairs a 3-D landscape (coolwarm surface floating over its filled contour projection, own vertical colourbar) with the averaged convergence curves of five metaheuristics, two blocks carrying a first-ten-iteration zoom inset linked by a magenta callout arrow | 群智能算法主图、基准函数+收敛对比一图流、投顶刊的核心图, "算法收敛对比大图" |
| 144 | `fig_benchmark_surfaces.py` | 2x4 gallery of the eight test-function landscapes from one common viewpoint; RdYlBu_r surfaces with fine mesh edges, white-contoured floor projection, global-optimum star, per-panel value range and one shared normalised colourbar | 测试函数地形展示、基准套件介绍章节, "基准函数曲面画廊" |
| 145 | `fig_convergence_grid.py` | 3x2 log-scale convergence grid; geometric-mean best-so-far curves for five algorithms wrapped in min-max envelopes over 30 runs, with a ranked final-fitness strip printed under every panel | 收敛速度对比、30次独立运行统计、算法优劣速览, "收敛曲线网格图" |
| 146 | `fig_algo_boxplot_stats.py` | 3x2 box-plot panel of the final fitness of 30 runs; log axis, jittered per-run dots, geometric-mean diamonds, Wilcoxon rank-sum brackets against the proposed algorithm and a +/=/- win-tie-loss tally per function | 统计显著性检验、算法稳定性对比、秩和检验可视化, "算法箱线图统计" |
| 147 | `fig_exploration_exploitation.py` | 2x3 exploration-exploitation balance panels; twin-axis blue/orange filled areas that cross at the annotated balance point, each stacked over a log-scale population-diversity decay strip | 探索开发平衡分析、种群多样性衰减、算法行为解释, "探索开发平衡图" |
| 148 | `fig_search_trajectory.py` | Classic four-panel qualitative behaviour analysis: iteration-coloured search path over the 2-D contour map, first-dimension oscillation with contracting envelope, average swarm fitness and mean pairwise distance | 搜索轨迹分析、个体运动可视化、算法定性分析, "搜索轨迹四联图" |
| 149 | `fig_swarm_snapshots.py` | 2x4 population snapshots at iterations 1 to 200 over the Rastrigin contour map; agents coloured by fitness, dashed spread circle, incumbent-best star, and magnified insets once the swarm has collapsed | 种群收敛过程、聚集行为演示、算法动画式截图, "种群分布快照" |
| 150 | `fig_friedman_ranking.py` | Friedman and Nemenyi analysis in three parts: mean-rank bars with the critical difference band, a Demsar critical-difference diagram with clique bars, and the full algorithm-by-function rank heatmap with mean rank and overall placing | Friedman检验、Nemenyi事后检验、多算法总排名, "Friedman排名图" |
| 151 | `fig_multiobj_pareto3d.py` | Three-objective Pareto board: large 3-D scatter of five non-dominated sets against the translucent analytical front, three 2-D objective-plane projections, and IGD / hypervolume / spacing indicator bars | 多目标优化、Pareto前沿对比、IGD与HV指标, "三目标帕累托前沿" |
| 152 | `fig_param_sensitivity.py` | Parameter study in five parts: bicubic grid-search heatmap with iso-rank contours and a star on the selected setting, three single-parameter sweeps with inter-quartile bands, and a functional-ANOVA importance bar chart | 参数敏感性分析、网格搜索调参、超参数重要性, "参数敏感性分析" |
