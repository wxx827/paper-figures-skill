# 数据库系统 · Database systems  `854-863`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 854 | `fig_fir_evacuation_flow.py` | Evacuation floor plan with flow-weighted corridor lanes, exit queue density hotspots, cumulative egress under three scenarios and exit flow rates | 疏散模拟与出口瓶颈分析, "疏散路径, 出口瓶颈, 排队密度, RSET, 人员流率" |
| 855 | `fig_str_pushover_curve.py` | Static pushover of three retrofit schemes with IO/LS/CP performance bands, equal-energy bilinear, storey drift profiles and plastic hinge sequence | 抗震性能化评估与推覆分析, "静力推覆, 基底剪力, 等效双折线, 层间位移角, 塑性铰" |
| 856 | `fig_str_fragility_curves.py` | Lognormal seismic fragility for four damage states with design-intensity readout, damage-state probability stack summing to one, and median capacity by typology | 地震易损性与风险评估, "易损性曲线, 对数正态, 超越概率, 破坏状态, 中位能力" |
| 857 | `fig_str_moment_diagram.py` | Three-span continuous beam solved by stiffness method: load case with support symbols, shear and tension-side moment diagram, patterned envelope and reactions | 结构力学内力分析与配筋包络, "弯矩图, 剪力图, 连续梁, 反弯点, 支座反力" |
| 858 | `fig_orx_lp_feasible_region.py` | Two-variable production LP: feasible polytope with hatched infeasible half-planes, objective contour family, solved optimal vertex, per-row slack and an objective-coefficient sensitivity sweep | 线性规划教学与产能配比讲解, "可行域", "单纯形最优顶点", "影子价格与灵敏度" |
| 859 | `fig_orx_branch_bound_tree.py` | Depth-first branch-and-bound search tree over an integer programme: LP bounds in every node box, branch conditions on the edges, crossed-out pruned nodes, bound-gap convergence and nodes per depth | 整数规划求解过程讲解, "分支定界", "剪枝", "上下界收敛" |
| 860 | `fig_orx_vrp_routes.py` | Capacitated vehicle routing built by nearest neighbour plus 2-opt: equal-aspect depot map with demand-scaled customers, running load along each closed tour, per-vehicle fill rate and distance saved against the order-book plan | 配送路径规划与车队调度汇报, "VRP", "2-opt", "载重利用率" |
| 861 | `fig_sch_jobshop_gantt.py` | Job-shop Gantt from a non-delay SPT dispatcher: machine rows, one colour per job across machines, routing links between consecutive operations, hatched idle bands, makespan rule, machine utilisation and job lateness | 车间作业排程与产线节拍分析, "甘特图", "makespan", "机器利用率" |
| 862 | `fig_sch_critical_path.py` | Activity-on-node project network with forward and backward CPM passes: ES/EF/LS/LF in every node, red zero-float critical spine, float ranking bar chart and a resource histogram before and after exact float-window levelling | 项目进度管理与工期压缩评审, "关键路径", "浮时", "资源平准化" |
| 863 | `fig_orx_queue_performance.py` | M/M/c performance: mean wait and queue length against utilisation on twin log axes for four server counts, the rho to 1 asymptote, an operating reference point, interarrival versus service-time densities and the wait-breach service level | 服务台配置与排队系统容量规划, "排队论", "利用率拐点", "服务水平" |
