# 脑影像 fMRI · Neuroimaging  `604-613`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 604 | `fig_wat_reservoir_operation.py` | Daily reservoir mass-balance simulation on a shared time axis: inflow versus total release, pool level against normal, dead and phased flood-limit levels, generation bars, flood-season band and detected spill days in red, with the year replayed on a rule-zone operating chart | 水库调度方案展示、防洪与发电调度复盘、水库运行年报, "水库调度图, 汛限水位, 弃水, 入库出库流量, 发电出力" |
| 605 | `fig_wat_supply_demand.py` | Monthly supply stacked by source against a total demand line with the deficit filled in red, a sector-demand donut with non-crossing elbow leaders, and shortage-rate bars re-run at four hydrological design frequencies | 区域供需水平衡分析、缺水风险评估、水资源配置规划, "供需平衡, 缺水率, 保证率, 分水源供水, 需水结构" |
| 606 | `fig_spo_pitch_heatmap.py` | Full-size football pitch with player activity heat map, average-position markers, zone touch shares and per-player distance covered | 足球跑动与控球分析、球队阵型报告, "球场热图, 活动密度, 触球分区, 跑动距离" |
| 607 | `fig_spo_pass_network.py` | Directed pass network over average positions, with the pass matrix and eigenvector-centrality ranking | 传球组织与核心球员识别、战术复盘, "传球网络, 传球矩阵, 中心性, 传球成功率" |
| 608 | `fig_spo_shot_map.py` | Half-pitch shot map over an xG surface, with cumulative xG versus goals and conversion by zone | 射门质量与终结效率评估、赛后报告, "射门分布, xG 期望进球, 转化率, 半场球场图" |
| 609 | `fig_spo_player_radar.py` | Six-axis player ability radar over a league-average baseline, with percentile bars and minutes played | 球员能力画像、转会引援与选材对比, "能力雷达图, 百分位排名, 联赛基准, 综合评分" |
| 610 | `fig_spo_season_form.py` | Season ladder with inverted rank axis, result ribbon, goal-difference bars, home-away split and points accumulation | 赛季走势复盘、联赛排名报告, "排名曲线, 胜平负色带, 净胜球, 主客场对比" |
| 611 | `fig_spo_load_injury.py` | Acute and chronic training load with ACWR danger bands, injury events, wellness strips and an ACWR risk curve | 训练负荷监控与伤病风险预警, "急慢性负荷, ACWR, 伤病风险, 疲劳睡眠条带" |
| 612 | `fig_cry_glacier_massbalance.py` | Distributed glacier surface mass balance map with the equilibrium line drawn as the zero contour, a balance-versus-elevation gradient profile and a 44-year cumulative loss series | 冰川学监测报告、山地冰川物质平衡评估、气候变化影响图, "冰川物质平衡, 平衡线高度, 消融区, 积累区, 累计物质亏损" |
| 613 | `fig_cry_seaice_extent.py` | Arctic sea-ice concentration on an equal-area polar disk with the 15 % ice edge against the 1991-2020 median, the family of annual cycles and the March/September extreme trends | 极地海冰监测、气候变化通报、北极航道评估, "海冰范围, 海冰密集度, 极地投影, 冰缘线, 九月最小值, 年循环" |
