# 大气化学 · Atmospheric chemistry  `674-683`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 674 | `fig_bch_gas_fee_dynamics.py` | EIP-1559 base fee versus block utilisation with congestion bands, bid-latency quantiles and fee split | 链上经济/Gas 费分析, "Gas 费动态" "EIP-1559" "拥堵" "双轴时序" "费用构成" |
| 675 | `fig_hci_eye_gaze_ui.py` | Gaze heat map and numbered scan path over a UI wireframe with per-AOI first-fixation and dwell bars | 眼动实验/界面可用性报告, "眼动热图" "扫视路径" "AOI 兴趣区" "线框图" "首次注视时间" |
| 676 | `fig_hci_usability_metrics.py` | Task completion versus time bubbles against an acceptance window, SUS distribution, radar and learning curve | 可用性评估/用户测试汇报, "SUS 量表" "完成率" "气泡图" "维度雷达" "学习曲线" |
| 677 | `fig_hci_interaction_flow.py` | Onboarding funnel Sankey with grey drop-off ribbons, per-step dwell bars and exit-reason breakdown | 用户旅程/转化漏斗汇报, "桑基图" "交互流程" "转化漏斗" "流失分析" "停留时长" |
| 678 | `fig_sct_urban_dashboard.py` | City operations dashboard: KPI card strip with sparklines and year-on-year arrows, trip-density map, three stacked diurnal service series, full-width incident timeline, banded background cards for visual zoning | 智慧城市运行监测中心大屏, 城市体征综合看板, "KPI 卡片, 迷你趋势线, 热力地图, 事件时间轴, 分区看板" |
| 679 | `fig_sct_multi_source_fusion.py` | Multi-source urban data fusion: three-tier source diagram annotated with update interval and native resolution, fused 500 m city grid, inverse-variance source weights, error before and after fusion | 多源异构城市数据融合, 遥感物联网政务数据协同, "数据源分层, 更新频率, 空间分辨率, 融合权重, 精度提升" |
| 680 | `fig_sct_emergency_response.py` | Emergency response reachability: travel-time isochrones over the street grid with target and escalation contours, station and incident markers, cumulative response-time curve, duration boxplots by incident class | 城市应急响应能力评估, 消防急救站点选址与覆盖分析, "等时圈, 可达性, 达标率, 响应时长分布, 路网" |
| 681 | `fig_mfg_flexible_line.py` | Flexible assembly line: U-shaped ten-station layout with material-flow arrows and an AGV corridor, stations shaded by load under the current order, changeover-visible shift Gantt, balance rate against mix evenness | 柔性制造产线布局与节拍分析, 多品种混线生产排程, "工位占用, 物料流, AGV 路径, 换型时间, 产线平衡率" |
| 682 | `fig_mfg_quality_traceability.py` | Quality genealogy trace: four-level batch tree from raw lots to finished goods with the recall path highlighted in red, node-level inspection verdicts, defect Pareto by process step, coating control chart | 质量追溯与批次召回分析, 制造过程缺陷根因定位, "追溯树, 问题批次, 检验结果, 帕累托图, 控制图" |
| 683 | `fig_mfg_energy_optimization.py` | Production energy optimisation: hourly energy stacked by equipment with output on the right axis and tariff bands behind, saving waterfall by measure, specific energy ranking before and after | 生产能耗监测与节能优化, 峰谷电价下的负荷管理, "堆叠面积图, 峰谷时段, 瀑布图, 能效排行, 节能率, 成本节约" |
