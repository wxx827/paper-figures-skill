# 计算机图形 · Computer graphics  `824-833`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 824 | `fig_mph_allometry_scaling.py` | Log-log allometry with RMA slopes and bootstrap CIs, isometry reference at exactly 45 deg, slope forest plot and residual diagnostics | 异速生长分析、类群间标度关系比较, "异速生长, RMA回归, 等速生长线, 森林图, 自举置信区间" |
| 825 | `fig_pln_stratigraphic_range.py` | Stratigraphic range chart: horizontal taxon ranges with sampling whiskers and abundance shading, geological period band, mass-extinction rules, per-period turnover and range-duration panels | 古生物延限图、化石记录时序展示、类群更替分析, "地层延限, 首现末现, 地质年代带, 灭绝事件" |
| 826 | `fig_pln_diversity_curve.py` | Phanerozoic diversity curve, raw versus sampling-standardised with an ensemble band, big-five extinction markers, origination/extinction rate curves and ranked event severity | 多样性曲线、灭绝事件强度对比、抽样标准化演示, "多样性曲线, 五次大灭绝, 起源率, 灭绝率" |
| 827 | `fig_pln_phylo_timetree.py` | Time-calibrated rectangular timetree with 95% HPD node bars, fossil calibrations, extinct-lineage crosses, node-age posterior ridgelines and a log-axis LTT plot | 分子钟系统树、分歧时间估计、谱系累积曲线展示, "时间标定树, HPD区间, 定标化石, LTT曲线" |
| 828 | `fig_met_phase_diagram_binary.py` | Eutectic binary phase diagram with tinted phase fields, a cooling path with lever-rule tie lines, plus cooling curve and stacked phase fractions | 材料热力学教学与合金凝固分析, "二元相图", "共晶点", "杠杆定律", "液相线固相线" |
| 829 | `fig_met_ttt_curve.py` | Double-C isothermal transformation diagram on a log time axis with Ms/Mf lines, three continuous cooling paths, Jominy hardenability sweep and product inventory | 钢热处理工艺设计与淬火选型, "TTT曲线", "CCT曲线", "马氏体贝氏体珠光体", "淬透性" |
| 830 | `fig_met_ellingham.py` | Ellingham diagram of oxide formation free energies with the two carbon lines, solved reduction temperatures, pO2 and CO/CO2 nomographic rays, plus ranking and stability bars | 冶金还原热力学与炉料设计, "埃林厄姆图", "氧化物稳定性", "碳还原温度", "氧分压标尺" |
| 831 | `fig_mnp_particle_size_dist.py` | Particle size distribution of a hydrocyclone circuit: log-size frequency with cumulative passing on a twin axis, d10/d50/d80/d90 callouts, Rosin-Rammler fit and Tromp partition curve | 磨矿分级流程评价与筛析报告, "粒度分布", "累积通过率", "Rosin-Rammler", "Tromp分级效率" |
| 832 | `fig_mnp_flotation_recovery.py` | Grade-recovery curves for three reagent schemes over separation-efficiency contours with the theoretical limit, plus first-order kinetics and mineral floatability bars | 浮选药剂制度对比与选厂指标谈判, "品位回收率曲线", "分离效率", "浮选动力学", "可浮性" |
| 833 | `fig_mnp_grade_tonnage.py` | Grade-tonnage curves cut from one lognormal block model: tonnage falling and average grade rising on twin axes, with peak produced metal and price-scenario sensitivity | 矿床储量评估与边界品位优化, "品位吨位曲线", "边界品位", "金属量最优点", "价格敏感性" |
