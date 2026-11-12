# 森林碳汇 · Forest carbon  `704-713`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 704 | `fig_cat_turnover_kinetics.py` | Conversion versus time on stream at five bed temperatures with a deactivation-free reference and an Arrhenius inset; auxiliary reaction-order log-log plot and site-activity decay curves. | 反应动力学表征、催化剂寿命评估, "转化率曲线, Arrhenius 活化能, 反应级数, 失活半衰期, TOF/TON" |
| 705 | `fig_ech_fuel_cell_polarization.py` | Fuel-cell polarisation curve with activation, ohmic and transport regions shaded from computed crossovers, power density on a second axis with its peak marked; auxiliary loss stack and operating-condition comparison. | 燃料电池性能报告、电堆工况对比, "极化曲线, 三段损失, 功率密度峰值, 面比电阻, 极限电流" |
| 706 | `fig_ech_electrolyzer_efficiency.py` | Electrolyser Faradaic and voltage efficiency with their product and the hydrogen output on a second axis; auxiliary specific-energy curve with its optimum and a four-technology comparison against the thermoneutral floor. | 制氢系统效率评估、电解技术选型, "法拉第效率, 电压效率, 制氢速率, 单位能耗 kWh/kg, 最优工作点" |
| 707 | `fig_ech_tafel_analysis.py` | Tafel plot of four OER catalysts with per-catalyst linear fits, slope-to-mechanism key and extrapolated exchange current densities; auxiliary benchmark-overpotential bars and a 24 h stability test. | 电催化剂活性对比、OER/HER 表征, "Tafel 斜率, 交换电流密度, 过电位@10 mA/cm2, 稳定性测试, 机理判据" |
| 708 | `fig_syn_genetic_circuit.py` | SBOL-glyph toggle-switch circuit with steady-state truth table and bistable state graph | 合成生物学线路设计, "基因线路, 双稳态开关, SBOL 符号" |
| 709 | `fig_syn_expression_dynamics.py` | Reporter induction time courses over six inducer doses with growth curve, Hill dose response and expression-noise scaling | 诱导表达实验汇报, "表达动力学, 剂量响应, Hill 拟合, 表达噪声" |
| 710 | `fig_syn_metabolic_flux.py` | Central-carbon flux map with lane width set by flux, node split pies and yield comparison before and after engineering | 代谢工程改造效果, "通量分析, 代谢网络, 通量分配, 产物得率" |
| 711 | `fig_bio_fermentation_profile.py` | Three-row fed-batch fermentation profile on a shared time axis with phase bands, specific growth rate and cumulative yields | 发酵过程数据汇报, "发酵曲线, 分批补料, 溶氧 pH 转速, 比生长速率" |
| 712 | `fig_bio_bioreactor_scaleup.py` | Parallel-coordinates comparison of kLa, mixing time, shear and power input across five reactor scales, with titre loss attribution | 反应器放大论证, "放大准则, kLa, 混合时间, 偏差归因" |
| 713 | `fig_bio_downstream_purification.py` | Yield and purity staircase across six downstream unit operations, with chromatography elution trace and SDS-PAGE lane schematic | 下游纯化工艺汇报, "纯化流程, 收率纯度, 洗脱曲线, SDS-PAGE" |
