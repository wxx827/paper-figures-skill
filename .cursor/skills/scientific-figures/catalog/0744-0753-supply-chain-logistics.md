# 供应链物流 · Supply chain & logistics  `744-753`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 744 | `fig_am_meltpool_thermal.py` | Rosenthal melt-pool temperature field in top view and longitudinal section, liquidus isotherm marking the trailing pool, with centreline temperature and cooling-rate profiles | 增材制造熔池热场讲解, 激光/电弧工艺参数汇报, "熔池", "温度场", "冷却速率" |
| 745 | `fig_am_scan_strategy.py` | Six laser scan strategies on one layer footprint, vectors coloured by scan order on a shared ruler, with peak residual stress and relative density compared underneath | 粉床增材扫描路径对比, 工艺策略选型汇报, "扫描策略", "棋盘格", "残余应力" |
| 746 | `fig_am_process_window.py` | Printability map over laser power and scan speed with balling, lack-of-fusion, keyhole and printable regions, iso-energy lines, coupon scatter and power-headroom curves | 增材工艺窗口标定, 参数优化报告, "工艺窗口", "能量密度", "缺陷区" |
| 747 | `fig_am_porosity_defects.py` | CT-reconstructed pore cloud in a build cylinder sized by diameter and coloured by sphericity, with log-scale diameter histogram and porosity against build height | CT 孔隙检测报告, 增材缺陷分析, "孔隙率", "球形度", "未熔合" |
| 748 | `fig_wld_heat_affected_zone.py` | Arc weld cross-section split into fusion zone and three HAZ bands by hatch and tint, hardness traverse below on the same axis, thermal cycles and CCT cooling paths alongside | 焊接热影响区教学, 焊接工艺评定, "热影响区", "硬度分布", "CCT 曲线" |
| 749 | `fig_wld_residual_stress.py` | Longitudinal residual stress map around a weld run on a zero-centred diverging scale with stress contours, transverse profile against yield and peak stress by welding sequence | 焊接残余应力评估, 变形与开裂风险汇报, "残余应力", "发散色标", "焊接顺序" |
| 750 | `fig_pal_ice_core_record.py` | Four stacked ice-core proxy records on a shared reversed age axis with marine isotope stage bands, plus a cross-proxy correlation matrix | 冰芯多指标古气候重建, "冰期-间冰期旋回", "MIS 阶段", "代用指标相关性" |
| 751 | `fig_pal_orbital_cycles.py` | Eccentricity, obliquity and precession against insolation and an ice-volume proxy, with Milankovitch power spectra and band-passed phase lags | 轨道驱动与米兰科维奇理论, "偏心率/斜率/岁差", "功率谱", "100 ka 问题" |
| 752 | `fig_pal_pollen_diagram.py` | Percentage pollen diagram: ten taxon columns grouped by growth form on a downward depth axis, with concentration, charcoal and CONISS-style zones | 湖芯孢粉与古植被重建, "孢粉图", "花粉带划分", "炭屑与人类活动" |
| 753 | `fig_iso_delta_crossplot.py` | Carbon-oxygen crossplot by stratigraphic unit with 95 percent ellipses, a diagenetic trend line and a covariation screening test | 碳酸盐同位素成岩筛查, "delta-13C 与 delta-18O 交会图", "置信椭圆", "协变性检验" |
