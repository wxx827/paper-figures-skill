# 土壤科学 · Soil science  `714-723`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 714 | `fig_wnd_shear_profile.py` | Wind shear profile: log-law vs power-law fit with rotor swept band, stability family and shear-exponent distribution | 风资源评估报告, 测风塔数据分析, "风切变" "廓线拟合" "轮毂高度风速" |
| 715 | `fig_wnd_turbine_power_curve.py` | Turbine power curve with cut-in/rated/cut-out regions, thrust coefficient on a right axis, Weibull distribution and per-bin annual energy | 风机选型与功率曲线验收, 发电量测算, "功率曲线" "推力系数" "威布尔分布" "年发电量" |
| 716 | `fig_wnd_wake_field.py` | Single-turbine wake: plan-view velocity field with expansion cone and contours, centreline recovery and lateral cuts | 尾流建模与CFD结果展示, 机位间距论证, "尾流场" "速度亏损" "尾流恢复" |
| 717 | `fig_wnd_farm_layout.py` | Wind farm layout: turbines coloured by annual yield with wake shadows, inset wind rose and scale bar, row-wise loss and layout comparison | 风场微观选址, 尾流损失评估, "风场布局" "阵列效率" "尾流损失" "风玫瑰" |
| 718 | `fig_abl_stability_regimes.py` | Boundary-layer stability: potential-temperature soundings for stable/neutral/unstable states with layer structure, diurnal depth cycle and Obukhov stability histogram | 大气边界层教学与观测报告, 稳定度分类, "边界层结构" "位温廓线" "混合层高度" "奥布霍夫长度" |
| 719 | `fig_abl_turbulence_spectrum.py` | Turbulence spectra at four heights on log-log axes with energy-containing, inertial and dissipation ranges, minus five thirds reference, intensity profile and vertical coherence | 湍流特性分析, 载荷与疲劳输入, "湍流谱" "惯性副区" "湍流强度" "相干函数" |
| 720 | `fig_cat_hazard_footprint.py` | Tropical-cyclone wind footprint as the running maximum of an asymmetric modified-Rankine field, house-red wind classes over a grey population-density surface, recurving track with 12 h markers and a widening uncertainty cone, plus intensity history and cumulative exposed population by wind class | 台风灾害影响范围评估、风场足迹制图、承灾体暴露度分析, "台风路径, 风场足迹, 风圈分级, 不确定性锥, 人口暴露度" |
| 721 | `fig_cat_vulnerability_curve.py` | Log-normal vulnerability curve family for five building classes with epistemic bands and marked half-damage intensities, a conditional beta distribution of damage ratio at one intensity slice, and stock share against expected-loss share per class | 建筑易损性曲线族对比、损失比条件分布、承灾体存量与损失贡献分析, "易损性曲线, 损失比, 中位损失强度, 次生不确定性, 建筑存量占比" |
| 722 | `fig_cat_exceedance_probability.py` | Log-log EP curves from a 200 000-year Poisson-lognormal simulation with AEP and OEP read at the 10, 50, 100 and 250-year return periods in an inline table, a generalised Pareto tail fitted to event severity, and PML and TVaR capital bars against average annual loss | 巨灾模型超越概率曲线、重现期损失读取、资本充足与再保定价, "超越概率曲线, EP曲线, 重现期, PML, TVaR, 年均损失" |
| 723 | `fig_cat_portfolio_exposure.py` | Insured locations as sum-insured bubbles coloured by line of business over a filled 100-year gust field with the worst accumulation zone outlined, plus region-by-line stacked exposure bars and a Lorenz concentration curve carrying the Gini and Herfindahl index | 保险组合风险敞口制图、风险累积区集中度分析、再保分入组合审阅, "风险敞口, 保额分布, 风险集中度, 累积区, 基尼系数, 赫芬达尔指数" |
