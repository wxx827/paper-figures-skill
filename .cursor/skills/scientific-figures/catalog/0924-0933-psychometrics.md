# 心理测量 · Psychometrics  `924-933`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 924 | `fig_law_sentencing_disparity.py` | Sentence-length density bands by judge against the advisory guideline range, with covariate-adjusted deviations and an OLS forest of sentencing factors | 量刑一致性分析, "量刑差异" "指南偏离" "法官效应" |
| 925 | `fig_law_case_flow_sankey.py` | Five-stage alluvial of a civil docket from filing to enforcement with asserted flow conservation, plus stage-duration boxes and reversal rate by cause of action | 案件流转与司法效率, "桑基图" "案件分流" "上诉改判率" |
| 926 | `fig_law_citation_network.py` | Directed precedent-citation network laid out by judgment year and court tier, with a power-law tail of times cited and citation half-life curves | 判例引用网络分析, "先例影响力" "幂律分布" "引用半衰期" |
| 927 | `fig_pcy_budget_allocation.py` | Airtight stacked shares of five programme areas across 25 budget years with the envelope on a second axis, spend-per-head versus outcome trajectories and execution rates | 财政预算配置演变, "预算份额" "人均支出产出" "预算执行率" |
| 928 | `fig_pcy_regulatory_impact.py` | Cumulative discounted cost against benefit with an interpolated payback year and a discount-rate family, plus a cost-benefit waterfall and an assumption tornado | 监管影响评估, "成本收益" "回本时点与NPV" "敏感性分析" |
| 929 | `fig_pcy_equity_incidence.py` | Benefit by income decile in dollars and as a share of income on twin axes, with pre and post Lorenz curves and the benefit concentration curve behind the Kakwani index | 政策受益分布与公平性, "累进性" "洛伦兹曲线与基尼" "集中曲线" |
| 930 | `fig_wt_piv_vector_field.py` | PIV plane behind a NACA 0012 near stall: vorticity background with a decimated quiver, ringed vortex cores and a masked model silhouette, plus wake-deficit and Reynolds-shear-stress profiles on three traverses | 风洞 PIV 速度场, 翼型尾迹涡街, "涡量云图" "矢量箭头" "涡核识别" "尾迹亏损" "雷诺应力" |
| 931 | `fig_wt_boundary_layer_profile.py` | Law of the wall in viscous units at three Reynolds numbers with the u+=y+ and log-law reference lines and a Coles wake, plus Blasius-versus-turbulent outer profiles and the shape factor along a transitioning plate | 边界层剖面, 壁面律标志图, "y+ u+ 对数律" "粘性底层" "尾迹区" "Blasius" "形状因子 H" "动量厚度" |
| 932 | `fig_wt_polar_lift_drag.py` | Drag polar at three Reynolds numbers with the drag-bucket floor and the origin tangent that locates maximum lift-to-drag, an inset lift curve through stall, plus glide ratio and pitching moment against incidence | 气动极曲线, 天平测力, "升阻极曲线" "最小阻力点" "最大升阻比切线" "失速" "力矩系数" "气动中心" |
| 933 | `fig_wt_inlet_distortion.py` | Total-pressure recovery contoured at the engine face with the 40-probe rake and the searched worst 60-degree sector, plus unwrapped ring traces and a DC(60) sweep against mass-flow ratio | 进气道畸变, 发动机进口截面, "总压恢复" "耙式测点" "DC60 畸变指数" "周向压力分布" "流量比" |
