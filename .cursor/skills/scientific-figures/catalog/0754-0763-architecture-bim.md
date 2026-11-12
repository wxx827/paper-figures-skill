# 建筑与 BIM · Architecture & BIM  `754-763`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 754 | `fig_iso_mixing_model.py` | Three end-member mixing triangle in dual-isotope space with iso-contribution grid, posterior contribution violins and end-member sensitivity | 同位素源解析与混合模型, "三端元混合三角", "贡献比例后验", "端元敏感性" |
| 755 | `fig_iso_age_depth_model.py` | Bayesian age-depth model with 95 percent credible ribbon, calibrated date densities, one rejected outlier, sedimentation rate and cumulative dry mass | 沉积岩芯年代框架建立, "年代-深度模型", "校正年代概率分布", "异常测年点剔除", "沉积速率" |
| 756 | `fig_rs_spectral_library.py` | Field spectral signature library for five land-cover classes over 400-2500 nm with +/-1 sigma envelopes, water-vapour masks, red-edge and chlorophyll markers, broad-band means and a class separability matrix | 地物光谱库展示, 高光谱建库与波段选择, "光谱曲线, 红边, 水汽吸收带, 类间可分性" |
| 757 | `fig_rs_hsi_datacube.py` | Hyperspectral data cube drawn as a pseudo-3D block: false-colour top face, two wavelength-by-space cut faces, wavelength ruler, plus representative pixel spectra and PCA explained variance | 高光谱数据立方示意, 成像光谱仪数据结构讲解, "数据立方, 假彩色, 主成分, 波段数" |
| 758 | `fig_rs_ndvi_timeseries.py` | Annual NDVI trajectories of four parcels with cloud gaps, gap-filled reconstruction, green-up/peak/senescence callouts, phenology-date boxplots and season NDVI integrals | 植被物候时序分析, 农作物长势监测, "NDVI 时序, 云污染缺口, 物候参数, 生长积分" |
| 759 | `fig_rs_bitemporal_change.py` | Bi-temporal change detection: two co-registered dates plus a diverging dNDVI change map with change polygons, and area gain/loss bars with a class transition matrix in square kilometres | 变化检测成果展示, 城市扩张与毁林制图, "双时相, 发散色表, 变化图斑, 转移矩阵" |
| 760 | `fig_rs_unmixing_endmember.py` | Linear spectral unmixing: abundance map rendered as a barycentric colour blend with a ternary key, endmember spectra with one mixed pixel, the PCA simplex with a data convex hull, and reconstruction residuals | 光谱解混与亚像元制图, 端元提取教学, "丰度图, 端元光谱, 单形体, 重建残差" |
| 761 | `fig_rs_sar_interferogram.py` | Wrapped C-band interferogram on a cyclic colour map with low-coherence pixels greyed out, plus the unwrapped line-of-sight velocity profile along A-A' and the coherence histogram | InSAR 形变监测展示, 干涉条纹与相干性讲解, "缠绕相位, 循环色表, 相干性掩膜, 形变剖面" |
| 762 | `fig_ecat_volcano_plot.py` | Sabatier volcano of exchange current density against the dG(O*)-dG(OH*) descriptor, both limiting branches extended past the apex, thirteen labelled metal and alloy surfaces on leader lines; auxiliary OOH*/OH* scaling scatter with fit and R2, plus theoretical overpotential bars against the scaling floor. | 电催化剂描述符筛选、OER/ORR 计算催化综述, "火山图, 吸附能描述符, Sabatier 最优点, 标度关系, 理论过电位" |
| 763 | `fig_cat_free_energy_diagram.py` | Four-electron OER free-energy staircase for three catalysts at the equilibrium potential plus the ideal catalyst and one onset-potential trace, potential-determining step of each surface marked with a double-headed arrow; auxiliary elementary-step bars and an overpotential split into scaling floor and excess. | 多相/电催化机理讲解、DFT 反应路径对比, "自由能台阶图, 中间体, 决速步, 过电位分解, 标度极限" |
