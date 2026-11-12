# 虚拟现实 · Virtual reality  `814-823`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 814 | `fig_seis_avo_gather.py` | AVO analysis: wiggle variable-area angle gather with picked target amplitudes and an inset sin-squared regression, an intercept-gradient crossplot with fluid quadrants and background trend, and the four AVO class templates | 含气性检测、AVO属性分析, "角道集, 振幅随偏移距变化, 截距梯度交会, 流体因子, III类AVO" |
| 815 | `fig_seis_velocity_model.py` | Layer velocity model with Snell-traced ray fan and an inch-pinned colour bar, a 1D interval/RMS/average velocity column, and a before-after migration pair showing the fault-tip diffraction collapse | 速度建模、偏移成像效果对比, "层速度模型, 射线追踪, 均方根速度, 偏移前后对比, 绕射收敛" |
| 816 | `fig_dnt_periodontal_chart.py` | Full-mouth periodontal chart: unrolled FDI arches with six probing sites per tooth, recession line and attachment-loss band, plus site-depth histogram and per-quadrant CAL bars | 牙周检查记录, 探诊深度与附着丧失, "牙周图, 探诊深度, 附着丧失, 牙位图" |
| 817 | `fig_dnt_caries_prevalence.py` | Caries epidemiology: DMFT stacked into decayed/missing/filled by age group for two cohorts with total lines, plus prevalence curves with 95% bands and a risk-factor odds-ratio forest | 龋病流行病学, 队列对比, "DMFT指数, 患龋率, 森林图, 比值比" |
| 818 | `fig_dnt_implant_stress.py` | Dental implant stress analysis: 2-D von Mises field across the bone-implant section with cortical/cancellous layering, plus interface stress profile and peak stress vs load angle and diameter | 种植体生物力学, 有限元应力云图, "von Mises应力, 皮质骨, 骨结合, 应力集中" |
| 819 | `fig_dnt_ortho_cephalometric.py` | Orthodontic cephalometric tracing: lateral skull outline with landmarks and SNA/SNB/ANB/FMA arcs, pre- and post-treatment overlay, plus normal-band comparison and treatment change bars | 正畸头影测量分析, 治疗前后对比, "头影测量, 标志点, SNA, ANB, 正常值范围" |
| 820 | `fig_dnt_occlusal_force_map.py` | Occlusal force distribution: maxillary arch contact map with size-and-colour coded pressure, centre-of-force cross and midline offset, plus load share bars and a contact timing gantt | 咬合力分布分析, 咬合平衡评估, "咬合接触点, 压力热图, 合力中心, 接触时序" |
| 821 | `fig_dnt_bite_cycle_emg.py` | Masticatory EMG: four muscle channels on one shared microvolt scale over chewing cycles with jaw opening trace and phase shading, plus symmetry index bars and cycle variability scatter | 咀嚼肌电分析, 咬合周期, "表面肌电, 咬肌, 颞肌, 对称指数, 开闭口相" |
| 822 | `fig_mph_landmark_gpa.py` | Generalized Procrustes superimposition: aligned landmark cloud with per-landmark 95% ellipses, consensus outline, before/after Procrustes-distance shift and per-landmark variance bars | 几何形态测量教学、标本形状配准结果展示, "普氏叠加, 地标点, 形状变异, 置信椭圆" |
| 823 | `fig_mph_shape_pca_warp.py` | Shape-space PCA with convex hulls and group centroids, four thin-plate-spline warp insets at the PC extremes, scree plot and PC1 deformation series | 形态空间分析、形变网格演示、组间形状差异汇报, "形态空间, PCA, 薄板样条, 变形网格, 碎石图" |
