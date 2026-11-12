# 水文与海洋 · Hydrology & oceanography  `213-222`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 213 | `fig_hydrograph.py` | Storm hydrograph with an inverted hyetograph hanging from the top axis, two-tone baseflow separation fill, flood-peak call-out and a dimensioned basin-lag arrow between the rainfall centroid and the peak | 降雨-径流响应过程、洪水预报与产流分析, "流量过程线, 降雨倒挂图, 基流分割" |
| 214 | `fig_flow_duration.py` | Semi-log flow duration curves for four gauges with high/mid/low flow bands, Q10-Q50-Q90 markers on a reference gauge, plus a 58 %-height card comparing the same characteristic flows across gauges | 径流特性对比、水资源可利用量与调节能力评价, "流量历时曲线, 特征流量, 保证率" |
| 215 | `fig_flood_frequency.py` | Flood frequency curve on a logarithmic return-period axis with a mirrored exceedance-probability axis, Gumbel and Log-Pearson III fits, a 90 % confidence band and a small fitted-vs-observed goodness-of-fit card | 设计洪水计算、重现期与工程标准论证, "洪水频率曲线, 重现期, 置信带" |
| 216 | `fig_ocean_currents.py` | Surface current map over a masked coastline and island: speed shading, speed-weighted streamlines, four dashed mesoscale eddies, a shelf-break jet axis, bar scale and north arrow | 海域环流态势、涡旋识别与流场展示, "海流矢量场, 中尺度涡, 流线图" |
| 217 | `fig_wave_rose_spectrum.py` | Wave rose of height-by-direction occurrence beside a polar frequency-direction spectrum with separated swell and wind-sea peaks, over a short one-dimensional spectrum panel | 波浪气候统计、涌浪与风浪分离, "波浪玫瑰图, 方向谱, 频谱曲线" |
| 218 | `fig_tidal_harmonics.py` | Observed against harmonically predicted water level with dimensioned spring and neap ranges, a shared-axis residual panel highlighting a storm surge, and a small amplitude-and-phase panel with a fit summary card | 潮汐调和分析、风暴增水与潮位预报检验, "调和分析, 分潮振幅相位, 余水位" |
| 219 | `fig_ocean_section.py` | Three-row hydrographic section (temperature, salinity, dissolved oxygen) sharing distance and depth ranges, with labelled isolines, station triangles drawn once and an identical grey sea-bed mask | 断面观测数据展示、水团与缺氧带分析, "海洋断面图, 温盐氧剖面, 等值线填色" |
| 220 | `fig_ts_diagram.py` | Temperature-salinity scatter coloured by sample depth over labelled sigma-t isopycnals, with four covariance-fitted dashed water-mass envelopes and a dashed mean mixing path | 水团判别、混合过程与密度结构分析, "温盐图, 等密度线, 水团划分" |
| 221 | `fig_groundwater_contour.py` | Water-table map with 1 m shading, labelled 2 m equipotentials, unit hydraulic-gradient arrows normal to the contours, a pumping cone, monitoring-well triangles and a small hydrograph card | 地下水流场分析、降落漏斗与井网监测, "地下水位等值线, 流向箭头, 监测井" |
| 222 | `fig_water_quality_panel.py` | Two-by-three water-quality dashboard sharing one layout: monthly mean, min-max envelope, dashed regulatory limit (minimum, band or maximum) and rust exceedance wash, with one shared legend | 水质达标评价、多参数月度监测汇报, "水质多参数面板, 标准限值, 超标高亮" |
