# 地理空间与制图 · Geospatial & cartography  `55-62`

8 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 55 | `fig_choropleth_grid.py` | 3x2 choropleth small multiples over Lloyd-relaxed Voronoi regions, one panel per year, shared horizontal colorbar, white region borders, per-panel inset histogram + mean badge | 分区指标的多年份对比、行政区面板地图, "分区着色地图, 多年份小多图, 区域指标" |
| 56 | `fig_bivariate_map.py` | Bivariate choropleth: two indicators split into terciles and crossed into a 3x3 Stevens colour matrix, with legend matrix, tercile-gridded scatter and two univariate mini-maps | 两个空间变量的耦合关系展示（如城市化 x 热暴露）, "双变量地图, 3x3色块图例, 双变量配色" |
| 57 | `fig_hexgrid_map.py` | Hexagonal binning of point events clipped to a coastline, with vertical colorbar flush against a rotated cell-count histogram and a Lorenz-style spatial concentration curve | 点事件密度聚合、观测样点热点分析, "六边形格网, 点密度聚合, 蜂窝地图" |
| 58 | `fig_map_timeseries_combo.py` | Large regional choropleth linked by colour-matched dashed leader lines to four stacked time-series panels with confidence bands and decadal trend labels | 地图定位 + 典型区域时间序列的组合版式, "地图加时序组合, 引导线关联, 区域轨迹" |
| 59 | `fig_flow_map.py` | Origin-destination flow map: gravity-model Bezier arcs with width by volume and hue by direction, node size by throughput, plus a diverging net-balance bar panel | 人口迁徙、贸易/物流流向、区域间联系强度, "流向地图, 迁徙弧线, OD流量图" |
| 60 | `fig_terrain_hillshade.py` | Synthetic fractal DEM rendered as soft-blended hillshade (315/45 deg) with translucent hypsometric ramp, 200 m contours, A-A' profile inset, slope map and hypsometric curve | 地形晕渲、DEM 高程与坡度展示, "地形晕渲图, 山体阴影, 高程剖面" |
| 61 | `fig_cluster_map_grid.py` | 3x2 grid of spatial clustering solutions (k-means at several k, Ward linkage, Gaussian mixture) with shared discrete palette, silhouette badges and 100% stacked cluster-share bars | 空间分区/类型划分方法与 k 值敏感性对比, "空间聚类分区, k值对比, 聚类占比" |
| 62 | `fig_change_detection_map.py` | Before / after / difference triptych with a shared sequential colorbar, symmetric diverging change map, significance contours, change histogram and gain-stable-loss area shares | 两期遥感/土地覆被变化检测与增减统计, "变化检测图, 前后对比, 差值发散色带" |
