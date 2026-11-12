# 气候与大气 · Climate & atmosphere  `79-86`

8 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 79 | `fig_hovmoller.py` | Two time-latitude anomaly rasters (diverging, zero-centred) with a latitude-mean time series on top, a time-mean meridional profile on the right and one shared colorbar | 气候场的时空演变、遥相关传播、纬向异常带迁移, "霍夫默勒图, 时间-纬度剖面, 异常场" |
| 80 | `fig_wavelet_power.py` | Morlet wavelet power spectrum on a log2 period axis with 95 % red-noise significance contours, hatched cone of influence, source time series above and global wavelet spectrum on the right | 周期性诊断、非平稳信号的时频分析、ENSO/太阳周期识别, "小波功率谱, 时频分析, 显著性检验" |
| 81 | `fig_anomaly_bands.py` | Three stacked anomaly series sharing one x-axis, red/blue fills against a fixed baseline, 13-month running means and ENSO warm/cold episodes as background bands | 多指标异常联动、事件年份标注、耦合变量对比, "距平带状图, 正负填色, 事件色带" |
| 82 | `fig_climate_stripes.py` | Two regional rows, each pairing an Ed Hawkins stripe band with the annual anomaly series, a smoothing curve, long-term and post-1975 trend lines and a colour-matched distribution histogram | 增暖故事讲述、区域放大对比、科普与论文兼用图, "气候条纹, 增暖条带, 距平分布" |
| 83 | `fig_windrose_grid.py` | 2x3 polar grid of stacked wind roses (16 sectors x 7 speed classes) with a shared radial scale, per-panel calm frequency in the donut hole and one speed-class legend | 风资源评估、季节性风向对比、多站点风况, "风玫瑰图, 极坐标堆叠, 风向频率" |
| 84 | `fig_vertical_profile.py` | Temperature, humidity and wind profiles at four local times on a shared inverted log-pressure axis, with +-1 sigma envelopes, annotated inversion layers, PBL top and low-level jet, plus a mirrored height axis | 探空廓线、边界层日变化、垂直结构诊断, "垂直廓线, 气压对数轴, 逆温层" |
| 85 | `fig_ensemble_projection.py` | Historical ensemble and observations followed by three emission scenarios drawn as ensemble means with 25-75 % and 5-95 % fans, warming-threshold guides, crossing-year table and end-of-century violins | 情景预估、不确定性传播、升温阈值评估, "集合预估, 分位数扇形, 情景对比" |
| 86 | `fig_flux_diurnal.py` | 2x3 grid of mean diurnal flux cycles by month with day-to-day interquartile bands, a zero line separating uptake from release, night-time shading and markers on the daily extremum | 通量观测、日变化季节演变、碳收支诊断, "日变化曲线, 四分位带, 昼夜背景" |
