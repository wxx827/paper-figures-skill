# 时间序列预测 · Time series & forecasting  `354-363`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 354 | `fig_ts_decomposition.py` | STL decomposition of a monthly series into trend, season and remainder | 时间序列预测 |
| 355 | `fig_ts_arima_diagnostics.py` | Four-panel residual diagnostic sheet for a fitted ARIMA(1,1,1) model | 时间序列预测 |
| 356 | `fig_ts_forecast_fan.py` | Probabilistic forecast fan with an interval-growth companion panel | 时间序列预测 |
| 357 | `fig_ts_rolling_backtest.py` | Rolling-origin backtest: five forecast folds laid over the realised series | 时间序列预测 |
| 358 | `fig_ts_anomaly_detection.py` | Anomaly detection read as three stacked, x-aligned views of one series | 时间序列预测 |
| 359 | `fig_ts_changepoint.py` | Offline change-point detection by greedy binary segmentation | 时间序列预测 |
| 360 | `fig_ts_seasonal_subseries.py` | Seasonal subseries plot: twelve month-groups on one shared y axis | 时间序列预测 |
| 361 | `fig_ts_acf_grid.py` | ACF / PACF identification grid for three candidate processes | 时间序列预测 |
| 362 | `fig_ts_horizon_error.py` | How forecast error grows with the horizon, for four competing models | 时间序列预测 |
| 363 | `fig_ts_ensemble_forecast.py` | Forecast combination: six base learners, inverse-error weights, holdout gain | 时间序列预测, "集成预测" |
