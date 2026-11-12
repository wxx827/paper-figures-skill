# 环境毒理 · Environmental toxicology  `664-673`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 664 | `fig_spo_vo2max_test.py` | Three stacked panels sharing one ramp-test time axis (VO2 and VCO2 with the plateau, ventilation and RER, heart rate and lactate) with VT1 and VT2 ruled through all of them, beside the V-slope scatter that fits the VT1 breakpoint | 最大摄氧量测试、心肺运动试验、通气阈判定, "VO2max, 递增负荷, 通气阈 VT1 VT2, V-slope, 血乳酸" |
| 665 | `fig_spo_training_zones.py` | Five colour-banded training zones on a percent-HRmax axis carrying their physiology, the athlete annual hours per band and a cumulative-share step line, beside the fat / carbohydrate / protein substrate mix and the periodised zone allocation | 训练强度分区、周期化训练计划、耐力训练监控, "训练分区, 极化训练, 心率区间, 底物利用, 周期化" |
| 666 | `fig_dt_architecture.py` | Digital twin architecture: mirrored physical/virtual columns, bidirectional sync channel with rate and latency, four enabling layers, sync latency distribution | 数字孪生系统架构总览, 工业设备虚实映射, "架构图, 双向数据流, 同步延迟" |
| 667 | `fig_dt_model_calibration.py` | Twin calibration: simulated vs measured time series before and after calibration, parameter prior/posterior densities, residual against operating load | 仿真模型标定与验证, 数字孪生精度评估, "模型标定, 后验分布, RMSE 下降" |
| 668 | `fig_dt_predictive_maintenance.py` | Predictive maintenance: fleet health-index degradation with widening forecast fan and RUL bracket, RUL error convergence, maintenance policy cost comparison | 设备健康管理与剩余寿命预测, 维修策略经济性对比, "退化曲线, 置信扇形, RUL, 失效阈值" |
| 669 | `fig_iot_sensor_network.py` | IoT sensor deployment on a plant floor plan: marker-coded sensor classes, coverage discs, gateway routing tree coloured by RSSI, battery vs report rate, hop distribution | 工业物联网传感器布点, 无线网络覆盖与链路质量, "厂区平面图, 信号强度, 网关, 跳数分布" |
| 670 | `fig_iot_production_takt.py` | Production takt: per-station cycle time split into processing/waiting/transfer against the takt line with the bottleneck banded, OEE factors, output per shift vs capacity | 生产线节拍与瓶颈分析, 产线平衡与产能差距, "节拍时间, 瓶颈工位, OEE, 理论产能" |
| 671 | `fig_iot_alarm_management.py` | Alarm management: alarms per 10 min stacked by priority with EEMUA reference rates and flood episodes banded, tag Pareto with cumulative share, acknowledgement time distribution | 报警泛滥分析与报警优化, 操作员响应绩效, "报警管理, EEMUA, 帕累托, 响应时间" |
| 672 | `fig_bch_transaction_graph.py` | Address-level transaction graph with a traced laundering chain, degree CCDF and Lorenz curve | 区块链取证/反洗钱汇报, "交易图谱" "资金流追踪" "地址聚类" "幂律度分布" "洛伦兹曲线" |
| 673 | `fig_bch_consensus_timeline.py` | BFT consensus swim-lane timeline with a failed split-vote round, latency histogram and fork rate | 共识协议讲解/分布式系统汇报, "共识时间线" "泳道图" "分叉" "出块延迟" "拜占庭容错" |
