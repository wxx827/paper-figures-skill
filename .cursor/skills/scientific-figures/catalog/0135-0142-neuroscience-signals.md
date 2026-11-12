# 神经科学与信号 · Neuroscience & signals  `135-142`

8 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 135 | `fig_raster_psth.py` | Two-neuron 2x2 grid: spike raster with condition colour strips over an aligned PSTH showing mean firing rate with SEM band and a shaded stimulus epoch | Single-unit electrophysiology, event-locked spiking, condition comparison, "光栅图" "PSTH" "峰电位" "发放率" |
| 136 | `fig_erp_topomap.py` | Grand-average ERP waveforms for three conditions with labelled N1/P2/N2/P300 components and significance bars, above a row of five scalp topographies sharing one colorbar | EEG/MEG event-related potentials, component latency and scalp distribution, "ERP" "脑电" "地形图" "诱发电位" |
| 137 | `fig_spectrogram_panel.py` | Four-part time-frequency composite: raw LFP trace, log-frequency spectrogram with event markers, marginal band-shaded PSD profile, and delta-to-gamma band-power time courses | Oscillatory dynamics, event-related spectral perturbation, LFP/EEG time-frequency analysis, "时频图" "频谱图" "脑电节律" |
| 138 | `fig_connectivity_matrix.py` | Module-sorted connectivity matrix with network colour bands and diverging scale, paired with a circular connectogram whose chord width encodes edge strength | Functional/structural brain networks, resting-state modularity, correlation networks, "脑连接" "连接矩阵" "环形图" "connectome" |
| 139 | `fig_multichannel_traces.py` | 32-channel EEG waterfall with per-channel labels, highlighted epileptiform and alpha channels, event lines, and an L-shaped microvolt/second scale bar replacing the axes | Raw multichannel recordings, clinical EEG review, LFP/ECoG traces, "多通道波形" "脑电导联" "比例尺" |
| 140 | `fig_tuning_curves.py` | 2x3 polar tuning grid: per-neuron mean firing rate with SEM error bars, von Mises fit, preferred-direction arrow, and DSI/OSI/HWHM annotations | Orientation and direction selectivity, sensory neuron characterisation, receptive-field tuning, "调谐曲线" "方向选择性" "极坐标" |
| 141 | `fig_filter_response.py` | 2x2 digital filter characterisation across three elliptic orders: magnitude with ripple inset, group delay with passband inset, pole-zero plot on the unit circle, impulse and step responses | Filter design review, signal-preprocessing documentation, DSP teaching material, "滤波器" "频率响应" "零极点" "群延迟" |
| 142 | `fig_neural_decoding.py` | Four-panel decoding report: accuracy time course with chance line and significance bar, temporal generalisation matrix, annotated confusion matrix, and per-ROI accuracy bars with significance stars | MVPA/decoding results, classifier evaluation over time, region-wise information content, "神经解码" "时间泛化" "混淆矩阵" |
