# 世界模型 · 效率与系统 Efficiency & systems  `1189-1194`

6 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 1189 | `fig_wm_sys_latency_breakdown.py` | Inference latency broken down by stage, three serving configs |  |
| 1190 | `fig_wm_sys_memory_footprint.py` | Device memory against context length, and where each config dies |  |
| 1191 | `fig_wm_sys_parallelism.py` | Four ways to cut one model across four GPUs |  |
| 1192 | `fig_wm_sys_quantization.py` | Quantisation: rollout throughput against rollout quality |  |
| 1193 | `fig_wm_sys_roofline.py` | Roofline of one world-model inference step |  |
| 1194 | `fig_wm_sys_batch_schedule.py` | Continuous batching: one timeline, and the sweep it came from |  |
