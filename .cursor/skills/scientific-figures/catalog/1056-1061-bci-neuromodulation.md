# 脑机接口与神经调控 · BCI & neuromodulation  `1056-1061`

6 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 1056 | `fig_bci_dbs_therapeutic_window.py` | Deep brain stimulation programming map over the pulse-width x amplitude plane: symptom improvement as a truncated warm field, side-effect iso-lines of the neighbouring capsule in house blue, the therapeutic window hatched and outlined, and a callout on the cell that maximises benefit minus risk, beside the two Lapicque strength-duration curves with the window ratio on a twin axis and a rate sweep of benefit, risk and mean power | 深部脑刺激参数编程、治疗窗与副作用边界讲解、脉宽-幅度权衡, "深部脑刺激, 治疗窗, 强度-时间曲线, 时值, 脉宽, 激活组织体积, 副作用阈值" |
| 1057 | `fig_bci_tes_field_focality.py` | Transcranial electric stimulation on a true-scale axial head slice: the field magnitude of a 4x1 HD ring painted inside the brain compartment with equipotential lines, the half-maximum focality contour and the scalp pads drawn where current enters and leaves, beside the radial decay of three montages and a focality-versus-target-intensity trade-off swept over the ring half-angle | 经颅电刺激建模、电极构型对比、靶区聚焦度与强度权衡, "经颅电刺激, tDCS, 电场分布, 头模型, 聚焦度, 高精度环形电极, 等势线" |
| 1058 | `fig_bci_closed_loop_beta_trigger.py` | Closed-loop stimulation run drawn as three offset rows on one time base: local field potential, the causal beta envelope the controller sees against its trigger and release thresholds, and the delivered current, with every epoch shaded through all three rows, beside the onset-aligned envelope against a matched open-loop average and a threshold sweep of duty cycle versus suppression | 闭环/自适应神经刺激讲解、生物标志物触发时序、占空比与抑制效果权衡, "闭环刺激, 自适应DBS, beta爆发, 触发阈值, 迟滞, 占空比, 事件对齐平均" |
| 1059 | `fig_bci_ssvep_frequency_id.py` | SSVEP: four flicker frequencies, one occipital electrode, one decision |  |
| 1060 | `fig_bci_p300_speller_matrix.py` | P300 speller: from single flashes to one letter on the screen |  |
| 1061 | `fig_bci_motor_imagery_erd.py` | Motor-imagery ERD: mu-band envelope, time-frequency map, and laterality |  |
