# 半导体 · Semiconductors  `514-523`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 514 | `fig_bio_balance_cop.py` | Plantar pressure map on a 5 mm sensor grid with the time-coloured centre-of-pressure path, a 95 % sway ellipse with area and path statistics, and the sway-velocity spectrum with median power frequency | 平衡功能评估、足底压力、姿势控制研究, "压力中心" "COP" "足底压力" "平衡" |
| 515 | `fig_bio_motion_capture.py` | 3D motion-capture stick figures at five instants of one stride, time-ramped with ground grid and cast shadows, beside the lower-limb and upper-body joint angle curves that pose them | 动作捕捉、三维步态、运动学可视化, "动作捕捉" "骨架" "三维姿态" |
| 516 | `fig_drug_virtual_screening.py` | Virtual screening cascade funnel from a 2.1 M compound library to confirmed hits, with ROC-style enrichment curves for four scoring functions and their EF@1% / EF@5% factors | 虚拟筛选流程与打分函数富集能力对比, "虚拟筛选" "富集因子" "打分函数" "命中率" |
| 517 | `fig_drug_sar_landscape.py` | Structure-activity landscape: 2D chemical-space projection of 220 analogues coloured by pIC50, activity cliffs drawn between near-identical pairs, potency histogram and top SALI pairs | 先导优化中的构效关系与活性悬崖分析, "构效关系" "活性悬崖" "化学空间" "SALI" |
| 518 | `fig_drug_admet_radar.py` | ADMET desirability radar over six axes with the target band shaded, per-property score bars for three candidates plus a reference drug, and a Lipinski rule-of-five verdict table | 候选化合物成药性与 ADMET 综合评估, "ADMET" "雷达图" "类药五规则" "成药性" |
| 519 | `fig_drug_pk_profile.py` | Oral pharmacokinetics: semi-log plasma concentration-time curves for three dose groups with two-compartment fits and Cmax / Tmax / T1/2 / AUC call-outs, q12h accumulation to steady state, dose-proportionality check and an exposure metric table | 临床前药代动力学与给药方案支持, "药代动力学" "血药浓度" "半衰期" "稳态累积" |
| 520 | `fig_drug_docking_pose.py` | Docking pose schematic: hand-drawn binding-pocket cross-section with labelled residues, skeletal ligand, hydrogen bonds with distances and hydrophobic contact arcs, plus stacked score decomposition and pose metrics | 分子对接结果展示与结合模式解读, "分子对接" "结合口袋" "氢键" "打分函数" |
| 521 | `fig_drug_pipeline_funnel.py` | Development pipeline funnel where band width tracks live programmes and band height tracks stage success, with per-stage cycle time and cost on twin axes, indication donut, and cumulative probability of approval | 研发管线组合管理与阶段淘汰分析, "研发管线" "漏斗图" "成功率" "适应症分布" |
| 522 | `fig_prot_ramachandran.py` | Ramachandran plot of backbone dihedrals: phi-psi scatter over favoured / allowed density envelopes, glycine marked separately, the beta, alpha-R and alpha-L basins annotated in place, and a companion bar panel of basin occupancy | 蛋白质骨架二面角质量检查、结构验证报告主图, "拉氏图, 二面角, 允许区, 甘氨酸, 二级结构" |
| 523 | `fig_prot_structure_align.py` | Structure superposition: two synthetic C-alpha backbones overlaid in 3-D with the model coloured by local deviation, aligned core and divergent loop called out, plus companion per-residue deviation and per-element mean-deviation panels on one shared scale | 预测模型与实验结构叠合、AlphaFold 精度评估, "结构叠合, RMSD, 主链骨架, 三维结构, 残基偏差" |
