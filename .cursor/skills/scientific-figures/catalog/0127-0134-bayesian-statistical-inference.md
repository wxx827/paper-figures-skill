# 贝叶斯与统计推断 · Bayesian & statistical inference  `127-134`

8 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 127 | `fig_trace_posterior.py` | ArviZ-style MCMC diagnostics: per-chain marginal posterior KDE with 94% HDI on the left, multi-chain trace on the right, split-Rhat and bulk-ESS in a side column, one row per parameter | 贝叶斯模型收敛诊断、MCMC 采样质量报告, "轨迹图, 后验分布, 收敛诊断, R-hat" |
| 128 | `fig_posterior_pairs.py` | Posterior pair plot: hexbin joint density with 50%/95% credible contours and joint mode in the lower triangle, marginal posterior with HDI shading on the diagonal, correlation value on a diverging background in the upper triangle | 检查后验参数相关性与后验几何形状（漏斗、脊）, "配对图, 联合后验, 参数相关性" |
| 129 | `fig_prior_posterior.py` | 2x3 prior-to-posterior update panels: dashed pale prior against solid filled posterior, both means with 95% intervals, shift arrow, plus KL(posterior‖prior) in bits and interval shrinkage per parameter | 展示数据带来的信息增益、先验敏感性分析, "先验后验对比, 信息增益, KL散度" |
| 130 | `fig_ppc_check.py` | Posterior predictive check: observed density against replicate densities with 50%/95% envelopes, a dedicated tail-zoom panel, and four test-statistic panels (mean, sd, min, max) with observed value and Bayesian p-value | 贝叶斯模型拟合优度检验、发现尾部/离散度失配, "后验预测检验, PPC, 贝叶斯p值" |
| 131 | `fig_hdi_forest.py` | HDI forest plot for a hierarchical model: ridgeline posterior per row, thick 50% HDI, thin 95% HDI, median dot, zero reference line, colour-coded by whether the HDI excludes zero, with a numeric column for median / HDI / Pr(>0) | 分层模型分组效应汇总、多研究/多站点效应对比, "森林图, HDI, 分层模型, 区间估计" |
| 132 | `fig_bootstrap_dist.py` | 2x3 nonparametric bootstrap panels: replicate histogram plus KDE, normal reference curve, original-sample estimate, shaded percentile 95% CI with labelled endpoints, bias and bootstrap SE per statistic | 非参数区间估计、统计量抽样分布与偏差评估, "Bootstrap, 重抽样, 置信区间, 标准误" |
| 133 | `fig_power_analysis.py` | Power analysis: family of noncentral-t power curves against log sample size with the 80% target and crossing points, required-n versus effect-size curve on a log axis, and three alpha-sensitivity panels | 实验设计样本量规划、审稿人要求的功效说明, "功效分析, 样本量估计, 效应量" |
| 134 | `fig_causal_dag.py` | Causal DAG drawn with patches and FancyArrowPatch: highlighted treatment and outcome nodes, dashed unobserved confounder, red backdoor paths, teal causal paths, collider warning, plus a side panel listing backdoor paths, the minimal adjustment set and the d-separation statement | 因果识别策略说明、混杂控制与调整集论证, "因果图, DAG, 后门路径, 调整集" |
