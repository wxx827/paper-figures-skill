# 机器学习与模型解释 · Machine learning & interpretability  `63-70`

8 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 63 | `fig_shap_interaction.py` | 2x3 SHAP dependence scatter grid; x = feature value, y = SHAP value, points coloured by the interacting feature, LOWESS main-effect curve with CI band, auto-placed interaction-strength box, shared Low-High colourbar | 模型可解释性、特征交互效应、SHAP 依赖分析, "SHAP交互图" |
| 64 | `fig_pdp_ice_grid.py` | 3x2 PDP + ICE grid; translucent per-individual ICE curves, +/-1 SD ribbon, 10th/90th ICE percentile envelope, thick average PDP curve, and a marginal feature-distribution strip with decile rules under each panel | 部分依赖分析、个体异质性、黑箱模型解释, "PDP-ICE图" |
| 65 | `fig_confusion_grid.py` | 2x3 grid of row-normalised confusion matrices; each cell shows percentage over raw count, diagonal cells outlined, per-panel Accuracy/Macro-F1 in the title, one shared colourbar | 多模型分类效果对比、错分结构诊断, "混淆矩阵组图" |
| 66 | `fig_calibration_curves.py` | Reliability diagram with binomial error bars and the ideal diagonal, a stacked predicted-probability histogram on the shared x-axis, and a Brier-score bar panel truncated at the oracle floor with ECE annotations | 概率校准评估、风险预测模型验证, "校准曲线图" |
| 67 | `fig_feature_ablation.py` | Cumulative feature-removal curves under three removal orders with 95% CI ribbons, signed leave-one-out contribution bars with SD error bars, and a model-family x top-k-subset AUC heatmap with the best subset outlined | 特征消融实验、变量重要性验证、精简模型选择, "消融研究图" |
| 68 | `fig_hyperparam_heatmap.py` | 2x3 hyper-parameter search landscape; each panel pairs an interpolated AUC heatmap with white iso-AUC contours, a star at the slice optimum, and top/right marginal best-score profiles, all on one shared colourbar | 超参数调优、网格/随机搜索可视化, "超参数热图" |
| 69 | `fig_embedding_scatter.py` | t-SNE/UMAP style embedding; class-coloured scatter with core convex hulls and haloed centroid labels, three perplexity-comparison thumbnails inset in the free lower-left corner, plus a composition bar panel that doubles as the colour key | 降维可视化、聚类结构展示、单细胞/表征分析, "嵌入散点图" |
| 70 | `fig_model_benchmark_panel.py` | Four-in-one benchmark board: multi-metric radar, per-dataset rank heatmap with average rank, training-cost vs accuracy bubble scatter with Pareto frontier, and headline-AUROC bars with 95% CI and per-seed dots | 模型综合评测、论文主图、一图看懂全部对比, "模型对比面板" |
