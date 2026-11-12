# 边缘计算 · Edge computing  `894-903`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 894 | `fig_edm_irt_icc_curves.py` | 3PL item characteristic curves with a/b/c leader lines, item-bank parameter map and item information | 教育测量与心理测量学, 题目质量分析, "项目特征曲线" "IRT" "区分度" |
| 895 | `fig_edm_wright_map.py` | Wright map: examinee ability and item difficulty on one shared logit ruler, with targeting gap and Rasch separation | 教育测量与心理测量学, 试卷与人群匹配诊断, "怀特图" "Rasch" "分离信度" |
| 896 | `fig_edm_test_information.py` | Test information as a stack of item contributions with the SEM = 1/sqrt(I) twin axis, plus long / short / adaptive form comparison against a target curve | 教育测量与心理测量学, 测验精度与组卷设计, "测验信息函数" "测量标准误" "自适应测验" |
| 897 | `fig_edm_learning_curve.py` | Per-learner power-law learning curves with group means, 95% CI bands, mastery-threshold crossings, the right-skewed cost distribution and rate-against-start scatter | 学习分析与智能导学, 技能掌握进度追踪, "学习曲线" "掌握度" "练习次数" |
| 898 | `fig_edm_knowledge_graph_mastery.py` | Layered prerequisite DAG with mastery-coded nodes, practice-count sizing, unlocked-concept rings and a recommended study path, plus mastery ranking and accumulation | 学习分析与智能导学, 知识图谱与学习路径推荐, "知识图谱" "掌握度" "先修关系" |
| 899 | `fig_edm_dif_analysis.py` | Group-specific ICC pairs for uniform and non-uniform DIF with integrated difference areas, an ETS-banded effect-size forest and a Mantel-Haenszel funnel plot | 教育测量与心理测量学, 试题公平性与偏差检验, "项目功能差异" "DIF" "Mantel-Haenszel" |
| 900 | `fig_fsh_stock_assessment.py` | Kobe phase plane with a 95% terminal ellipse, biomass and F series against reference points, and the surplus production curve | 渔业资源评估与管理汇报, "Kobe图, 资源状态, 生物量, 捕捞死亡率, MSY, 剩余产量" |
| 901 | `fig_fsh_length_frequency.py` | Monthly length-frequency ridgeline with age components and tracked modes, a von Bertalanffy fit to those modes, and log-log length-weight allometry | 渔业生物学与年龄生长分析, "体长频率, 年龄组, 世代追踪, von Bertalanffy, 渐近体长, 体长体重关系, 异速生长" |
| 902 | `fig_fsh_cpue_standardization.py` | Nominal versus GLM-standardised CPUE with 95% contrast bands and effort bars, factor effect sizes with sequential deviance, and a residual QQ plot | 渔获努力量标准化与资源指数构建, "CPUE标准化, 名义CPUE, GLM, 努力量, 因子效应, 偏差解释率, 残差QQ图" |
| 903 | `fig_fsh_trophic_network.py` | Food web laid out on a solved trophic-level axis with biomass-scaled nodes and flow-scaled edges, a biomass pyramid and level-to-level transfer efficiency | 生态系统食物网与营养级分析, "食物网, 营养级, 能量流, 生物量金字塔, 传递效率, 营养级联, 关键种" |
