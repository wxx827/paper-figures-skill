# 编译与程序分析 · Compilers & program analysis  `864-873`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 864 | `fig_tac_pitch_heatmap.py` | Single-player activity KDE on a full 105 x 68 m pitch, with marginal length/width profiles and defensive/middle/attacking third touch shares | 球员个人活动热区分析, "热力图, 核密度, 球场标线, 边缘分布, 三区触球占比" |
| 865 | `fig_tac_pass_network.py` | Directed pass network with curved volume-weighted arcs, unit lines and team centroid, plus a receptions/completion butterfly and a PageRank hub ranking | 传球网络与组织核心识别, "传球网络, 有向图, PageRank, 传球成功率, 阵型线" |
| 866 | `fig_tac_shot_map_xg.py` | Half-pitch shot map over a labelled expected-goals contour surface, with a cumulative xG versus goals step chart and per-zone conversion bars | 射门质量与期望进球评估, "射门图, xG, 等值线, 累积曲线, 转化率" |
| 867 | `fig_tac_pressing_zones.py` | 5 x 3 pressing grid colour-coded by recoveries with in-cell PPDA, a press trigger line and 5 s transition arrows, plus PPDA by period and recovery height | 逼抢强度与由守转攻分析, "逼抢, PPDA, 分区网格, 夺回球, 体能下降" |
| 868 | `fig_tac_player_radar.py` | Nine-axis percentile radar comparing two players against a 240-player league, with per-metric violin distributions and a profile similarity ranking | 球员能力画像与选材对比, "雷达图, 百分位, 小提琴图, 联赛分布, 相似度" |
| 869 | `fig_tac_match_momentum.py` | Zero-centred match momentum curve with goals, cards, substitutions and half time, plus back-to-back shots per 15 min and a rolling possession curve | 比赛进程与势头复盘, "势头曲线, 发散填充, 进球时刻, 控球率, 分时段射门" |
| 870 | `fig_rtx_dvh_curves.py` | Cumulative dose-volume histogram for a five-structure plan, with constraint pass/fail markers, Dmean/Dmax bars against planning limits, and the DVH shift of a sparing re-plan | 放疗计划评估与计划比较, "剂量体积直方图", "DVH", "危及器官约束", "V20", "D95" |
| 871 | `fig_rtx_dose_distribution.py` | Axial dose colour wash over a grey-scale slice with 95/80/50/30 % isodose lines, structure contours and beam-entry arrows, plus a lateral profile with penumbra and per-isodose conformity indices | 放疗剂量分布展示与适形度评价, "剂量分布", "等剂量线", "半影", "适形度指数", "射野方向" |
| 872 | `fig_rtx_depth_dose.py` | Percent depth dose for 6/15 MV photons, a 9 MeV electron beam, a pristine Bragg peak and a least-squares spread-out Bragg peak, with off-axis ratio profiles and output factors | 射线质对比与束流特性教学, "百分深度剂量", "PDD", "布拉格峰", "SOBP", "离轴比", "输出因子" |
| 873 | `fig_rtx_gamma_analysis.py` | Two-dimensional gamma index map with a cool pass / warm fail colour break, measured-versus-calculated isodose overlay, cumulative gamma distribution and pass rates under four acceptance criteria | 放疗剂量验证与质控报告, "伽马分析", "gamma index", "通过率", "3%/3mm", "患者特异性QA" |
