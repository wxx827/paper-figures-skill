# 综合版式与图形摘要 · Composite layouts & graphical abstracts  `87-94`

8 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 87 | `fig_abstract_workflow.py` | Graphical-abstract layout: five rounded pipeline boxes (Data to Output) linked by arrows, each holding a live thumbnail chart, above three key-result panels (grouped bars, trajectories with CI, importance bars) | 论文图形摘要、项目开篇总览, "图形摘要" "工作流" "流程图+结果" |
| 88 | `fig_composite_overview.py` | Irregular grid study overview: large spatial field with station bubbles and inset histogram spanning two rows, two compact time-series panels, ranked bar chart, attribution donut, and a full-width project timeline ribbon | 研究总览大图、结题汇报首图, "综述总览" "不规则网格" "时间轴" |
| 89 | `fig_conceptual_diagram.py` | Layered mechanism schematic drawn purely with patches: atmosphere / canopy / organic soil / mineral soil / bedrock bands with flux arrows whose shaft width encodes magnitude, plus flux bars, a carbon budget and a width key | 机制示意图、概念模型说明, "概念图" "通量箭头" "分层示意" |
| 90 | `fig_multipanel_story.py` | Narrative 2x3 grid of six heterogeneous panel types (trend with CI, gridded map, grouped boxes with significance, correlation scatter, causal-loop schematic, radar) unified by one palette and a takeaway line under each heading | 讲故事式多面板、Figure 1, "多面板叙事" "异构组图" "统一配色" |
| 91 | `fig_method_pipeline.py` | Five-stage horizontal method pipeline with geometric stage icons, per-stage input/output shape and parameter cards, chevron arrows, and a Sankey-style ribbon tracking the data-volume reduction underneath | 方法流程图、系统架构说明, "方法管线" "流程+参数" "桑基带状图" |
| 92 | `fig_results_dashboard.py` | Academic BI dashboard: four KPI cards with big numbers, delta chips and sparklines, an annotated diverging skill heatmap beside a directly-labelled lead-time decay plot, and a full-width stacked bar with a twin-axis overlay | 结果仪表盘、基准评测汇总, "结果看板" "KPI卡片" "高信息密度" |
| 93 | `fig_comparison_matrix.py` | Method-by-criterion matrix where every cell is a mini chart (fold bars, kernel density, calibration scatter, bullet gauge, scaling curve, rating dots), with a criterion-weight strip on top and weighted total bars on the right | 方法对比矩阵、消融/基线对照, "对比矩阵" "格内小图" "加权总分" |
| 94 | `fig_hero_figure.py` | Cover-grade composite: dominant regime map with critical boundary, site trajectories and a zoom callout, supported by a ridgeline, a directional rose, an effect-size forest plot, site time series and a driver decomposition, linked by a dashed cross-panel leader | 门面大图、封面/Figure 1, "门面图" "主次层次" "跨面板引导线" |
