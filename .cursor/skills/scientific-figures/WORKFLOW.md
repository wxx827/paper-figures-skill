# WORKFLOW —— 用这个 skill 做出顶刊级图

本文是 **操作手册**。`SKILL.md` 回答「有哪些图」，本文回答「怎么用它们把活干完」。

```
scientific-figures/
├─ SKILL.md            # 入口：怎么检索、怎么换数据、样式要点
├─ catalog/            # 图表目录 —— 权威清单
│  ├─ INDEX.md            # 135 个分组及其编号段，先读这个
│  └─ <段号>-<主题>.md    # 每组一个文件，列全组每张图
├─ WORKFLOW.md         # 本文：怎么用
├─ DESIGN_SYSTEM.md    # 视觉规范（字体、字号、配色、版式）—— 权威标准
├─ scripts/
│  ├─ pubstyle.py         # 共享样式模块（字体栈、主题、配色、图头、savefig）
│  ├─ figkit.py           # 构造模块（英寸布局、缩放插图、卡片、箭头）
│  ├─ fig_*.py            # 每个脚本 = 一张图，独立可跑
│  ├─ plot_data.py        # 表格数据的前门（14 种图型，csv/xlsx）
│  ├─ run_all.py          # 全量重跑（可 --vector 出矢量）
│  ├─ build_reference.py  # 图改了以后重建 reference/thumbs/
│  ├─ new_figure.py       # 新图脚手架
│  ├─ check_figures.py    # 批量质量检查（产物、DPI、体积）
│  └─ check_typography.py # 字体与标题审计（PNG 查不到的那一层）
├─ fonts/              # 自带 Tinos（SIL OFL），Times 度量兼容兜底
├─ reference/          # thumbs/ 全部 1219 张缩略图；curated/ 33 张精选原图
├─ output/             # 所有产物（NN_name.png，--vector 时另有 .pdf）
└─ figure_health.md    # check_figures.py 生成的体检报告
```

**一条铁律**：这些脚本值钱的部分是 *样式代码*，不是数据。改数据、别改样式。

---

## 1. 快速上手

所有命令都在 `scripts/` 目录下执行：

```bash
cd scripts
```

### 场景 A：我只想看看有什么图 / 直接跑现成的

```bash
python fig_lollipop.py          # 跑单张 -> output/40_lollipop.png
python run_all.py               # 全部重跑（几分钟）
```

先翻 `reference/thumbs/` 里的缩略图挑图，再到 `catalog/` 对应的分组文件里定位
脚本名（从 `catalog/INDEX.md` 进）。每个脚本自带合成数据，
开箱即出图，可以直接拿去当排版占位或风格演示。

### 场景 B：套我自己的数据（最常见）

**先分清你的数据是"表格"还是"模型"。这决定了走哪条路，走错会白费很多时间。**

**B1 · 表格数据（CSV / Excel / Parquet）——走正门，别改脚本**

```bash
cd scripts
python plot_data.py --data ../examples/sales.csv --describe
python plot_data.py --data ../examples/sales.csv --kind scatter_fit \
       --x spend --y revenue --group region \
       --title "Revenue scales with spend in every region"
```

`--describe` 会先给出列画像（类型、缺失率、基数、样例），再**只列出这张表真正
支持的图型**，所以第二条命令基本不用猜。14 种图型见第 6 节。

加载层是按真实客户文件写的，不是按教科书 CSV：自动探测编码（中文 Windows 导出的
GBK 文件直接能开）、嗅探分隔符、吃掉 BOM，并把 `"1,234"`、`"12.5%"` 还原成数字。
中文列名与分类值会被检测到并自动启用中文字体，西文数字仍是 Times。

**B2 · 想改现成的 1219 张图之一——先查它的数据在哪**

```bash
python plot_data.py --contract fig_threshold_regression.py
```

**本节旧版写的是"每个脚本都把数据关在一个 `gen_data()` 里，换掉它就行"。这句话
不成立**，实测分布：

| 数据生成方式 | 脚本数 |
|---|---:|
| 有 `gen_data()`（旧文档描述的形态） | 19 |
| 内联在 `main()` 里，与绘图代码交织 | 656 |
| 无随机数据——图本身就是个模型 | 245 |
| 其他自定义函数名（`simulate` / `synth` / `make_*`） | 299 |

`--contract` 会告诉你落在哪一类、数据从第几行开始、以及有哪些模块常量值得先调。
落在第三类（比如 `fig_abl_turbulence_spectrum.py` 画的是 Kaimal 湍流谱）的图，
**没有 CSV 可以塞进去**——它值得读的是样式，不是数据通道。

改完记得同步标签：`set_xlabel` / `set_ylabel` / `panel_title` / 图例文字 / 单位。
5. 重跑脚本，看 `output/` 里的 PNG。

**不要**改 `set_pub_style()`、颜色常量、`spines`、`tight_layout` 参数、`savefig` 调用——
这些就是「贵」的来源。

### 场景 C：从零做一张新图

```bash
python new_figure.py 90 my_new_chart "One-line description of the figure"
```

生成 `scripts/fig_my_new_chart.py`，已经接好 `pubstyle`、`gen_data()`、`main()`、
`savefig(fig, "90_my_new_chart.png")`。然后：

1. 在 `gen_data()` 里换成真实数据。
2. 在 `main()` 里把示例散点换成你要的图形——**从最接近的 `fig_*.py` 里抄绘图段**，
   比从空白开始快得多，也更容易保持风格一致。
3. `python fig_my_new_chart.py` 出图。
4. `python build_reference.py --range 90-90` 更新这张图在 `reference/thumbs/` 里的缩略图。
5. 在 `catalog/` 对应的分组文件里补一行（编号、脚本名、复现了什么、什么时候用）；
   新开一组的话，同时在 `catalog/INDEX.md` 里加一行。

编号规则：`output/` 里的文件名前缀 = 图号，两位数补零（`90_my_new_chart.png`）。
`new_figure.py` 会检查这个号是否已被别的脚本占用并给出警告。

---

## 2. 选图决策树

先问自己：**这张图要让读者一眼看出什么？** 按「表达意图」而不是「数据长什么样」来选。

```
你的一句话结论是……
├─ "A 比 B 大 / 有差异"            → 比较类
├─ "这批数据长这样 / 分布偏了"      → 分布类
├─ "X 越大 Y 越……"                → 关系类
├─ "整体由这几块构成"              → 构成类
├─ "随时间在变"                    → 时序类
├─ "空间上有格局"                  → 空间类
├─ "从这里流到那里"                → 流向类
├─ "谁和谁有联系"                  → 网络/结构类
└─ "我的模型好不好 / 为什么这么判"  → 模型评估与解释类
```

### 比较类（组间差异、排序）

| 想说什么 | 用哪个 |
|---|---|
| 分组均值差异 + 显著性 | `fig_barerr_grid.py` (32) 误差棒分组柱 + 星号 |
| 干净的多组对比 | `fig_box_jitter_grid.py` (47) 箱线+抖动+均值菱形 |
| 组间分布形状也重要 | `fig_violin_grid.py` (30)、`fig_raincloud.py` (12) |
| 排序（谁第一谁最后） | `fig_lollipop.py` (40) 棒棒糖、`fig_dotplot.py` (44) Cleveland 点图+CI |
| 前后 / 两时点配对 | `fig_dumbbell.py` (39) 哑铃、`fig_slope_chart.py` (49) 斜率图 |
| 多指标画像对比 | `fig_radar.py` (15) 雷达、`fig_parallel_coords.py` (41) 平行坐标 |
| 环形排版的分组柱 | `fig_circular_bars.py` (09) |
| 多研究/多亚组效应量汇总 | `fig_forest_plot.py` (77) 森林图 |
| 生存/时间到事件 | `fig_survival_curves.py` (74) K-M 曲线 |

> 三组以上 + 只比均值 → 优先点图/棒棒糖，比柱状图省墨且更易读排序。

### 分布类

| 想说什么 | 用哪个 |
|---|---|
| 单变量形状 + 均值/中位数 | `fig_hist_kde_grid.py` (31) 直方+KDE+统计框 |
| 多组分布随序列漂移 | `fig_ridgeline.py` (13) 山脊图 |
| 分布 + 原始点 + 箱体（信息量最大） | `fig_raincloud.py` (12) |
| 正态性检查 | `fig_qq_grid.py` (46) Q-Q + 置信包络 |
| 二维分布 / 点太多糊成一团 | `fig_hexbin.py` (18)、`fig_contour_density.py` (24) 2D KDE |

### 关系类（相关、回归、阈值）

| 想说什么 | 用哪个 |
|---|---|
| 存在拐点/突变阈值 | `fig_threshold_regression.py` (01) 分段回归 + 双段 R² |
| 散点 + 边缘分布 | `fig_scatter_marginal_grid.py` (43)、`fig_marginal_violin.py` (02) |
| 相关矩阵筛变量 | `fig_corrplot.py` (05) 圆点相关矩阵、`fig_bubble_matrix.py` (53) |
| 多变量两两关系 | `fig_pairplot.py` (06) 对角 KDE + 下三角散点 |
| 相关矩阵 + 聚类结构 | `fig_clustermap.py` (14) 带树状图热图 |
| 上/下边界而非平均趋势 | `fig_constraint_boundary.py` (07) 约束线分析 |
| 带置信带的趋势 | `fig_trend_ci.py` (29) |
| 降维后看样本+载荷 | `fig_pca_biplot.py` (75) PCA 双标图 |
| 剂量-反应曲线 | `fig_dose_response.py` (115) |

### 构成类（占比、层级）

| 想说什么 | 用哪个 |
|---|---|
| 层级份额 | `fig_treemap.py` (23) 矩形树图 |
| 占比随时间变化 | `fig_stacked_area_grid.py` (37) 100% 堆叠面积 |
| 逐级流失 | `fig_funnel.py` (50) 漏斗 |
| 正负项累计分解 | `fig_waterfall.py` (20) 瀑布 |
| 多集合重叠（Venn 装不下） | `fig_upset.py` (26) UpSet |
| 周期性构成 | `fig_polar_rose.py` (52) 玫瑰图 |

### 时序类

| 想说什么 | 用哪个 |
|---|---|
| 趋势 + 显著性 | `fig_trend_ci.py` (29) 散点+OLS+95%CI+斜率/星号 |
| 两个量纲一起看 | `fig_dualaxis_grid.py` (36) 双轴 |
| 排名此消彼长 | `fig_bump.py` (19) |
| 成分流动（美观优先） | `fig_streamgraph.py` (16) |
| 日尺度长序列 | `fig_calendar_heatmap.py` (25) 日历热图 |
| 月×年季节格局 | `fig_heatmap_month_grid.py` (42) |
| 训练/实验曲线 + 方差带 | `fig_learning_curves.py` (28) |
| 项目排期 | `fig_gantt.py` (51) 甘特 |
| 信息图风格的趋势 | `fig_gradient_area_grid.py` (54) 渐变面积 |
| 阶段划分 + 占比饼 | `fig_coupling_zones.py` (08) |

### 空间类

| 想说什么 | 用哪个 |
|---|---|
| 多年份栅格对比 | `fig_spatial_grid.py` (27) 3×2 + 共享发散色标 + 指北针 + 比例尺 |
| 行政单元填色 | `fig_choropleth_grid.py` (55) |
| 两个变量同时上色 | `fig_bivariate_map.py` (56) 双变量地图 |
| 点太密的空间聚合 | `fig_hexgrid_map.py` (57) 六边形格网 |
| 纬度/位置 × 时间 | `fig_hovmoller.py` (79) Hovmöller |
| 矢量场 / 流场 | `fig_vector_field.py` (95)、`fig_streamline_flow.py` (96) |

### 流向类

| 想说什么 | 用哪个 |
|---|---|
| 多阶段流转 | `fig_sankey.py` (22) 桑基 |
| 类别间双向转移矩阵 | `fig_chord.py` (17) 和弦 |
| 样本筛选/入组流程 | `fig_consort_flow.py` (111) CONSORT |
| 方法/流程示意 | `fig_conceptual_diagram.py` (89)、`fig_abstract_workflow.py` (87) |

### 网络 / 结构类

| 想说什么 | 用哪个 |
|---|---|
| 变量相关网络 | `fig_corr_network.py` (48) 弹簧布局 + 正负边色 |
| 多面板总览拼版 | `fig_composite_overview.py` (88) |

### 模型评估与解释类

| 想说什么 | 用哪个 |
|---|---|
| 多模型预测 vs 实测 | `fig_ml_prediction_grid.py` (03) 1:1 线 + R²/RMSE/MAE |
| 分类性能 | `fig_roc_pr_grid.py` (34) ROC/PR、`fig_confusion_grid.py` (65) 混淆矩阵 |
| 概率输出可信吗 | `fig_calibration_curves.py` (66) 校准曲线 |
| 回归诊断 | `fig_residual_grid.py` (35) 残差 / Q-Q / scale-location |
| 多模型综合技巧 | `fig_taylor.py` (21) 泰勒图 |
| 消融 / 超参搜索 | `fig_feature_ablation.py` (67)、`fig_hyperparam_heatmap.py` (68) |
| 特征重要性 | `fig_pred_importance.py` (04)、`fig_lollipop.py` (40) |
| SHAP 解释 | `fig_shap_pdp.py` (11)、`fig_shap_donut.py` (33)、`fig_shap_interaction.py` (63) |
| 部分依赖 / ICE | `fig_pdp_ice_grid.py` (64) |
| 差异分析（组学风） | `fig_volcano.py` (38)、`fig_volcano_grid.py` (72)、`fig_manhattan.py` (45) |
| 富集分析 | `fig_enrichment_bubble.py` (73) |
| 贝叶斯后验 | `fig_trace_posterior.py` (127)、`fig_posterior_pairs.py` (128) |
| 因果/政策评估 | `fig_did_event_study.py` (104)、`fig_lorenz_gini.py` (103) |

### 扩展区专题地图（编号 55+）

01–54 是通用核心库；55 以后按**学科专题**成块编号。找不到通用图时先按学科定位到块，
再进块里挑：

| 编号段 | 专题 | 块内容 |
|---|---|---|
| 55–61 | 空间 / 地图 | 分区填色、双变量地图、六边形格网、地图+时序组合、流向地图、地形晕渲、空间聚类 |
| 63–68 | 模型解释与调参 | SHAP 交互、PDP/ICE、混淆矩阵、校准曲线、特征消融、超参热图 |
| 71–78 | 组学 / 生物医学统计 | 注释热图、火山图组、富集气泡、生存曲线、PCA 双标图、成分堆叠、森林图、多组学面板 |
| 79–83 | 气候 / 时频 | Hovmöller、小波功率谱、距平带、气候条纹、风玫瑰组 |
| 87–92 | 示意图 / 拼版 | 图形摘要、总览拼版、概念图、多面板叙事、方法流程、结果看板 |
| 95–98 | 物理场 | 矢量场、流线、应力等值线、相图 |
| 103–108 | 经济 / 社会 | 洛伦兹曲线+基尼、DID 事件研究、K 线面板、人口金字塔、投入产出矩阵、LMDI 分解 |
| 111–115 | 临床 / 医学 | CONSORT 流程、列线图、泳道图、瀑布响应图、剂量-反应 |
| 119–124 | 材料 / 光谱 / 电化学 | XRD 堆叠、光谱瀑布、XPS 分峰、CV-EIS 面板、电池循环、Ashby 图 |
| 127–131 | 贝叶斯 | 轨迹+后验、后验配对、先验-后验对比、后验预测检查、HDI 森林图 |
| 135–139 | 神经 / 电生理 | Raster+PSTH、ERP 地形图、时频谱面板、连接矩阵、多通道波形 |

> 编号段中间的空号是**留给后续补充的**，所以块内可能还不满。
> **当前完整清单以 `catalog/` 下的分组文件为准**（新图要手工补一行）。
> 库里实际有哪些产物，用这条盘点：
> ```bash
> python check_figures.py --quick
> ```

### 还是选不出来？

- **一图一结论**。一张图塞两个结论 → 拆成两个 panel，用 `letter_panels(axes)` 打 (a)(b)。
- **审稿人视角**：他们最先找「误差/不确定性」。有 CI 带、误差棒、置信包络的版本
  永远优先于没有的版本。
- 名字里带 `_grid` 的都是 3×2 六联版（27–37、42–43、46–47、54 等），
  共享配色/标题/图例/色标，适合「同一种图重复六次」的排版需求。

---

## 3. 换成真实数据的标准步骤

每个脚本都把**合成数据生成隔离在一个函数里**，这是整个库的设计约定。
换数据 = 换那一个函数，其它一行不动。

### 第 1 步：定位数据函数

最省事的办法是让工具替你找：

```bash
python plot_data.py --contract fig_xxx.py
```

它按 AST 定位 RNG 创建点，报告数据是在 `gen_data()` 里、在 `main()` 里内联，
还是根本没有随机数据。手工找的话，看这几个特征之一：

- 函数名是 `gen_data`、`make_data`、`make_corr`、`simulate`、`synth`、`panel`；
- 函数体里有 `np.random.default_rng(seed)`；
- 它的返回值直接被 `main()` 解包。

**注意只有 18 个脚本是标准的 `gen_data()` 形态**，656 个把数据内联在 `main()`
里，这类没有"一个函数可换"，只能在每个数组第一次被用到的地方替换。

```python
def gen_data(seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 160)
    y = 2.2 * np.sin(0.9 * x) + rng.normal(0, 0.42, x.size)
    group = rng.choice(["A", "B"], x.size)
    return x, y, group          # <- 记住这个契约
```

### 第 2 步：照着契约改写

**只改函数体，保留函数名、参数签名和返回值的形状/顺序/dtype。**

```python
import pandas as pd

def gen_data(seed=7, path="../data/measurements.csv"):
    """Load real data. Returns (x, y, group) exactly like the synthetic version."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["ndvi", "biomass", "site"])
    return (df["ndvi"].to_numpy(float),
            df["biomass"].to_numpy(float),
            df["site"].to_numpy(str))
```

要点：

- **返回 numpy 数组**，不要返回 `pd.Series`——部分脚本用布尔掩码索引，Series 的
  索引对齐会带来意外行为。`.to_numpy()` 一步到位。
- **先 `dropna`**。`np.polyfit`、`scipy.stats` 遇到 `NaN` 会静默给出 `nan` 结果，
  图上表现为「线没了」。
- **保留 `seed` 参数**（哪怕用不到），这样 `run_all.py` 和 `check_figures.py` 
  的调用方式不变。
- 路径用相对于 `scripts/` 的相对路径（脚本从 `scripts/` 运行），或用
  `os.path.join(os.path.dirname(__file__), ...)` 做成绝对路径更稳。

### 第 3 步：常见数据契约速查

| 脚本类型 | 典型返回 | 你要准备的 CSV |
|---|---|---|
| 散点/回归类 | `(x, y)` 或 `(x, y, group)` | 一行一个样本，两列数值 + 可选分组列 |
| 分组分布类 | `dict{label: values}` 或 `(values, labels)` | 长表：`group, value` |
| 矩阵/热图类 | `(matrix, row_labels, col_labels)` | 宽表，`df.set_index(...).to_numpy()` |
| 时序类 | `(years, series_dict)` | 宽表：第一列时间，其余每列一条序列 |
| 3×2 六联版 | 函数被调 6 次，每次返回一个 panel 的数据 | 准备 6 份，或改成从 6 个子集切片 |

宽表转矩阵的标准写法：

```python
df = pd.read_csv("corr_input.csv", index_col=0)
matrix = df.to_numpy(float)
row_labels = df.index.tolist()
col_labels = df.columns.tolist()
```

### 第 4 步：改标签，别改样式

要改的：`set_xlabel` / `set_ylabel` / `set_title` / `legend` 文字 / 单位 / 类别名 /
注释框里的变量名。

**不要动的**（这些是复现「贵」质感的关键）：

- `set_pub_style()` 调用及其参数
- `PALETTES[...]` / `get_palette(...)` 的取色
- `spines`、`tick_params`、`linewidth`、`alpha`、`zorder`
- `figsize`（除非投稿要求换栏宽，见第 7 节）
- `fig.tight_layout(...)` 的 `h_pad`/`w_pad`
- `savefig(fig, "NN_name.png")` —— 文件名前缀是图库的编号，别改

### 第 5 步：数值范围对不上时的最小调整

真实数据量级和合成数据不同，可能出现刻度难看、注释框压住数据、点太密。
按这个顺序调，都属于「允许动」的范围：

```python
ax.set_xlim(...) / ax.set_ylim(...)        # 先给注释和图例留出 10-15% 余量
ax.margins(x=0.03, y=0.10)
s=18 -> s=8                                 # 点变多就调小 markersize
alpha=0.75 -> alpha=0.35                    # 重叠严重就降透明度
annotate_box(ax, txt, loc="lower right")    # 换个不挡数据的角
```

点数超过 ~5000 个还是糊 → 直接换 `fig_hexbin.py` (18) 或 `fig_contour_density.py` (24)。

---

## 4. 顶刊图审美 checklist

出图后逐条过一遍。**按你自己最终印刷尺寸看图**（不是屏幕 100% 放大看）。

### 版式

- [ ] 一张图只讲一个结论；多结论已拆成 (a)(b)(c) 子图
- [ ] 图宽等于目标期刊的单栏或双栏宽度（第 7 节表），不是随手写的 `figsize`
- [ ] 高度不超过版心（Nature 类约 247 mm），留出图注空间
- [ ] 子图之间对齐：共享 x 轴的面板用 `sharex=True`，刻度线对得上
- [ ] 面板间距靠 `tight_layout(h_pad=, w_pad=)` 统一控制，不是手工 `subplots_adjust` 拼凑
- [ ] 数据墨水比高：没有 3D 效果、没有阴影、没有渐变背景、没有网格线满屏
- [ ] 子图字母 (a)(b)(c) 位置一致，用 `letter_panels(axes)` 统一打

### 字体

- [ ] 最小字号 ≥ 目标期刊要求（一般 5–7 pt，缩放后仍要够大）
- [ ] 全图字体一致，不混用；`set_pub_style()` 已统一，别单独 `fontsize=` 乱改
- [ ] 只有面板标题和统计标注加粗，坐标轴标签不加粗
- [ ] 坐标轴标签写全「量 + 单位」，如 `Biomass (Mg ha⁻¹)`，不写 `y`
- [ ] 刻度标签不重叠、不旋转 90°（实在长就旋 30–45° 或改横向布局）
- [ ] 数字用 mathtext 上下标（`set_pub_style` 已开 `axes.formatter.use_mathtext`）

### 配色

- [ ] 只从 `PALETTES` 取色，没有 matplotlib 默认蓝橙绿
- [ ] 类别色 ≤ 6 种；更多就分组或改用位置/形状编码
- [ ] 色盲安全：用 `cb_safe` / `cb_muted` / `cb_diverging`（不要红绿对立）
- [ ] 发散型数据用发散色标且**中点为 0**（`vmin=-v, vmax=v`）
- [ ] 顺序型数据用单向色标（`cb_sequential_*`），不用 jet / rainbow
- [ ] 灰度打印仍可区分（必要时叠加线型/marker 区分）
- [ ] 强调色只用在真正要强调的元素上，其余降到中性灰 `PALETTES["neutral"]`

### 标注

- [ ] 有不确定性表达：CI 带 / 误差棒 / 置信包络 / 分布，三者至少其一
- [ ] 统计量标在图上（R²、RMSE、p、n），用 `annotate_box(ax, txt)` 的白底圆角框
- [ ] 显著性用 `sig_stars(p)` + `add_sig_bracket(...)`，并在图注说明阈值
- [ ] 样本量 n 出现在图上或图注里
- [ ] 图例不遮数据；同一批面板共用一个图例（`fig.legend`）
- [ ] 参考线（1:1 线、零线、阈值线）用虚线 + 中性色，压在数据下层（`zorder` 小）
- [ ] 坐标轴不从非零处偷偷起跳；如果截断了，明确标出

### 导出

- [ ] 通过 `savefig(fig, "NN_name.png")` 导出（已内置 300 dpi + `bbox_inches="tight"`）
- [ ] 需要投稿矢量版时：`savefig(fig, "NN_name", formats=("png", "pdf"))`
- [ ] PNG 打开检查：没有被裁掉的标签、没有多余白边
- [ ] 文件命名 `NN_name.png`，编号与 `catalog/` 里登记的一致
- [ ] 跑一遍 `python check_figures.py --quick` 确认尺寸/DPI/体积正常
- [ ] 跑一遍 `python check_typography.py --only <name>` 确认字体与标题合规

---

## 5. 常见问题排查

### 中文显示成方框 □□□

`pubstyle` 默认用 **Times New Roman** 衬线栈（不含中文字形）。中文标签要显式开：

```python
from pubstyle import set_pub_style, set_cn_font
set_pub_style()
set_cn_font()          # 返回 True 表示找到了中文字体
```

`set_cn_font()` 会**先找宋体系**（`SimSun` / `Songti SC` / `STSong` /
`Noto Serif CJK SC` / `Source Han Serif SC`），找不到再退到黑体系
（`Microsoft YaHei` / `SimHei` / `Noto Sans CJK SC` / ...），并顺手关掉
`axes.unicode_minus`（否则负号会变成方框）。传 `set_cn_font(prefer_serif=False)`
可以反过来优先黑体。

**它不会把整张图切成无衬线。** 西文和数字仍然是 Times，只有汉字取自中文字体。
旧版本直接改写 `font.family="sans-serif"`，那会把每一个刻度数字也一并换掉。

### 关键细节：中文字体必须写进 `font.family`，不能塞进 `font.serif`

这一条踩过坑，写下来免得再踩：

```python
# 不行 —— 不报错、findfont 也不警告，但汉字全是空心方框
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = SERIF_STACK + ["SimSun"]

# 可以 —— 具体字体名直接列在 font.family 上
mpl.rcParams["font.family"] = ["Times New Roman", "Tinos", "SimSun"]
```

matplotlib 只在 `font.family` 本身是一串**具体字体名**时才逐字形回退；
`font.family="serif"` 这条泛型别名路径不会沿着 `font.serif` 往下找缺失的字形。
两种写法都能通过 `findfont`，所以这个错误不会以异常的形式暴露，只会以满屏豆腐块
的形式出现在图上。`set_cn_font()` 现在用的是后一种写法，并且会把栈过滤成本机
真实存在的字体，顺带消掉那几条 `findfont: Font family 'Nimbus Roman' not found`
的噪音警告。

`plot_data.py` 会自动检测列名与分类值里的非拉丁字符并调用 `set_cn_font()`，
所以客户直接上传中文表格即可，不需要手动开。用 `--no-cn-font` 可以关掉。

投稿图一律用英文标签。

返回 `False` = 一个都没装。排查：

```python
from matplotlib import font_manager
print(sorted({f.name for f in font_manager.fontManager.ttflist
              if any(k in f.name for k in ("YaHei", "SimHei", "Hei", "Song"))}))
```

装了字体但仍然找不到 → matplotlib 字体缓存过期，删掉缓存目录重跑：

```python
import matplotlib
print(matplotlib.get_cachedir())     # 删掉这个目录下的 fontlist-*.json
```

**投稿建议**：正式投稿图一律用英文标签。中文只用于组会/汇报版本。

### `tight_layout` 警告：not compatible with tight_layout

典型触发：图里有 `fig.add_axes([...])` 手工放的色标轴、`inset_axes`、
`make_axes_locatable` 造的边缘轴。`tight_layout` 不认识这些轴。

三种解法，按推荐度：

1. 手工轴**最后加**：先 `fig.tight_layout()`，再 `fig.add_axes(...)`。
2. 换 `fig.subplots_adjust(left=..., right=..., bottom=..., top=..., hspace=..., wspace=...)`
   显式控制，完全不调 `tight_layout`。
3. 改用 constrained layout：`plt.subplots(..., layout="constrained")`，
   然后**不要**再调 `tight_layout`（两者互斥）。

只是警告、图没坏的话可以忽略——`savefig` 里的 `bbox_inches="tight"` 会兜住裁剪。

### colorbar 位置错乱 / 把某个子图挤小了

`fig.colorbar(im, ax=ax)` 只从**那一个** `ax` 里抠空间，六联版里就会出现
「第 6 个子图比别人窄一截」。

共享色标的正确写法——把整组轴都传进去：

```python
from pubstyle import add_colorbar
add_colorbar(fig, im, ax=axes.ravel().tolist(), label="Δ NDVI",
             fraction=0.025, pad=0.02)
```

要精确控制位置就自己开一根轴（记得此时别用 `tight_layout`）：

```python
fig.subplots_adjust(right=0.88)                 # 先腾出右侧空间
cax = fig.add_axes([0.90, 0.15, 0.015, 0.70])   # [left, bottom, width, height]
cb = fig.colorbar(im, cax=cax, label="Δ NDVI")
cb.outline.set_linewidth(0.6)                   # 保持发丝描边的房子风格
cb.outline.set_edgecolor("#333333")
cb.ax.tick_params(labelsize=7, width=0.7, length=2.5, color="#333333")
```

横向色标放底部：`orientation="horizontal"`，并把 `fraction` 调到 ~0.05、
`aspect` 调到 ~40，否则会变成又粗又短的一条。

### 图例遮挡数据

按情况选：

```python
# 1) 挪到画布外
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

# 2) 多面板共用一个图例（推荐，省空间又整齐）
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01),
           frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 1])      # 给图例腾出底部 5%

# 3) 数据在上半部就给顶部留白，而不是挪图例
ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
```

注释框被压住同理：`annotate_box(ax, txt, loc=...)` 支持 9 个方位，换一个角就行。

### 导出模糊

- `savefig()` 已经是 300 dpi，**模糊几乎都是二次处理造成的**：不要把 PNG 拖进
  Word/PPT 再拉大，不要截图，不要用画图工具另存。
- 要更高解析度：`savefig(fig, "x.png", dpi=600)`。
- **最优解是给期刊矢量图**（线条图无损、文字可搜索）：
  ```python
  savefig(fig, "27_spatial_grid", formats=("png", "pdf"))   # 同时出 PNG 和 PDF
  ```
- 例外：栅格地图/热图/大点云的 PDF 会非常大且渲染慢，这类保持 PNG @ 600 dpi 更实际。
- LaTeX 里用 `\includegraphics[width=\linewidth]{fig.pdf}`，别指定 `scale=`。

### 字太小 / 字太大

关键认知：**字号是相对最终印刷宽度的**。一张 `figsize=(7.2, 4.5)` 的图被压到
89 mm 单栏，等于缩到 49%，7 pt 的字变成 3.4 pt —— 直接被拒。

正确做法是**按最终尺寸设计**，而不是事后缩放：

```python
from pubstyle import set_pub_style, FIGSIZES
MM = 1 / 25.4

set_pub_style(theme="nature")                       # 7 pt 基准字号
fig, ax = plt.subplots(figsize=(89 * MM, 70 * MM))  # 直接按单栏宽建图
```

可选主题（`pubstyle.THEMES`）：

| theme | 基准字号 | 默认画布 | 用途 |
|---|---|---|---|
| `default` | 9 pt | — | 通用、屏幕查看 |
| `nature` | 7 pt | 3.5 × 2.6 in | Nature 系单栏 |
| `science` | 8 pt | 3.6 × 2.7 in | Science 系 |
| `ieee` | 7 pt | 3.3 × 2.4 in | IEEE 双栏，刻度朝内 |
| `presentation` | 13 pt | 10 × 5.6 in | 投影/汇报，粗线大字 |

也可以只微调基准字号：`set_pub_style(base_font=8)`；
或临时覆盖任意 rcParams：`set_pub_style(**{"axes.grid": True})`。
`FIGSIZES` 里备好了 `single` / `single_tall` / `double` / `double_tall` /
`square` / `slide` 六种常用画布。

### 其它零碎

| 现象 | 原因 / 处理 |
|---|---|
| 保存后标签被裁掉 | `savefig` 已带 `bbox_inches="tight"`；仍被裁说明标签在轴外且用了 `add_axes`，改用 `tight_layout` 布局 |
| 负号显示成方框 | 调了中文字体但没关 `axes.unicode_minus`——用 `set_cn_font()` 而不是手动设 `font.family` |
| 曲线/拟合线消失 | 数据里有 `NaN`；`gen_data` 里先 `dropna` |
| 图例里出现重复条目 | 循环里每次都传了 `label=`；只在第一次传，或用 `dict(zip(labels, handles))` 去重 |
| 颜色和参考图不一样 | 没走 `PALETTES`，落回 matplotlib 默认色环 |
| 运行很慢 | KDE / 网络布局 / 大网格插值本来就慢，见 `figure_health.md` 的耗时排行 |
| 多脚本并行跑时报错 | 用 `check_figures.py --jobs 1` 串行复现，确认是不是真错 |

---

## 6. 工具用法

### `run_all.py` —— 全量重跑

```bash
cd scripts
python run_all.py
python run_all.py --parallel 8        # 8 个子进程并发
python run_all.py --range 100-200     # 只跑编号 100-200
python run_all.py --only rs_          # 文件名子串过滤
python run_all.py --vector            # 每张 PNG 旁边再出一份 PDF
python run_all.py --formats pdf,svg   # 指定额外格式
python run_all.py --list              # 只列出选中项，不执行
```

每个脚本跑在独立子进程里（`MPLBACKEND=Agg`），所以**单张图崩了不会带崩整轮**，
失败清单在结尾统一打印。

`--vector` / `--formats` 通过环境变量 `PUBSTYLE_FORMATS` 生效，`pubstyle.savefig`
读它并追加格式——不需要去改 1219 个 `savefig()` 调用点。投稿要矢量图时用它：

```bash
python run_all.py --parallel 8 --vector      # output/ 里同时得到 PNG 和 PDF
```

### `build_reference.py` —— 重建缩略图

```bash
python build_reference.py                 # output/ 里所有编号 PNG
python build_reference.py --range 0-54    # 只更新改过的那一段
python build_reference.py --out some/dir  # 写到别处，用来和现有的对比
```

- 把 `output/NN_name.png` 缩成 520 px 宽的 JPEG，写进 `reference/thumbs/`。
- `reference/thumbs/` 是随包提交的，图改了它不会自己更新；改完图跑一次，
  和脚本一起提交。
- `reference/curated/` 是手挑的原图，这个脚本不动它；要更新就从 `output/` 手工拷。
- 只依赖 Pillow，不 import matplotlib。

**建议**：每次新增或翻新图之后跑一次，翻 `reference/thumbs/` 做视觉回归对比。

### `new_figure.py` —— 新图脚手架

```bash
python new_figure.py 90 my_new_chart "One-line description"
python new_figure.py 90 my_new_chart "..." --force   # 允许覆盖（默认拒绝）
```

- 生成 `scripts/fig_my_new_chart.py`，包含模块 docstring、`gen_data()`、
  `main()`、`set_pub_style()`、`savefig(fig, "90_my_new_chart.png")`、
  `if __name__ == "__main__"` 守卫。
- 名称必须是小写蛇形（`^[a-z][a-z0-9_]*$`），否则直接报错退出。
- 目标文件已存在 → 拒绝覆盖并提示。
- 图号已被别的脚本占用 → 打印警告（不阻塞，方便你有意复用）。
- 生成后打印五步「下一步」清单。

### `check_figures.py` —— 批量体检

```bash
python check_figures.py                 # 全量：逐个跑 + 检查产物
python check_figures.py --quick         # 秒级：只查已有 PNG，不重跑脚本
python check_figures.py --only heatmap  # 只查文件名含 heatmap 的脚本
python check_figures.py --jobs 8        # 并行度（默认 4，设 1 则串行）
```

检查内容：

| 维度 | 判定 |
|---|---|
| 退出码 | 非 0 记为失败，附 stderr 末尾 6 行 |
| 耗时 | 单脚本超时 300 s 判失败，报告里给耗时排行 |
| 警告 | 从 stderr 抓 `*Warning`，过滤环境噪音 |
| 产物存在 | 解析脚本里的 `savefig(...)` 得到期望文件名，逐个查 `output/` |
| 文件大小 | < 20 KB 报警（通常意味着画了个空图） |
| 像素尺寸 | Pillow 读取，报告里列出 |
| DPI | < 290 报警；无 DPI 元数据也报警 |

输出：终端汇总表 + `figure_health.md`（失败列表 / PNG 问题表 / 警告清单 /
耗时排行 / 全量明细）。

**它只读不写脚本**——发现问题不会去改 `fig_*.py`。多人并行改库时的临时失败
如实记录即可，重跑一次通常就好了。

**它看不见字体和标题。** PNG 里没有字体元数据，所以哪怕 matplotlib 悄悄回退到
了 DejaVu，上面每一项照样全绿。那一层交给 `check_typography.py`。

### `plot_data.py` —— 客户数据的正门

```bash
python plot_data.py --list-kinds                      # 14 种图型
python plot_data.py --data f.csv --describe           # 列画像 + 可用图型建议
python plot_data.py --data f.csv --kind scatter_fit --x a --y b --group g
python plot_data.py --contract fig_xxx.py             # 某张图的数据从哪来
```

| 图型 | 用途 |
|---|---|
| `scatter_fit` | 散点 + 拟合线（OLS 或 `--smooth` 走 LOWESS）+ 边缘分布 |
| `bar_group` | 分组柱状图，`--stacked` 切堆叠 |
| `line_ci` | 折线 / 时间序列 + 95% 置信带（同 x 有重复值时自动算） |
| `dist` | 分组分布：小提琴 + 箱线 + 原始点 |
| `corr` | 数值列相关矩阵热图，`--method spearman/kendall` |
| `hist` | 直方图 + 核密度 |
| `ecdf` | 分组经验累积分布 |
| `area` | 堆叠面积 |
| `donut` | 环形构成图 + 占比表 |
| `forest` | 森林图，需 `--y --lo --hi`，`--ref` 设参考线 |
| `lollipop` | 排序棒棒糖图 |
| `errorbar` | 组均值 + 95% CI + 背后的原始点 |
| `survival` | Kaplan-Meier 阶梯曲线，`--event` 给删失列 |
| `pairs` | 数值列散点矩阵 |

通用参数：`--agg`（mean/median/sum/count/min/max）、`--top`（类别上限）、
`--bins`、`--title/--subtitle/--footer/--corner`、`--theme`、`--formats`、
`--out/--outdir`、`--sheet`（Excel 工作表）、`--encoding/--delimiter`（覆盖嗅探）。

**样式不可配置，这是刻意的**——样式来自 `pubstyle`，那是这个库值钱的部分。
标题请自己写 `--title`，并且**写数据支持的结论**，别复述图表类型。

### `check_typography.py` —— 字体与标题审计

```bash
python check_typography.py                 # 环境 + 源码 + 渲染探针
python check_typography.py --metrics       # 证明兜底字体与 Times 度量一致
python check_typography.py --source-only   # 只扫源码，不渲染
python check_typography.py --render 40     # 探针改渲染 40 张（默认 12）
python check_typography.py --pdf ../output/01_threshold_regression.pdf
python check_typography.py --strict        # 警告也算失败
python check_typography.py --source-only --fix   # 把低于下限的字号抬到 6.0
```

检查内容：

| 维度 | 判定 |
|---|---|
| 字体栈解析 | `set_pub_style()` 后正文落到 DejaVu = error |
| 度量兼容 | 逐字比较字宽；`STIXGeneral` 是已知例外 |
| PDF 嵌入字体 | 渲染成 PDF 后读 `/BaseFont`，命中 DejaVu = error |
| 字体嵌入 | 有 `/BaseFont` 但无 `/FontFile` = error（期刊排版会替换字体） |
| findfont 警告 | 渲染时 stderr 出现 `findfont` = error |
| 字号阶梯 | < 6.0 pt = error；> 12.5 pt = warning |
| rcParams 逃逸 | `fig_*.py` 里直写 `rcParams` = error |
| 标题大小写 | 疑似 Title Case、标题带句号 = warning |
| 面板字母 | 非小写 `(a)` 形式 = warning |

**为什么必须渲染 PDF 才能查字体**：PNG 是像素，不记录字体；PDF 会把实际使用的
字体嵌进去。这是唯一能确认"你以为用的字体"和"渲染器真正用的字体"是同一个的办法。

`--fix` 只修一类问题：低于下限的字号字面量（这类有唯一正确答案，可机械修复）。
Title Case、"标题复述图表类型"这些需要人重新写文案，一律只报 warning。

### 典型组合

```bash
# 日常：改完一张图
python fig_xxx.py && python check_figures.py --only xxx

# 收尾：交付前完整体检 + 字体审计 + 重建缩略图
python check_figures.py --jobs 4 --typography && python build_reference.py

# 快速盘点：现在库里有多少张、有没有缺件
python check_figures.py --quick

# 换了机器 / 换了环境，第一件事：确认字体栈还解析得到
python check_typography.py --metrics

# 投稿包：矢量图 + 全量审计
python run_all.py --parallel 8 --vector && python check_typography.py --strict
```

---

## 7. 投稿规格速查

> 下表是常见档位，**投稿前务必对照目标期刊当年的 Author Guidelines 核对**——
> 各刊会微调，特刊/子刊也可能不同。

### 尺寸与字号

| 期刊 / 出版社 | 单栏宽 | 1.5 栏 | 双栏宽 | 最大高度 | 最小字号 | 栅格 DPI | 首选格式 |
|---|---|---|---|---|---|---|---|
| Nature 系 | 89 mm | 120 mm | 183 mm | 247 mm | 5–7 pt | 300（照片）/ 600+（线条） | AI, EPS, PDF；可 TIFF/PNG |
| Science 系 | 57 mm | 121 mm | 184 mm | 233 mm | 5–6 pt | 300 / 600（线条） | AI, EPS, PDF, TIFF |
| Cell 系 | 85 mm | 114 mm | 174 mm | 240 mm | 5–7 pt | 300（彩/灰）/ 1000（线条） | AI, EPS, PDF, TIFF |
| PNAS | 87 mm | 114 mm | 178 mm | 235 mm | 6–8 pt | 300–600 | TIFF, EPS, PDF |
| Elsevier（多数刊） | 90 mm | 140 mm | 190 mm | 240 mm | 7 pt（6–8） | 300（半调）/ 500（组合）/ 1000（线条） | TIFF, EPS, PDF |
| Springer / Nature 子刊 | 84 mm | 129 mm | 174 mm | 234 mm | 6–8 pt | 300 / 600（线条） | TIFF, EPS, PDF |
| Wiley（多数刊） | 80 mm | 120 mm | 166 mm | 230 mm | 7–8 pt | 300 / 600 | TIFF, EPS, PDF |
| IEEE 期刊 | 88.9 mm (3.5″) | — | 181.6 mm (7.16″) | 按版面 | 8–10 pt | 300（照片）/ 600（组合）/ 1200（线条） | PS, EPS, PDF, TIFF |
| ACS | 83 mm (3.3″) | 114 mm | 178 mm (7″) | 240 mm | 4.5–8 pt | 300（彩）/ 600（灰）/ 1200（线条） | TIFF, EPS, PDF |
| MDPI（Remote Sensing 等） | 85 mm | — | 170 mm | — | 8 pt | ≥300（宽 ≥1000 px） | PNG, TIFF, EPS, PDF |
| AGU（JGR 等） | 95 mm | 140 mm | 190 mm | 230 mm | 6–8 pt | 300–600 | PDF, EPS, PNG, TIFF |

### mm → matplotlib

```python
MM = 1 / 25.4
fig, ax  = plt.subplots(figsize=(89 * MM, 70 * MM))     # Nature 单栏
fig, axs = plt.subplots(2, 3, figsize=(183 * MM, 120 * MM))  # Nature 双栏六联
```

常用换算：

| mm | inch |
|---:|---:|
| 57 | 2.24 |
| 85 | 3.35 |
| 89 | 3.50 |
| 90 | 3.54 |
| 114 | 4.49 |
| 120 | 4.72 |
| 170 | 6.69 |
| 174 | 6.85 |
| 178 | 7.01 |
| 183 | 7.20 |
| 190 | 7.48 |

### 主题速配

| 目标 | 一行配置 |
|---|---|
| Nature 单栏 | `set_pub_style(theme="nature")` + `figsize=(89*MM, 70*MM)` |
| Science 双栏 | `set_pub_style(theme="science")` + `figsize=(184*MM, 110*MM)` |
| IEEE 双栏 | `set_pub_style(theme="ieee")` + `figsize=(181.6*MM, 100*MM)` |
| Elsevier 双栏 | `set_pub_style(base_font=7)` + `figsize=(190*MM, 120*MM)` |
| 组会 / 答辩 | `set_pub_style(theme="presentation")` + `figsize=FIGSIZES["slide"]` |

### 投稿前最后三件事

1. **按 100% 打印尺寸打一张纸**（或在 PDF 里按实际 mm 显示），确认最小字号读得清。
2. **灰度打印一份**，确认分类还能区分。
3. **同时导出矢量版**备用：
   ```python
   savefig(fig, "01_threshold_regression", formats=("png", "pdf"))
   ```
   多数刊在初审接受 PNG，但录用后会索要矢量或高 DPI TIFF。
