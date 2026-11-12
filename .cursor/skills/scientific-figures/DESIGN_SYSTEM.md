# 图库设计规范 (Design System)

本文件是 `scientific-figures` 图库的**跨图视觉统一标准**。规范不是凭空拟定的，
而是从全库源码里扫描出真实分布后，取"高频事实标准"固化而成
（扫描脚本 `scripts/_style_scan.py`，首轮统计见 `scripts/_audit_style_unity.md`）。

> **本文所有统计数字的口径**：**2026-08-12 17:00 收口重扫**，覆盖
> `scripts/fig_*.py` 全部 **354 个脚本（编号 00–353）**，扫描时全库无并行改动，
> 因此本轮数字是**终态**而非中间态（上一版的说明相反，已作废）。
> 首轮审计是在 190 张左右时做的，那一轮的数字（842 种 hex / 52 种字号 /
> 156 种 figsize）只保留在 `_audit_style_unity.md` 里做对比，不再出现在本文。
>
> 工具口径提醒：`_style_scan.py` 只认**字面量**。凡是把值先赋给标识符
> （`HEAT = "jet"`、`def f(..., fs=5.8)`）再传进去的写法它一律扫不到，本轮就是靠
> AST 回溯才补齐的——见 `_audit_final_pass.md`。引用本文数字时，凡涉及"零残留"
> 的结论都应该按 AST 口径复核，别只信正则。

> **关于本文引用的 `_audit_*.md` 与 `_style_scan.py`**：那些是建库期间的审计日志和
> 一次性扫描脚本，不随本包分发。文中提到它们的地方只是在交代某个数字的来历，
> 不影响任何规范条款的执行——规范的唯一权威实现始终是 `scripts/pubstyle.py`。

**适用范围**：所有 `scripts/fig_*.py`。
**唯一权威实现**：`scripts/pubstyle.py`。规范里给的每个数值都应该能在 `pubstyle`
里找到对应的常量或默认参数；不要在图脚本里重新发明。

> 判定标准始终是：**把 20 张图并排放在一张 contact sheet 上，它们像不像同一支
> 团队做的。** 单图好看不算数。自检用 `python scripts/build_reference.py`
> 重建 `reference/thumbs/`，然后在文件管理器里用大图标并排看。

---

## 0. 三条铁律

1. **每个脚本第一件事是 `set_pub_style()`**，且**永远不要直接改 `rcParams`**。
   （当前全库 354/354 合规，零处 `rcParams` 逃逸；这是本图库最健康的一项，
   别破坏它。）
2. **颜色先查 `PALETTES`，查不到再想自己配。** 禁止 matplotlib `tab10` 默认色
   （`#1f77b4` `#d62728` `#2ca02c` …）、Office/Flat-UI 色（`#c0392b` `#27ae60`
   `#2e74b5` `#c0504d`）、纯饱和霓虹色（`#ff4fa3` `#00b0f0` `#ffd23f` …）。
3. **禁止彩虹与高饱和色表**：`jet` / `rainbow` / `turbo` / `hot` / `RdYlGn` /
   `RdYlBu` / `nipy_spectral` / `hsv` / `Spectral`。见 §2.4 的替代方案。
   这一条**要连 `HEAT = "jet"` 这种存进常量再用的写法一起查**——它正是三轮审计
   都漏掉这五个脚本的原因。

---

## 1. 画布尺寸档位 (Canvas)

### 1.1 档位表

全库真实宽度分布始终是**双峰**的：316 次 `figsize=` 调用里，7.2–7.8 in 占
147 次、9.6–11.4 in 占 109 次，两峰合计 81%。把它固化成四档，新图**只从这四档
里选**：

| 档位 | figsize | 面板容量 | 使用场景 |
|---|---|---|---|
| `single` | `(3.5, 2.6)` | 1 | 单栏小图、嵌入正文的示意 |
| `double` | **`(7.2, 4.5)`** | 1×2 / 2×2 | **默认档**。双栏跨页主图，绝大多数图应该用这一档 |
| `wide` | **`(10.4, 6.4)`** | 2×3 / 3×3 | 面板多、每格需要独立坐标轴标签 |
| `full` | **`(12.6, 7.6)`** | ≥ 3×4，或含 3D / 地图 | 密集多面板；用之前先问能不能拆成两张 |

`pubstyle.FIGSIZES` 已提供 `single / single_tall / double / double_tall /
square / slide`；上表的 `wide` / `full` 是本规范新增的口语档名，写代码时直接写
数值即可。

```python
fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))   # wide
```

### 1.2 高宽比才是观感的关键

contact sheet 会把每张图缩放到统一格子里，**绝对英寸看不出来，长宽比一眼就看
出来**。所以尺寸规范的硬约束是 aspect = h/w：

- **正常带**：`0.55 ≤ h/w ≤ 0.80`。新图默认落在这里。
- **允许的例外**（收口审计确认的八类，只有这八类）：

| 代号 | 例外 | 为什么不能靠改 figsize 解决 |
|---|---|---|
| E1 | `set_aspect("equal")` 锁死的地图 / 影像 / 相关阵 / 圆形图 | 画幅由数据自身长宽比决定，加高只加白边 |
| E2 | 含 3D 面板（`projection="3d"`） | 投影盒本身浪费竖向空间，宽画布是标准做法 |
| E3 | 极坐标 / 圆形主面板（radar、rose、Taylor、chord） | 圆只能横向平铺 |
| E4 | 英寸画布架构图（`fig_net_*`，§12.8 方案 A） | 画布即图形外接框，补白等于纯白边 |
| E5 | 行数驱动的图（gantt、forest、ridgeline、swimmer、funnel、consort flow、堆叠式仪表板） | 高度由数据行数决定，不是版式选择 |
| E6 | 矩阵型图，`h/w ≈ 1.0`（pairplot、posterior pairs、clustermap、confusion / attention grid） | 对角对称的内容天然是方的 |
| E7 | 多行堆叠，行数 ≥ 3 且列数 ≤ 2 | 压扁会让每个面板变成细长条 |
| E8 | 单行多列对比网格，列数 ≥ 3 | 拉进 0.55 会把每格挤成竖条，比越界更糟 |

**当前合规率：263 / 354 = 74.3 %**（越界 91 张，逐张核对后全部落在 E1–E8）。
收口那一轮把 20 张**不属于任何例外**的图改回了合规带；名单见
`_audit_final_pass.md`。判断顺序是**先问属不属于 E1–E8，不属于就改 figsize**，
不要反过来先编一个例外。

### 1.3 尺寸微调的纪律

不要出现 `7.3` / `7.35` / `7.42` 这种为了挤下一行标签而随手 +0.1 的尺寸。
**先减内容、再调 `tight_layout` 的 pad、最后才换档位。** 全库目前有 265 种不同
的 figsize（354 次调用，每个脚本恰好一次；把 `FW, FH = …` 这类间接写法也解析进
来了，所以比只数字面量的 `_style_scan.py` 多出十几种）。

**但不要再拿"figsize 种数"当 KPI。** contact sheet 会把每张图缩到统一格子，
绝对英寸根本看不见，种数多寡不影响观感；真正该管的是 §1.2 的高宽比。
最常用的两档确实已经是本表里的 `(10.4, 6.4)`（16 次）和 `(7.2, 4.5)`（9 次），
第三名 `(11.0, 6.6)`（7 次）不在四档里，新图别跟它。

四个越界尺寸（宽 > 14 in 或高 > 11 in）已知且都是刻意为之：
`00_style_showcase (12.4×14.2)`、`143_benchmark_convergence (13.0×15.4)`、
`144_benchmark_surfaces (14.2×7.6)`、`151_multiobj_pareto3d (14.6×8.6)`。
新图不要再加进这份名单。

---

## 2. 调色板规范 (Colour)

### 2.1 官方色板

以下 hex 是**从全库高频色里提炼的事实标准**，括号里是当前使用该色的脚本数。
新图**优先从这张表里取色**。

#### 中性灰阶（骨架色 · 用得最多，先记熟这一组）

| 用途 | hex | 说明 |
|---|---|---|
| 正文/标题墨色 | `#1a1a1a` (169) | 面板标题、强调注释 |
| 轴线 / 刻度 / 主注释 | `#333333` (200) | **全库第一高频色**，`pubstyle` 默认边框色 |
| 次级说明文字 | `#555555` (148) | 副标题、脚注 |
| 弱化说明文字 | `#777777` (80) / `#8a8a8a` (78) | 图内旁注、单位 |
| 参考线 / 基线 | `#808080` (11) | 中位线、零线、baseline 系列 |
| 网格 / 分隔线 | `#cccccc` (62) / `#d5d5d5` (28) | `pubstyle` 默认 `grid.color = #d9d9d9` |
| 背景分区块 | `#f0f0f0` (16) / `#f5f5f5` (14) | 见 §7.2 |
| 深色实体填充 | `#3a3a3a` (53) | 障碍物、遮罩等大面积深色。**不要用近黑 `#1c1c1c` 或 `#000000` 做大面积填充**，在 contact sheet 上会明显比其他图"重" |

灰阶里还有一条**没被收编**的尾巴：`#666666` (101)、`#9a9a9a` (48)、
`#444444` (46)、`#5a5a5a` (40)、`#2b2b2b` (38)、`#bbbbbb` (31)、
`#8c8c8c` (31)、`#4a4a4a` (27) 都是上表某一档的邻居，肉眼分不出来，
只是各写各的。**新图不要再引入新的灰**；上表八档够用。

#### 定性分类色（Categorical）

按顺序取用，最多 5 类；超过 5 类改用分组 + 灰色"其他"。

| 序位 | hex | 名称 | 来源 |
|---|---|---|---|
| 1 | `#b23a48` (160) | house red | `PALETTES["blue_red"][1]` |
| 2 | `#3b5b92` (93) | house blue | `PALETTES["blue_red"][0]` |
| 3 | `#2e7d5b` (95) | house green | `PALETTES["greens"][1]` |
| 4 | `#c9a227` (79) | house gold | `PALETTES["earth"][2]` |
| 5 | `#8e6bbf` (53) | house purple | `PALETTES["pastel4"][3]` |

这五个序位是收编最成功的一项：`#b23a48` 从 42 个脚本涨到 160 个，已经是全库
第三高频色，仅次于两档墨色。

补充色（需要第 6、7 类，或需要与上表同族但更深/更浅时）：
`#8c2f2f` (30) 深红 · `#c0603f` (38) 赭 · `#3d6b8e` (19) 灰蓝 ·
`#2a7f7a` (14) 青 · `#7b3294` (11) 深紫。

**需要色盲安全时**改用整组 `get_palette("cb_safe", n)`（Okabe-Ito）。
不要把 Okabe-Ito 里的色和上面的 house 色混着用——两套色的饱和度不一样，
混用会显脏。

#### 顺序色（Sequential）

```python
from pubstyle import get_cmap, get_palette
cmap = get_cmap("cb_sequential_blue")      # 蓝，默认首选
get_palette("cb_sequential_green", 5)      # 取 5 个离散台阶
```

可选：`cb_sequential_blue` / `cb_sequential_green` / `cb_sequential_warm` /
`cb_sequential_purple`。
外部 cmap 只允许感知均匀的 `viridis` / `magma` / `cividis` / `gray`
（`viridis` 是全库最常用的**彩色**具名 cmap，7 个脚本；只有中性的 `gray`
更多，9 个）。

#### 发散色（Diverging）

```python
cmap = get_cmap("cb_diverging")       # 蓝 <-> 橙，首选
cmap = get_cmap("cb_diverging_pg")    # 紫 <-> 绿，需要第二条发散通道时
```

发散色**必须对称归一化**，否则白色中点会骗人：

```python
import matplotlib as mpl
norm = mpl.colors.TwoSlopeNorm(vcenter=0.0, vmin=-m, vmax=m)
```

#### 强调色（Accent）

全图**最多一个**强调对象（"我们的方法"/异常点/阈值）。用 `#b23a48`，
其余序列全部退到中性灰阶。不要用第二个强调色。

### 2.2 自建色表的规矩

全库 1665 种 hex 里只有 55 种（3%）来自 `PALETTES`，其中 1226 种是**只出现一次
的孤儿色**。这不全是坏事——地物分类、岩性、土地利用这类图的语义色是领域惯例，
本来就该自建。但自建必须守两条：

1. **写成模块级常量并注明语义**，不要散落在调用处：
   ```python
   LANDCOVER = {"water": "#2c5b59", "forest": "#1f6b57", "crop": "#c9a227"}
   ```
2. **明度单调**。顺序色表相邻两档的明度差要肉眼可辨；用
   `pubstyle.lighten()` / `darken()` 从一个基色派生，别手搓一串 hex。

### 2.3 明确禁止的色

| 禁用 | 替换为 | 原因 |
|---|---|---|
| `#1f77b4` `#d62728` `#2ca02c` `#7f7f7f` | `#3b5b92` `#b23a48` `#2e7d5b` `#808080` | matplotlib `tab10` 默认色，比本库整体饱和度高一档 |
| `#c0392b` `#27ae60` `#2e74b5` `#c0504d` | `#b23a48` `#2e7d5b` `#3b5b92` `#b23a48` | Flat-UI / Office 配色，观感"PPT 味" |
| `#ff4fa3` `#00b0f0` `#ffd23f` `#e8232a` `#e8342a` `#f6a71c` | `#b23a48` `#3d6b8e` `#c9a227` … | 纯饱和霓虹 |
| `#000000` / `#1c1c1c` 大面积填充 | `#3a3a3a` | 见上表 |

这张表现在是**空的**：收口重扫时 `tab10`、Flat-UI、霓虹三组在全库一个都不剩。
唯一剩下的 `#000000` 字面量在 `90_multipanel_story` L262，是面板标签 `(f)` 的
文字色，和同图其它面板标签走 `add_panel_label()` 拿到的 matplotlib 默认
`text.color`（也是纯黑）一致——**不是大面积填充，本条禁令不适用**，刻意保留。
这是三轮统一收编最干净的成果，别把它们放回来。

### 2.4 色表禁令与替代

| 禁用 cmap | 替代 |
|---|---|
| `turbo` / `jet` / `rainbow` | 单极量 → `magma`、`get_cmap("cb_sequential_warm")` 或 `viridis` |
| `hot`（含 PET / 热成像伪彩） | `magma` —— 同样是暗→亮的暖调，但感知均匀，不会烧出荧光黄 |
| `RdYlGn` / `RdYlGn_r`（好/坏语义） | `get_cmap("cb_diverging")`；低值=好时用默认，高值=好时加 `reverse=True` |
| `Spectral` / `nipy_spectral` / `hsv` | 同上 |

**这一条上一版报了假的"已清零"。** 上一轮用的是字面量扫描，而
`HEAT = "jet"` / `HEAT = "turbo"` 这种先赋给模块常量、再 `cmap=HEAT` 传进去的
写法它看不见。收口审计按标识符回溯，抓出 **5 个漏网脚本**并已全部改成 `magma`：

| 图号 | 脚本 | 原 cmap |
|---|---|---|
| 273 | `fig_gradcam_grid` | `jet` |
| 274 | `fig_attention_maps` | `turbo` |
| 278 | `fig_attention_rollout` | `turbo` |
| 279 | `fig_saliency_compare` | `jet` |
| 253 | `fig_cv_fusion_grid` | `hot`（PET 伪彩） |

改完后全库具名 cmap 的完整清单是：`gray(_r)` 10 · `viridis` 7 · `magma(_r)` 10 ·
`Blues(_r)` 4 · `bone(_r)` 3 · `coolwarm` 2 · `plasma` 2 · `YlGnBu` 2 ·
`Reds` / `YlOrBr` / `RdPu` / `YlGn` / `inferno` / `cividis` 各 1 ·
`twilight_shifted` 2。全部感知均匀或单调，无彩虹。
（其中 `cividis` 和一处 `twilight_shifted` 同样是写在常量里的，只有 AST 口径看得见。）
两处需要备案的：

- `twilight_shifted`（`rs_insar_deformation` 的缠绕相位、`rs_validation_scatter`
  的年内日序）画的是**周期量**，循环色表是唯一正确选择，属正当例外。
- `plasma` / `inferno` 感知均匀但饱和度偏高，能换 `magma` 就换。

**给下一个人的规矩**：新图别把 cmap 名字藏进常量再传。要么直接写
`cmap=get_cmap("cb_sequential_blue")`，要么让常量名以 `_CMAP` 结尾，
这样正则和人眼都拦得住。

---

## 3. 字体与字号 (Typography)

### 3.0 字体家族 (Font family)

**本库的正文字体是 Times New Roman。** 这一条以前只存在于 `pubstyle.py` 的
rcParams 里，本文档从未写明，于是文档与实现各说各话了很久。现在以此处为准。

```python
# scripts/pubstyle.py — SERIF_STACK
"font.family": "serif",
"font.serif": [
    "Times New Roman",   # Windows / 装了 MS 核心字体的机器
    "Tinos",             # 仓库自带 fonts/，SIL OFL，度量与 Times 完全一致
    "Nimbus Roman",      # URW 克隆，多数 Linux 发行版自带
    "Liberation Serif",  # 多数 Linux 发行版自带
    "TeX Gyre Termes",   # TeX Live 自带
    "Times",             # macOS
    "STIXGeneral",       # matplotlib 自带，最后防线
],
```

**为什么要自带字体。** Times New Roman 是专有字体，不能随开源仓库分发。没有它
的机器上，matplotlib 会静默回退到 DejaVu Serif——比 Times 宽约 8%，每一个标签
都会重排，你生成的图和 `reference/` 里的对不上，而且**所有既有检查都照样通过**。
所以 `fonts/` 里放了 Tinos：它是为 Times 度量兼容而设计的，字宽逐字相同。
可自行验证：

```bash
python check_typography.py --metrics
#   vs Tinos    identical    (Tinos-Regular.ttf)
```

**硬约束**

- 图脚本里**永远不要写** `rcParams["font.family"]`，只调用一次 `set_pub_style()`。
  `check_typography.py` 会把直写 rcParams 判为 error。
- 字体栈里**故意不含 DejaVu**。`pubstyle.assert_font_ok()` 在解析结果落到
  DejaVu 时直接抛异常，而不是静默降级。唯一例外是统计标注框里的
  `DejaVu Sans Mono`——那里要的是等宽数字，是有意为之。
- 数学用 `mathtext.fontset="stix"`（衬线，与 Times 同源），别改成默认的无衬线，
  否则公式和正文一眼就能看出不是一套字。
- 导出用 `pdf.fonttype=42`（真嵌入轮廓）、`svg.fonttype="none"`（保留可编辑文本）。
- 中文标签调 `set_cn_font()`，西文数字仍是 Times；不要手动改 `font.family`。
  注意它把中文字体写进 **`font.family`** 而不是 `font.serif`——matplotlib 只在
  `font.family` 列出具体字体名时才逐字形回退，走 `serif` 泛型别名那条路汉字会
  变成空心方框，且不报任何错。详见 `WORKFLOW.md` §5。投稿图一律用英文标签。

### 3.1 字号阶梯 (Type scale)

字号的单一事实源是 `pubstyle.TYPE_SCALE` 这个 dict，不是本表——本表是它的说明。

`set_pub_style(base_font=9)` 已经把 `axes.titlesize=10` / `axes.labelsize=9` /
`xtick.labelsize=8` / `legend.fontsize=8` 设好了。**能靠默认值就别写
`fontsize=`。** 需要显式指定时，只用下面这把尺子：

| 层级 | 字号 | 加粗 | `TYPE_SCALE` 键 | 说明 |
|---|---|---|---|---|
| 图头主标题 | **11.0** | `fontweight="bold"` | `header_title` | 全图唯一，见 §5 |
| 图头副标题 | **8.0** | 常规 | `header_subtitle` | 颜色 `#555555` |
| 面板标签 `(a)` | **11** | bold | `panel_label` | 见 §4，用默认值，别改 |
| 面板标题 `set_title` | **9.0** | bold | `panel_title` | 短句，见 §5。用 `panel_title()` 别手写字号 |
| 轴标签 | **8.0** | 常规 | 必带单位 |
| 刻度标签 | **7.0 – 7.5** | 常规 | 全库最高频档（7.0 出现 371 次、7.5 出现 197 次） |
| 图例 | **7.0 – 7.5** | 常规 | 与刻度同级 |
| 图内注释 / 统计框 | **6.6 – 7.0** | 常规 | `annotate_box()` 默认 `font.size - 1` |
| 密集矩阵格内数字 / 等值线标注 | **6.0 – 6.5** | 常规 | **6.0 是全库硬下限** |
| 图底脚注 | **6.5** | 常规 | 颜色 `#777777` |

**硬约束**

- **下限 6.0 pt。** 低于 6 pt 在 300 dpi 下缩到 contact sheet 尺寸就糊了。
  这条曾经宣称"零残留"，但那是基于只覆盖编号 ≤192 的旧扫描器
  （`_apply_unity_fixes.py` 硬性截断在 193）。`check_typography.py` 全库重扫后
  查出 **168 处** 5.4–5.9 pt 的漏网之鱼，已统一抬到 6.0。现在归零是真的，
  并且由 CI 持续守着：

  ```bash
  python check_typography.py --source-only          # 低于下限 = error
  python check_typography.py --source-only --fix    # 机械抬到 6.0
  ```

  **注意间接写法**：`def chip_legend(..., fs=5.8)` 这种"函数默认参数 →
  `fontsize=` 右值是标识符"的链条，字面量扫描看不见（同 §2.4 那批 cmap 的病根），
  自查时要一起查。
- **上限 12.5 pt**，仅 `suptitle` 和仪表盘式大数字可用。正文任何元素不得超过 11。
  重扫时还有 29 处超过 12.5：绝大多数是 `263–272` 架构图组和几张仪表盘图里的
  13.5–15.5 pt 标题，以及 `00` / `92` / `244` 的 19–22 pt 大数字。属于已知例外，
  但新图别再扩大这份名单。
- 只有图头主标题、面板标签、面板标题用 `bold`。**轴标签、刻度、图例一律不加粗**。

---

## 4. 面板标签约定 (Panel labels)

这是全库目前最统一的一项：147 个脚本共 288 次 `add_panel_label`，其中写成字面量
的 199 个标签**全部**是 `(a)` 小写加半角圆括号，`(A)` / `a)` / `A` 一个都没有。
**照抄即可，不要发挥**：

```python
from pubstyle import add_panel_label, letter_panels

add_panel_label(ax, "(a)")            # 单个
letter_panels(axes)                   # 整个 axes 数组，自动 (a)(b)(c)...
```

| 项 | 值 |
|---|---|
| 写法 | `(a)` —— 小写字母 + 半角圆括号。**不用** `a)` / `A` / `(A)` / `a.` |
| 位置 | `x=-0.12, y=1.05`（axes 坐标，即左上角外侧） |
| 字号 | `size=11` |
| 字重 | `weight="bold"` |
| 颜色 | 继承默认墨色 |
| 编号顺序 | 先行后列（reading order） |

**偏移什么时候可以改**：只在 y 轴标签特别长、或该面板没有 y 轴刻度时。
调整幅度限制在 `x ∈ [-0.20, -0.08]`、`y ∈ [1.01, 1.06]`，并且**同一张图里所有
面板用同一个偏移值**——参差不齐比偏一点难看得多。

跨列的宽面板（`gridspec` 里 `colspan > 1`）：`x` 按宽度反比缩放，
`x_wide ≈ -0.12 / colspan`，这样标签到画布左边缘的**物理距离**才一致。

**新图直接用英寸定位，别再手算这个反比。** `panel_labels_inches()` 按英寸偏移，
跨列面板自动对齐，不需要 `colspan` 修正：

```python
from pubstyle import panel_labels_inches
panel_labels_inches(fig, axes)              # 默认 dx=-0.34, dy=0.10 英寸
```

三个 helper（`add_panel_label` / `letter_panels` / `panel_labels_inches`）现在
共用同一份实现。前两个的默认单位仍是轴分数，签名和默认值都没动，既有图渲染
结果不变；传 `units="inches"` 即可切到英寸语义。

---

## 5. 标题规范 (Titles)

### 5.1 图头：用 `add_figure_header()`，不要用 `suptitle`

**本节曾经规范的是 `fig.suptitle(y=0.985)`。那份规范是死的。**
`_strip_fig_chrome.py` 当年删光了全库的 `fig.suptitle`，此后 1219 个
`fig_*.py` 里 `suptitle` 出现次数为 **0**，也就是说本文档描述了一个没有任何代码
在执行的标题层。WM3/WM4/WM5 三轮批次审计连续要求补一个统一的图头 helper，
现在它落地了：

```python
from pubstyle import add_figure_header, header_height, panel_labels_inches

head = header_height(fig, subtitle=True)          # 先算出需要多少留白（英寸）
fig.tight_layout(rect=[0, 0.03, 1, 1 - head / fig.get_size_inches()[1]])
add_figure_header(
    fig,
    "Loop closure removes unbounded drift",       # 主标题：说结论
    subtitle="n = 240 trajectories, 8 sequences", # 副标题：样本量 / 单位 / 版本
    footer="Synthetic data, seed=0",              # 页脚：出处
    corner="KITTI-08")                            # 角标：数据集 tag
panel_labels_inches(fig, axes)
```

| 层 | 字号 | 颜色 | 字重 |
|---|---|---|---|
| 主标题 | 11.0 | `#1a1a1a` | bold |
| 副标题 | 8.0 | `#555555` | 常规 |
| 角标 | 8.0 | `#777777` | 常规 |
| 页脚 | 6.5 | `#777777` | 常规 |

**为什么用英寸定位。** 图头按距画布边缘的英寸数落位，所以 3.5 in 单栏图和
12.6 in 仪表盘上，标题距边缘的**物理距离**一致。轴分数偏移做不到这点——面板
字母的 `x` 从 `-0.045` 一路飘到 `-0.155`，病根就在这儿。

**先留白，再盖章。** `header_height()` 默认把面板字母那一行也算进去
（`panel_labels=True`）。少算这一行，`(a)` 就会直接压在副标题上——这正是历次
审计反复报告的那类缺陷。单面板无字母的图传 `panel_labels=False`。

### 5.2 措辞

- **主标题说结论，不复述图表类型**：
  ✅ `Loop closure removes unbounded drift`
  ❌ `SLAM trajectory plot`
  并且结论必须是图上数据支持的。审计里反复出现"标题说 still accelerating，
  柱子却是 730→725→745"这种，属于硬伤。
- **统一 Sentence case**：只有首字母和专有名词/缩写大写。
  ✅ `Path-length optimality across all successful runs`
  ❌ `Path-Length Optimality Across All Successful Runs`
  `check_typography.py` 会把疑似 Title Case 报为 warning（专有名词会误报，
  人工判断）。
- **标题不写句号。**
- 面板标题短于 8 个词；放不下的信息移到轴标签或脚注。
- 面板标题用 `panel_title(ax, "...")`，别手写 `fontsize=`——全库曾经
  8.5 / 8.6 / 9.0 三个值混用于同一角色。

---

## 6. 图例与色条 (Legend & colorbar)

### 6.1 图例

- **`frameon=False` 是默认，也是首选**（全库 233 处显式 `False` vs 35 处 `True`）。
- 位置优先级：
  1. `loc="upper center"` + `bbox_to_anchor` 放到**画布顶部**、`ncol=N` 排成一行
     —— 多面板共享一套图例时**只画一次**，这是本库的招牌做法。
  2. `loc="best"` 之外的**显式** `loc`：`upper left` → `lower right` → `upper right`。
  3. 直接给曲线**端点打标签**（`ax.annotate`），完全不要图例——线数 ≤ 4 时最优。
- 只有当图例必须压在填充色块/影像上时才用边框，此时用统一规格：
  ```python
  ax.legend(frameon=True, framealpha=0.9, facecolor="white",
            edgecolor="#cccccc", borderpad=0.4)
  ```
  （`pubstyle` 已把 `legend.framealpha/facecolor/edgecolor` 设成这套值。）

### 6.2 色条

**新图一律走 helper**，不要裸调 `fig.colorbar`（当前 114 个脚本裸调、14 个用
helper，比例三轮没动过）。

> **但存量的 114 处不要为了统一去批量替换。** 收口审计把全库 128 个含色条的脚本
> 全部渲染出来、按 `cax._colorbar.mappable.axes` 找到真正的父轴逐个量了尺寸：
> **§12.4 那种"色条飘在图外"的真实缺陷已经一处不剩**（静态扫出的 5 个嫌疑全部
> 实测 gap ≤ 0.10 in、长度差 ≤ 0.02 in），剩下的差异都只是 API 风格。
> 拿 114 处的回归风险换零视觉收益，不划算。

标准参数（`add_colorbar` 的默认值，不要覆盖）：

| 参数 | 值 |
|---|---|
| `fraction` | `0.046` |
| `pad` | `0.03` |
| `shrink` | `0.92` |
| `aspect` | `22` |
| 轮廓 | `linewidth=0.6`, `edgecolor="#333333"` |
| 刻度 | `labelsize = font.size - 1`, `width=0.7`, `length=2.5` |

- `label` **必填**，格式 `量名 (单位)`，小写开头，例如 `von Mises stress (MPa)`、
  `traversal cost (-)`。无量纲写 `(-)`。
- 多面板共享同一色标时，色条**只画一个**，用 `ax=axes.ravel().tolist()` 挂在整行/
  整列上；不要每个面板配一个。
- 横向色条用 `orientation="horizontal", fraction=0.06, pad=0.10`。

**⚠️ `aspect=22` 在矮面板上会把色条压成发丝。** `aspect` 锁的是"长 : 宽"，
所以色条**宽度 = 长度 / aspect**，而长度跟着父轴走。默认值是照 3 in 高的面板
调的（→ 0.13 in 宽，合适）；挂到 1 in 高的附属面板上就只剩 **0.045 in ≈ 3 pt**。
`343_vlm_image_text_align` 的矩阵面板就是这么来的，收口时改成 `aspect=9`。

**别只看 `fraction`/`pad` 的数值判断粗细。** 它们是相对父轴的比例：挂在
`ax=axes` 整组上时 `fraction=0.020` 完全正常，挂在单轴上就偏细。
唯一可靠的口径是**实测英寸**——本库实测区间 0.13–0.18 in，越出这个区间才要管：

```python
cax = cb.ax.get_position()
print(cax.width * fig.get_size_inches()[0])   # 竖条厚度，单位 in
```

---

## 7. 留白与间距 (Spacing)

### 7.1 常用值

```python
fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.6, w_pad=2.6)
```

| 场景 | 参数 |
|---|---|
| 无 suptitle、无脚注 | `fig.tight_layout()` |
| 有脚注 | `rect=[0, 0.035, 1, 1]` |
| 有 suptitle | `rect=[0, 0, 1, 0.95]` + `suptitle(y=0.985)` |
| 两者都有 | `rect=[0, 0.03, 1, 0.95]` |
| 多面板需要喘息 | 追加 `h_pad=2.6, w_pad=2.6`（全库最高频组合） |
| 面板要贴紧（共享轴） | `h_pad=0.6, w_pad=0.6` + `sharex/sharey=True` |

`savefig` 时 `bbox_inches="tight"` + `pad_inches=0.04` 已由 `pubstyle` 设定，
**不要在脚本里再传**。

### 7.2 背景分区块

用浅灰块把画布分成语义区（如"输入 | 方法 | 结果"）时：

```python
fig.patches.append(Rectangle((x0, y0), w, h, transform=fig.transFigure,
                             facecolor="#f5f5f5", edgecolor="none", zorder=0))
```

只用 `#f5f5f5` / `#f0f0f0` 两档，**不加边框**，`zorder=0`。区块之间留
≥ 0.02 figure 坐标的缝。

---

## 8. 层级表达 (Visual hierarchy)

一张图里所有面板同样大 = 没有主次 = 读者不知道看哪。

- **主面板 : 附属面板 = 1 : 0.55–0.60**（面积或高度比）。用 `gridspec` 的
  `width_ratios` / `height_ratios` 实现：

  ```python
  gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.58, 0.58],
                        height_ratios=[1.0, 0.58])
  ```

- 附属面板的字号跟着降一档（刻度 7.0 → 6.5），但**不得低于 6.0**。
- 主面板承载"结论"，附属面板承载"证据/敏感性/统计量"。
- 插图（inset）用 `ax.inset_axes([x, y, w, h])`，`w/h` 取 0.28–0.40，
  边框 `linewidth=0.8`、颜色 `#333333`。

---

## 9. 共享轴规范 (Shared axes)

**同类面板必须共享轴范围。** 这是"看起来像一套东西"里最容易被忽略、代价最高的
一条——四个并排的收敛曲线如果各有各的 y 轴上限，读者会得到完全错误的印象。

**必须共享的情形**

- 同一物理量在不同条件/组别下的重复面板（时间序列、收敛曲线、分布）。
- 网格里同一行/同一列的面板。
- 前后对比（before/after、T1/T2）。

**可以不共享**：量纲不同、或跨越多个数量级且已各自用 log 轴。

**实现方式**：先遍历全部数据求全局范围，再统一设定——不要靠 `sharey=True`
碰运气，因为它只在 `plt.subplots` 里有效，`gridspec` 手搓的面板管不到。

```python
vals = np.concatenate([d for d in series.values()])
lo, hi = np.nanmin(vals), np.nanmax(vals)
m = 0.05 * (hi - lo)
for ax in axes.ravel():
    ax.set_ylim(lo - m, hi + m)
```

热图/影像共享色标同理——先求全局 `vmin/vmax`，再传给每个 `imshow`：

```python
vmin, vmax = np.nanmin(stack), np.nanmax(stack)
for ax, img in zip(axes, stack):
    im = ax.imshow(img, cmap=get_cmap("cb_sequential_blue"),
                   vmin=vmin, vmax=vmax)
```

内侧面板隐藏重复的刻度标签：`ax.tick_params(labelleft=False)`，
但**保留刻度线**，否则读者会以为轴不同。

---

## 10. 新图完成前检查清单

提交前逐条勾选。任何一条打叉，都会让这张图在 contact sheet 上"跳出来"。

**结构**
- [ ] 第一行调用了 `set_pub_style()`，全脚本没有直接写 `rcParams`
- [ ] `figsize` 取自 §1.1 四档之一，`h/w` 落在 0.55–0.80，或**能报出 §1.2 里
      E1–E8 中的哪一条**（说不出代号就不是例外，是超标）
- [ ] 通过 `pubstyle.savefig(fig, "NN_name.png")` 输出

**颜色**
- [ ] 没有 `tab10` / Flat-UI / 霓虹色（对照 §2.3 搜一遍 hex）
- [ ] 没有 `turbo` / `jet` / `hot` / `RdYlGn` / `Spectral`——**连
      `HEAT = "jet"` 这种先存进常量再用的写法一起查**（§2.4）
- [ ] 定性色按 §2.1 序位取用，类别 ≤ 5
- [ ] 发散色做了对称归一化：优先 `figkit.shared_norm(..., symmetric=True)`，
      用 `TwoSlopeNorm` 时刻度不得越界（§12.5）
- [ ] 全图只有一个强调对象

**文字**
- [ ] 没有任何 `fontsize < 6.0`——**包括**藏在函数默认参数和模块常量里的
      （`def f(..., fs=5.8)` 这类，正则扫不到）
- [ ] 正文元素没有 `fontsize > 11`（`suptitle` ≤ 12.5）
- [ ] 只有 suptitle / 面板标签 / 面板标题加粗
- [ ] 标题是 Sentence case，无句号
- [ ] 每个轴标签都带单位

**构件**
- [ ] 面板标签是 `(a)`，`x=-0.12, y=1.05, size=11, bold`，全图偏移一致
- [ ] 图例 `frameon=False`；多面板共享图例只画一次
- [ ] 色条走 `add_colorbar()`，`label` 已填且带单位
- [ ] 同类面板已共享轴范围 / 色标范围（§9）
- [ ] 主辅面板比例 ≈ 1 : 0.58（§8）

**收尾**
- [ ] `tight_layout` 参数取自 §7.1
- [ ] 脚本 exit code 0
- [ ] **重跑 `python scripts/build_reference.py`，在 `reference/thumbs/` 里并排确认这张图不跳脱**

---

## 11. 自动化自检

```powershell
cd scripts
python check_typography.py --source-only   # 字号阶梯、rcParams 逃逸、标题、面板字母
python check_figures.py --quick            # 已有 PNG 的尺寸、DPI、体积
python build_reference.py                  # 重建 reference/thumbs/，然后用眼睛看
```

以下是建库期间的记录。`_style_scan.py`、`_style_scan2.py`、`_apply_unity_fixes.py`
是那时的一次性工具，不随本包分发；留着这几段是因为结论（正则口径会漏）仍然成立。

`_apply_unity_fixes.py` 只做**可逆的机械替换**（违禁色 → house 色、
`fontsize` 抬到 6.0 下限、彩虹 cmap → house ramp），并自动跳过编号 ≥ 193
以及尚未渲染出 PNG 的脚本。新图批量入库后跑一次即可。

⚠️ 那个 `≥ 193` 的截断是首轮审计时写死的，当时图库到编号 192 为止。现在图库有
354 张（编号 00–353），**这个工具只覆盖其中 193 张、约 54%**；`193–353` 段靠的是
后两轮人工统一收编，不是它。要用它跑新段落，得先放宽那个上界。

⚠️ **更要紧的是：`_style_scan.py` / `_style_scan2.py` / `_apply_unity_fixes.py`
三个工具全是正则口径，只认字面量。** 值先赋给标识符再传进去就完全隐身——
本轮就是靠这条缝隙同时藏住了 1 处 5.8 pt 字号和 5 个彩虹 cmap，而三轮审计都报了
"零残留"。它们跑出来的"0"只能证明**字面量**干净。要真的确认，得按 AST 回溯
`fontsize=` / `cmap=` 右值的标识符（做法见 `_audit_final_pass.md`）。

---

## 12. matplotlib 图层与裁剪陷阱 (Layering & clipping traps)

本章的十一条不是从文档里抄的，是本轮 30 多个代理各自撞出来、然后在源码里留下
绕行代码的坑。共同特征是：**不报错、不警告，只是画错**，而且大多只在
"多面板 / equal aspect / 多图例" 这类稍微复杂一点的图上才发作，
单面板 demo 永远复现不出来。（§12.10 是唯一的例外：它**会**报警，
但报的恰恰是一条假警报——见该节。）

所有解法都已经封装进 `scripts/figkit.py`，能用 helper 就别手搓。
本章的版本基准是 **matplotlib 3.10.8**（下述现象均已在该版本上实测复现）。

---

### 12.1 `add_artist()` 加的第二个图例会被从中间切断

**现象**　一张图有两个图例（比如一个说颜色、一个说线型），导出后**第二个**
图例的文字被竖直切掉一截，像是被一把看不见的刀削过。只有一个图例时永远正常。

**成因**　两件事叠加。`ax.legend()` 返回的图例被 axes 特殊登记，
而用 `ax.add_artist(leg)` 挂上去的那个只是一个普通子 artist：
它**继承了 axes 的 clip path**，超出轴框的部分会被裁掉；同时
`bbox_inches="tight"` 在算裁剪边界时**不把它计入**，于是画布还会朝它的方向收紧。
两个效应方向相同，合起来就是"从中间切断"。

**解法**　清掉图例本身、它的文字和图元的 clip，并把它重新计入 layout。

```python
from figkit import unclip_legend

leg1 = ax.legend(h1, l1, loc="upper left")
leg2 = ax.legend(h2, l2, loc="lower right")   # 这一句会顶掉 leg1
ax.add_artist(leg1)                            # 所以要手动加回去
unclip_legend(leg1)                            # <- 少了这行就会被切
```

> 顺带一提：`ax.legend()` 第二次调用会**替换**掉第一个图例，
> 所以"两个图例"的写法必然要走 `add_artist`，也就必然会踩这个坑。

---

### 12.2 figure 级背景块会盖住它本想衬托的面板

**现象**　用浅灰块给面板分组（"输入 | 方法 | 结果"），结果灰块把面板整个糊住了，
只剩一片灰。把 `zorder` 设成 `0` 没用。

**成因**　`zorder` 相同时 matplotlib 用**稳定排序**，也就是按
`fig.get_children()` 里的先后决定谁后画。这个列表的顺序是：

```
fig.patch -> fig.artists -> fig.axes -> fig.lines -> fig.patches -> fig.texts -> ...
```

axes 的默认 `zorder` 正好也是 `0`。于是（实测）：

| 添加方式 | 同为 zorder=0 时 | 结果 |
|---|---|---|
| `fig.patches.append(rect)` | 排在 axes **之后** | **盖在面板上** |
| `fig.add_artist(rect)` | 排在 axes **之前** | 在面板下面 |

也就是说 `zorder=0` 不是"最底层"，它是"平局"，而平局的裁判恰好判了背景块赢。
§7.2 早期给的 `fig.patches.append(..., zorder=0)` 写法正是踩雷版本。

**解法**　用 `fig.add_artist` **并且**给严格负的 zorder（双保险），
同时把它移出 layout 计算，免得 `bbox_inches="tight"` 被它撑大。

```python
from figkit import background_card

background_card(fig, (0.04, 0.06, 0.42, 0.40),      # figure 分数坐标
                facecolor="#f5f5f5", zorder=-5, radius=0.012)
```

手写版本必须是：

```python
rect = Rectangle((x0, y0), w, h, transform=fig.transFigure,
                 facecolor="#f5f5f5", edgecolor="none", zorder=-5)
fig.add_artist(rect)
rect.set_in_layout(False)
```

---

### 12.3 `inset_axes` 会被父 axes 的数据 artist 盖住

**现象**　给影像加了放大插图，插图却被主图的 `imshow` / 散点 / 填充盖掉，
或者只露出一个边框。

**成因**　`ax.inset_axes()` 返回的子 axes 默认 `zorder=0`，
而父 axes 里 `imshow` 的 `AxesImage` 是 `0`、patch 是 `1`、`plot` 是 `2`、
`scatter` 是 `3`。子 axes 和这些数据 artist 在同一个排序队列里，
所以只要主图里有任何 `zorder >= 0` 的东西压过来，插图就被埋了。

**解法**　显式把插图抬到所有数据之上（`figkit.add_zoom_inset` 已经这么做）。

```python
axi = ax.inset_axes([0.58, 0.02, 0.40, 0.40])
axi.set_zorder(7)              # 高于主图任何数据 artist
axi.patch.set_visible(True)    # 否则插图背景透明，主图会透上来
```

---

### 12.4 `fig.colorbar(ax=...)` 在 equal-aspect 图上必然跑偏

**现象**　给一张 `set_aspect("equal")` 的影像/地图配色条，色条既比影像高一截，
又飘在影像右边一大段空白之外，看起来根本不属于这个面板。

**成因**　`fig.colorbar(mappable, ax=ax)` 走 `make_axes`，
它是从 axes 的**名义矩形**（你传给 `add_axes` / `gridspec` 的那个）里切一刀，
而不是从 equal aspect 缩小后**实际绘制**的那个框里切。实测（6×3 画布、
axes 名义框 `[0.08, 0.12, 0.80, 0.80]`、20×20 影像）：

```
影像实际绘制框 : (0.28,   0.12, 0.40,   0.80)
色条落点       : (0.8432, 0.12, 0.0368, 0.80)   <- 离影像右缘 0.16 ~ 1 inch
```

而且色条高度取的是名义高度，比影像高。更糟的是名义框被切窄之后
equal-aspect 影像会**重新居中**（0.28 -> 0.4192），两者又被推得更开一点。

**解法**　先 `apply_aspect()` 拿到真实框，再把色条按**英寸**钉在它旁边。

```python
from figkit import safe_colorbar

im = ax.imshow(field, cmap=get_cmap("cb_sequential_blue"), norm=norm)
ax.set_aspect("equal")
safe_colorbar(fig, im, ax, label="NDVI (-)", width=0.11, pad=0.07)
```

普通（非 equal-aspect）图继续用 `pubstyle.add_colorbar()`，不需要换。

---

### 12.5 `TwoSlopeNorm` + 越界刻度 = 整条 colorbar 静默消失

**现象**　发散色条上原本该有 `-2 -1 0 1 2` 五个刻度，导出后**一个标签都没有**，
甚至整条色条看起来是空的。没有报错，没有 warning。

**成因**　`TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)` 对定义域外的值返回
`±inf` 而不是抛错：

```python
TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)(np.array([-2., 0., 2.]))
# -> array([-inf,  0.5,  inf])
```

色条把刻度位置喂给这个 norm 做坐标变换，`inf` 进了 transform，
整条刻度轴的定位就崩了。实测 `cb.get_ticks()` 仍然老实返回 `[-2 -1 0 1 2]`
（所以你自查时看不出问题），但 `cb.ax.get_yticklabels()` 返回**空列表**——
标签在渲染阶段被静默丢弃。

**解法**　两条，任选：

```python
# A. 刻度必须落在 [vmin, vmax] 内，且先算范围再定刻度
lo, hi = -m, m
cb.set_ticks([t for t in ticks if lo <= t <= hi])

# B. 更省事：不用 TwoSlopeNorm，用对称的普通 Normalize
from figkit import shared_norm
norm = shared_norm(arrays, symmetric=True)   # -> Normalize(-m, +m)
```

方案 B 同样保证白色中点落在 0 上（§2.1 的要求），但没有越界失效这个雷。
只有**中点不在数据中位、且两侧量级差很大**时才真的需要 `TwoSlopeNorm`。

---

### 12.6 `mplot3d` 的散点不会被坐标轴范围裁剪

**现象**　3D 图设了 `set_xlim/ylim/zlim`，超范围的点却照样画出来，
飘在立方体外面，读者以为那是数据。

**成因**　`clip_on` 默认是 `True`，但它裁的是**二维 axes 矩形**，
不是三维数据盒。超范围的点经过投影以后落点往往还在那个矩形内，
裁剪条件不成立，点就被画出来了。3D 的轴范围只影响刻度和投影矩阵，
根本不参与裁剪。

**解法**　**在传给 `scatter` 之前自己过滤数据**，别指望 matplotlib。

```python
m = ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1))
ax.scatter(x[m], y[m], z[m], s=6, depthshade=False)
```

同理 `plot3D` 的折线也不裁；跨界的线段要自己打断成 `NaN`。

---

### 12.7 3D axes 即使 `set_axis_off()` 仍然打一个白洞

**现象**　把一个 3D 面板叠在一张浅色底板/背景卡片上，已经
`ax3d.set_axis_off()` 了，底板还是被挖掉一块白的。

**成因**　`set_axis_off()` 只关掉坐标轴装饰，**不动 axes 的背景 patch**。
实测 3D axes 在 `set_axis_off()` 之后：

```
ax.patch.get_visible()   -> True
ax.patch.get_facecolor() -> (1.0, 1.0, 1.0, 1.0)     # 不透明白
```

**解法**　背景 patch 要单独关，而且三个面板（pane）也要单独清。

```python
ax.set_axis_off()
ax.patch.set_alpha(0.0)          # 或 ax.patch.set_visible(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_visible(False)
```

挖洞的是 axes patch，不是 figure patch，所以 `savefig(transparent=False)`
照旧，不用改。

---

### 12.8 `Circle` 在非等比 axes 上会被拉成椭圆

**现象**　架构图里的 `+` 残差符号、流程图里的节点圆，在有些面板上是正圆，
在另一些面板上是明显的椭圆。

**成因**　`Circle` 的半径是**数据单位**。只有 `ax.set_aspect("equal")`
（x、y 两个方向的数据-像素比相同）时它才是正圆；普通图轴的比例由数据范围决定，
圆自然被拉扁。

**解法**　两选一：

```python
# A. 画在英寸画布上（数据单位=英寸，天然等比）—— 架构图一律走这条
from figkit import inch_canvas, op_circle
fig, ax = inch_canvas(7.2, 3.6)
op_circle(ax, 3.1, 1.8, "+", r=0.135)

# B. 必须画在普通数据轴上时，改用固定 pt 的 marker，尺寸与数据范围无关
ax.plot([x], [y], marker="o", ms=9, mfc="white", mec="#333333", mew=1.0)
```

不要用"调 `width/height` 凑一凑"的 `Ellipse` 补救——换个数据范围就又歪了。

---

### 12.9 `inset_axes` 的比例是在轴框被 aspect 缩小**之后**才生效的

**现象**　正方形图块上放 `inset_axes([x, y, 0.4, 0.4])`，插图框是正方形，
但塞进去的裁剪窗口被拉变形；或者反过来，插图里出现上下黑边（letterbox）。

**成因**　`inset_axes` 的 `(w, h)` 是**父 axes 实际绘制框**的分数，
不是名义框的分数。所以在正方形图块上 `w == h` 确实给出正方形插图——
这一步没问题。问题在下一步：`imshow` 默认 `aspect="equal"`，
如果**裁剪窗口本身不是正方形**，它就会在正方形插图里居中留白。
把 `aspect` 改成 `"auto"` 能填满，但非正方形裁剪窗会被**拉伸变形**。

一句话：**正方形图块配正方形插图，就必须配正方形裁剪窗。**

**解法**　让裁剪窗口的像素长宽比 = 插图框的纸面长宽比，再用 `aspect="auto"`
填满：既不留白也不变形。`figkit.add_zoom_inset(match_aspect=True)` 会自动
把裁剪框以中心为基准扩到匹配比例（并 clamp 在图像边界内）。

```python
from figkit import add_zoom_inset

add_zoom_inset(ax, img, box_xywh=(96, 22, 52, 40),   # 比例不对没关系
               inset_loc=(0.58, 0.02, 0.40, 0.40),
               color="#e8352a", match_aspect=True)
```

另外两个必须记住的细节：ROI 矩形要画在 `(x0-0.5, y0-0.5)`
（`imshow` 把像素 (0,0) 的**中心**放在数据坐标 (0,0)，
画在整数角点就会偏半个像素），
以及**同一行所有方法必须用同一个裁剪窗**，否则这排对比图没有任何意义。

---

### 12.10 contour 的 rasterize：先 `AttributeError`，再假警报

这一条有**两层**坑，第二层比第一层隐蔽得多。

**第一层 · `.collections` 没了**　老代码
`for c in cs.collections: c.set_rasterized(True)` 在 3.10 上直接
`AttributeError`。3.8 起 `ContourSet` 本身就是一个 `Collection`，
`.collections` 在 3.8 弃用、**3.10 移除**。实测
`hasattr(cs, "collections") -> False`。改成在 contour set 对象上直接设即可。

**第二层 · 改对了照样警告**　换成官方推荐的 `cs.set_rasterized(True)` 之后，
3.10.8 会对**每一次调用**丢一条

```
UserWarning: Rasterization of '<matplotlib.contour.QuadContourSet ...>'
will be ignored
```

**这条警告是假的。** `Artist.set_rasterized` 靠内省判断能不能栅格化：
它查 `getattr(self.draw, "_supports_rasterization", False)`，
而这个标记是 `@artist.allow_rasterization` 装饰器打上去的。
`ContourSet` 重写了 `draw()` 却**没有重新套这个装饰器**，
于是内省查不到标记就报警——但重写版最终仍然调用被装饰过的
`Collection.draw`，栅格化其实**照常生效**。实测（3.10.8，8 级 contourf 存 SVG）：

| | SVG 体积 | 内含 `<image>` |
|---|---|---|
| 不设 rasterized | 62 KB | 0 |
| `set_rasterized(True)`（并报警） | 14 KB | 1 |

有填充 hatch 的分支结论相同。所以**不要因为这条警告就把 rasterize 去掉**，
那才是真的丢掉了栅格化。

**解法**　该设还是设，只把这条已知的假警报就地屏蔽掉：

```python
import warnings

def rasterize(artist):
    """3.10 的 ContourSet 会误报 'rasterization will be ignored'。"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)

cs = ax.contourf(X, Y, Z, levels=12, cmap=get_cmap("cb_sequential_blue"))
rasterize(cs)                        # 不要再遍历 .collections
cs.set_edgecolor("face")             # 消除矢量导出时等值线之间的白缝
```

参考实现见 `scripts/fig_vector_field.py`（编号 95）。
矢量导出（PDF/SVG）时 `contourf` 一定要 rasterize，否则文件动辄几十 MB，
而且在阅读器里会显现出网格状白缝。只输出 PNG 时它是空操作，留着也无害。

---

### 12.11 长脚注把画布撑大：`bbox_inches="tight"` 是扩展，不是裁剪

**现象**　源码里写着 `figsize=(10.4, 10.4)`，导出的 PNG 却有 14.8 in 宽。
一次全库像素扫描命中 14 张这样的图，实际宽度比标称宽 **12 %–42 %**。
后果不是内容溢出被切掉，而是**每个面板被按比例压小**：画布横向被撑开，
面板占的份额随之变小，字号相对于画幅也跟着缩水。单看一张不容易察觉，
和同批次的图排在一起时，这几张明显"虚"一号。

**成因**　`savefig.bbox = "tight"`（`pubstyle.set_pub_style` 的默认值）的语义是
**把画布扩展到能装下所有 artist**，而不是把越界的内容裁掉。
于是任何伸到标称画布之外的元素都会把导出尺寸顶大：

| 成因 | 典型写法 | 本轮 13 张命中 |
|---|---|---|
| 长单行脚注 | `fig.text(0.5, 0.008, "……很长的一整句……")` | 11 |
| 越界的左对齐标题 | `ax.set_title(..., loc="left", x=-0.22)` | 1 |
| 当副标题用的宽横幅 | `fig.text(0.5, 0.945, "……")` | 1 |
| 推到画布外的图例 | `bbox_to_anchor` 超出 `(0, 1)` 区间 | 0 |

绝大多数是第一种。脚注在源码里靠隐式字符串拼接写成好几行，
读代码时看着很短，**渲染出来却是一整行**——这正是它藏得住的原因。

**为什么源码审计查不出来**　`figsize` 只是声明值，越不越界取决于**渲染之后**
的文本外框，只有比对像素才看得见。自查方法是把导出宽度换算回英寸，
再和声明值相除：

```python
from PIL import Image

px_w, _ = Image.open("output/146_algo_boxplot_stats.png").size
ratio = (px_w / 300.0) / 10.4        # 300 = savefig.dpi, 10.4 = 声明的宽
```

`ratio > 1.12` 即判越界。全库批量版见 `scripts/_scan_canvas_inflation.py`，
在 `scripts/` 下直接 `python _scan_canvas_inflation.py` 即可。
要定位**具体是哪个 artist** 顶出去的，画完之后逐个比对它的 tight bbox：

```python
fig.canvas.draw()
r = fig.canvas.get_renderer()
W, _ = fig.get_size_inches()
for a in fig.get_children():
    b = a.get_tightbbox(r).transformed(fig.dpi_scale_trans.inverted())
    if b.x0 < 0.0 or b.x1 > W:
        print(type(a).__name__, b.x0, b.x1)
```

**四种修法**（按优先级）

1. **长脚注换行**——首选，也最有效。用 `textwrap.fill(text, width)` 折成 2 行：

```python
import textwrap

fig.text(0.5, 0.008, textwrap.fill(
         "Boxes span the inter-quartile range with the median as a black "
         "rule; whiskers reach 1.5 IQR. ...", 150),
         ha="center", va="bottom", fontsize=7.4, linespacing=1.45)
```

`width` 的量级取 `(figsize_w - 1.0) * 182 / fontsize`
（Times New Roman 平均字宽 0.395 em，约合 `182 / fontsize` 字符每英寸。
本节旧值 `145` 是按 DejaVu Sans 的 0.499 em 算的——那是文档描述的字体，
不是代码实际用的字体，于是这里一直把可容纳字符数低估了约 26%）；
这是上界，**宁可折窄不要折宽**，左对齐的脚注还要再扣掉起始 x 偏移。
句子里含 mathtext 的改用手写 `\n` 断行，别让 `textwrap` 从公式中间切开。

2. **轴外标注收回**——`annotate` / `text` 跑到 axes 之外的，挪回画布内，
或者加 `clip_on=True`。左对齐标题的负 `x` 偏移是重灾区：
`loc="left", x=-0.22` 落在 6.7 in 宽的 axes 上就是往左伸出 1.5 in。

3. **图例位置**——被 `bbox_to_anchor` 推到画布外的图例，改放进图内空白区，
或者减小 `fontsize`、增大 `ncol` 让它横向收窄。

4. **suptitle 过宽**——缩短，或在语义断点处手动换行，
例如把参数括号 `($v$ = 24 m/s, $R_{min}$ = 250 m)` 单独放到第二行。

5. **兜底**——内容确实需要那么宽，就把 `figsize` 调大到与实际相符，
让标称值与实际值一致、高宽比重新可审计，并在 docstring 里写明理由。
但**不要**图省事把 `bbox_inches="tight"` 关掉：那才是真的把内容裁掉。

**换行之后必须复查底部间距**　脚注从 1 行变 2 行是**向上长**的，
本轮 13 张里有 5 张因此压到了 x 轴标签上。配套动作有两个：
把 `GridSpec(..., bottom=...)` 抬高 0.03–0.04 给出空间；
`fig.text` 显式写 `va="bottom"`——默认的 baseline 锚点会让第二行
掉到画布下沿之外，等于把横向膨胀换成了纵向膨胀。

本轮 13 张的逐图成因、修法与前后膨胀比，见
`scripts/_audit_canvas_inflation.md`。

---

## 13. 让图不说谎的三条规则 (Three rules against lying figures)

前面 12 章管的是"好不好看"，这一章管的是"对不对"。
一张违反本章的图，越精致越有害——它会让读者更相信一个错误的结论。
**这三条的优先级高于本文档其余所有条款。**

---

### 13.1 同类面板必须共享标尺

**规则**　只要多个面板画的是**同一个物理量**，就先遍历全部数据求统一范围，
再画。任何一个面板都不许自动缩放。

**为什么**　matplotlib 默认把每个面板各自撑满自己的框。于是"响应很弱"和
"响应很强"在图上**长得一模一样**，差异被自动缩放吃掉了。

> **真实反例（本轮出现过）**：一张神经元调谐曲线图，几个极坐标面板里
> 有一个是"未调谐"的对照神经元。它用了自己的径向轴范围，
> 结果这个**本该最平的**对照组在图上反而是看起来响应**最强**的那个，
> 和它正下方的说明文字 "untuned control" 直接矛盾。
> 曲线数值全对，图完全错。

**做法**

```python
from figkit import shared_limits, apply_shared_limits, shared_norm, safe_colorbar

# 折线 / 柱状 / 极坐标：先求统一范围，再统一设定
lo, hi = shared_limits(list(series.values()), pad=0.05)
for ax in axes.ravel():
    ax.set_ylim(lo, hi)

# 或者一行搞定
apply_shared_limits(axes, list(series.values()), axis="y", pad=0.05)

# 影像 / 热图：共享 norm，并且只画一个色条
norm = shared_norm(stack, robust=(2, 98))
for ax, img in zip(axes, stack):
    im = ax.imshow(img, cmap=get_cmap("cb_sequential_blue"), norm=norm)
safe_colorbar(fig, im, axes[-1], label="error (m)")
```

不要靠 `sharey=True` 碰运气：它只在 `plt.subplots` 里有效，
手搓 `gridspec` 的面板它管不到（§9 已述）。

**什么时候该放弃共享**　量级差**一个数量级以上**时，强行共享会把小的那个
压成贴着轴的一条直线——那不叫诚实，那叫用另一种方式抹掉信息。
此时**不要**退回各画各的，而应该改变被画的量，让它重新可比：

```python
# 差： 三条曲线量级 0.01 / 0.3 / 12，共享 y 轴后前两条是两条平线
# 也差：各用各的 y 轴（读者一定会横向比较，然后得到错误结论）
# 好： 改画相对量，量纲统一了，共享标尺自然成立
gain = curve / uniform_baseline          # "相对均匀基线的增益"
ax.set_ylabel("gain over uniform baseline (x)")
ax.axhline(1.0, color="#808080", lw=0.8, ls="--")   # 基线必须画出来
```

其他等效手段：统一取 `log` 轴（此时**更要**共享，否则比线性轴还误导）、
画相对于对照组的 Δ%、或者干脆**拆成两张图**并在标题里说明量级不同。
唯一不可接受的是：量纲相同、量级相近，却各画各的。

---

### 13.2 标注的数字必须从数据实时算出来

**规则**　图上任何一个数字——百分比、峰值、比值、R²、n——都必须由**画图用的
那份数据**在运行时计算并格式化进字符串。**不许**把数字硬写进文案。

**为什么**　硬写的数字和曲线是两条独立的信息流，只要有一条被改过
（换了随机种子、改了模拟参数、调了单位），两者就会失配。
而这种失配**没有任何自动检查能发现**，只有审稿人会发现。

> **真实反例（本轮出现过两处）**：
> 电池图的注释写着 `97.2% capacity recovery`，而图上画出来的曲线终点是
> **86%**；XPS 图的化学计量比是照着文献"拍脑袋"写死的，
> 而不是从图上那几个峰的面积积出来的。

**做法**

```python
# 差
ax.annotate("97.2% capacity recovery", xy=(x[-1], y[-1]))

# 好：数字来自被画的那个数组本身
recovery = y[-1] / y[0]
ax.annotate(f"{recovery:.1%} capacity recovery", xy=(x[-1], y[-1]))

# 好：化学计量比从峰面积积出来
area_a = np.trapezoid(peak_a, energy)
area_b = np.trapezoid(peak_b, energy)
ax.text(..., f"O : Ti = {area_a / area_b:.2f} : 1")
```

自查方法：**在脚本里把随机种子换一个再跑一遍**，
所有标注数字都应该跟着变；任何一个纹丝不动的数字都是硬写的。

顺带三条同源纪律：

- 统计量（p 值、R²、n）必须来自实际检验，别手抄。显著性星号用
  `pubstyle.sig_stars(p)` 从 p 值生成，别自己数星星。
- 图注里的样本量写 `f"n={len(x)}"`，不要写死。
- 拟合线和它的方程必须来自**同一次** `polyfit` / 回归调用。

---

### 13.3 附属面板要用尺寸表达层级

**规则**　主面板和附属面板**不能一样大**。三选一：
① 附属面板占主面板 **55–60%**；② 附属面板独立成行/列；
③ 给附属面板加浅灰背景卡片划成一组。

**为什么**　根因是"**作者心里有主次，但网格几何里没有**"。
等分的 `GridSpec` 是在明明白白地告诉读者"这几块是平级的、请依次同等阅读"，
哪怕作者本意是"左边是结论、右边三个是敏感性分析"。
读者只会相信几何，不会相信作者的本意。

**做法**

```python
# ① 比例法（最常用）：主 1.0，附属 0.58
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.58, 0.58],
                      height_ratios=[1.0, 0.58],
                      hspace=0.28, wspace=0.24)
ax_main = fig.add_subplot(gs[0, 0])          # 结论
ax_aux1 = fig.add_subplot(gs[0, 1])          # 证据
ax_aux2 = fig.add_subplot(gs[0, 2])          # 敏感性

# ② 独立成行：主面板整行跨列，附属面板缩在下面一行
ax_main = fig.add_subplot(gs[0, :])

# ③ 背景卡片分组：几何上是平级，但视觉上被圈成"附录区"
from figkit import background_card
background_card(fig, (0.03, 0.05, 0.94, 0.34), facecolor="#f5f5f5",
                zorder=-5, radius=0.008)
```

配套约束：

- 附属面板的字号跟着降一档（刻度 7.0 → 6.5），但**不得低于 6.0**（§3）。
- **层级只做一层。** "主 → 次 → 次次"三级会退化成没有层级。
- 主面板承载**结论**，附属面板承载**证据 / 敏感性 / 统计量**。
  如果你说不清哪个是主面板，那说明这张图应该拆成两张。
- 面板标签用 `figkit.panel_labels_inches()`：跨列的宽面板用 axes 分数偏移
  会把标签推出双倍距离（§4 的经验值只对等宽面板成立），按英寸算才对齐。

---

### 13.4 提交前的诚实性自查

在 §10 的清单之外，再过一遍这三条：

- [ ] 同一物理量的所有面板，`set_ylim` / `norm` 来自**一次**全局计算（§13.1）
- [ ] 量级差 ≥ 1 个数量级时，已改画相对量或 log 轴，而不是各画各的（§13.1）
- [ ] 换一个随机种子重跑，**每一个**标注数字都跟着变了（§13.2）
- [ ] 显著性星号来自 `sig_stars(p)`，不是手打的（§13.2）
- [ ] 能一句话说出哪个是主面板，且它在几何上确实更大（§13.3）
- [ ] 发散色标的白色中点确实落在 0（或真正的中性值）上（§2.1 / §12.5）
