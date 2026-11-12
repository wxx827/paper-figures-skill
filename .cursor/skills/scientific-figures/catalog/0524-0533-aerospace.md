# 航空航天 · Aerospace  `524-533`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 524 | `fig_prot_contact_map.py` | Residue-residue contact map: reference contacts shaded by C-alpha distance above the diagonal, top-L predicted hits and misses below it, secondary-structure tracks on two sides, dashed domain blocks, and a companion precision-versus-list-depth curve | 接触图预测评估、共进化耦合分析、结构域划分, "接触图, 残基接触, 预测精度, 结构域, β折叠配对" |
| 525 | `fig_prot_folding_funnel.py` | Folding free-energy funnel: a rugged 3-D surface over two conformational coordinates with the native basin at the bottom, automatically detected intermediates and a kinetic trap on the walls, an inward-spiralling folding trajectory, and a companion energy-versus-progress profile with the accessible-energy band | 蛋白折叠能量地形讲解、机理示意主图, "折叠漏斗, 自由能面, 能量地形, 中间态, 天然态" |
| 526 | `fig_prot_secondary_topology.py` | TOPS-style topology cartoon of a Rossmann-like fold: beta strands as sheet-plane arrows, helices as cylinder glyphs, the chain path drawn as curved connectors showing the right-handed beta-alpha-beta crossovers, plus a companion sequence track of the same elements | 折叠拓扑讲解、二级结构排布示意、结构分类, "拓扑图, TOPS, 二级结构, 折叠类型, β片层" |
| 527 | `fig_prot_bfactor_flexibility.py` | Backbone flexibility putty: a 3-D C-alpha tube whose width and house sequential colour both track the B factor so the rigid core reads thin and dark, plus companion panels for B factor along the sequence with the three most flexible stretches marked and the B distribution by secondary-structure type, all on one shared scale | 结构柔性分析、温度因子着色、动力学热点标注, "B因子, 柔性, 温度因子, 主链着色, 柔性环" |
| 528 | `fig_sc_umap_clusters.py` | UMAP embedding with eight labelled cell types plus a 2x3 marker-gene overlay block on one shared expression scale | 单细胞分群展示, "UMAP分群" |
| 529 | `fig_sc_marker_dotplot.py` | Marker dot plot where dot area encodes fraction expressing and colour encodes mean expression, genes grouped by cell type | 标志基因表达, "点图" |
| 530 | `fig_sc_trajectory_pseudotime.py` | Principal-curve trajectory over pseudotime with a bifurcation marker, branch-wise gene programmes and a density strip | 发育轨迹推断, "拟时序" |
| 531 | `fig_sc_cell_communication.py` | Bezier chord diagram of ligand-receptor signalling with an unfolded sender-receiver bubble map and ranked pathways | 细胞通讯分析, "配受体" |
| 532 | `fig_sc_spatial_transcriptomics.py` | Hexagonal spots clipped to a tissue outline coloured by spatial domain, with four spatially variable genes on a shared scale | 空间转录组, "空间域" |
| 533 | `fig_sc_qc_metrics.py` | Six-panel QC dashboard: gene/UMI/mito distributions, UMI-gene scatter with filter box, retention bars and doublet scores | 单细胞质控, "QC面板" |
