# 图神经网络 · Graph neural networks  `444-453`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 444 | `fig_gnn_message_passing.py` | Message passing in a GNN layer, drawn as three stages on one small graph | 图神经网络 |
| 445 | `fig_gnn_node_embedding.py` | Node embeddings tightening layer by layer, on one shared coordinate frame | 图神经网络 |
| 446 | `fig_gnn_attention_graph.py` | Graph attention weights: edge thickness on the graph, heads compared aside | 图神经网络 |
| 447 | `fig_gnn_oversmoothing.py` | Over-smoothing: deeper propagation makes every node look like every other | 图神经网络 |
| 448 | `fig_gnn_subgraph_sampling.py` | Two ways to fit a graph into GPU memory, drawn on one fixed layout | 图神经网络 |
| 449 | `fig_gnn_heterogeneous.py` | A heterogeneous academic graph: three node types, three relation types | 图神经网络 |
| 450 | `fig_gnn_link_prediction.py` | Link prediction on a graph with a fifth of its edges hidden | 图神经网络 |
| 451 | `fig_gnn_graph_classification.py` | Graph-level classification: eight molecule-like graphs and the scoreboard | 图神经网络 |
| 452 | `fig_gnn_explainability.py` | Explanation subgraphs from three methods, on three graphs with a planted motif | 图神经网络 |
| 453 | `fig_gnn_depth_ablation.py` | Component tick/cross matrix, aligned accuracy-gain heat map over five datasets, and a subordinate depth sweep showing oversmoothing | 图神经网络消融, 过平滑, "深度对比" |
