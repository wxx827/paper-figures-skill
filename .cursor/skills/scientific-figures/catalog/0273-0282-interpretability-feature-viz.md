# 深度学习可解释性与特征可视化 · Interpretability & feature viz  `273-282`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 273 | `fig_gradcam_grid.py` | Three samples by five views (input, three backbone depths, ours) with translucent jet CAMs over the grayscale input, per-cell predicted class and confidence tags, and one shared activation colour bar | Grad-CAM 逐层对比、可解释性主图、证据聚焦程度论证, "类激活图, Grad-CAM, 热力图叠加, 可解释性" |
| 274 | `fig_attention_maps.py` | One tokenised input with a red query patch above a 3x4 grid of the twelve heads of a transformer block, each plotted as attention gain over the uniform 1/256 baseline on one shared turbo scale | 多头注意力分工分析、ViT 注意力头可视化, "注意力图, 多头注意力, 查询块, 注意力头" |
| 275 | `fig_feature_evolution.py` | Six backbone stages by four channels, resolution falling 96 to 3 pixels, oriented-edge responses dissolving into part-level blobs, with a shallow-to-deep arrow rail and one shared viridis scale | 特征图逐层演化、浅层到深层语义变化说明, "特征图, 逐层可视化, 感受野, 语义特征" |
| 276 | `fig_tsne_embedding.py` | Three t-SNE panels (raw, contrastive baseline, ours) with class-coloured scatter and measured silhouette scores, over a subordinate strip of inter-class separation matrices on one shared magma scale | 表征学习效果对比、嵌入可分性论证, "t-SNE, 降维可视化, 类别可分性, 轮廓系数" |
| 277 | `fig_filter_visualization.py` | All 64 first-layer kernels rendered as RGB tiles under one shared weight normalisation, beside twelve deep units visualised by activation maximisation with per-unit labels | 卷积核可视化、激活最大化、深层单元偏好展示, "卷积核, Gabor滤波器, 激活最大化, 特征可视化" |
| 278 | `fig_attention_rollout.py` | Input plus four transformer blocks and the accumulated rollout as turbo overlays with per-panel Shannon entropy, over a subordinate entropy-versus-depth curve marked with the rollout result | 注意力累积、ViT 逐层聚焦过程、注意力熵下降, "注意力累积, Attention Rollout, 注意力熵, ViT" |
| 279 | `fig_saliency_compare.py` | Two samples by six columns comparing vanilla gradients, SmoothGrad, integrated gradients, Grad-CAM and ours on one shared jet scale, over a subordinate deletion/insertion AUC bar panel | 显著性方法横向对比、归因忠实度评估, "显著性图, 归因方法, SmoothGrad, 积分梯度, 忠实度" |
| 280 | `fig_feature_similarity.py` | Linear CKA of a ResNet-50 against itself with bright stage blocks, a rectangular ResNet-versus-ViT similarity matrix, and a depth curve whose markers are coloured on the same shared scale | 表征相似度分析、跨架构对比、CKA 层间结构, "CKA, 表征相似度, 层间相似性, 跨模型对比" |
| 281 | `fig_prototype_retrieval.py` | Five queries in blue frames with their top-five neighbours framed green or red by correctness, cosine similarity tags in every tile, and a row-aligned subordinate precision@5 bar strip | 检索结果定性展示、近邻可视化、错检案例分析, "近邻检索, 原型, 检索精度, 定性结果" |
| 282 | `fig_latent_traversal.py` | Five latent dimensions swept from -3 to +3 sigma in eight columns, starting from the encoder input, with the blue-framed z = 0 reconstruction as reference and named semantics on the left | 隐空间遍历、解耦表征验证、生成模型可控性, "隐空间遍历, 解耦表征, VAE, 潜变量语义" |
