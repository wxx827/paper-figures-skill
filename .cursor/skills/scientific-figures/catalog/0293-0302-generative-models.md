# 生成模型 · Generative models  `293-302`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 293 | `fig_diffusion_process.py` | Forward and reverse diffusion strips, 8 shared-noise steps each, arrow bars carrying q and p_theta, attached cosine log-SNR panel | Method section opener for a diffusion paper, "什么是扩散模型" |
| 294 | `fig_sampling_steps_compare.py` | Sampler x step-budget matrix (DDPM / DDIM / DPM-Solver++ at 10-250 NFE) from one latent, FID per tile, attached FID-vs-NFE curve | Fast-sampling ablation, "采样步数对比" |
| 295 | `fig_conditional_generation.py` | Four conditioning modalities (text, class, Canny edges, depth) x four seeds, condition tile on the left of each row | ControlNet-style conditioning demo, "条件生成一致性与多样性" |
| 296 | `fig_latent_interpolation.py` | Five endpoint pairs interpolated over nine frames by slerp, framed real endpoints, attached latent-trajectory projection | Latent-space smoothness evidence, "隐空间插值" |
| 297 | `fig_gan_training_dynamics.py` | Four-panel dashboard: G/D losses with spread band, FID and IS, checkpoint sample grid, mode-collapse diagnostics, one collapse episode tracked across all panels | Training analysis page, "GAN 训练动态与模式坍塌" |
| 298 | `fig_generation_quality_grid.py` | Five model families decoding the same six latents, each with its own failure signature, metric chips with normalised bars on the right | Headline qualitative comparison, "生成质量大网格" |
| 299 | `fig_inpainting_results.py` | Three masked samples x six views (input, mask, two baselines, ours, GT) with red zoom boxes, 2.7x seam insets and PSNR / LPIPS per tile | Restoration comparison table, "图像修复对比" |
| 300 | `fig_style_transfer_matrix.py` | Content x style matrix: empty corner, five style references (orange), four contents (blue), twenty transfers with per-style operators | Style transfer results page, "内容风格矩阵" |
| 301 | `fig_noise_schedule.py` | Schedule study 2x2: alpha-bar curves, log-SNR, sample appearance at four time fractions, FID bars at two step budgets, colour keyed to schedule | Schedule ablation, "噪声调度分析" |
| 302 | `fig_mode_coverage.py` | Real / collapsed baseline / ours on one shared coordinate frame with missed modes ringed in red, attached per-mode recall bars and precision-recall curves | Diversity and coverage claim, "模式覆盖与模式坍塌" |
