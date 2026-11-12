# 医学影像 · Medical imaging  `374-383`

10 figures. Run any of them from `scripts/`; the PNG lands in `output/` under the same number.

| # | Script | What it draws | Use when |
|---|--------|---------------|----------|
| 374 | `fig_med_ct_windowing.py` | Figure 374 - CT display windows: one HU volume slice, five renderings | 医学影像 |
| 375 | `fig_med_mri_sequences.py` | Figure 375 - Multi-sequence MRI of one lesion, eight contrasts | 医学影像 |
| 376 | `fig_med_segmentation_overlay.py` | Figure 376 - Tumour segmentation, three slices by three networks | 医学影像 |
| 377 | `fig_med_lesion_detection.py` | Figure 377 - Brain metastasis detection on a contrast-enhanced T1w slice | 医学影像 |
| 378 | `fig_med_registration.py` | Figure 378 - Deformable registration of a follow-up scan onto a baseline | 医学影像 |
| 379 | `fig_med_radiotherapy_dose.py` | Figure 379 - Radiotherapy plan: isodose wash on CT beside its DVH | 医学影像 |
| 380 | `fig_med_petct_fusion.py` | Figure 380 - FDG PET/CT: anatomy, function, fusion and the SUV read-out | 医学影像 |
| 381 | `fig_med_radiomics.py` | Figure 381 - Radiomics, read left to right: ROI, feature matrix, model | 医学影像 |
| 382 | `fig_med_orthogonal_views.py` | Axial / coronal / sagittal reformats of one isotropic volume with colour-coded cross hairs, a depth-shaded surface rendering, an isometric plane locator and the viewer read-out header | 三维影像定位与多平面重建说明, "三视图" "正交切面" "MPR" |
| 383 | `fig_med_qc_panel.py` | Weekly scanner QC dashboard: SNR, uniformity, geometric-accuracy and N/2-ghost maps, a trend panel plotting every metric as a fraction of its own action limit, and four pass/fail verdict cards | 设备质控周报与验收测试, "影像质控" "SNR" "均匀性" "伪影" |
