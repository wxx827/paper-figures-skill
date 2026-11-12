"""Figure 381 - Radiomics, read left to right: ROI, feature matrix, model.

The feature matrix is the main panel because it is where the claim lives; the
ROI on its left says what was measured and the two small panels on its right
say what the measurement was worth.  Cohen's d, the ranking and both AUCs are
computed from the same matrix that is drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import (image_axes, axes_inches, corner_stamp, shared_norm,
                    safe_colorbar, panel_labels_inches)

N = 200
MM_PX = 200.0 / N
ACCENT = "#b23a48"
BAR_C = "#3b5b92"
CLASS_C = ["#8a8a8a", "#c9a227"]        # non-responder, responder

FAMILIES = [
    ("Shape", ["sphericity", "elongation", "surf/vol", "max diameter",
               "flatness", "compactness"]),
    ("First order", ["mean", "entropy", "skewness", "kurtosis", "p90",
                     "energy"]),
    ("GLCM", ["contrast", "correlation", "joint entropy", "IDM",
              "cluster shade", "Imc2"]),
    ("Wavelet", ["LLH mean", "LHL entropy", "HHH energy", "LLL skew",
                 "HLH p90", "LHH contrast"]),
]
N_SAMPLE = 60


# ---------------------------------------------------------------------------
# Synthetic brain phantom (layered ellipses + gyral folding, as in figure 253)
# ---------------------------------------------------------------------------
def _sm(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _sym(a):
    return 0.5 * (a + a[:, ::-1])


def _grid(n):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (xx - (n - 1) / 2) / (n / 2), (yy - (n - 1) / 2) / (n / 2)


def _rad(x, y, cx, cy, rx, ry, ang=0.0):
    t = np.deg2rad(ang)
    xr = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    yr = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return np.hypot(xr / rx, yr / ry)


def roi_slice(seed=21):
    rng = np.random.default_rng(seed)
    x, y = _grid(N)
    wob = 1.0 + 0.015 * _sym(_sm(rng, N, 9))
    rr = _rad(x, y, 0.0, 0.0, 0.80, 0.92) * wob

    img = np.full((N, N), 0.02)
    img = np.where(rr <= 1.000, 0.34, img)
    img = np.where(rr <= 0.962, 0.09, img)
    img = np.where(rr <= 0.900, 0.12, img)
    brain = rr <= 0.875
    img = np.where(brain, 0.52 + 0.035 * _sym(_sm(rng, N, 3)), img)
    rw = _rad(x, y, 0.0, 0.03, 0.62, 0.74) * wob
    img = np.where(brain & (rw <= 1.0), 0.36 + 0.025 * _sm(rng, N, 5), img)
    g1, g2 = _sym(_sm(rng, N, 7.0)), _sym(_sm(rng, N, 15.0))
    sulci = (np.sin(8.0 * g1 + 4.2 * g2) > 0.62) & brain & (rr > 0.58)
    img = np.where(sulci, 0.10, img)
    for vx, vy, vrx, vry, va in [(0.115, -0.02, 0.055, 0.22, -13),
                                 (-0.115, -0.02, 0.055, 0.22, 13)]:
        img = np.where(_rad(x, y, vx, vy, vrx, vry, va) * wob <= 1.0, 0.06, img)

    dl = _rad(x, y, 0.28, -0.24, 0.185, 0.160, -16)
    dl = dl * (1.0 + 0.13 * _sm(rng, N, 8))
    roi = (dl <= 1.0) & brain
    texture = 0.70 + 0.16 * _sm(rng, N, 1.6) + 0.10 * _sm(rng, N, 4.0)
    img = np.where(roi, texture, img)
    img = gaussian_filter(img, 0.8) + 0.012 * rng.standard_normal((N, N))
    return np.clip(img, 0, 1), roi


def feature_matrix(seed=3):
    """Correlated feature families with a real but partial class effect."""
    rng = np.random.default_rng(seed)
    names = [nm for _, group in FAMILIES for nm in group]
    n_feat = len(names)
    label = np.repeat([0, 1], N_SAMPLE // 2)
    x = np.empty((N_SAMPLE, n_feat))
    for f in range(len(FAMILIES)):
        latent = rng.standard_normal(N_SAMPLE)
        for k in range(6):
            x[:, 6 * f + k] = (0.78 * latent
                               + 0.62 * rng.standard_normal(N_SAMPLE))
    # only texture-derived features actually separate the two outcome groups
    effect = np.zeros(n_feat)
    for idx, d in [(7, 1.05), (13, -0.92), (14, 1.18), (16, 0.80),
                   (19, 1.32), (21, -0.86), (10, 0.55), (2, 0.42)]:
        effect[idx] = 0.55 * d
    x = x + label[:, None] * effect[None, :]
    x = (x - x.mean(0)) / x.std(0)
    return x, label, names


def roc(score, label):
    order = np.argsort(-score)
    y = label[order]
    tpr = np.concatenate([[0.0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / (1 - y).sum()])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def main():
    set_pub_style(base_font=8)
    img, roi = roi_slice()
    x, label, names = feature_matrix()

    pos, neg = x[label == 1], x[label == 0]
    pooled = np.sqrt(0.5 * (pos.var(0, ddof=1) + neg.var(0, ddof=1)))
    d = (pos.mean(0) - neg.mean(0)) / pooled

    row_order = leaves_list(linkage(x, method="ward"))
    col_order = leaves_list(linkage(x.T, method="ward"))
    mat = x[np.ix_(row_order, col_order)]
    lab_sorted = label[row_order]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_roi = axes_inches(fig, (0.30, 1.85, 3.20, 3.20))
    ax_mat = axes_inches(fig, (4.32, 1.72, 2.83, 3.72))
    ax_imp = axes_inches(fig, (8.10, 3.52, 2.02, 1.92))
    ax_roc = axes_inches(fig, (8.10, 0.92, 2.02, 1.92))

    # -- step 1: the region the features come from ---------------------------
    image_axes(ax_roi, color="#d5d5d5", lw=0.6)
    ax_roi.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    rgba = np.zeros(roi.shape + (4,))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0.70, 0.23, 0.28
    rgba[..., 3] = 0.26 * roi
    ax_roi.imshow(rgba, interpolation="nearest", zorder=2)
    ax_roi.contour(roi.astype(float), levels=[0.5], colors=[ACCENT],
                   linewidths=1.8, zorder=3)
    area = roi.sum() * MM_PX ** 2
    corner_stamp(ax_roi, f"ROI {roi.sum()} px = {area / 100:.1f} cm2",
                 loc="lower left", fontsize=6.8)
    ax_roi.set_title("Tumour ROI on T2w", fontsize=8.8, fontweight="bold",
                     pad=4.0)

    # -- step 2: the feature matrix (main panel) -----------------------------
    norm = shared_norm([mat], robust=(2, 98), symmetric=True)
    im = ax_mat.imshow(mat, cmap=get_cmap("cb_diverging"), norm=norm,
                       aspect="auto", interpolation="nearest")
    ax_mat.set_xticks(range(len(names)))
    ax_mat.set_xticklabels([names[i] for i in col_order], rotation=90,
                           fontsize=6.0)
    ax_mat.set_yticks([])
    ax_mat.set_ylabel(f"patients, Ward-clustered (n = {N_SAMPLE})",
                      fontsize=8.0)
    ax_mat.set_title("Standardised feature matrix", fontsize=8.8,
                     fontweight="bold", pad=4.0)
    ax_mat.tick_params(length=1.8)
    for s in ax_mat.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#d5d5d5")
    safe_colorbar(fig, im, ax_mat, label="feature z-score (-)", side="bottom",
                  width=0.10, pad=0.80, shrink=0.86)

    for k, cls in enumerate(lab_sorted):                 # outcome side bar
        ax_mat.add_patch(Rectangle((-1.9, k - 0.5), 1.2, 1.0,
                                   facecolor=CLASS_C[cls], edgecolor="none",
                                   clip_on=False, zorder=4))
    ax_mat.set_xlim(-2.0, len(names) - 0.5)
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=4.5, color=c,
                          label=nm)
               for nm, c in zip(["non-responder", "responder"], CLASS_C)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.545, 0.938),
               ncol=2, frameon=False, fontsize=7.0)

    # -- step 3a: which features carry the signal ----------------------------
    top = np.argsort(-np.abs(d))[:8][::-1]
    ax_imp.barh(np.arange(len(top)), np.abs(d[top]), height=0.66,
                color=BAR_C, edgecolor="white", linewidth=0.6)
    ax_imp.set_yticks(np.arange(len(top)))
    ax_imp.set_yticklabels([names[i] for i in top], fontsize=6.2)
    ax_imp.set_xlabel("|Cohen's d| (-)", fontsize=7.5)
    ax_imp.tick_params(labelsize=6.5)
    ax_imp.grid(True, axis="x", lw=0.5, alpha=0.7)
    ax_imp.set_title("Top discriminative features", fontsize=8.5,
                     fontweight="bold", pad=3.0)

    # -- step 3b: what they are worth ----------------------------------------
    w = d[top]
    f_all, t_all, auc_all = roc(x[:, top] @ w, label)
    shape_idx = np.arange(6)
    f_sh, t_sh, auc_sh = roc(x[:, shape_idx] @ d[shape_idx], label)
    ax_roc.plot([0, 1], [0, 1], color=COLORS["baseline"], lw=0.8, ls=":")
    ax_roc.plot(f_sh, t_sh, color=COLORS["baseline"], lw=1.2, ls="--",
                label=f"shape only, AUC {auc_sh:.2f}")
    ax_roc.plot(f_all, t_all, color=ACCENT, lw=1.7,
                label=f"top-8 panel, AUC {auc_all:.2f}")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("false positive rate (-)", fontsize=7.5)
    ax_roc.set_ylabel("true positive rate (-)", fontsize=7.5)
    ax_roc.tick_params(labelsize=6.5)
    ax_roc.grid(True, lw=0.5, alpha=0.7)
    ax_roc.legend(loc="lower right", frameon=False, fontsize=6.6)
    ax_roc.set_title("Response prediction", fontsize=8.5, fontweight="bold",
                     pad=3.0)

    for a in (ax_roi, ax_mat, ax_imp, ax_roc):
        a.apply_aspect()
    panel_labels_inches(fig, [ax_roi, ax_mat, ax_imp, ax_roc], dx=-0.30,
                        dy=0.13)

    for xf, txt in ((0.30 / 10.4, "1  segment"), (4.32 / 10.4, "2  extract"),
                    (8.10 / 10.4, "3  predict")):
        fig.text(xf, 0.930, txt, fontsize=7.0, color=COLORS["muted"],
                 ha="left", va="bottom")

    savefig(fig, "381_med_radiomics.png")


if __name__ == "__main__":
    main()
