"""SHAP beeswarm summary + partial-dependence panels.

Left: a SHAP beeswarm with features ordered by mean |SHAP|, points coloured by
the (scaled) feature value and the mean |SHAP| printed at the right of each
row.  Right: six partial-dependence panels, each with the smoothed response,
a bootstrap band, the fitted change point and a marginal count strip.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

CMAP = LinearSegmentedColormap.from_list("brown_teal",
                                         ["#8c6b4a", "#ded7cb", "#2a8f86"])
C_TREND, C_THR, C_PTS = "#2b2b2b", "#b06a3a", "#9db8b0"

# feature, unit, effect size, threshold (in feature units), direction
FEATS = [
    ("Soil organic carbon", "%", 0.115, 0.40, -1),
    ("Mean annual temperature", "$^{\\circ}$C", 0.092, -0.20, -1),
    ("Precipitation", "mm", 0.074, -0.35, +1),
    ("NDVI", "-", 0.058, 0.10, +1),
    ("Elevation", "m", 0.044, -0.55, -1),
    ("Population density", "km$^{-2}$", 0.031, 0.25, -1),
    ("Slope", "degrees", 0.022, -0.10, +1),
    ("Soil pH", "-", 0.014, 0.05, -1),
]


def beeswarm(fig, ax, rng):
    n = 320
    rows = []
    for name, _unit, eff, thr, sign in FEATS:
        fval = rng.normal(0, 1, n)
        shap = sign * eff * np.tanh(1.4 * (fval - thr)) \
            + rng.normal(0, 0.22 * eff, n)
        rows.append((name, fval, shap))
    rows.sort(key=lambda r: np.mean(np.abs(r[2])))       # weakest at bottom

    for i, (name, fval, shap) in enumerate(rows):
        q = (np.argsort(np.argsort(fval)) / (n - 1))     # rank-scaled colour
        # vertical spread proportional to local density -> beeswarm look
        dens = np.histogram(shap, bins=28)[0]
        edges = np.histogram_bin_edges(shap, bins=28)
        w = dens[np.clip(np.digitize(shap, edges) - 1, 0, 27)]
        jit = rng.uniform(-1, 1, n) * 0.20 * np.sqrt(w / w.max())
        ax.scatter(shap, i + jit, c=q, cmap=CMAP, vmin=0, vmax=1, s=7,
                   alpha=0.8, linewidths=0.15, edgecolors="white", zorder=3)
        ax.text(0.995, i, f"{np.mean(np.abs(shap)):.3f}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.2, color="#666666")

    ax.axvline(0, color="#9a9a9a", lw=0.7, zorder=2)
    span = max(np.abs(np.concatenate([r[2] for r in rows])))
    ax.set_xlim(-1.12 * span, 1.42 * span)      # room for the |SHAP| column
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("SHAP value (impact on predicted biomass, mg g$^{-1}$)",
                  fontsize=7.5)
    ax.set_title("SHAP summary (mean |SHAP| at right)", fontsize=8.5,
                 fontweight="bold", pad=5)
    ax.tick_params(labelsize=7)
    style_axes(ax)

    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), ax=ax,
                      orientation="horizontal", fraction=0.032, pad=0.11,
                      ticks=[0, 1])
    cb.ax.set_xticklabels(["low", "high"], fontsize=7)
    cb.set_label("Feature value (rank-scaled)", fontsize=7, labelpad=1)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor("#666666")
    cb.ax.tick_params(length=2, width=0.6)


def pdp_panel(fig, subspec, rng, feat, unit, eff, thr, sign, tag, show_y):
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=subspec,
                                 height_ratios=[4, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)

    n = 400
    x = rng.normal(0, 0.55, n)
    x = np.clip(x, -1.4, 1.4)
    y = 21.9 + sign * 3.2 * eff * np.tanh(2.6 * (x - thr)) \
        + rng.normal(0, 0.045, n)

    ax.scatter(x, y, s=6, color=C_PTS, alpha=0.5, edgecolors="none", zorder=2)

    order = np.argsort(x)
    xs, ys_raw = x[order], y[order]
    xg = np.linspace(xs.min(), xs.max(), 140)
    h = 0.16 * (xs.max() - xs.min())

    def smooth(vals):
        w = np.exp(-0.5 * ((xs[None, :] - xg[:, None]) / h) ** 2)
        s = w.sum(axis=1)
        s[s == 0] = 1.0
        return (w @ vals) / s

    yg = smooth(ys_raw)
    boot = np.array([smooth(ys_raw[rng.integers(0, n, n)]) for _ in range(80)])
    band = np.percentile(boot, [2.5, 97.5], axis=0)
    ax.fill_between(xg, band[0], band[1], color="#9a9a9a", alpha=0.25,
                    linewidth=0, zorder=3)
    ax.plot(xg, yg, color=C_TREND, lw=1.5, zorder=4)
    ax.axvline(thr, color=C_THR, ls=(0, (2, 2)), lw=1.0, zorder=1)

    ax.text(0.03, 0.06, f"change point = {thr:+.2f}", transform=ax.transAxes,
            fontsize=6.3, color=C_THR, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e0d0c4",
                      lw=0.5, alpha=0.92))
    if show_y:
        ax.set_ylabel("Predicted biomass\n(mg g$^{-1}$)", fontsize=7,
                      linespacing=1.2)
    ax.tick_params(labelbottom=False, labelsize=6.5)
    ax.margins(x=0.02)
    style_axes(ax)
    add_panel_label(ax, f"({tag})", x=-0.14, y=1.0, size=9)

    axh.hist(x, bins=34, color="#a9c4bc", edgecolor="white", linewidth=0.25)
    axh.set_xlabel(f"{feat} (z-score)", fontsize=7)
    axh.set_yticks([])
    axh.tick_params(labelsize=6.5, length=2)
    axh.spines["left"].set_visible(False)
    style_axes(axh)


def main():
    set_pub_style()
    rng = np.random.default_rng(8)
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = GridSpec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.34,
                  left=0.145, right=0.985, top=0.90, bottom=0.115)

    ax_bee = fig.add_subplot(gs[0])
    beeswarm(fig, ax_bee, rng)
    add_panel_label(ax_bee, "(a)", x=-0.34, y=1.01, size=10)

    inner_l = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.48)
    inner_r = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2], hspace=0.48)
    picks = [FEATS[i] for i in (0, 1, 2, 3, 4, 5)]
    for k in range(3):
        name, unit, eff, thr, sign = picks[k]
        pdp_panel(fig, inner_l[k], rng, name, unit, eff, thr, sign,
                  chr(98 + k), show_y=True)
    for k in range(3):
        name, unit, eff, thr, sign = picks[k + 3]
        pdp_panel(fig, inner_r[k], rng, name, unit, eff, thr, sign,
                  chr(101 + k), show_y=False)

    handles = [
        Line2D([], [], marker="o", ls="none", color=C_PTS, ms=4,
               label="Observations"),
        Line2D([], [], color=C_TREND, lw=1.5, label="Partial dependence"),
        Patch(facecolor="#9a9a9a", alpha=0.25, label="95% bootstrap CI"),
        Line2D([], [], color=C_THR, ls=(0, (2, 2)), lw=1.0,
               label="Fitted change point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.62, 0.005), columnspacing=1.8)
    savefig(fig, "11_shap_pdp.png")


if __name__ == "__main__":
    main()
