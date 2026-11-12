"""Partial-dependence (PDP) plus individual-conditional-expectation (ICE) grid.

3x2 layout. Every cell stacks a main axis over a thin marginal strip:
  * thin translucent grey ICE curves - one per sampled individual,
  * a +/-1 SD ribbon across individuals,
  * a thick coloured average PDP curve with its 10th/90th ICE decile markers,
  * a bottom strip with the observed feature histogram and a rug of deciles.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

ICE_C = "#8d949b"
# (feature, unit, lo, hi, shape, colour, sampling beta, effect magnitude)
# ordered by descending partial-dependence range
SPECS = [
    ("Credit utilisation", "%", 0, 100, "sigmoid", "#2f6f8f", "2.0,2.6", 1.00),
    ("Payment history", "score", 300, 850, "plateau", "#a8493f", "3.0,1.9", 0.84),
    ("Debt-to-income", "ratio", 0.05, 0.75, "rise", "#b0653a", "2.2,3.0", 0.70),
    ("Account age", "months", 3, 180, "decay", "#4a7c59", "1.6,2.8", 0.54),
    ("Recent inquiries", "count / 12 mo", 0, 12, "step", "#8a5f9c", "1.4,3.2", 0.40),
    ("Loan-to-value", "%", 30, 120, "ushape", "#3f5a8a", "2.6,2.4", 0.27),
]


def shape_fn(kind, u):
    if kind == "sigmoid":
        return 1 / (1 + np.exp(-8.5 * (u - 0.52)))
    if kind == "rise":
        return u ** 1.45
    if kind == "decay":
        return np.exp(-2.9 * u)
    if kind == "step":
        return 1 / (1 + np.exp(-16 * (u - 0.30)))
    if kind == "plateau":
        return 1 - 1 / (1 + np.exp(-9.5 * (u - 0.34)))
    if kind == "ushape":
        return 0.25 + 2.6 * (u - 0.46) ** 2
    raise ValueError(kind)


def panel(fig, spec_cell, rng, spec, label, show_ylabel):
    feat, unit, lo, hi, kind, col, dist, mag = spec
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec_cell,
                                 height_ratios=[5.0, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    m = 90                                      # grid resolution
    n_ice = 70                                  # sampled individuals
    u = np.linspace(0, 1, m)
    x = lo + u * (hi - lo)

    base_shape = shape_fn(kind, u)
    base_shape = (base_shape - base_shape.min()) / np.ptp(base_shape)

    # heterogeneous individuals: own intercept, own effect magnitude, and a
    # small idiosyncratic ripple so the ICE bundle is not a rigid translation
    intercept = rng.beta(2.0, 3.4, n_ice)[:, None] * 0.40 + 0.07
    slope = (0.35 + 1.30 * rng.beta(2.2, 2.2, n_ice))[:, None]
    phase = rng.uniform(0, 2 * np.pi, n_ice)[:, None]
    ripple = 0.024 * np.sin(2.3 * np.pi * u[None, :] + phase) \
        * rng.uniform(0.3, 1.0, n_ice)[:, None]
    ice = intercept + slope * 0.46 * mag * base_shape[None, :] + ripple
    ice += rng.normal(0, 0.006, ice.shape).cumsum(axis=1) * 0.50
    ice = np.clip(ice, 0.005, 0.995)

    pdp = ice.mean(axis=0)
    sd = ice.std(axis=0)
    q10, q90 = np.percentile(ice, [10, 90], axis=0)

    for row in ice:
        ax.plot(x, row, color=ICE_C, lw=0.50, alpha=0.42, zorder=1)
    ax.fill_between(x, pdp - sd, pdp + sd, color=col, alpha=0.16, lw=0,
                    zorder=2)
    ax.plot(x, q10, color=col, lw=0.75, ls=(0, (3, 2)), alpha=0.75, zorder=3)
    ax.plot(x, q90, color=col, lw=0.75, ls=(0, (3, 2)), alpha=0.75, zorder=3)
    ax.plot(x, pdp, color=col, lw=2.2, zorder=4,
            path_effects=[pe.Stroke(linewidth=3.6, foreground="white"),
                          pe.Normal()])

    delta = pdp.max() - pdp.min()
    ax.text(0.035, 0.955, f"$\\Delta$PDP = {delta:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
            fontweight="bold", color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#c8c8c8",
                      lw=0.6, alpha=0.92))
    ax.set_title(f"{feat}", fontsize=8.6, fontweight="bold", pad=3)
    if show_ylabel:
        ax.set_ylabel("Predicted default risk", fontsize=7.8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(lo, hi)
    ax.tick_params(labelbottom=False, labelsize=6.8, length=2.6)
    style_axes(ax)
    add_panel_label(ax, label, x=-0.115 if show_ylabel else -0.065,
                    y=1.02, size=9.5)

    # ---- marginal strip: observed feature distribution -------------------
    a, b = (float(t) for t in dist.split(","))
    obs = lo + rng.beta(a, b, 4000) * (hi - lo)
    axr.hist(obs, bins=42, range=(lo, hi), color=col, alpha=0.35,
             edgecolor="white", linewidth=0.25)
    for q in np.percentile(obs, np.arange(10, 100, 10)):
        axr.axvline(q, color="#555555", lw=0.5, alpha=0.6)
    axr.set_xlabel(f"{feat} ({unit})", fontsize=7.6)
    axr.set_yticks([])
    axr.tick_params(labelsize=6.8, length=2.6)
    axr.spines["left"].set_visible(False)
    style_axes(axr)


def main():
    set_pub_style()
    rng = np.random.default_rng(707)
    fig = plt.figure(figsize=(9.4, 8.3), layout="constrained")
    fig.get_layout_engine().set(rect=(0.005, 0.015, 0.99, 0.912),
                                h_pad=0.08, w_pad=0.08)
    gs = GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.18)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for k, spec in enumerate(SPECS):
        panel(fig, gs[k // 2, k % 2], rng, spec, labels[k],
              show_ylabel=(k % 2 == 0))

    handles = [
        Line2D([], [], color=ICE_C, lw=0.9, alpha=0.9,
               label="ICE curve (n = 70 individuals)"),
        Line2D([], [], color="#444444", lw=2.2, label="Average PDP"),
        Line2D([], [], color="#444444", lw=0.8, ls=(0, (3, 2)),
               label="10th / 90th ICE percentile"),
        Patch(facecolor="#7c7c7c", alpha=0.35, edgecolor="none",
              label="$\\pm$1 SD across individuals"),
        Patch(facecolor="#7c7c7c", alpha=0.50, edgecolor="white", lw=0.4,
              label="Feature distribution (deciles ruled)"),
    ]
    savefig(fig, "64_pdp_ice_grid.png")


if __name__ == "__main__":
    main()
