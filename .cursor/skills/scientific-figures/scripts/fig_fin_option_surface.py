"""Implied volatility surface of a synthetic equity index option book.

(a) the surface over maturity and moneyness with its contours projected on the
    floor, so the smile (across strikes) and the term structure (across
    maturities) can be read separately; (b) smile slices at four maturities,
    showing the wing flattening as expiry lengthens; (c) the ATM level and the
    90-110 skew as functions of maturity.

The surface is one analytic function; the slices and the term-structure curves
are sampled from that same array, so no panel can drift from another.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers "3d")

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, get_palette
from figkit import panel_labels_inches, safe_colorbar

BLUE = "#3b5b92"
RED = "#b23a48"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ececec"

SLICES = (0.083, 0.25, 0.75, 2.0)      # years: 1M, 3M, 9M, 2Y
SURF_CMAP = get_cmap("cb_sequential_blue")


def iv_surface(t, k):
    """Analytic vol surface: humped term structure plus a decaying smile."""
    T, K = np.meshgrid(t, k, indexing="ij")
    m = np.log(K)
    atm = 15.0 + 9.5 * np.exp(-4.2 * T) + 4.0 * (1.0 - np.exp(-1.1 * T))
    curv = 62.0 * np.exp(-2.1 * T) + 14.0
    skew = 30.0 * np.exp(-1.7 * T) + 7.0
    return atm + curv * m ** 2 - skew * m


def panel_surface(fig, ax, t, k, iv):
    T, K = np.meshgrid(t, k, indexing="ij")
    zmin = np.floor(iv.min() / 5.0) * 5.0 - 6.0
    # A plain min-max norm would spend most of the ramp on the short-dated
    # spike and leave the rest of the surface uniformly pale.
    norm = plt.Normalize(vmin=float(np.percentile(iv, 2)),
                         vmax=float(np.percentile(iv, 96)))
    surf = ax.plot_surface(T, K, iv, cmap=SURF_CMAP, norm=norm, rstride=1,
                           cstride=1, linewidth=0.0, edgecolors="none",
                           alpha=0.98, antialiased=True, shade=False,
                           zorder=3)
    ax.contour(T, K, iv, levels=11, zdir="z", offset=zmin, cmap=SURF_CMAP,
               norm=norm, linewidths=0.7, alpha=0.95)
    ax.set_zlim(zmin, iv.max() * 1.02)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(k[0], k[-1])
    ax.set_xticks([0.25, 0.75, 1.25, 1.75])
    ax.set_yticks([0.85, 0.95, 1.05, 1.15])
    ax.set_xlabel("maturity $T$ (years)", fontsize=8.0, labelpad=1.0)
    ax.set_ylabel("moneyness $K/S$ (-)", fontsize=8.0, labelpad=1.0)
    ax.set_zlabel("implied vol $\\sigma$ (%)", fontsize=8.0, labelpad=1.0)
    ax.tick_params(labelsize=6.8, pad=0.6)
    ax.view_init(elev=24, azim=-56)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("#d9d9d9")
        pane.set_alpha(1.0)
    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="#e3e3e3", linewidth=0.5)
    ax.set_box_aspect((1.24, 1.0, 0.68))
    ax.set_title("Implied volatility surface, contours projected on the floor",
                 fontsize=8.8, fontweight="bold", pad=-2)
    return surf


def panel_smiles(ax, t, k, iv):
    cols = get_palette("cb_sequential_blue", 6)[1:6][::-1]
    atm_i = int(np.argmin(np.abs(k - 1.0)))
    for c, tau in zip(cols, SLICES):
        i = int(np.argmin(np.abs(t - tau)))
        lab = ("%.0fM" % (tau * 12) if tau < 1.0 else "%.0fY" % tau)
        ax.plot(k, iv[i], color=c, lw=1.35, label="$T$ = " + lab)
        ax.plot([k[atm_i]], [iv[i, atm_i]], marker="o", ms=3.0, color=c)
    ax.axvline(1.0, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)))
    ax.text(1.005, ax.get_ylim()[1], " ATM", fontsize=6.5, color=MUTED,
            ha="left", va="top")
    i_short = int(np.argmin(np.abs(t - SLICES[0])))
    ax.annotate("shortest expiry: steepest wings,\npeak $\\Gamma$ near ATM",
                xy=(k[atm_i], iv[i_short, atm_i]),
                xytext=(1.025, iv[i_short].max() * 0.86),
                fontsize=6.6, color=RED, ha="left", va="top", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=RED,
                                connectionstyle="arc3,rad=0.28"))
    ax.set_xlabel("moneyness $K/S$ (-)")
    ax.set_ylabel("implied vol $\\sigma$ (%)")
    ax.set_xlim(k[0], k[-1])
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.6, handlelength=1.4,
              borderaxespad=0.25, ncol=2, columnspacing=1.0)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)


def panel_term(ax, t, k, iv):
    atm_i = int(np.argmin(np.abs(k - 1.0)))
    lo_i = int(np.argmin(np.abs(k - 0.90)))
    hi_i = int(np.argmin(np.abs(k - 1.10)))
    atm = iv[:, atm_i]
    skew = iv[:, lo_i] - iv[:, hi_i]
    ax.plot(t, atm, color=BLUE, lw=1.45, label="ATM $\\sigma$")
    ax.plot(t, skew, color=RED, lw=1.35, ls=(0, (4, 2)),
            label="90-110 skew $\\Delta\\sigma$")
    ax.fill_between(t, 0, skew, color=RED, alpha=0.10, lw=0)
    i1y = int(np.argmin(np.abs(t - 1.0)))
    ax.axvline(1.0, color="#9a9a9a", lw=0.7, ls=(0, (3, 2)))
    ax.text(1.03, ax.get_ylim()[1] * 0.98,
            "1Y ATM $\\sigma$ = %.1f%%\n1Y skew = %.1f vol pts"
            % (atm[i1y], skew[i1y]), fontsize=6.5, color=SUB, ha="left",
            va="top", linespacing=1.3)
    ax.set_xlabel("maturity $T$ (years)")
    ax.set_ylabel("vol (%) and skew\n(vol pts)", linespacing=1.3)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, max(atm.max(), skew.max()) * 1.30)
    ax.grid(axis="y", lw=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="center right", fontsize=6.6, handlelength=1.5,
              borderaxespad=0.4)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)
    return atm, skew


def main():
    set_pub_style()
    t = np.linspace(0.05, 2.0, 46)
    k = np.linspace(0.80, 1.20, 44)
    iv = iv_surface(t, k)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.60],
                          height_ratios=[1.0, 1.0], hspace=0.46,
                          wspace=0.30, left=0.045, right=0.965, top=0.905,
                          bottom=0.115)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_sm = fig.add_subplot(gs[0, 1])
    ax_tm = fig.add_subplot(gs[1, 1])

    surf = panel_surface(fig, ax3d, t, k, iv)
    safe_colorbar(fig, surf, ax3d, label="implied vol $\\sigma$ (%)",
                  side="bottom", width=0.12, pad=0.02, shrink=0.52)
    panel_smiles(ax_sm, t, k, iv)
    atm, skew = panel_term(ax_tm, t, k, iv)

    atm_i = int(np.argmin(np.abs(k - 1.0)))
    i1m = int(np.argmin(np.abs(t - SLICES[0])))

    panel_labels_inches(fig, [ax3d, ax_sm, ax_tm], dx=-0.40, dy=0.04)
    savefig(fig, "573_fin_option_surface.png")


if __name__ == "__main__":
    main()
