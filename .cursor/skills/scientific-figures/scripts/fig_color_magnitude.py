"""Colour-magnitude diagrams of three star clusters of increasing age.

Each panel is a synthetic CMD: main-sequence stars drawn from a power-law mass
function up to the age-dependent turn-off mass, an equal-mass binary sequence
0.75 mag above it, a subgiant/red-giant branch built as a quadratic Bezier
from the turn-off, a red clump, and — for the oldest cluster — a horizontal
branch.  All three panels deliberately share one pair of axis limits and one
membership colour scale so the turn-off migrates visibly with age.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

MEMBER_COLORS = ["#bcd7e2", "#7fb0c6", "#4a80a6", "#2c5480", "#1f3355"]

CLUSTERS = [("NGC 2168", 0.15, "young open cluster"),
            ("NGC 6819", 2.40, "intermediate-age open cluster"),
            ("M 5", 12.00, "old globular cluster")]

MV_SUN = 4.83


# empirical colour-temperature sequence, interpolated in log Teff
_TEFF = np.array([3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000,
                  10000, 14000, 20000, 30000], dtype=float)
_BV = np.array([1.66, 1.50, 1.35, 1.02, 0.81, 0.68, 0.58, 0.35, 0.17,
                -0.02, -0.14, -0.22, -0.32])


def bv_from_teff(teff):
    return np.interp(np.log10(teff), np.log10(_TEFF), _BV)


def ms_track(m):
    teff = 5772.0 * m ** 0.53
    mv = MV_SUN - 2.5 * (3.55 * np.log10(m))
    return bv_from_teff(teff), mv


def bezier(p0, p1, p2, n=160):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def isochrone(age_gyr):
    """Return (colour, magnitude) along the full isochrone."""
    m_to = (10.0 / age_gyr) ** 0.40
    m = np.linspace(0.26, m_to, 260)
    bv, mv = ms_track(m)
    p0 = np.array([bv[-1], mv[-1]])
    tip = np.array([1.52, max(-3.0, p0[1] - 3.6)])
    ctrl = np.array([p0[0] + 0.42, p0[1] - 0.15])
    post = bezier(p0, ctrl, tip)
    return (np.concatenate([bv, post[:, 0]]),
            np.concatenate([mv, post[:, 1]]), m_to, p0)


def sample_cluster(rng, age_gyr, n_ms):
    bvi, mvi, m_to, to_pt = isochrone(age_gyr)
    a, mlo = 1.8, 0.30
    u = rng.random(n_ms)
    m = ((m_to ** (1 - a) - mlo ** (1 - a)) * u + mlo ** (1 - a)) \
        ** (1 / (1 - a))
    bv, mv = ms_track(m)
    bv = bv + rng.normal(0, 0.022, m.size)
    mv = mv + rng.normal(0, 0.075, m.size)
    binary = rng.random(m.size) < 0.24
    mv[binary] -= 0.72 + rng.normal(0, 0.06, binary.sum())
    bv[binary] += 0.035

    # sparse post-main-sequence population along the Bezier segment
    n_post = max(6, int(0.045 * n_ms))
    t = rng.random(n_post) ** 1.7
    idx = (t * 159).astype(int)
    post = bezier(to_pt, np.array([to_pt[0] + 0.42, to_pt[1] - 0.15]),
                  np.array([1.52, max(-3.0, to_pt[1] - 3.6)]))
    pbv = post[idx, 0] + rng.normal(0, 0.030, n_post)
    pmv = post[idx, 1] + rng.normal(0, 0.10, n_post)

    extra_bv, extra_mv = [], []
    if age_gyr > 1.0:                                     # red clump
        k = 10 if age_gyr < 5 else 16
        extra_bv.append(rng.normal(1.02, 0.035, k))
        extra_mv.append(rng.normal(0.68, 0.10, k))
    if age_gyr > 8.0:                                     # horizontal branch
        k = 18
        u = rng.random(k)
        extra_bv.append(-0.05 + 0.62 * u + rng.normal(0, 0.03, k))
        extra_mv.append(0.55 + 0.20 * np.sin(np.pi * u) +
                        rng.normal(0, 0.07, k))
    if extra_bv:
        pbv = np.concatenate([pbv] + extra_bv)
        pmv = np.concatenate([pmv] + extra_mv)

    cbv = np.concatenate([bv, pbv])
    cmv = np.concatenate([mv, pmv])
    prob = np.clip(rng.beta(6.0, 1.6, cbv.size), 0.5, 1.0)

    n_f = 45
    fbv = rng.uniform(-0.25, 1.85, n_f)
    fmv = rng.uniform(-2.5, 9.2, n_f)
    return (cbv, cmv, prob, fbv, fmv, bvi, mvi, to_pt)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(71)

    data = [sample_cluster(rng, age, n) for (_, age, _), n in
            zip(CLUSTERS, (520, 430, 620))]

    # ---- one pass over every panel to fix the shared axis ranges -----------
    all_bv = np.concatenate([np.concatenate([d[0], d[3], d[5]]) for d in data])
    all_mv = np.concatenate([np.concatenate([d[1], d[4], d[6]]) for d in data])
    xlo, xhi = np.floor(all_bv.min() * 10) / 10, np.ceil(all_bv.max() * 10) / 10
    ylo, yhi = np.floor(all_mv.min()), np.ceil(all_mv.max())
    xlim = (xlo - 0.08, xhi + 0.08)
    ylim = (yhi + 0.4, ylo - 0.5)              # inverted: bright at the top

    cmap = mpl.colors.LinearSegmentedColormap.from_list("mem", MEMBER_COLORS)
    norm = mpl.colors.Normalize(0.5, 1.0)

    fig = plt.figure(figsize=(10.2, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.34], hspace=0.30,
                          wspace=0.075, left=0.062, right=0.895,
                          top=0.870, bottom=0.075)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_l = fig.add_subplot(gs[1, :])

    for i, (ax, (name, age, kind)) in enumerate(zip(axes, CLUSTERS)):
        cbv, cmv, prob, fbv, fmv, bvi, mvi, to_pt = data[i]
        ax.plot(fbv, fmv, ls="none", marker="x", ms=2.8, mew=0.6,
                color="#c3c7cc", zorder=2)
        sc = ax.scatter(cbv, cmv, c=prob, cmap=cmap, norm=norm, s=5.5,
                        linewidths=0.0, zorder=3)
        ax.plot(bvi, mvi, color=GOLD, lw=1.5, zorder=4)
        ax.plot(*to_pt, marker="*", ms=11, mfc=RUST, mec="white", mew=0.8,
                ls="none", zorder=6)
        ax.annotate("turn-off\n$M_V = %.1f$" % to_pt[1], xy=to_pt,
                    xytext=(to_pt[0] + 0.62, to_pt[1] + 1.5), fontsize=6.8,
                    color=RUST, ha="center", va="center", linespacing=1.4,
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.26", fc="white",
                              ec=RUST, lw=0.55, alpha=0.92),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color=RUST,
                                    shrinkB=5,
                                    connectionstyle="arc3,rad=-0.25"))
        if age > 1.0:
            ax.annotate("red clump", xy=(1.02, 0.68), xytext=(1.34, -1.35),
                        fontsize=6.6, color=CYAN, ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.7, color=CYAN,
                                        shrinkB=4,
                                        connectionstyle="arc3,rad=0.25"))
        if age > 8.0:
            ax.annotate("horizontal branch", xy=(0.30, 0.62),
                        xytext=(0.30, -1.55), fontsize=6.6, color=INDIGO,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                        color=INDIGO, shrinkB=4))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(np.arange(0.0, 1.81, 0.5))
        ax.set_yticks(np.arange(-2, 10, 2))
        ax.tick_params(labelsize=7.0, length=2.6)
        ax.grid(True, lw=0.4, color="#eceef1")
        ax.set_axisbelow(True)
        ax.set_xlabel("$B-V$ (mag)", fontsize=8.2, labelpad=2)
        if i == 0:
            ax.set_ylabel("absolute magnitude  $M_V$ (mag)", fontsize=8.2,
                          labelpad=3)
        else:
            ax.set_yticklabels([])
        age_txt = ("%.0f Myr" % (age * 1000)) if age < 1 else \
            ("%.1f Gyr" % age)
        ax.set_title("%s  —  %s" % (name, age_txt), fontsize=8.8,
                     fontweight="bold", pad=4)
        ax.text(0.972, 0.028, "%s\n%d members\n"
                "$\\log(\\mathrm{age/yr}) = %.2f$"
                % (kind, cbv.size, np.log10(age * 1e9)),
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.6, linespacing=1.45,
                zorder=8, bbox=dict(boxstyle="round,pad=0.28", fc="white",
                                    ec="#cccccc", lw=0.55, alpha=0.94))

    # ---- shared colourbar ---------------------------------------------------
    pos = axes[-1].get_position()
    cax = fig.add_axes([0.908, pos.y0 + 0.16 * pos.height, 0.015,
                        0.68 * pos.height])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("membership probability  $p$", fontsize=7.4, labelpad=4)
    cb.ax.tick_params(labelsize=6.6, width=0.6, length=2.4)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")

    # ---- demoted strip: luminosity functions on the shared magnitude axis --
    ax_l.set_facecolor("#f4f5f7")
    edges = np.arange(np.floor(ylim[1]), np.ceil(ylim[0]) + 0.5, 0.5)
    cols = [CYAN, GOLD, INDIGO]
    for (name, age, _), d, c in zip(CLUSTERS, data, cols):
        ax_l.hist(d[1], bins=edges, histtype="step", lw=1.1, color=c,
                  label=name)
        ax_l.plot(0.5 * (edges[1:] + edges[:-1]),
                  np.histogram(d[1], bins=edges)[0], ls="none", marker="o",
                  ms=2.0, mfc=c, mec="none", alpha=0.7)
    ax_l.set_xlim(ylim[1], ylim[0])
    ax_l.set_yscale("log")
    ax_l.set_ylim(0.7, 400)
    ax_l.set_xlabel("absolute magnitude  $M_V$ (mag)   —  fainter to the "
                    "right", fontsize=7.8, labelpad=1.8)
    ax_l.set_ylabel("stars / 0.5 mag", fontsize=7.4, labelpad=2)
    ax_l.tick_params(labelsize=6.6, length=2.2)
    ax_l.grid(True, lw=0.4, color="#e3e5e8")
    ax_l.set_axisbelow(True)
    for s in ax_l.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_l.legend(loc="upper left", fontsize=6.8, frameon=False, ncol=3,
                columnspacing=1.6, handlelength=1.6)
    ax_l.set_title("Supporting statistic  |  V-band luminosity functions of "
                   "the same three samples", fontsize=7.2, style="italic",
                   color="#555555", pad=2.5)

    for ax, lab in zip(axes + [ax_l], ["(a)", "(b)", "(c)", "(d)"]):
        p = ax.get_position()
        add_panel_label(ax, lab, x=-0.030 / p.width, y=1.0 + 0.026 / p.height,
                        size=10)

    handles = [Line2D([], [], color=GOLD, lw=1.6, label="isochrone"),
               Line2D([], [], marker="*", ls="none", ms=9, mfc=RUST,
                      mec="white", mew=0.7, label="main-sequence turn-off"),
               Line2D([], [], marker="o", ls="none", ms=4, mfc=NAVY,
                      mec="none", label="cluster member ($p>0.5$)"),
               Line2D([], [], marker="x", ls="none", ms=4, mew=0.7,
                      color="#c3c7cc", label="field contamination")]
    savefig(fig, "237_color_magnitude.png")


if __name__ == "__main__":
    main()
