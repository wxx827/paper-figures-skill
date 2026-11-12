"""Temperature-salinity diagram with isopycnals and water-mass envelopes.

Every CTD sample is plotted in salinity-temperature space and coloured by the
depth at which it was collected, so the mixing path from warm fresh surface
water down to cold saline deep water is read as a single curve. Grey contours
are lines of constant potential density anomaly (sigma-t) computed from a
simplified equation of state; dashed envelopes mark the four water masses.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"

# salinity (psu), temperature (degC), depth (m) along the mean mixing path
PATH = np.array([
    [33.72, 24.8, 0], [34.10, 21.6, 45], [34.62, 19.2, 90],
    [35.02, 17.4, 135], [34.96, 14.2, 215], [34.72, 10.6, 330],
    [34.48, 8.0, 470], [34.40, 6.3, 640], [34.50, 4.4, 860],
    [34.62, 3.0, 1120], [34.70, 2.1, 1450],
])

MASSES = [
    dict(name="Surface Water", band=(0, 85), lx=33.52, ly=18.9, ha="left",
         tx=33.94, ty=21.6),
    dict(name="Subtropical Underwater\n(salinity maximum)", band=(115, 250),
         lx=35.36, ly=22.6, ha="right", tx=34.99, ty=18.9),
    dict(name="Intermediate Water\n(salinity minimum)", band=(360, 700),
         lx=33.54, ly=9.6, ha="left", tx=34.33, ty=7.4),
    dict(name="Deep Water", band=(950, 1500), lx=35.36, ly=5.4, ha="right",
         tx=34.72, ty=3.0),
]


def sigma_t(s, t):
    """Potential density anomaly, simplified Millero-style polynomial."""
    return (28.152 - 0.0735 * t - 0.00469 * t ** 2
            + (0.802 - 0.002 * t) * (s - 35.0))


def sample(rng, n=1500):
    u = np.sort(rng.random(n)) ** 0.85
    k = np.linspace(0, 1, len(PATH))
    s = np.interp(u, k, PATH[:, 0])
    t = np.interp(u, k, PATH[:, 1])
    d = np.interp(u, k, PATH[:, 2])
    spread = 0.9 * np.exp(-d / 260.0) + 0.10
    s = s + rng.normal(0, 0.075, n) * spread
    t = t + rng.normal(0, 0.95, n) * spread
    return s, t, d


def main():
    set_pub_style()
    rng = np.random.default_rng(31)
    s, t, d = sample(rng)

    fig = plt.figure(figsize=(8.0, 6.4))
    ax = fig.add_axes([0.088, 0.098, 0.700, 0.775])
    cax = fig.add_axes([0.836, 0.170, 0.019, 0.560])

    slim, tlim = (33.45, 35.42), (0.0, 27.5)
    sg = np.linspace(*slim, 220)
    tg = np.linspace(*tlim, 220)
    SG, TG = np.meshgrid(sg, tg)
    dens = sigma_t(SG, TG)

    cs = ax.contour(SG, TG, dens, levels=np.arange(20.0, 29.01, 0.5),
                    colors="#b9bfc4", linewidths=0.6, zorder=2)
    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6.4, fmt="%.1f",
              inline_spacing=4)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "depth", ["#dceaf2", "#a9cbdd", "#6ea6c2", "#3f7fa3", "#255c7e",
                  "#123c5a", "#0a2438"])
    norm = mpl.colors.Normalize(0, 1500)
    sc = ax.scatter(s, t, c=d, cmap=cmap, norm=norm, s=11, lw=0.0,
                    alpha=0.88, zorder=4)
    ax.plot(PATH[:, 0], PATH[:, 1], color="#ffffff", lw=2.4, zorder=5,
            alpha=0.85, solid_capstyle="round")
    ax.plot(PATH[:, 0], PATH[:, 1], color=C_ACC, lw=1.1, zorder=6,
            ls=(0, (4.5, 2.2)), label="mean mixing path")

    # ---- water-mass envelopes, fitted in axes space so tilts look right ---
    def to_frac(sv, tv):
        return ((sv - slim[0]) / (slim[1] - slim[0]),
                (tv - tlim[0]) / (tlim[1] - tlim[0]))

    for m in MASSES:
        sel = (d >= m["band"][0]) & (d < m["band"][1])
        fx, fy = to_frac(s[sel], t[sel])
        pts = np.column_stack([fx, fy])
        mu = pts.mean(axis=0)
        val, vec = np.linalg.eigh(np.cov(pts.T))
        order = np.argsort(val)[::-1]
        val, vec = val[order], vec[:, order]
        ang = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
        ax.add_patch(Ellipse(mu, 4.4 * np.sqrt(val[0]), 4.4 * np.sqrt(val[1]),
                             angle=ang, transform=ax.transAxes, fill=False,
                             edgecolor="#2b4a5e", linewidth=0.9,
                             linestyle=(0, (3.2, 2.2)), alpha=0.9, zorder=7))
        ax.annotate(m["name"], xy=(m["tx"], m["ty"]),
                    xytext=(m["lx"], m["ly"]),
                    fontsize=7.4, color="#22404f", ha=m["ha"], va="center",
                    linespacing=1.4, zorder=9,
                    bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                              edgecolor="#d3dade", linewidth=0.6, alpha=0.94),
                    arrowprops=dict(arrowstyle="-", color="#5c7a8a", lw=0.7,
                                    connectionstyle="arc3,rad=0.18"))

    ax.set_xlim(*slim)
    ax.set_ylim(*tlim)
    ax.set_xticks(np.arange(33.5, 35.45, 0.25))
    ax.set_yticks(np.arange(0, 27.6, 5))
    ax.set_xlabel("Salinity (psu)")
    ax.set_ylabel("Potential temperature (\u00b0C)")
    ax.grid(color="#f0f0f0", lw=0.5, zorder=1)
    ax.set_axisbelow(True)
    style_axes(ax)

    ax.text(0.018, 0.055,
            "grey contours: $\\sigma_t$ (kg m$^{-3}$)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            color="#7a8288")
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.012, 0.088),
                    fontsize=7.4, handlelength=2.0)
    leg.set_clip_on(False)

    cb = fig.colorbar(sc, cax=cax, ticks=np.arange(0, 1501, 250))
    cb.set_label("Sample depth (m)", fontsize=8.2, labelpad=4)
    cb.ax.tick_params(labelsize=7.4, width=0.7, length=2.5)
    cb.ax.invert_yaxis()
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.solids.set_alpha(1.0)

    savefig(fig, "220_ts_diagram.png")


if __name__ == "__main__":
    main()
