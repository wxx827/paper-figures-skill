"""Seventy-two-hour back trajectories arriving at one receptor site.

Air parcels are integrated backwards from a single receptor and coloured by
the fine-particle concentration measured on arrival, which turns a tangle of
lines into a transport-pathway diagram: the slow south-westerly bundle is the
one that arrives dirty, the fast north-westerly bundle is the one that
ventilates.  The map is drawn in an equidistant projection about the receptor
so ``set_aspect("equal")`` is metrically true and the scale bar means what it
says; the two auxiliary panels are cluster statistics computed from exactly
the trajectories that are drawn.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import add_scale_bar, panel_labels_inches, safe_colorbar, halo_text

INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
ACCENT = "#b23a48"          # house red: receptor star and the exceedance line
BAR_C = "#3b5b92"           # house blue: neutral bars

# Domain-semantic map colours (DESIGN_SYSTEM section 2.2: declared as
# module-level constants with their meaning).
LAND = "#f6f5f2"
SEA = "#e7eef4"
COAST = "#9aa7b1"
GRAT = "#dcdcdc"

REC_LON, REC_LAT = 116.4, 39.9
LAT0 = 36.0
KM_LAT = 110.57
KM_LON = 111.32 * np.cos(np.deg2rad(LAT0))
LIMIT = 75.0                # ug m^-3, 24 h PM2.5 limit value

# (name, backward bearing deg, speed km/h, n members, mean arrival PM2.5)
CLUSTERS = [
    ("fast north-westerly from Mongolia", "NW fast", 312.0, 32.0, 6, 52.0),
    ("northerly under the Siberian ridge", "N", 349.0, 24.0, 5, 34.0),
    ("westerly off the Loess plateau", "W", 274.0, 20.0, 5, 92.0),
    ("slow south-westerly over the N China plain", "SW slow",
     224.0, 11.0, 6, 148.0),
    ("easterly off the Yellow Sea", "E marine", 104.0, 16.0, 4, 29.0),
]

COAST_LONLAT = np.array([
    [135.0, 55.0], [133.0, 50.2], [130.8, 47.4], [128.4, 45.6],
    [126.6, 43.2], [124.2, 40.9], [121.6, 39.4], [120.9, 37.6],
    [119.4, 36.3], [120.6, 34.6], [121.6, 32.2], [121.0, 29.8],
    [119.2, 27.4], [117.4, 25.6], [113.0, 22.6], [109.0, 21.2],
])


def to_km(lon, lat):
    return (np.asarray(lon) - REC_LON) * KM_LON, (np.asarray(lat) - REC_LAT) * KM_LAT


def trajectory(rng, bearing, speed, hours=72, dt=1.0):
    """Backward parcel path in kilometres from the receptor."""
    n = int(hours / dt) + 1
    b = np.deg2rad(bearing + rng.normal(0.0, 9.0))
    omega = np.deg2rad(rng.normal(0.0, 0.32))        # rad per hour of curving
    v = speed * float(rng.normal(1.0, 0.14))
    x, y = [0.0], [0.0]
    for i in range(1, n):
        b = b + omega * dt + np.deg2rad(rng.normal(0.0, 1.1))
        vv = v * (1.0 + 0.22 * np.sin(0.06 * i))
        x.append(x[-1] + vv * dt * np.sin(b))
        y.append(y[-1] + vv * dt * np.cos(b))
    return np.array(x), np.array(y)


def main():
    set_pub_style()
    rng = np.random.default_rng(555)

    traj, conc, member_of = [], [], []
    for ci, (_nm, _sh, bear, spd, n, pm) in enumerate(CLUSTERS):
        for _ in range(n):
            traj.append(trajectory(rng, bear, spd))
            conc.append(float(pm * rng.lognormal(0.0, 0.20)))
            member_of.append(ci)
    conc = np.array(conc)
    member_of = np.array(member_of)

    cmap = get_cmap("cb_sequential_warm")
    norm = plt.Normalize(vmin=float(conc.min()), vmax=float(conc.max()))

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58], left=0.030, right=0.972,
                          top=0.880, bottom=0.130, hspace=0.46, wspace=0.10)
    ax = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    # ---- basemap, sized so no trajectory is cut off at the frame ----------
    allx = np.concatenate([t[0] for t in traj])
    ally = np.concatenate([t[1] for t in traj])
    mx, my = 0.06 * float(np.ptp(allx)), 0.07 * float(np.ptp(ally))
    xlo, xhi = float(allx.min()) - mx, float(allx.max()) + mx
    ylo, yhi = float(ally.min()) - my, float(ally.max()) + my

    ax.add_patch(plt.Rectangle((xlo, ylo), xhi - xlo, yhi - ylo,
                               facecolor=LAND, edgecolor="none", zorder=0))
    cx, cy = to_km(COAST_LONLAT[:, 0], COAST_LONLAT[:, 1])
    far = max(abs(xhi), abs(xlo), abs(yhi), abs(ylo)) * 3.0
    poly = np.column_stack([np.r_[cx, far, far], np.r_[cy, -far, far]])
    ax.add_patch(plt.Polygon(poly, closed=True, facecolor=SEA,
                             edgecolor="none", zorder=1))
    ax.plot(cx, cy, color=COAST, lw=0.9, zorder=2)

    for lon in range(80, 141, 5):
        gx, _ = to_km(lon, LAT0)
        if not xlo + 0.03 * (xhi - xlo) < gx < xhi:
            continue
        ax.plot([gx, gx], [ylo, yhi], color=GRAT, lw=0.5, zorder=1)
        halo_text(ax, gx, ylo + 0.012 * (yhi - ylo), f"{lon}$^\\circ$E",
                  ha="center", va="bottom", fontsize=6.0, color=MUTED,
                  halo="white", halo_lw=1.6, zorder=3)
    for lat in range(20, 61, 5):
        _, gy = to_km(REC_LON, lat)
        if not ylo < gy < yhi - 0.03 * (yhi - ylo):
            continue
        ax.plot([xlo, xhi], [gy, gy], color=GRAT, lw=0.5, zorder=1)
        halo_text(ax, xlo + 0.010 * (xhi - xlo), gy, f"{lat}$^\\circ$N",
                  ha="left", va="bottom", fontsize=6.0, color=MUTED,
                  halo="white", halo_lw=1.6, zorder=3)

    # ---------------------------------------------------------- trajectories
    order = np.argsort(conc)                     # dirtiest drawn last, on top
    for i in order:
        x, y = traj[i]
        col = cmap(norm(conc[i]))
        ax.plot(x, y, color=col, lw=1.15, alpha=0.92, zorder=4,
                solid_capstyle="round")
        ax.plot(x[24::24], y[24::24], marker="o", ls="none", ms=2.8,
                mfc=col, mec="white", mew=0.5, zorder=5)

    ax.plot([0.0], [0.0], marker="*", ms=15.0, mfc=ACCENT, mec="white",
            mew=0.9, zorder=8)
    halo_text(ax, 0.016 * (xhi - xlo), 0.014 * (yhi - ylo),
              "receptor supersite", ha="left", va="bottom", fontsize=7.0,
              color=INK, halo="white", halo_lw=2.0, zorder=8)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_edgecolor("#c9c6bf")
    ax.set_title("72 h back trajectories coloured by arrival $PM_{2.5}$",
                 fontsize=9.0, fontweight="bold", color=INK)

    add_scale_bar(ax, 500.0, "500 km", loc=(0.040, 0.048), height=0.020,
                  n_seg=2, color="#333333", text_color="#1a1a1a",
                  halo="white", fontsize=6.2)

    # north arrow, in axes fractions so it never moves with the data
    ax.annotate("", xy=(0.930, 0.905), xytext=(0.930, 0.828),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333333",
                                mutation_scale=9))
    halo_text(ax, 0.930, 0.912, "N", transform=ax.transAxes, ha="center",
              va="bottom", fontsize=7.4, color="#333333", halo="white",
              halo_lw=2.0, zorder=8)

    ax.legend(handles=[Line2D([], [], marker="*", ls="none", ms=8.0,
                              mfc=ACCENT, mec="white", label="receptor"),
                       Line2D([], [], marker="o", ls="none", ms=3.4,
                              mfc="#8a8a8a", mec="white",
                              label="24 h time marker")],
              loc="lower right", fontsize=6.8, frameon=True, framealpha=0.90,
              facecolor="white", edgecolor="#cccccc", borderpad=0.4)

    safe_colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax,
                  side="bottom", label="arrival $PM_{2.5}$ ($\\mu$g m$^{-3}$)",
                  width=0.11, pad=0.09, shrink=0.72, labelsize=7.0)

    # ------------------------------------------------- cluster statistics
    names = [c[0] for c in CLUSTERS]
    short = [c[1] for c in CLUSTERS]
    n_mem = np.array([int((member_of == k).sum()) for k in range(5)])
    share = n_mem / n_mem.sum() * 100.0
    mean_pm = np.array([float(conc[member_of == k].mean()) for k in range(5)])
    sd_pm = np.array([float(conc[member_of == k].std(ddof=1)) for k in range(5)])
    rank = np.argsort(-mean_pm)                  # same order in both panels
    ypos = np.arange(5)[::-1]

    ax1.barh(ypos, share[rank], height=0.56, color=BAR_C, alpha=0.90,
             edgecolor="white", linewidth=0.6, zorder=3)
    for yy, v, n in zip(ypos, share[rank], n_mem[rank]):
        ax1.text(v + 0.9, yy, f"{v:.0f}%  (n={n})", va="center", ha="left",
                 fontsize=6.3, color=SUB)
    ax1.set_xlim(0, share.max() * 1.52)
    ax1.set_xlabel("share of trajectories (%)", fontsize=7.5)
    ax1.set_title("Cluster frequency", fontsize=8.5, fontweight="bold",
                  color=INK)

    bar_cols = [ACCENT if mean_pm[k] > LIMIT else BAR_C for k in rank]
    ax2.barh(ypos, mean_pm[rank], height=0.56, color=bar_cols, alpha=0.90,
             edgecolor="white", linewidth=0.6, zorder=3,
             xerr=sd_pm[rank], error_kw=dict(ecolor="#555555", elinewidth=0.7,
                                             capsize=1.8))
    ax2.axvline(LIMIT, color=ACCENT, lw=1.0, ls=(0, (3.2, 1.8)), zorder=4)
    ax2.text(LIMIT + 3.0, -0.62, f"24 h limit {LIMIT:.0f}", fontsize=6.3,
             color=ACCENT, ha="left", va="bottom")
    for yy, v, s in zip(ypos, mean_pm[rank], sd_pm[rank]):
        ax2.text(v + s + 5.0, yy, f"{v:.0f}", va="center", ha="left",
                 fontsize=6.3, color=SUB)
    ax2.set_xlim(0, float((mean_pm + sd_pm).max()) * 1.30)
    ax2.set_xlabel("mean arrival $PM_{2.5}$ ($\\mu$g m$^{-3}$)", fontsize=7.5)
    ax2.set_title("Concentration by cluster", fontsize=8.5, fontweight="bold",
                  color=INK)

    for a in (ax1, ax2):
        a.set_yticks(ypos)
        a.set_yticklabels([short[k] for k in rank], fontsize=7.0)
        a.set_ylim(-0.72, 4.72)
        a.tick_params(labelsize=6.8)
        a.grid(axis="x", color="#ececec", lw=0.55, zorder=0)

    worst = int(rank[0])
    panel_labels_inches(fig, [ax, ax1, ax2], dx=-0.24, dy=0.12)

    savefig(fig, "555_atm_back_trajectory.png")


if __name__ == "__main__":
    main()
