"""Figure 680 - Emergency response reachability.

Main panel: travel-time isochrones from the nearest response station, drawn
over the street grid, with the eight-minute service target picked out as a
single heavy green contour and everything past fifteen minutes outlined in
house red.  Stations, and the incident under investigation, sit on top.

Right column at 58 % width: (b) the cumulative distribution of response time
over simulated calls, (c) the same calls split by incident class.  Both are
sampled *from the field drawn in (a)* - the coverage percentage in the map
caption, the median in (b) and every box in (c) come from one array, so they
cannot drift apart (DESIGN_SYSTEM 13.2).
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import axes_inches, safe_colorbar, panel_label_inches, halo_text

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"
BLUE, RED, GREEN = "#3b5b92", "#b23a48", "#2e7d5b"

W_KM, H_KM = 12.0, 10.4
TARGET_MIN = 8.0            # statutory arrival target
LATE_MIN = 12.0             # beyond this the cell is escalated
DETOUR = 1.40               # street distance / straight-line distance
DISPATCH_MIN = 2.0          # call handling plus turnout
V_BASE = 26.0               # km/h on an unimpeded arterial

# x km, y km, marker, class label
STATIONS = [(2.1, 8.3, "s", "Fire"), (9.2, 8.9, "s", "Fire"),
            (4.6, 5.4, "^", "Ambulance"), (10.3, 4.1, "^", "Ambulance"),
            (6.9, 2.0, "o", "Police"), (1.6, 2.6, "o", "Police")]
CLASSES = [("Medical", 0.0), ("Traffic", 1.1), ("Fire", 2.3),
           ("Public order", 3.4)]
INCIDENT = (7.8, 6.7)


def rasterize(artist):
    """3.10 mis-reports ContourSet rasterisation as ignored; it is not."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(680)
    fig = plt.figure(figsize=(10.4, 6.4))

    nx, ny = 260, 226
    xs = np.linspace(0.0, W_KM, nx)
    ys = np.linspace(0.0, H_KM, ny)
    X, Y = np.meshgrid(xs, ys)

    rough = gaussian_filter(rng.normal(0.0, 1.0, (ny, nx)), 14.0,
                            mode="nearest")
    rough = rough / (np.abs(rough).max() or 1.0)
    speed = np.clip(V_BASE * (1.0 + 0.42 * rough), 13.0, 40.0)

    tt = np.full((ny, nx), np.inf)
    for sx, sy, _, _ in STATIONS:
        d = np.hypot(X - sx, Y - sy) * DETOUR
        tt = np.minimum(tt, DISPATCH_MIN + 60.0 * d / speed)

    covered = 100.0 * float((tt <= TARGET_MIN).mean())

    # ---- simulated calls, drawn from the very field plotted in (a) --------
    dens = gaussian_filter(rng.random((ny, nx)), 12.0, mode="nearest")
    dens = (dens - dens.min()) ** 2
    p = (dens / dens.sum()).ravel()
    idx = rng.choice(p.size, 640, p=p)
    r_i, c_i = np.unravel_index(idx, tt.shape)
    base_t = tt[r_i, c_i] * np.exp(rng.normal(0.0, 0.13, idx.size))
    cls = rng.integers(0, len(CLASSES), idx.size)
    resp = base_t + np.array([CLASSES[k][1] for k in cls])
    within = 100.0 * float((resp <= TARGET_MIN).mean())

    # =====================================================================
    # (a) isochrone map
    # =====================================================================
    axm = axes_inches(fig, (0.82, 0.95, 4.90, 4.247))
    levels = np.arange(0.0, 33.0, 3.0)
    cf = axm.contourf(X, Y, np.clip(tt, 0.0, 31.5), levels=levels,
                      cmap=get_cmap("cb_sequential_blue", reverse=True),
                      zorder=1)
    rasterize(cf)
    cf.set_edgecolor("face")

    for k in range(1, 9):
        axm.plot([0.0, W_KM], [k * 1.16 + 0.1 * np.sin(k), k * 1.16],
                 lw=0.55, color="#ffffff", alpha=0.55, zorder=2)
    for k in range(1, 11):
        axm.plot([k * 1.09, k * 1.09 + 0.14 * np.cos(k)], [0.0, H_KM],
                 lw=0.55, color="#ffffff", alpha=0.55, zorder=2)
    axm.plot([0.3, 11.7], [3.2, 7.6], lw=1.0, color="#ffffff", alpha=0.70,
             zorder=2)
    axm.plot([1.0, 10.9], [9.9, 1.2], lw=1.0, color="#ffffff", alpha=0.70,
             zorder=2)

    axm.contour(X, Y, tt, levels=[TARGET_MIN], colors=[GREEN],
                linewidths=1.9, zorder=4)
    axm.contour(X, Y, tt, levels=[LATE_MIN], colors=[RED], linewidths=1.3,
                linestyles=[(0, (4.0, 2.2))], zorder=4)
    axm.plot([], [], lw=1.9, color=GREEN,
             label="%.0f min target isochrone" % TARGET_MIN)
    axm.plot([], [], lw=1.3, color=RED, ls=(0, (4.0, 2.2)),
             label="%.0f min escalation limit" % LATE_MIN)

    seen = set()
    for sx, sy, mk, lab in STATIONS:
        axm.plot([sx], [sy], marker=mk, ms=6.2, mfc="white", mec=INK,
                 mew=1.1, zorder=6,
                 label=None if lab in seen else "%s station" % lab)
        seen.add(lab)
    ix, iy = INCIDENT
    axm.plot([ix], [iy], marker="*", ms=13.0, mfc=RED, mec="white", mew=0.9,
             zorder=7, label="incident under review")
    best = int(np.argmin([np.hypot(ix - s[0], iy - s[1]) for s in STATIONS]))
    axm.plot([ix, STATIONS[best][0]], [iy, STATIONS[best][1]], lw=1.1,
             color=RED, ls=(0, (2.4, 1.8)), zorder=6)
    t_here = float(tt[int(round(iy / H_KM * (ny - 1))),
                      int(round(ix / W_KM * (nx - 1)))])
    halo_text(axm, ix + 0.25, iy + 0.30, "%.1f min" % t_here, ha="left",
              va="bottom", fontsize=6.8, color="white", halo=RED, zorder=8)

    axm.plot([0.55, 2.55], [0.55, 0.55], lw=2.6, color=INK,
             solid_capstyle="butt", zorder=7)
    halo_text(axm, 1.55, 0.75, "2 km", ha="center", va="bottom",
              fontsize=6.4, color="white", halo=INK, zorder=8)

    axm.set_xlim(0.0, W_KM)
    axm.set_ylim(0.0, H_KM)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    leg = axm.legend(loc="upper left", fontsize=6.6, frameon=True,
                     framealpha=0.92, facecolor="white", edgecolor=HAIR,
                     borderpad=0.4, handletextpad=0.4, labelspacing=0.28)
    leg.set_zorder(9)
    safe_colorbar(fig, cf, axm, label="response travel time (min)",
                  width=0.11, pad=0.07, labelsize=6.8,
                  ticks=np.arange(0.0, 33.0, 6.0))
    axm.set_title("Nearest-station isochrones, %.0f %% of area inside "
                  "%.0f min" % (covered, TARGET_MIN), fontsize=9.0,
                  fontweight="bold", pad=4.0)

    # =====================================================================
    # (b) cumulative distribution of response time
    # =====================================================================
    axc = axes_inches(fig, (7.10, 3.44, 2.90, 1.76))
    srt = np.sort(resp)
    cdf = 100.0 * np.arange(1, srt.size + 1) / srt.size
    axc.fill_between(srt, 0.0, cdf, where=srt <= TARGET_MIN,
                     facecolor=lighten(GREEN, 0.82), edgecolor="none",
                     zorder=1)
    axc.plot(srt, cdf, lw=1.6, color=BLUE, zorder=3)
    axc.axvline(TARGET_MIN, color=GREEN, lw=1.2, ls=(0, (4.0, 2.2)), zorder=4)
    axc.axhline(90.0, color=BASE, lw=0.9, ls=(0, (2.4, 1.8)), zorder=4)
    p90 = float(np.percentile(resp, 90))
    med = float(np.median(resp))
    axc.plot([TARGET_MIN], [within], marker="o", ms=4.0, mfc="white",
             mec=GREEN, mew=1.3, zorder=5)
    axc.annotate("%.0f %% of calls met the\n%.0f min target"
                 % (within, TARGET_MIN), xy=(TARGET_MIN, within),
                 xytext=(10.0, -26.0), textcoords="offset points",
                 ha="left", va="center", fontsize=6.6, color=GREEN)
    axc.text(0.25, 91.0, "90th percentile %.1f min" % p90, ha="left",
             va="bottom", fontsize=6.4, color=SUB)
    axc.set_xlim(0.0, min(resp.max(), 32.0))
    axc.set_ylim(0.0, 104.0)
    axc.set_xlabel("response time (min)", fontsize=7.4)
    axc.set_ylabel("calls answered (%)", fontsize=7.4)
    axc.tick_params(labelsize=6.5, length=2.2)
    axc.grid(axis="y", color=GRID, lw=0.55)
    axc.set_title("Cumulative response time, n=%d calls" % resp.size,
                  fontsize=8.5, fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    # =====================================================================
    # (c) response time by incident class
    # =====================================================================
    axb = axes_inches(fig, (7.10, 0.95, 2.90, 1.76))
    groups = [resp[cls == k] for k in range(len(CLASSES))]
    bp = axb.boxplot(groups, vert=True, widths=0.56, patch_artist=True,
                     showfliers=False, zorder=3)
    for k, (box, grp) in enumerate(zip(bp["boxes"], groups)):
        late = float(np.median(grp)) > TARGET_MIN
        box.set_facecolor(lighten(RED if late else BLUE, 0.68))
        box.set_edgecolor(RED if late else BLUE)
        box.set_linewidth(0.9)
    for key, col in (("whiskers", SUB), ("caps", SUB), ("medians", INK)):
        for a in bp[key]:
            a.set_color(col)
            a.set_linewidth(1.0)
    axb.axhline(TARGET_MIN, color=GREEN, lw=1.2, ls=(0, (4.0, 2.2)), zorder=4,
                label="%.0f min target" % TARGET_MIN)
    axb.set_xticks(np.arange(1, len(CLASSES) + 1))
    axb.set_xticklabels([c[0] for c in CLASSES], fontsize=6.5)
    top = min(np.percentile(resp, 99) * 1.22, 34.0)
    axb.set_ylim(0.0, top)
    for k, grp in enumerate(groups):
        axb.text(k + 1, top * 0.985, "n=%d" % grp.size, ha="center",
                 va="top", fontsize=6.2, color=MUTED)
    axb.legend(loc="lower left", fontsize=6.5, frameon=False,
               handlelength=1.6)
    axb.set_ylabel("response time (min)", fontsize=7.4)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="y", color=GRID, lw=0.55)
    axb.set_title("Class-specific duration, same calls", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    for ax, lab in ((axm, "(a)"), (axc, "(b)"), (axb, "(c)")):
        panel_label_inches(fig, ax, lab, dx=-0.44, dy=0.12)
    savefig(fig, "680_sct_emergency_response.png")


if __name__ == "__main__":
    main()
