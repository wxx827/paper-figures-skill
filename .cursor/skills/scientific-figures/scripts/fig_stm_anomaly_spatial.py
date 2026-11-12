"""Spatio-temporal anomaly detection over a week of city sensor records.

One event list drives all three panels, so they cannot disagree: the map is
the severity-weighted kernel density of those events, the time series is
their hourly total, and the strip chart is the events themselves.  The heavy
outline on the map and the shaded bands on the time series are the same
detection rule seen in space and in time - three incident clusters rise
above a threshold that is drawn, not asserted.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, get_cmap, COLORS
from figkit import add_scale_bar, panel_labels_inches, safe_colorbar

W, H = 6.4, 6.0                      # city extent in km
T_END = 168.0                        # one week, in hours
NCOL, NROW = 4, 3                    # region grid -> 12 reporting zones
K_SD = 2.1                           # detection threshold, in s.d. of score
INK = COLORS["ink"]
SUB = COLORS["sub"]
MUTED = COLORS["muted"]
ACCENT = "#b23a48"                   # the single emphasis colour
CALM = "#8fa8bf"
FRAME = "#c9c6bf"

# x, y, spread (km), peak hour, duration (h), n events, label anchor, label
INCIDENTS = [
    (1.45, 4.35, 0.50, 33.0, 6.0, 26, (1.45, 5.62), "A1 depot outage"),
    (4.85, 1.95, 0.44, 88.0, 4.6, 23, (4.85, 0.62), "A2 stadium egress"),
    (2.25, 1.70, 0.40, 133.0, 4.2, 24, (0.95, 2.95),
     "A3 flooded underpass"),
]


def region_id(x, y):
    """Zone index R01..R12 for a point, row-major from the north-west."""
    c = np.clip((x / W * NCOL).astype(int), 0, NCOL - 1)
    r = np.clip(((H - y) / H * NROW).astype(int), 0, NROW - 1)
    return r * NCOL + c


def build_events(rng):
    """Background chatter plus three concentrated incident clusters."""
    n_bg = 165
    x = rng.uniform(0.2, W - 0.2, n_bg)
    y = rng.uniform(0.2, H - 0.2, n_bg)
    t = rng.uniform(0.0, T_END, n_bg)
    s = 0.8 + rng.gamma(1.5, 0.36, n_bg)
    for cx, cy, sp, t0, dur, n, _anchor, _lab in INCIDENTS:
        x = np.r_[x, np.clip(rng.normal(cx, sp, n), 0.1, W - 0.1)]
        y = np.r_[y, np.clip(rng.normal(cy, sp, n), 0.1, H - 0.1)]
        t = np.r_[t, np.clip(rng.normal(t0, dur / 2.2, n), 0.0, T_END)]
        s = np.r_[s, 2.6 + rng.gamma(2.4, 0.85, n)]
    return x, y, t, s


def main():
    set_pub_style()
    rng = np.random.default_rng(485)
    ex, ey, et, es = build_events(rng)
    reg = region_id(ex, ey)

    # ---- spatial score: severity-weighted kernel density of the events ----
    gx = np.linspace(0, W, 200)
    gy = np.linspace(0, H, int(200 * H / W))
    X, Y = np.meshgrid(gx, gy)
    bw = 0.40
    field = np.zeros_like(X)
    for xi, yi, si in zip(ex, ey, es):
        field += si * np.exp(-(((X - xi) ** 2 + (Y - yi) ** 2)
                               / (2 * bw ** 2)))
    field = 100.0 * field / field.max()
    s_thr = 62.0

    # ---- temporal score: hourly severity total, lightly smoothed ---------
    hb = np.arange(0, T_END + 1.0, 1.0)
    hourly, _ = np.histogram(et, bins=hb, weights=es)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    k = k / k.sum()
    score = np.convolve(np.r_[hourly[:6][::-1], hourly, hourly[-6:][::-1]],
                        k, mode="same")[6:-6]
    hc = 0.5 * (hb[:-1] + hb[1:])
    t_thr = float(np.mean(score) + K_SD * np.std(score))
    flag = score > t_thr

    fig = plt.figure(figsize=(10.4, 6.4))
    axm = fig.add_axes([0.030, 0.062, 0.545, 0.818])
    axt = fig.add_axes([0.655, 0.585, 0.315, 0.295])
    axe = fig.add_axes([0.655, 0.088, 0.315, 0.310])

    # ------------------------------------------------------- (a) score map
    cmap = get_cmap("cb_sequential_warm")
    im = axm.imshow(field, origin="lower", extent=[0, W, 0, H], cmap=cmap,
                    vmin=0.0, vmax=100.0, interpolation="bilinear", zorder=1)
    for c in range(1, NCOL):
        axm.axvline(c * W / NCOL, color="#b9b5ae", lw=0.55,
                    ls=(0, (2.5, 2.0)), zorder=2)
    for r in range(1, NROW):
        axm.axhline(r * H / NROW, color="#b9b5ae", lw=0.55,
                    ls=(0, (2.5, 2.0)), zorder=2)
    for i in range(NCOL * NROW):
        cx = (i % NCOL + 0.5) * W / NCOL
        cy = H - (i // NCOL + 0.5) * H / NROW
        axm.text(cx, cy, f"R{i + 1:02d}", fontsize=6.4, color="#8a8a8a",
                 ha="center", va="center", alpha=0.85, zorder=3)
    axm.contour(gx, gy, field, levels=[s_thr], colors=[ACCENT],
                linewidths=1.9, zorder=5)
    axm.scatter(ex, ey, s=2.6 + 3.4 * es, facecolors="none",
                edgecolors="#5a5a5a", linewidths=0.35, alpha=0.55, zorder=4)

    for cx, cy, sp, _t0, _d, _n, anchor, lab in INCIDENTS:
        edge = cy + sp * 1.1 if anchor[1] > cy else cy - sp * 1.1
        axm.annotate(lab, xy=(cx, edge), xytext=anchor,
                     fontsize=6.7, color="#7a2f2a", fontweight="bold",
                     ha="center", va="center", zorder=8,
                     bbox=dict(boxstyle="round,pad=0.24", fc="white",
                               ec="#e0cfcb", lw=0.6, alpha=0.92),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT,
                                     shrinkA=1, shrinkB=3))

    axm.set_xlim(0, W)
    axm.set_ylim(0, H)
    axm.set_aspect("equal")
    axm.set_xticks([])
    axm.set_yticks([])
    for sp in axm.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_edgecolor(FRAME)
    add_scale_bar(axm, 1.0, "1 km", loc=(0.032, 0.042), height=0.014,
                  color="#333333", text_color="#1a1a1a", halo="white",
                  fontsize=6.2)
    safe_colorbar(fig, im, axm, label="anomaly score (-)", width=0.11,
                  pad=0.07, labelsize=6.8)
    axm.text(0.988, 0.020, f"heavy outline: score = {s_thr:.0f}; "
             f"circles: {len(es)} events sized by severity",
             transform=axm.transAxes, ha="right", va="bottom", fontsize=6.3,
             color=SUB, zorder=9)

    # -------------------------------------------------- (b) temporal score
    axt.fill_between(hc, 0.0, score, color=CALM, alpha=0.32, zorder=2)
    axt.plot(hc, score, color="#3b5b92", lw=1.0, zorder=4)
    axt.axhline(t_thr, color=ACCENT, lw=1.0, ls=(0, (3.2, 2.0)), zorder=5)
    edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        axt.axvspan(hc[a] - 0.5, hc[b - 1] + 0.5, color=ACCENT, alpha=0.14,
                    lw=0, zorder=1)
    axt.text(2.0, t_thr * 1.06, f"threshold = mean + {K_SD:.1f} s.d. "
             f"({t_thr:.0f})", fontsize=6.4, color=ACCENT, ha="left",
             va="bottom", zorder=6)
    axt.set_xlim(0, T_END)
    axt.set_ylim(0, score.max() * 1.30)
    axt.set_xticks(np.arange(0, T_END + 1, 24))
    axt.set_xticklabels([f"d{i}" for i in range(8)], fontsize=6.6)
    axt.set_xlabel("time (day of week)", fontsize=7.4, labelpad=2)
    axt.set_ylabel("hourly score (-)", fontsize=7.4, labelpad=2)
    axt.yaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))
    axt.set_title("City-wide score over one week", fontsize=8.6,
                  fontweight="bold", color=INK, pad=4)
    axt.tick_params(labelsize=6.6, length=2.4)
    axt.grid(axis="y", color="#ececec", lw=0.55, zorder=0)

    # ----------------------------------------------- (c) event strip chart
    sev_thr = float(np.quantile(es, 0.88))
    hot = es > sev_thr
    axe.scatter(et[~hot], reg[~hot] + 1, s=3.0 + 5.0 * es[~hot],
                facecolors=CALM, edgecolors="white", linewidths=0.3,
                alpha=0.80, zorder=3)
    axe.scatter(et[hot], reg[hot] + 1, s=3.0 + 5.0 * es[hot],
                facecolors=ACCENT, edgecolors="white", linewidths=0.35,
                alpha=0.90, zorder=4)
    for a, b in zip(edges[::2], edges[1::2]):
        axe.axvspan(hc[a] - 0.5, hc[b - 1] + 0.5, color=ACCENT, alpha=0.10,
                    lw=0, zorder=1)
    axe.set_xlim(0, T_END)
    axe.set_ylim(0.3, NCOL * NROW + 0.7)
    axe.set_xticks(np.arange(0, T_END + 1, 24))
    axe.set_xticklabels([f"d{i}" for i in range(8)], fontsize=6.6)
    axe.set_yticks(np.arange(1, NCOL * NROW + 1, 2))
    axe.set_yticklabels([f"R{i:02d}" for i in range(1, NCOL * NROW + 1, 2)],
                        fontsize=6.4)
    axe.set_xlabel("time (day of week)", fontsize=7.4, labelpad=2)
    axe.set_ylabel("region", fontsize=7.4, labelpad=2)
    axe.set_title("Events in space and time", fontsize=8.6,
                  fontweight="bold", color=INK, pad=4)
    axe.tick_params(labelsize=6.6, length=2.4)
    axe.grid(axis="y", color="#f0efec", lw=0.5, zorder=0)

    hs = [Line2D([], [], marker="o", ls="none", ms=3.6, mfc=CALM,
                 mec="white", mew=0.3, label="routine"),
          Line2D([], [], marker="o", ls="none", ms=5.0, mfc=ACCENT,
                 mec="white", mew=0.3,
                 label=f"severity > {sev_thr:.1f} (top 12 %)")]
    axe.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.40),
               ncol=2, fontsize=6.5, frameon=False, handletextpad=0.4,
               columnspacing=1.0)
    axe.text(0.0, -0.52, "marker area: event severity (-)",
             transform=axe.transAxes, fontsize=6.3, color=MUTED, ha="left",
             va="top")

    panel_labels_inches(fig, [axm, axt, axe], dx=-0.20, dy=0.30)

    savefig(fig, "485_stm_anomaly_spatial.png")


if __name__ == "__main__":
    main()
