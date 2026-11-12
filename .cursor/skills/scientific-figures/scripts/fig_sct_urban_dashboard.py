"""Figure 678 - City operations dashboard.

A deliberately heterogeneous layout: five KPI cards on top, a trip-density
map beside three stacked diurnal series in the middle, and one full-width
incident timeline at the bottom.  Panels of that many different shapes only
read as one board if the grouping is drawn, so each band sits on its own
figure-level card (figkit.background_card, which sorts *behind* the axes -
the popular fig.patches.append recipe paints over them, DESIGN_SYSTEM 12.2)
and every axes keeps a white face so the card survives as a gutter.

Every printed number - the KPI values, the year-on-year deltas, the hotspot
density, the evening peak, the incident counts - is reduced from the arrays
that are plotted, so re-seeding moves all of them together
(DESIGN_SYSTEM 13.2).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, lighten
from figkit import (axes_inches, background_card, safe_colorbar, halo_text,
                    add_scale_bar, image_axes)

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
GRID, BASE = "#d9d9d9", "#808080"
BLUE, RED, GREEN = "#3b5b92", "#b23a48", "#2e7d5b"
GOLD, PURPLE = "#c9a227", "#8e6bbf"
CARD, CARD_ALT = "#f0f0f0", "#f5f5f5"

FW, FH = 12.6, 7.6

# name, unit, level, month-on-month noise, annual drift, +1 if up is good, fmt
KPIS = [
    ("Population flow", "M trips per day", 4.80, 0.005, 0.038, +1, "{:.2f}"),
    ("Traffic index", "congestion index (-)", 6.35, 0.006, 0.031, -1, "{:.2f}"),
    ("Air quality", "AQI (-)", 71.0, 0.009, -0.064, -1, "{:.0f}"),
    ("Grid energy", "GWh per day", 41.5, 0.005, 0.023, -1, "{:.1f}"),
    ("Incidents", "events per day", 318.0, 0.008, -0.047, -1, "{:.0f}"),
]

LANES = ["Road accident", "Urban flooding", "Power outage", "Public safety"]
STATUS = [("closed", GREEN), ("in progress", GOLD), ("escalated", RED)]

HOTSPOTS = [(9.5, 7.0, 1.9, 8.6), (15.2, 8.6, 1.4, 6.4),
            (5.6, 4.2, 1.2, 5.0), (19.4, 4.8, 1.1, 4.4),
            (12.4, 10.9, 1.0, 3.6), (7.8, 9.6, 0.9, 2.8),
            (17.6, 11.0, 0.8, 2.2), (21.2, 8.6, 0.8, 2.0),
            (3.0, 8.2, 0.9, 2.4), (13.6, 3.4, 1.0, 3.0),
            (9.0, 1.9, 0.8, 2.0), (22.0, 2.4, 0.7, 1.6)]
DISTRICTS = [(9.5, 7.0, "Central"), (15.6, 8.9, "Riverside"),
             (5.4, 3.7, "West port"), (19.4, 4.6, "Tech park")]


def monthly(rng, level, sigma, drift, n=24):
    """24 monthly readings: exponential drift, annual season, lognormal noise."""
    t = np.arange(n)
    season = 1.0 + 0.035 * np.sin(2.0 * np.pi * (t + 2.0) / 12.0)
    trend = np.exp(drift * t / 12.0)
    return level * trend * season * np.exp(rng.normal(0.0, sigma, n))


def diurnal(h, peaks, widths, amps, floor):
    y = np.full_like(h, floor)
    for p, w, a in zip(peaks, widths, amps):
        y = y + a * np.exp(-0.5 * ((h - p) / w) ** 2)
    return y


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(678)
    fig = plt.figure(figsize=(FW, FH))

    def rect(x, y, w, h):
        return (x / FW, y / FH, w / FW, h / FH)

    def ftext(x, y, s, **kw):
        return fig.text(x / FW, y / FH, s, **kw)

    def zone_label(x, y, s):
        ftext(x, y, s, fontsize=11, fontweight="bold", color=INK,
              ha="left", va="bottom")

    # ---- band cards: the only thing separating three unlike layouts --------
    background_card(fig, rect(0.34, 5.70, 11.92, 1.30), facecolor=CARD,
                    zorder=-6, radius=0.006)
    background_card(fig, rect(0.34, 2.10, 11.92, 3.48), facecolor=CARD_ALT,
                    zorder=-6, radius=0.006)
    background_card(fig, rect(0.34, 0.30, 11.92, 1.62), facecolor=CARD_ALT,
                    zorder=-6, radius=0.006)

    # =====================================================================
    # (a) KPI strip
    # =====================================================================
    cw, gap = 2.216, 0.14
    yoys = {}
    for i, (name, unit, level, sig, drift, good, fmt) in enumerate(KPIS):
        x0 = 0.48 + i * (cw + gap)
        y0 = 5.82
        background_card(fig, rect(x0, y0, cw, 1.06), facecolor="white",
                        edgecolor=HAIR, linewidth=0.6, zorder=-4,
                        radius=0.005)
        s = monthly(rng, level, sig, drift)
        yoy = 100.0 * (s[-1] / s[-13] - 1.0)
        yoys[name] = yoy
        col = GREEN if yoy * good > 0.0 else RED
        arrow = r"$\uparrow$" if yoy > 0.0 else r"$\downarrow$"

        ftext(x0 + 0.15, y0 + 0.93, name, fontsize=7.0, color=SUB,
              ha="left", va="top")
        ftext(x0 + 0.15, y0 + 0.47, fmt.format(s[-1]), fontsize=12.0,
              fontweight="bold", color=INK, ha="left", va="center")
        ftext(x0 + 0.15, y0 + 0.18, unit, fontsize=6.4, color=MUTED,
              ha="left", va="center")
        ftext(x0 + cw - 0.13, y0 + 0.93,
              "%s %.1f %% yoy" % (arrow, abs(yoy)), fontsize=6.8, color=col,
              ha="right", va="top")

        axs = axes_inches(fig, (x0 + 1.30, y0 + 0.16, 0.78, 0.52))
        axs.plot(np.arange(s.size), s, lw=1.0, color=col, zorder=3)
        axs.fill_between(np.arange(s.size), s.min() - 0.06 * np.ptp(s), s,
                         facecolor=lighten(col, 0.80), edgecolor="none",
                         zorder=2)
        axs.plot([s.size - 1], [s[-1]], marker="o", ms=2.6, color=col,
                 zorder=4)
        axs.set_xlim(-0.5, s.size - 0.5)
        axs.set_ylim(s.min() - 0.08 * np.ptp(s), s.max() + 0.08 * np.ptp(s))
        axs.set_xticks([])
        axs.set_yticks([])
        axs.patch.set_alpha(0.0)
        for sp in axs.spines.values():
            sp.set_visible(False)
        if i == len(KPIS) - 1:
            axs.text(s.size - 1, s.min() - 0.10 * np.ptp(s), "24 months",
                     ha="right", va="top", fontsize=6.0, color=MUTED)

    zone_label(0.42, 7.04, "(a)")
    ftext(0.90, 7.06, "Key indicators, latest month against the same month "
          "one year earlier", fontsize=8.5, fontweight="bold", color=INK,
          ha="left", va="bottom")

    # =====================================================================
    # (b) trip-density map
    # =====================================================================
    nx, ny = 132, 74
    xs = np.linspace(0.0, 24.0, nx)
    ys = np.linspace(0.0, 13.5, ny)
    X, Y = np.meshgrid(xs, ys)
    field = 0.55 + 0.45 * rng.random((ny, nx))
    for cx, cy, sd, amp in HOTSPOTS:
        field += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) /
                              (2.0 * sd ** 2))
    field = gaussian_filter(field, 1.0)

    axm = axes_inches(fig, (0.74, 2.42, 5.12, 2.88))
    im = axm.imshow(field, origin="lower", extent=(0.0, 24.0, 0.0, 13.5),
                    cmap=get_cmap("cb_sequential_blue"), interpolation="bicubic",
                    vmin=float(np.percentile(field, 1.0)),
                    vmax=float(np.percentile(field, 99.6)), zorder=1)
    image_axes(axm, color="#8a8a8a", lw=0.6)

    th = np.linspace(0.0, 2.0 * np.pi, 240)
    axm.plot(11.0 + 7.4 * np.cos(th), 7.0 + 4.1 * np.sin(th), lw=0.8,
             ls=(0, (4.0, 2.4)), color="#f5f5f5", alpha=0.85, zorder=3)
    axm.plot(xs, 2.6 + 1.1 * np.sin(xs / 3.4), lw=2.2, color="#9cc3dd",
             alpha=0.75, solid_capstyle="round", zorder=2)
    for cx, cy, name in DISTRICTS:
        halo_text(axm, cx, cy, name, ha="center", va="center", fontsize=6.6,
                  color="white", halo="#08263f", zorder=6)

    j, k = np.unravel_index(np.argmax(field), field.shape)
    axm.plot([xs[k]], [ys[j]], marker="o", ms=5.0, mfc="none", mec=RED,
             mew=1.3, zorder=7)
    halo_text(axm, xs[k] + 0.5, ys[j] + 0.7,
              "peak %.1f" % field[j, k], ha="left", va="bottom",
              fontsize=6.4, color="white", halo=RED, zorder=7)
    add_scale_bar(axm, 5.0, "5 km", loc=(0.035, 0.055), fontsize=6.2)
    safe_colorbar(fig, im, axm,
                  label=r"trip density ($10^3$ trips km$^{-2}$ d$^{-1}$)",
                  width=0.11, pad=0.07, labelsize=6.6)
    axm.set_title("Trip density, 24 h aggregate over a 24 x 13.5 km footprint",
                  fontsize=8.5, fontweight="bold", pad=3.0)
    zone_label(0.42, 5.34, "(b)")

    # =====================================================================
    # (c) three stacked diurnal series, shared hour axis
    # =====================================================================
    h = np.linspace(0.0, 24.0, 289)
    metro = diurnal(h, [8.0, 18.2], [1.5, 1.9], [78.0, 92.0], 12.0)
    metro = metro * np.exp(rng.normal(0.0, 0.035, h.size))
    road = diurnal(h, [7.6, 17.8], [2.1, 2.5], [64.0, 71.0], 22.0)
    road = road * np.exp(rng.normal(0.0, 0.035, h.size))
    aqi = 58.0 + 38.0 * np.exp(-0.5 * ((h - 8.6) / 2.2) ** 2) \
        + 46.0 * np.exp(-0.5 * ((h - 19.4) / 2.6) ** 2)
    aqi = aqi * np.exp(rng.normal(0.0, 0.030, h.size))
    load = 2.15 + 0.72 * np.sin(2.0 * np.pi * (h - 9.5) / 24.0) \
        + 0.55 * np.exp(-0.5 * ((h - 20.0) / 2.3) ** 2)
    load = load * np.exp(rng.normal(0.0, 0.014, h.size))

    ts_h, ts_gap, ts_x0, ts_w = 0.800, 0.16, 7.10, 4.92
    axt = [axes_inches(fig, (ts_x0, 2.58 + i * (ts_h + ts_gap), ts_w, ts_h))
           for i in range(3)]

    ax3 = axt[2]
    ax3.plot(h, metro, lw=1.3, color=BLUE, label="metro")
    ax3.plot(h, road, lw=1.3, color=GOLD, label="road")
    ax3.set_ylabel(r"flow ($10^3$ trips/h)", fontsize=7.0)
    ax3.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2,
               handlelength=1.3, columnspacing=1.0)

    ax2 = axt[1]
    ax2.plot(h, aqi, lw=1.3, color=PURPLE)
    ax2.axhline(100.0, color=RED, lw=0.9, ls=(0, (3.0, 2.0)))
    ax2.text(0.35, 103.0, "AQI 100 alert level", ha="left", va="bottom",
             fontsize=6.2, color=RED)
    ax2.set_ylabel("air quality (AQI)", fontsize=7.0)
    ax2.set_ylim(40.0, 150.0)

    ax1 = axt[0]
    ax1.plot(h, load, lw=1.3, color=GREEN)
    kp = int(np.argmax(load))
    ax1.plot([h[kp]], [load[kp]], marker="o", ms=3.4, color=GREEN)
    ax1.set_ylim(load.min() * 0.94, load.max() * 1.16)
    ax1.annotate("evening peak %.2f GW at %02d:%02d"
                 % (load[kp], int(h[kp]), int(round((h[kp] % 1.0) * 60))),
                 xy=(h[kp], load[kp]), xytext=(-7.0, 2.0),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=6.2, color=GREEN)
    ax1.set_ylabel("grid load (GW)", fontsize=7.0)
    ax1.set_xlabel("hour of day (h)", fontsize=7.4)

    for i, ax in enumerate(axt):
        ax.set_xlim(0.0, 24.0)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.grid(axis="y", color=GRID, lw=0.55)
        ax.tick_params(labelsize=6.5, length=2.2)
        if i > 0:
            ax.tick_params(labelbottom=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axt[2].set_title("Same day, three service systems", fontsize=8.5,
                     fontweight="bold", pad=3.0)
    zone_label(6.62, 5.34, "(c)")

    # =====================================================================
    # (d) incident timeline
    # =====================================================================
    n_ev = 54
    t_ev = np.concatenate([rng.normal(8.4, 1.7, n_ev // 3),
                           rng.normal(18.4, 2.1, n_ev // 3),
                           rng.uniform(0.0, 24.0, n_ev - 2 * (n_ev // 3))])
    t_ev = np.clip(t_ev, 0.2, 23.8)
    lane = rng.integers(0, len(LANES), t_ev.size)
    sev = rng.choice([1, 2, 3], t_ev.size, p=[0.56, 0.30, 0.14])

    axe = axes_inches(fig, (1.62, 0.70, 10.40, 0.92))
    for lo, hi in ((6.5, 9.5), (16.5, 19.5)):
        axe.axvspan(lo, hi, facecolor="#e8e8e8", edgecolor="none", zorder=0)
    axe.text(8.0, 3.62, "morning peak", ha="center", va="top", fontsize=6.2,
             color=MUTED)
    axe.text(18.0, 3.62, "evening peak", ha="center", va="top", fontsize=6.2,
             color=MUTED)
    for i in range(len(LANES)):
        axe.axhline(len(LANES) - 1 - i, color=HAIR, lw=0.6, zorder=1)
    for s_i, (sname, scol) in enumerate(STATUS):
        m = sev == s_i + 1
        axe.scatter(t_ev[m], len(LANES) - 1 - lane[m],
                    s=16.0 + 16.0 * sev[m], color=scol, edgecolor="white",
                    linewidth=0.5, zorder=4, label=sname)
    for i, name in enumerate(LANES):
        axe.text(-0.35, len(LANES) - 1 - i, name, ha="right", va="center",
                 fontsize=6.6, color=INK)
        axe.text(24.35, len(LANES) - 1 - i, "n=%d" % int((lane == i).sum()),
                 ha="left", va="center", fontsize=6.4, color=MUTED)
    axe.set_xlim(0.0, 24.0)
    axe.set_ylim(-0.7, 3.7)
    axe.set_xticks(np.arange(0, 25, 2))
    axe.set_yticks([])
    axe.set_xlabel("hour of day (h)", fontsize=7.4)
    axe.tick_params(labelsize=6.5, length=2.2)
    axe.grid(axis="x", color=GRID, lw=0.5)
    axe.legend(loc="lower right", bbox_to_anchor=(1.0, 1.03), ncol=3,
               fontsize=6.5, frameon=False, handlelength=1.0,
               columnspacing=1.4, handletextpad=0.35, scatterpoints=1)
    for s in ("top", "right", "left"):
        axe.spines[s].set_visible(False)
    zone_label(0.42, 1.70, "(d)")
    ftext(0.90, 1.72, "Incident timeline, %d events reported in 24 h "
          "(marker area scales with severity class)" % t_ev.size,
          fontsize=8.5, fontweight="bold", color=INK, ha="left", va="bottom")

    ftext(0.36, 0.09, "synthetic city telemetry, for layout reference only",
          ha="left", va="bottom", fontsize=6.5, color=MUTED)
    savefig(fig, "678_sct_urban_dashboard.png")


if __name__ == "__main__":
    main()
