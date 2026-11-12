"""Figure 720 - Tropical cyclone wind footprint over an exposed region.

One synthetic event drives all three panels.  A recurving track is sampled
every three hours; each node carries a maximum sustained wind and a radius of
maximum wind, and a modified-Rankine profile plus the vector translation speed
turns that node into an asymmetric wind field.  The footprint of panel (a) is
the running maximum of those fields, interpolated onto half-hourly steps so
the swath is a swath rather than a string of rings; the swath and the
intensity history of panel (b) are the same numbers read two ways.  Panel (c) accumulates
the gridded population living inside three wind thresholds as the running
maximum grows, which is why its curves can only ever step upward.

Everything quoted in the title, the map call-outs and the panel annotations is
computed from the arrays that are plotted.
"""
import textwrap
import warnings

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, savefig, lighten, darken, COLORS
from figkit import (add_scale_bar, axes_inches, halo_text, panel_label_inches,
                    safe_colorbar)

# ---- canvas geometry, in inches -------------------------------------------
FW, FH = 10.4, 6.4
MAIN_X, MAIN_W, MAIN_TOP = 0.62, 5.30, 5.50
AUX_X, AUX_W, AUX_GAP = 7.05, 2.95, 0.66

X1, Y1 = 900.0, 798.0               # map extent, km
NX, NY = 190, 168                   # hazard grid
T_END, DT = 72.0, 3.0               # forecast horizon and step, hours

ACCENT = "#b23a48"
# Saffir-Simpson-like class edges in m/s; one house-red ramp, monotone
# lightness (DESIGN_SYSTEM section 2.2).
WIND_LEVELS = [17.2, 24.5, 32.7, 41.5, 50.9, 60.0]
WIND_COLORS = [lighten(ACCENT, 0.80), lighten(ACCENT, 0.60),
               lighten(ACCENT, 0.40), lighten(ACCENT, 0.18), ACCENT]
POP_COLORS = ["#ffffff", "#f5f5f5", "#d9d9d9", "#b3b3b3", "#8a8a8a"]
THRESHOLDS = [17.2, 32.7, 41.5]

CITIES = [
    dict(name="Nanhu", xy=(215.0, 300.0), sig=52.0, peak=2100.0),
    dict(name="Beiport", xy=(470.0, 455.0), sig=44.0, peak=2650.0),
    dict(name="Qingzhou", xy=(690.0, 350.0), sig=38.0, peak=1450.0),
    dict(name="Wuling", xy=(360.0, 615.0), sig=40.0, peak=980.0),
    dict(name="Haixi", xy=(120.0, 585.0), sig=34.0, peak=760.0),
    dict(name="Tanyang", xy=(775.0, 620.0), sig=32.0, peak=640.0),
    dict(name="Shiwan", xy=(560.0, 175.0), sig=36.0, peak=880.0),
]


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def track(rng):
    """Recurving track sampled every DT hours, with intensity and size."""
    t = np.arange(0.0, T_END + 0.5 * DT, DT)
    x = 55.0 + 10.6 * t
    y = 105.0 + 0.058 * t ** 2
    vmax = 22.0 + 35.0 * np.exp(-((t - 34.0) / 25.0) ** 2)
    vmax += rng.normal(0.0, 0.9, t.size)
    rmax = 34.0 + 0.42 * t
    return t, x, y, vmax, rmax


def node_field(mx, my, x, y, vmax, rmax, vtx, vty):
    """Asymmetric modified-Rankine wind speed from one track node, m/s."""
    dx, dy = mx - x, my - y
    d = np.hypot(dx, dy)
    ds = np.maximum(d, 1e-6)
    core = np.where(d < rmax, vmax * d / rmax,
                    vmax * (rmax / ds) ** 0.62)
    # cyclonic tangential flow plus a share of the storm translation, which is
    # what puts the strongest wind on the right-hand side of the track
    wx = core * (-dy / ds) + 0.34 * vtx
    wy = core * (dx / ds) + 0.34 * vty
    return np.hypot(wx, wy)


def admin_lines():
    """Wobbly administrative boundaries, purely as a locational backdrop."""
    out = []
    for x0 in (250.0, 520.0, 715.0):
        yy = np.linspace(0.0, Y1, 120)
        out.append((x0 + 26.0 * np.sin(yy / 128.0 + x0 / 90.0)
                    + 11.0 * np.sin(yy / 46.0), yy))
    for y0 in (285.0, 565.0):
        xx = np.linspace(0.0, X1, 130)
        out.append((xx, y0 + 23.0 * np.sin(xx / 142.0 + y0 / 70.0)
                    + 9.0 * np.sin(xx / 53.0)))
    return out


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(7201)

    gx = np.linspace(0.0, X1, NX)
    gy = np.linspace(0.0, Y1, NY)
    mx, my = np.meshgrid(gx, gy)
    cell = (X1 / (NX - 1)) * (Y1 / (NY - 1))          # km^2 per cell

    # ---- population density, people per km^2 -------------------------------
    dens = 18.0 + 9.0 * np.sin(mx / 210.0) * np.cos(my / 185.0)
    for c in CITIES:
        r2 = (mx - c["xy"][0]) ** 2 + (my - c["xy"][1]) ** 2
        dens += c["peak"] * np.exp(-r2 / (2.0 * c["sig"] ** 2))
    dens += 120.0 * np.exp(-((my - 430.0) / 150.0) ** 2) \
        * np.exp(-((mx - 470.0) / 320.0) ** 2)
    dens = np.clip(dens, 0.0, None)
    pop_total = dens.sum() * cell / 1e6                # million people

    # ---- storm and its running-maximum footprint ---------------------------
    t, tx, ty, vmax, rmax = track(rng)
    vtx = np.gradient(tx, t) / 3.6                     # km/h -> m/s
    vty = np.gradient(ty, t) / 3.6
    v_tr = np.hypot(vtx, vty)

    # The swath is swept at 30-minute resolution; sampling it only at the
    # 3-hourly reporting nodes leaves a string of discrete wind rings.
    tf = np.arange(0.0, T_END + 1e-9, 0.5)
    fx, fy = np.interp(tf, t, tx), np.interp(tf, t, ty)
    fv, fr = np.interp(tf, t, vmax), np.interp(tf, t, rmax)
    fvx, fvy = np.interp(tf, t, vtx), np.interp(tf, t, vty)

    foot = np.zeros_like(mx)
    cum = np.zeros((len(THRESHOLDS), t.size))
    j = 0
    for i in range(tf.size):
        foot = np.maximum(foot, node_field(mx, my, fx[i], fy[i], fv[i],
                                           fr[i], fvx[i], fvy[i]))
        if j < t.size and tf[i] >= t[j] - 1e-9:
            for k, thr in enumerate(THRESHOLDS):
                cum[k, j] = dens[foot >= thr].sum() * cell / 1e6
            j += 1

    area_gale = (foot >= THRESHOLDS[0]).sum() * cell / 1e3   # 10^3 km^2
    peak_i = int(np.argmax(vmax))

    # ---- figure ------------------------------------------------------------
    fig = plt.figure(figsize=(FW, FH))
    main_h = MAIN_W * Y1 / X1
    main_y = MAIN_TOP - main_h
    ax = axes_inches(fig, (MAIN_X, main_y, MAIN_W, main_h))
    aux_h = (main_h - AUX_GAP) / 2.0
    ax_v = axes_inches(fig, (AUX_X, main_y + aux_h + AUX_GAP, AUX_W, aux_h))
    ax_p = axes_inches(fig, (AUX_X, main_y, AUX_W, aux_h))

    # ---- (a) footprint map -------------------------------------------------
    pop_cmap = mpl.colors.LinearSegmentedColormap.from_list("pop", POP_COLORS)
    p99 = float(np.percentile(dens, 99.4))
    ax.pcolormesh(mx, my, dens, cmap=pop_cmap, shading="auto",
                  norm=mpl.colors.PowerNorm(0.45, vmin=0.0, vmax=p99),
                  rasterized=True, zorder=1)
    for lx, ly in admin_lines():
        ax.plot(lx, ly, color="#8a8a8a", lw=0.65, ls=(0, (5, 2.4)), zorder=2)

    cmap = mpl.colors.ListedColormap(WIND_COLORS)
    cmap.set_over(darken(ACCENT, 0.42))
    norm = mpl.colors.BoundaryNorm(WIND_LEVELS, cmap.N)
    cs = ax.contourf(mx, my, foot, levels=WIND_LEVELS, cmap=cmap, norm=norm,
                     extend="max", alpha=0.76, zorder=3)
    rasterize(cs)
    cs.set_edgecolor("face")
    ax.contour(mx, my, foot, levels=WIND_LEVELS, colors="white",
               linewidths=0.45, alpha=0.75, zorder=4)

    # forecast-uncertainty cone, widening with lead time
    ux, uy = vtx / np.maximum(v_tr, 1e-6), vty / np.maximum(v_tr, 1e-6)
    rc = 32.0 + 2.05 * t
    left = np.column_stack([tx - uy * rc, ty + ux * rc])
    right = np.column_stack([tx + uy * rc, ty - ux * rc])
    ax.add_patch(Polygon(np.vstack([left, right[::-1]]), closed=True,
                         facecolor="none", edgecolor="#333333", lw=0.8,
                         ls=(0, (3.4, 2.0)), zorder=6))

    ax.plot(tx, ty, color="#1a1a1a", lw=1.7, solid_capstyle="round", zorder=7)
    mark = np.arange(0, t.size, int(round(12.0 / DT)))
    ax.scatter(tx[mark], ty[mark], s=17.0, facecolor="white",
               edgecolor="#1a1a1a", linewidth=0.9, zorder=8)
    for i in mark:
        halo_text(ax, tx[i] - 6.0, ty[i] + 13.0, f"+{t[i]:.0f} h",
                  fontsize=6.2, ha="right", va="bottom", color="#1a1a1a",
                  halo="white", halo_lw=2.2, zorder=9)
    halo_text(ax, tx[peak_i] + 12.0, ty[peak_i] - 16.0,
              f"$V_{{max}}$ = {vmax[peak_i]:.0f} m s$^{{-1}}$", fontsize=6.6,
              ha="left", va="top", color="#1a1a1a", halo="white",
              halo_lw=2.4, zorder=9)

    for c in CITIES[:4]:
        ax.plot([c["xy"][0]], [c["xy"][1]], marker="s", ms=3.6, mfc="white",
                mec="#1a1a1a", mew=0.9, zorder=8)
        halo_text(ax, c["xy"][0] + 11.0, c["xy"][1] - 2.0, c["name"],
                  fontsize=6.2, ha="left", va="center", color="#333333",
                  halo="white", halo_lw=2.2, zorder=9)

    ax.set_xlim(0.0, X1)
    ax.set_ylim(0.0, Y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.set_title("Wind footprint, track and population at risk", fontsize=8.8,
                 fontweight="bold", pad=4.0)
    add_scale_bar(ax, 200.0, "200 km", loc=(0.035, 0.045), height=0.019,
                  fontsize=6.2, n_seg=2)
    safe_colorbar(fig, mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax,
                  label="10-min sustained wind (m s$^{-1}$)", width=0.12,
                  pad=0.08, labelsize=6.8, ticks=WIND_LEVELS, extend="max")
    ax.legend(handles=[
        Line2D([0], [0], color="#1a1a1a", lw=1.7, label="storm centre track"),
        Line2D([0], [0], color="#333333", lw=0.8, ls=(0, (3.4, 2.0)),
               label="track uncertainty cone"),
        Line2D([0], [0], color="#8a8a8a", lw=0.65, ls=(0, (5, 2.4)),
               label="administrative boundary"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=3.6,
               markerfacecolor="white", markeredgecolor="#1a1a1a",
               markeredgewidth=0.9, label="major city"),
    ], loc="upper left", frameon=True, framealpha=0.92, facecolor="white",
        edgecolor="#cccccc", borderpad=0.4, fontsize=6.3, handlelength=1.8,
        labelspacing=0.3)
    halo_text(ax, 0.972, 0.045,
              f"population density 0-{p99:.0f} "
              "people km$^{-2}$ (grey)", transform=ax.transAxes, ha="right",
              va="bottom", fontsize=6.2, color="#333333", halo="white",
              halo_lw=2.4, zorder=9)

    # ---- (b) intensity evolution -------------------------------------------
    for lo, hi, col in zip(WIND_LEVELS[:-1], WIND_LEVELS[1:], WIND_COLORS):
        ax_v.axhspan(lo, hi, facecolor=col, alpha=0.30, lw=0.0, zorder=0)
    sig = 2.4 + 0.055 * t
    ax_v.fill_between(t, vmax - sig, vmax + sig, facecolor="#3a3a3a",
                      alpha=0.16, lw=0.0, zorder=2)
    ax_v.plot(t, vmax, color="#1a1a1a", lw=1.5, zorder=4)
    ax_v.plot([t[peak_i]], [vmax[peak_i]], marker="o", ms=3.8, mfc="white",
              mec=ACCENT, mew=1.2, zorder=5)
    ax_v.annotate(f"peak {vmax[peak_i]:.0f} m s$^{{-1}}$ at +{t[peak_i]:.0f} h",
                  xy=(t[peak_i], vmax[peak_i]), xytext=(5.0, 8.0),
                  textcoords="offset points", fontsize=6.3, color="#1a1a1a",
                  ha="left", va="bottom")
    ax_v.set_xlim(0.0, T_END)
    ax_v.set_ylim(12.0, 66.0)
    ax_v.set_xticks(np.arange(0.0, T_END + 1.0, 12.0))
    ax_v.set_xlabel("forecast lead time (h)", fontsize=7.4)
    ax_v.set_ylabel("$V_{max}$ (m s$^{-1}$)", fontsize=7.4)
    ax_v.tick_params(labelsize=6.6)
    ax_v.grid(axis="y", lw=0.5, alpha=0.45)
    ax_v.set_title("Intensity along the track", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_v.text(0.975, 0.05, "bands: wind classes of panel (a)",
              transform=ax_v.transAxes, fontsize=6.2, color=COLORS["sub"],
              ha="right", va="bottom")
    for sp in ("top", "right"):
        ax_v.spines[sp].set_visible(False)

    # ---- (c) cumulative exposed population ---------------------------------
    cols = [lighten(ACCENT, 0.46), ACCENT, darken(ACCENT, 0.38)]
    for k, (thr, col) in enumerate(zip(THRESHOLDS, cols)):
        ax_p.step(t, cum[k], where="post", color=col, lw=1.5, zorder=4,
                  label=f"$>${thr:.1f} m s$^{{-1}}$  ({cum[k, -1]:.1f} M)")
        ax_p.fill_between(t, 0.0, cum[k], step="post", facecolor=col,
                          alpha=0.13, lw=0.0, zorder=2)
    ax_p.set_xlim(0.0, T_END)
    ax_p.set_ylim(0.0, cum.max() * 1.34)
    ax_p.set_xticks(np.arange(0.0, T_END + 1.0, 12.0))
    ax_p.set_xlabel("forecast lead time (h)", fontsize=7.4)
    ax_p.set_ylabel("exposed population (million)", fontsize=7.2)
    ax_p.tick_params(labelsize=6.6)
    ax_p.grid(axis="y", lw=0.5, alpha=0.45)
    ax_p.legend(loc="upper left", frameon=False, fontsize=6.3,
                handlelength=1.6, labelspacing=0.26)
    ax_p.set_title("Cumulative population inside each class", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_p.text(0.975, 0.96,
              f"{cum[0, -1] / pop_total:.0%} of the {pop_total:.0f} M "
              "regional total", transform=ax_p.transAxes, fontsize=6.2,
              color=COLORS["sub"], ha="right", va="top")
    for sp in ("top", "right"):
        ax_p.spines[sp].set_visible(False)

    ax.apply_aspect()
    for a, lab in ((ax, "(a)"), (ax_v, "(b)"), (ax_p, "(c)")):
        panel_label_inches(fig, a, lab, dx=-0.30 if a is ax else -0.46,
                           dy=0.09)


    savefig(fig, "720_cat_hazard_footprint.png")


if __name__ == "__main__":
    main()
