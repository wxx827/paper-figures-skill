"""Interpreted 2D seismic line with variable-area wiggle display.

Sixty-four synthetic traces are built by convolving a layered reflectivity
series with a Ricker wavelet, then drawn in the conventional variable-area
form: the trace deflects to the right of its own centre line and the positive
lobe is filled solid, so bright reflectors read as continuous dark bands.  Five
interpreted horizons and a normal fault are overlain; the fault throw quoted on
the figure is measured from the horizon model that generated the data.  The
auxiliary panel is a semblance velocity spectrum sharing the two-way-time axis.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pubstyle import (set_pub_style, savefig, add_panel_label, add_colorbar,
                      get_cmap)

NTR = 52                 # traces
XMAX = 4.8               # km along the line
TMAX = 2.20              # s two-way time
DT = 0.002
F0 = 26.0                # Ricker peak frequency (Hz)

# interpreted horizons: base two-way time, dip, fold amplitude, wavelength,
# reflection coefficient, whether the fault cuts it, display colour, name
HORIZONS = [
    (0.34, 0.055, 0.022, 1.7, 0.85, False, "#c9a227", "H1 seabed"),
    (0.68, 0.090, 0.038, 2.3, 0.72, True, "#2e7d5b", "H2 top reservoir"),
    (1.02, 0.115, 0.030, 1.9, 0.95, True, "#3b5b92", "H3 base reservoir"),
    (1.46, 0.140, 0.026, 2.6, -0.62, True, "#8e6bbf", "H4 top carbonate"),
    (1.84, 0.165, 0.018, 2.1, 0.78, True, "#b23a48", "H5 basement"),
]
THROW = 0.062            # s of vertical separation across the fault
INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
FILL = "#3a3a3a"


def fault_x(t):
    """Listric normal fault: shallow trace at 2.05 km, steepening with depth."""
    return 2.05 + 0.42 * (t / TMAX) ** 1.35


def horizon_time(spec, x):
    t0, dip, amp, wl, *_ = spec
    return t0 + dip * (x / XMAX) + amp * np.sin(2 * np.pi * x / wl + 0.6)


def ricker(f, dt, n=161):
    tt = (np.arange(n) - n // 2) * dt
    a = (np.pi * f * tt) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def rasterize(artist):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def main():
    set_pub_style()
    rng = np.random.default_rng(20260813)

    t = np.arange(0.0, TMAX + DT, DT)
    xs = np.linspace(0.12, XMAX - 0.12, NTR)
    w = ricker(F0, DT)

    refl = rng.normal(0.0, 0.013, size=(NTR, t.size))   # thin-bed texture
    for j, x in enumerate(xs):
        down = x > fault_x(0.0)
        for spec in HORIZONS:
            th = horizon_time(spec, x)
            if spec[5] and x > fault_x(th):
                th += THROW
            k = int(round(th / DT))
            if 0 <= k < t.size:
                refl[j, k] += spec[4]
        _ = down

    traces = np.array([np.convolve(r, w, mode="same") for r in refl])
    traces /= np.abs(traces).max()
    gain = 2.3 * (xs[1] - xs[0])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.42], left=0.058,
                          right=0.925, top=0.855, bottom=0.155, wspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1], sharey=ax)

    # ---- variable-area wiggle --------------------------------------------
    for x, a in zip(xs, traces):
        v = x + gain * a
        ax.plot(v, t, color=FILL, lw=0.32, zorder=3)
        ax.fill_betweenx(t, x, v, where=(a > 0), facecolor=FILL, lw=0,
                         zorder=2, interpolate=True)

    # ---- interpretation ---------------------------------------------------
    xf = np.linspace(0.0, XMAX, 400)
    for spec in HORIZONS:
        th = horizon_time(spec, xf)
        cut = spec[5]
        left = xf <= fault_x(th)
        ax.plot(np.where(left, xf, np.nan), np.where(left, th, np.nan),
                color=spec[6], lw=1.7, zorder=6, solid_capstyle="round")
        tr = th + (THROW if cut else 0.0)
        right = xf > fault_x(tr)
        ax.plot(np.where(right, xf, np.nan), np.where(right, tr, np.nan),
                color=spec[6], lw=1.7, zorder=6, solid_capstyle="round")
        ax.text(XMAX - 0.04, horizon_time(spec, XMAX) +
                (THROW if cut else 0.0) - 0.028, spec[7].split()[0],
                ha="right", va="bottom", fontsize=6.6, color=spec[6],
                fontweight="bold", zorder=7)

    tfl = np.linspace(0.16, TMAX - 0.02, 200)
    ax.plot(fault_x(tfl), tfl, color=INK, lw=1.7, ls=(0, (5, 2.4)), zorder=8)
    for lab, dx in (("U", -0.14), ("D", 0.14)):
        ax.text(fault_x(1.94) + dx, 1.94, lab, ha="center", va="center",
                fontsize=7.0, fontweight="bold", color=INK, zorder=9,
                bbox=dict(boxstyle="circle,pad=0.16", facecolor="white",
                          edgecolor=INK, linewidth=0.7))
    ax.annotate(f"normal fault, {THROW * 1000:.0f} ms throw",
                xy=(fault_x(1.25), 1.25), xytext=(1.02, 1.62),
                fontsize=7.0, color=INK, ha="left", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.6, alpha=0.94),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                                connectionstyle="arc3,rad=0.28"))

    ax.set_xlim(-0.02, XMAX)
    ax.set_ylim(TMAX, 0.0)
    ax.set_xlabel("Distance along line (km)")
    ax.set_ylabel("Two-way travel time (s)")
    ax.set_yticks(np.arange(0.0, TMAX + 0.01, 0.25))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Migrated line SW-NE, variable-area wiggle",
                 fontsize=8.5, fontweight="bold")

    # ---- velocity spectrum (auxiliary) ------------------------------------
    vv = np.linspace(1500.0, 4600.0, 220)
    tt = np.linspace(0.0, TMAX, 260)
    V, T = np.meshgrid(vv, tt)
    vpick = 1680.0 + 2380.0 * (tt / TMAX) ** 0.80
    strength = 0.16 + np.zeros_like(tt)
    for spec in HORIZONS:
        th = horizon_time(spec, XMAX / 2)
        strength += 0.72 * abs(spec[4]) * np.exp(-((tt - th) / 0.055) ** 2)
    band = 150.0 + 130.0 * tt
    sem = (strength[:, None] *
           np.exp(-((V - vpick[:, None]) / band[:, None]) ** 2))
    sem += 0.06 * np.exp(-((V - vpick[:, None] * 0.72) / 320.0) ** 2)

    cs = ax_v.contourf(V, T, sem, levels=np.linspace(0.0, 1.0, 11),
                       cmap=get_cmap("cb_sequential_blue"), extend="max")
    rasterize(cs)
    cs.set_edgecolor("face")
    ax_v.plot(vpick, tt, color="#b23a48", lw=1.4, zorder=4)
    for spec in HORIZONS:
        th = horizon_time(spec, XMAX / 2)
        vp = float(np.interp(th, tt, vpick))
        ax_v.plot([vp], [th], marker="o", ms=4.0, mfc="white",
                  mec="#b23a48", mew=1.1, zorder=5)
        ax_v.axhline(th, color=spec[6], lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax_v.set_xlim(1500.0, 4600.0)
    ax_v.set_xlabel("Stacking velocity (m s$^{-1}$)")
    ax_v.set_xticks([2000, 3000, 4000])
    ax_v.tick_params(labelleft=False, labelsize=6.5)
    ax_v.xaxis.label.set_size(7.2)
    for s in ax_v.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax_v.set_title("Velocity spectrum at CDP 320", fontsize=8.5,
                   fontweight="bold")
    add_colorbar(fig, cs, ax_v, label="semblance (-)",
                 ticks=[0.0, 0.5, 1.0])

    handles = [Line2D([0], [0], color=s[6], lw=1.7, label=s[7])
               for s in HORIZONS]
    handles.append(Line2D([0], [0], color=INK, lw=1.7, ls=(0, (5, 2.4)),
                          label="normal fault"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.008), ncol=6, frameon=False,
               fontsize=7.0, handlelength=2.0, columnspacing=1.6)

    add_panel_label(ax, "(a)", x=-0.055, y=1.035)
    add_panel_label(ax_v, "(b)", x=-0.10, y=1.035)
    savefig(fig, "406_geo_seismic_section.png")


if __name__ == "__main__":
    main()
