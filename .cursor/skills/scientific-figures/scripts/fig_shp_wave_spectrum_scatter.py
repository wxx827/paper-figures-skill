"""Design sea states: the JONSWAP family, the scatter diagram, the long tail.

The main panel carries four parametric wave spectra plus the Pierson-Moskowitz
shape that shares one of their sea states, so the peak-enhancement factor can
be read as a ratio of two curves on the same axis.  Every sea state is quoted
by the numbers the spectrum actually has: the significant height is four times
the root of the plotted zeroth moment, the peak period is the reciprocal of the
plotted argmax, and the mean zero-crossing period comes from the second moment
of the same array.

The upper companion panel is the scatter diagram those spectra are drawn from -
the joint occurrence of significant height and peak period over a synthetic
twenty-year record, with the four spectra marked in their own cells and the
1/15 steepness limit drawn as the boundary the population cannot cross.  The
lower panel collapses the table to its height margin and reads the 10 %, 1 %
and 0.1 % exceedance heights straight off the plotted curve.

Illustrative synthetic data; not a real site.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, style_axes, get_cmap, add_footer
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

RED = "#b23a48"
STEEL = "#5b6b7a"
SEA = ["#9cc3dd", "#5f9dc6", "#2e74a8", "#134a76"]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.06, 5.26, 4.34)
AUX_T = (7.46, 3.62, 3.05, 1.78)
AUX_B = (7.46, 1.06, 3.05, 1.72)

G = 9.81
SEED = 20260813

# significant height (m), peak period (s), peak enhancement (dimensionless)
STATES = [dict(hs=3.0, tp=8.5, gam=1.0, tag="moderate", off=(0.40, 0.14)),
          dict(hs=4.5, tp=10.0, gam=3.3, tag="rough", off=(0.40, 0.14)),
          dict(hs=6.0, tp=12.0, gam=3.3, tag="very rough", off=(0.40, 0.14)),
          dict(hs=7.5, tp=13.5, gam=2.0, tag="storm", off=(0.40, 0.14))]


def jonswap(f, hs, tp, gamma):
    """JONSWAP density (m^2 s) on the frequency grid, rescaled to exactly hs."""
    fp = 1.0 / tp
    sig = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2.0 * sig ** 2 * fp ** 2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = (G ** 2 * (2.0 * np.pi) ** -4 * f ** -5.0
             * np.exp(-1.25 * (f / fp) ** -4.0) * gamma ** r)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    m0 = float(np.trapezoid(s, f))
    return s * (hs / (4.0 * np.sqrt(m0))) ** 2


def moments(f, s):
    """Return (Hs from 4 sqrt(m0), Tp from the argmax, Tm02 from m0/m2)."""
    m0 = float(np.trapezoid(s, f))
    m2 = float(np.trapezoid(s * f ** 2, f))
    return 4.0 * np.sqrt(m0), 1.0 / f[int(np.argmax(s))], np.sqrt(m0 / m2)


def scatter_table(rng, n=400000):
    """Joint occurrence of Hs and Tp, in counts per 1000 observations."""
    hs_edges = np.arange(0.0, 10.01, 0.5)
    tp_edges = np.arange(3.0, 18.01, 0.5)
    hs = 2.35 * rng.weibull(1.55, n)
    mu = np.log(4.55 * np.clip(hs, 0.05, None) ** 0.27 + 2.55)
    tp = np.exp(mu + 0.135 * rng.standard_normal(n))
    counts, _, _ = np.histogram2d(hs, tp, bins=[hs_edges, tp_edges])
    return hs_edges, tp_edges, counts / n * 1000.0


def check_layout(fig, foot, axes):
    """Warn if the footnote box touches an axis label, tick row or title."""
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for a in axes:
        ab = a.get_window_extent()
        arts = [a.xaxis.label, a.yaxis.label, a.title]
        arts += [t for t in a.get_xticklabels()
                 if ab.x0 - 3.0 <= t.get_window_extent().x0 <= ab.x1 + 3.0]
        for art in arts:
            if art.get_text() and fb.overlaps(art.get_window_extent()):
                print("WARNING: footnote overlaps:", art.get_text()[:44])


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    f = np.linspace(0.022, 0.46, 1400)
    spectra = [jonswap(f, st["hs"], st["tp"], st["gam"]) for st in STATES]
    stats = [moments(f, s) for s in spectra]
    pm = jonswap(f, STATES[2]["hs"], STATES[2]["tp"], 1.0)

    hs_e, tp_e, table = scatter_table(rng)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN)
    ax_s = axes_inches(fig, AUX_T)
    ax_x = axes_inches(fig, AUX_B)
    for a in (ax, ax_x):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.55, alpha=0.75, zorder=0)
        style_axes(a)
    style_axes(ax_s)

    # ---- main: the spectral family -----------------------------------------
    for st, s, (hs, tp, tm02), col in zip(STATES, spectra, stats, SEA):
        ax.plot(f, s, color=col, lw=1.9, zorder=6,
                label=("$H_s$ = %.2f m, $T_p$ = %.1f s, $\\gamma$ = %.1f"
                       % (hs, tp, st["gam"])))
        i = int(np.argmax(s))
        ax.plot([f[i]], [s[i]], marker="o", ms=3.8, mfc="white", mec=col,
                mew=1.2, zorder=8)

    ax.plot(f, pm, color=STEEL, lw=1.4, ls=(0, (5.0, 2.4)), zorder=5,
            label="Pierson-Moskowitz, same $H_s$ and $T_p$")
    ratio = float(spectra[2].max() / pm.max())
    i2 = int(np.argmax(spectra[2]))
    ax.annotate("peak enhancement measured off the two curves:\n"
                "$S_{max}$ ratio = %.2f at f = %.3f Hz" % (ratio, f[i2]),
                xy=(f[i2], spectra[2][i2]), xytext=(0.365, 0.745),
                textcoords="axes fraction", fontsize=6.8, color=INK,
                ha="left", va="top", linespacing=1.45, zorder=10,
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=0.9,
                                shrinkA=2, shrinkB=4))

    tm_txt = "\n".join(
        "%-12s $T_{m02}$ = %.1f s, $m_0$ = %.2f m$^2$"
        % (STATES[k]["tag"], stats[k][2],
           float(np.trapezoid(spectra[k], f)))
        for k in range(len(STATES)))
    ax.text(0.985, 0.505, tm_txt, transform=ax.transAxes, ha="right",
            va="top", fontsize=6.4, color=SUB, linespacing=1.5,
            family="DejaVu Sans Mono")

    s_max = max(float(np.max(s)) for s in spectra + [pm])
    ax.set_xlim(0.0, 0.40)
    ax.set_ylim(0.0, s_max * 1.14)
    ax.tick_params(labelsize=7.2)
    ax.set_xlabel("wave frequency f (Hz)", fontsize=8.4, labelpad=1.5)
    ax.set_ylabel("spectral density S(f) (m$^2$ s)", fontsize=8.4)
    ax.set_title("Four design sea states as JONSWAP densities",
                 fontsize=9.2, fontweight="bold", pad=5.0)
    ax.legend(loc="upper right", frameon=False, fontsize=6.9,
              handlelength=1.7, labelspacing=0.34, borderaxespad=0.35)

    # ---- aux (b): the scatter diagram --------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    shown = np.ma.masked_where(table <= 0.0, table)
    mesh = ax_s.pcolormesh(tp_e, hs_e, shown, cmap=cmap, vmin=0.0,
                           vmax=float(table.max()), shading="flat",
                           edgecolors="none", zorder=2)
    cax = ax_s.inset_axes([1.030, 0.045, 0.036, 0.910])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("occurrences per 1000 (%)", fontsize=6.2, color=INK,
                 labelpad=2.0)
    cb.ax.tick_params(labelsize=6.0, width=0.6, length=2.2)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(STEEL)

    tp_line = np.linspace(3.0, 18.0, 200)
    ax_s.plot(tp_line, G * tp_line ** 2 / (2.0 * np.pi * 15.0), color=RED,
              lw=1.1, ls=(0, (4.0, 2.2)), zorder=6)
    ax_s.text(8.10, 8.55, "steepness limit 1/15", fontsize=6.2, color=RED,
              ha="left", va="center", zorder=9)
    for st, (hs, tp, _), col in zip(STATES, stats, SEA):
        ax_s.plot([tp], [hs], marker="o", ms=5.0, mfc=col, mec="white",
                  mew=1.1, zorder=8)
        ax_s.text(tp + st["off"][0], hs + st["off"][1], st["tag"],
                  fontsize=6.1, color=INK, ha="left", va="bottom", zorder=9)
    ax_s.set_xlim(3.0, 18.0)
    ax_s.set_ylim(0.0, 10.0)
    ax_s.tick_params(labelsize=6.8)
    ax_s.set_xlabel("peak period $T_p$ (s)", fontsize=7.5, labelpad=1.5)
    ax_s.set_ylabel("$H_s$ (m)", fontsize=7.5)
    ax_s.set_title("Scatter diagram, 20-year synthetic record",
                   fontsize=8.5, fontweight="bold", pad=4.0)

    # ---- aux (c): the height margin and its tail ---------------------------
    per_hs = table.sum(axis=1)
    surv = np.concatenate([[per_hs.sum()],
                           per_hs.sum() - np.cumsum(per_hs)]) / 1000.0
    surv = np.clip(surv, 1e-6, 1.0)
    ax_x.step(hs_e, surv * 100.0, where="post", color=SEA[3], lw=1.9,
              zorder=6)
    ax_x.fill_between(hs_e, 1e-4, surv * 100.0, step="post",
                      color=SEA[0], alpha=0.40, zorder=3)
    for lev, col in ((10.0, SEA[1]), (1.0, SEA[3]), (0.1, RED)):
        h = float(np.interp(np.log(lev), np.log(surv[::-1] * 100.0),
                            hs_e[::-1]))
        ax_x.plot([h], [lev], marker="o", ms=4.4, mfc="white", mec=col,
                  mew=1.3, zorder=9)
        ax_x.annotate("%.0f %% exceeded above %.1f m" % (lev, h)
                      if lev >= 1.0 else
                      "%.1f %% exceeded above %.1f m" % (lev, h),
                      xy=(h, lev), xytext=(h + 0.45, lev * 2.6),
                      fontsize=6.2, color=col, ha="left", va="center",
                      zorder=10, arrowprops=dict(arrowstyle="-", color=col,
                                                 lw=0.7, shrinkA=1,
                                                 shrinkB=3))
    ax_x.set_yscale("log")
    ax_x.set_xlim(0.0, 10.0)
    ax_x.set_ylim(0.02, 260.0)
    ax_x.tick_params(labelsize=6.8)
    ax_x.set_xlabel("significant wave height $H_s$ (m)", fontsize=7.5,
                    labelpad=1.5)
    ax_x.set_ylabel("exceedance (%)", fontsize=7.5)
    ax_x.set_title("The tail sets the design height", fontsize=8.5,
                   fontweight="bold", pad=4.0)

    panel_labels_inches(fig, [ax, ax_s, ax_x], dx=-0.62, dy=0.10)
    foot = add_footer(fig, "Synthetic data, figure 996")
    check_layout(fig, foot, [ax, ax_s, ax_x])
    savefig(fig, "996_shp_wave_spectrum_scatter.png")


if __name__ == "__main__":
    main()
