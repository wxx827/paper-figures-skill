"""Lithography process window: a yield surface over exposure dose and focus
offset, the largest rectangular window that clears the yield spec, the target
point and the measured lots, with marginal yield curves alongside."""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import (set_pub_style, savefig, style_axes, get_cmap,
                      add_colorbar, annotate_box)
from figkit import panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

SPEC = 95.0                 # yield specification, %
DOSE0, FOCUS0 = 31.0, 0.0   # centre of the modelled process response
TARGET = (31.18, -14.0)     # where the tool is actually set up


def yield_surface(dose, focus):
    """Bossung-tilted yield response, %.

    A flat-topped power law rather than a Gaussian: a Gaussian peaked at 99%
    drops below a 95% spec within a fraction of its own sigma, which would
    leave a process window too small to draw.
    """
    u = (dose - DOSE0) / 1.90
    v = (focus - FOCUS0 - 26.0 * u) / 115.0
    r = np.hypot(u, v)
    y = 99.4 - 42.0 * r ** 3.2 - 5.0 * np.clip(u, 0.0, None) ** 2
    return np.maximum(y, 40.0)


def rasterize(artist):
    """3.10's ContourSet mis-reports 'rasterization will be ignored'."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Rasterization of", UserWarning)
        artist.set_rasterized(True)


def max_rect(mask, x, y):
    """Largest axis-aligned rectangle fitting inside a boolean mask.

    ``mask`` is indexed ``[iy, ix]``.  Returns ``(x0, x1, y0, y1, area)`` in
    the physical units of ``x`` and ``y``; the classic rectangular process
    window an exposure tool is actually set up against.
    """
    ny, nx = mask.shape
    best = (0.0, None)
    for i0 in range(nx):
        col = mask[:, i0].copy()
        for i1 in range(i0, nx):
            col &= mask[:, i1]
            if not col.any():
                break
            run = j0 = 0
            cur = start = 0
            for j in range(ny + 1):
                if j < ny and col[j]:
                    if cur == 0:
                        start = j
                    cur += 1
                else:
                    if cur > run:
                        run, j0 = cur, start
                    cur = 0
            area = (x[i1] - x[i0]) * (y[j0 + run - 1] - y[j0])
            if area > best[0]:
                best = (area, (x[i0], x[i1], y[j0], y[j0 + run - 1]))
    x0, x1, y0, y1 = best[1]
    return x0, x1, y0, y1, best[0]


def lots(rng, n=64):
    """Measured lots: correlated dose / focus excursions around the target."""
    d = TARGET[0] + rng.normal(0, 0.42, n)
    f = TARGET[1] + 30.0 * rng.normal(0, 1.0, n) + 26.0 * (d - TARGET[0])
    return d, f


def main_panel(fig, ax, D, F, Y, win, dl, fl):
    x0, x1, y0, y1, area = win
    cs = ax.contourf(D, F, Y, levels=np.arange(45, 101, 5),
                     cmap=get_cmap("cb_sequential_blue"), extend="min",
                     zorder=1)
    rasterize(cs)
    cs.set_edgecolor("face")
    line = ax.contour(D, F, Y, levels=[60, 75, 90], colors="white",
                      linewidths=0.6, zorder=2)
    ax.clabel(line, fmt="%d%%", fontsize=6.0, inline=True, colors="white")
    ax.contour(D, F, Y, levels=[SPEC], colors=[RED], linewidths=1.4, zorder=4)

    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                           edgecolor=RED, linewidth=1.4, linestyle=(0, (4, 2)),
                           zorder=6))
    ax.text((x0 + x1) / 2, y1 + 4, "rectangular process window", ha="center",
            va="bottom", fontsize=6.6, color=RED, fontweight="bold", zorder=7)

    inside = yield_surface(dl, fl) >= SPEC
    ax.plot(dl[inside], fl[inside], "o", ms=3.4, mfc="white", mec="#333333",
            mew=0.8, zorder=7, label=f"lot within spec (n={int(inside.sum())})")
    ax.plot(dl[~inside], fl[~inside], "o", ms=4.0, mfc=RED, mec="white",
            mew=0.6, zorder=8,
            label=f"lot out of spec (n={int((~inside).sum())})")
    ax.plot([TARGET[0]], [TARGET[1]], marker="*", ms=13, mfc=GOLD,
            mec="#3a3a3a", mew=0.8, zorder=9, label="target set-up")

    md = min(TARGET[0] - x0, x1 - TARGET[0])
    mf = min(TARGET[1] - y0, y1 - TARGET[1])
    ax.annotate("", xy=(x1, TARGET[1]), xytext=(TARGET[0], TARGET[1]),
                arrowprops=dict(arrowstyle="<->", color="#3a3a3a", lw=0.8,
                                shrinkA=0, shrinkB=0), zorder=9)
    ax.text((TARGET[0] + x1) / 2, TARGET[1] + 3.5,
            f"{x1 - TARGET[0]:.2f} mJ/cm$^2$", ha="center", va="bottom",
            fontsize=6.3, color="#3a3a3a", zorder=9)

    annotate_box(ax, f"window {x1 - x0:.2f} mJ/cm$^2$ \u00d7 {y1 - y0:.0f} nm"
                 f"\narea {area:.0f} mJ\u00b7nm/cm$^2$"
                 f"\ndose margin \u00b1{md:.2f} mJ/cm$^2$ "
                 f"({100 * md / (0.5 * (x1 - x0)):.0f}% of half-width)"
                 f"\nfocus margin \u00b1{mf:.0f} nm "
                 f"({100 * mf / (0.5 * (y1 - y0)):.0f}% of half-depth)",
                 loc="lower left", fontsize=6.4, linespacing=1.55)

    ax.set_xlim(D.min(), D.max())
    ax.set_ylim(F.min(), F.max())
    ax.set_xlabel("Exposure dose (mJ/cm$^2$)", fontsize=8.0)
    ax.set_ylabel("Focus offset (nm)", fontsize=8.0)
    ax.set_title("Yield response and qualified process window", fontsize=9.0,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=7.0)
    leg = ax.legend(loc="upper right", fontsize=6.6, frameon=True,
                    framealpha=0.92, facecolor="white", edgecolor="#cccccc",
                    borderpad=0.4, handlelength=1.0, borderaxespad=0.3)
    leg.set_zorder(10)
    style_axes(ax)
    cb = add_colorbar(fig, cs, ax, label="predicted die yield (%)",
                      shrink=0.60, aspect=15)
    cb.ax.tick_params(labelsize=6.5)
    return md, mf


def marginal_panel(ax, x, y, win_lo, win_hi, target, xlabel, unit, ylim,
                   color, title):
    ax.axvspan(win_lo, win_hi, color="#eef2f6", zorder=0)
    ax.plot(x, y, color=color, lw=1.6, zorder=4)
    ax.axhline(SPEC, color=RED, lw=0.9, ls=(0, (4, 2.5)), zorder=3)
    ax.text(x[0], SPEC - 1.2, f"spec {SPEC:.0f}%", ha="left", va="top",
            fontsize=6.3, color=RED)
    for v in (win_lo, win_hi):
        ax.axvline(v, color=BASE, lw=0.7, ls=(0, (2.2, 2.0)), zorder=3)
    ax.plot([target], [float(np.interp(target, x, y))], marker="*", ms=10,
            mfc=GOLD, mec="#3a3a3a", mew=0.7, zorder=6)
    ax.annotate(f"window {win_hi - win_lo:.2f} {unit}"
                if unit == "mJ/cm$^2$" else
                f"window {win_hi - win_lo:.0f} {unit}",
                xy=((win_lo + win_hi) / 2, ylim[0] + 0.06 * (ylim[1] - ylim[0])),
                ha="center", va="bottom", fontsize=6.4, color=SUB)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, fontsize=7.5)
    ax.set_ylabel("Die yield (%)", fontsize=7.5)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(9071)
    dose = np.linspace(28.4, 33.6, 261)
    focus = np.linspace(-150, 150, 261)
    D, F = np.meshgrid(dose, focus)
    Y = yield_surface(D, F)

    cd = np.linspace(28.4, 33.6, 96)
    cf = np.linspace(-150, 150, 96)
    CD, CF = np.meshgrid(cd, cf)
    win = max_rect(yield_surface(CD, CF) >= SPEC, cd, cf)
    x0, x1, y0, y1, _area = win
    dc, fc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    dl, fl = lots(rng, 64)
    y_dose = yield_surface(dose, np.full_like(dose, fc))
    y_focus = yield_surface(np.full_like(focus, dc), focus)
    ylim = shared_limits([y_dose, y_focus, np.array([SPEC, 100.0])], pad=0.06)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.52,
                          left=0.055, right=0.975, top=0.885, bottom=0.095)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[1, 1])

    md, mf = main_panel(fig, ax_main, D, F, Y, win, dl, fl)
    marginal_panel(ax_d, dose, y_dose, x0, x1, TARGET[0],
                   "Exposure dose (mJ/cm$^2$)", "mJ/cm$^2$", ylim, BLUE,
                   f"Dose margin at focus {fc:+.0f} nm")
    marginal_panel(ax_f, focus, y_focus, y0, y1, TARGET[1],
                   "Focus offset (nm)", "nm", ylim, GREEN,
                   f"Focus margin at dose {dc:.2f} mJ/cm$^2$")

    inside = yield_surface(dl, fl) >= SPEC

    panel_labels_inches(fig, [ax_main, ax_d, ax_f], dx=-0.36, dy=0.08)
    savefig(fig, "500_semi_process_window.png")


if __name__ == "__main__":
    main()
