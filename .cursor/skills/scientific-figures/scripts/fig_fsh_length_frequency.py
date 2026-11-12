"""Monthly length-frequency ridgeline with the cohorts tracked through it.

Each row is one month's length-frequency distribution, built as a four-age
mixture whose component means follow a von Bertalanffy curve, so the modes
march to the right as the year runs and the reader sees growth rather than
being told about it.  The coloured lines under each ridge are the age
components themselves; the red track joins, month by month, the *modes of the
plotted mixture* - found with find_peaks and matched to the nearest component
- so a cohort whose mode has merged into its neighbour simply drops out of
the track instead of being drawn where the model says it ought to be.

Both side panels are fitted to those tracked modes, not to the generating
parameters.  L_inf and K in panel (b) come from a curve_fit of the von
Bertalanffy form to the (age, modal length) pairs recovered from panel (a),
and the allometric exponent in panel (c) is the slope of a log-log
least-squares line through the sampled length-weight pairs.  Illustrative
values; this is a layout reference, not a stock assessment.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pubstyle import set_pub_style, savefig, get_palette, lighten, darken
from figkit import axes_inches, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
GREY = "#808080"
TRACK = "#b23a48"
FILL, EDGE = "#dce7f0", "#5b7fa6"

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.04, 4.86, 4.16)
AUX1 = (6.62, 3.14, 2.78, 2.06)
AUX2 = (6.62, 1.04, 2.78, 1.44)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
AGES = np.array([0.6, 1.6, 2.6, 3.6])      # age in years at the start of Jan
SHARE = np.array([0.40, 0.29, 0.19, 0.12])
LINF_T, K_T, T0_T = 42.0, 0.46, -0.18      # generating parameters
LGRID = np.linspace(4.0, 46.0, 700)


def von_bert(t, linf, k, t0):
    return linf * (1.0 - np.exp(-k * (t - t0)))


def build_lengths(rng):
    """Monthly mixtures on a shared density scale, plus their component means."""
    mu = np.empty((12, AGES.size))
    sd = np.empty((12, AGES.size))
    dens = np.empty((12, LGRID.size))
    comp = np.empty((12, AGES.size, LGRID.size))
    for m in range(12):
        age = AGES + m / 12.0
        mu[m] = von_bert(age, LINF_T, K_T, T0_T)
        sd[m] = 1.05 + 0.42 * age
        for j in range(AGES.size):
            g = np.exp(-0.5 * ((LGRID - mu[m, j]) / sd[m, j]) ** 2)
            comp[m, j] = SHARE[j] * g / (sd[m, j] * np.sqrt(2.0 * np.pi))
        dens[m] = comp[m].sum(axis=0)
    dens = dens * np.exp(rng.normal(0.0, 0.012, dens.shape))
    scale = dens.max()
    return dict(mu=mu, sd=sd, dens=dens / scale, comp=comp / scale)


def track_modes(lf):
    """Modes of each plotted mixture, matched to the nearest age component."""
    rows = []
    for m in range(12):
        idx, _ = find_peaks(lf["dens"][m], prominence=0.006)
        xs = LGRID[idx]
        for j in range(AGES.size):
            if xs.size == 0:
                continue
            k = int(np.argmin(np.abs(xs - lf["mu"][m, j])))
            if abs(xs[k] - lf["mu"][m, j]) <= 1.4:
                rows.append((m, j, float(xs[k]),
                             float(lf["dens"][m][idx[k]]),
                             float(AGES[j] + m / 12.0)))
    return rows


def sample_length_weight(rng, lf, n=420):
    j = rng.choice(AGES.size, size=n, p=SHARE / SHARE.sum())
    m = rng.integers(0, 12, size=n)
    length = rng.normal(lf["mu"][m, j], lf["sd"][m, j])
    length = np.clip(length, 5.0, 46.0)
    weight = 0.0098 * length ** 3.06 * np.exp(rng.normal(0.0, 0.085, n))
    return length, weight


def draw_ridgeline(ax, lf, modes):
    step = 0.62
    cols = get_palette("cb_sequential_blue", 6)[1:5]
    for m in range(12):
        base = m * step
        ax.fill_between(LGRID, base, base + lf["dens"][m], facecolor=FILL,
                        edgecolor="none", zorder=3 + m * 2)
        ax.plot(LGRID, base + lf["dens"][m], color=EDGE, lw=0.9,
                zorder=4 + m * 2)
        for j in range(AGES.size):
            ax.plot(LGRID, base + lf["comp"][m, j], color=cols[j], lw=0.65,
                    alpha=0.85, zorder=4 + m * 2)
        ax.axhline(base, color="#e0e0e0", lw=0.5, zorder=2)

    for j in range(AGES.size):
        pts = [(x, m * step + h) for m, jj, x, h, _ in modes if jj == j]
        if len(pts) < 2:
            continue
        px = [p[0] for p in pts]
        py = [p[1] for p in pts]
        ax.plot(px, py, color=TRACK, lw=1.0, ls=(0, (3.2, 1.8)), zorder=60)
        ax.scatter(px, py, s=8.0, color=TRACK, edgecolors="white",
                   linewidths=0.3, zorder=61)
        ax.annotate("age {:.0f}+".format(AGES[j] - 0.6),
                    xy=(px[-1], py[-1]), xytext=(0, 9),
                    textcoords="offset points", fontsize=6.4, color=TRACK,
                    ha="center", va="bottom", zorder=62,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_yticks([m * step for m in range(12)])
    ax.set_yticklabels(MONTHS)
    ax.set_ylim(-0.10, 11 * step + 1.55)
    ax.set_xlim(LGRID[0], LGRID[-1])
    ax.set_xlabel("fork length (cm)", fontsize=8.0)
    ax.set_ylabel("month of sampling", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title("Length-frequency by month, with age components and mode "
                 "tracks", fontsize=8.5, fontweight="bold")

    handles = [Line2D([], [], color=cols[j], lw=1.1,
                      label="age {:.0f}+ component".format(AGES[j] - 0.6))
               for j in range(AGES.size)]
    handles.append(Line2D([], [], color=TRACK, lw=1.0, ls=(0, (3.2, 1.8)),
                          marker="o", ms=3.0, label="tracked mode"))
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.6,
                    frameon=True, framealpha=0.94, borderpad=0.4,
                    handlelength=1.7, labelspacing=0.32)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(70)


def draw_vonbert(ax, modes):
    age = np.array([r[4] for r in modes])
    length = np.array([r[2] for r in modes])
    par, cov = curve_fit(von_bert, age, length, p0=[40.0, 0.4, -0.1],
                         maxfev=40000)
    linf, k, t0 = par
    grid = np.linspace(0.0, age.max() + 2.6, 240)
    ax.axhline(linf, color=GREY, lw=0.9, ls=(0, (4.0, 2.4)), zorder=3)
    ax.text(grid[-1], linf + 0.7,
            "L$_\\infty$ = {:.1f} cm".format(linf), fontsize=6.5,
            color=SUB, ha="right", va="bottom", zorder=6)
    ax.plot(grid, von_bert(grid, *par), color=darken(EDGE, 0.2), lw=1.5,
            zorder=5)
    ax.scatter(age, length, s=13.0, facecolors=lighten(TRACK, 0.45),
               edgecolors="white", linewidths=0.35, zorder=6)
    ax.text(0.975, 0.06,
            "L(t) = L$_\\infty$[1 - e$^{-K(t-t_0)}$]\n"
            + "K = {:.3f} /yr,  t$_0$ = {:.2f} yr".format(k, t0),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=SUB, linespacing=1.4)
    ax.set_xlim(0.0, grid[-1])
    ax.set_ylim(0.0, linf * 1.16)
    ax.set_xlabel("age (year)", fontsize=7.8)
    ax.set_ylabel("fork length (cm)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("von Bertalanffy fit to the tracked modes", fontsize=8.5,
                 fontweight="bold")
    return linf, k, age.size


def draw_allometry(ax, length, weight):
    lx, ly = np.log10(length), np.log10(weight)
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    r2 = 1.0 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    resid = np.sqrt(np.sum((ly - pred) ** 2) / (lx.size - 2))
    se_b = resid / np.sqrt(np.sum((lx - lx.mean()) ** 2))

    ax.scatter(length, weight, s=7.0, facecolors=lighten(EDGE, 0.42),
               edgecolors="white", linewidths=0.2, alpha=0.9, zorder=4)
    grid = np.linspace(length.min(), length.max(), 120)
    ax.plot(grid, 10.0 ** a * grid ** b, color=TRACK, lw=1.4, zorder=6)
    ax.plot(grid, 10.0 ** a * grid.mean() ** (b - 3.0) * grid ** 3.0,
            color=GREY, lw=0.9, ls=(0, (3.4, 2.2)), zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.text(0.035, 0.95,
            "W = aL$^b$,  b = {:.3f} $\\pm$ {:.3f}\nR$^2$ = {:.3f}   "
            "(dashed: b = 3)".format(b, se_b, r2),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            color=SUB, linespacing=1.4)
    ax.set_xlabel("fork length (cm, log)", fontsize=7.8)
    ax.set_ylabel("wet weight (g, log)", fontsize=7.8)
    ax.tick_params(labelsize=6.8, which="both")
    ax.set_title("Length-weight allometry", fontsize=8.5, fontweight="bold")
    return b, se_b, r2


def main():
    set_pub_style()
    rng = np.random.default_rng(9010)
    lf = build_lengths(rng)
    modes = track_modes(lf)
    length, weight = sample_length_weight(rng, lf)

    fig = plt.figure(figsize=(FW, FH))
    ax_r = axes_inches(fig, MAIN)
    ax_v = axes_inches(fig, AUX1)
    ax_a = axes_inches(fig, AUX2)

    draw_ridgeline(ax_r, lf, modes)
    linf, k_fit, n_mode = draw_vonbert(ax_v, modes)
    b, se_b, r2 = draw_allometry(ax_a, length, weight)

    panel_labels_inches(fig, [ax_r, ax_v, ax_a], dx=-0.46, dy=0.20)

    shift = np.array([r[2] for r in modes if r[1] == 0])
    grow = shift[-1] - shift[0] if shift.size > 1 else 0.0

    print("modes={:d} linf={:.2f} k={:.3f} b={:.3f} r2={:.3f} grow={:.2f}"
          .format(n_mode, linf, k_fit, b, r2, grow))
    savefig(fig, "901_fsh_length_frequency.png")


if __name__ == "__main__":
    main()
