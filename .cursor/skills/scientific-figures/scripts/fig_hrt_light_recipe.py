"""Six LED spectra crossed with two daily light integrals in a vertical farm.

One response surface in blue fraction, far-red fraction, green fraction and
daily light integral generates eight replicate plants per cell; the bars are
the means of those replicates, the whiskers their standard errors and the
translucent dots the replicates themselves, so nothing in the panel is drawn
twice from different numbers.  The spectra in the upper side panel are the
same recipes turned into photon flux per nanometre and scaled to the stated
photosynthetic photon flux density, and the light-use efficiency in the lower
one is the dry mass of those very plants divided by the photons they were
given.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
GOLD = "#c9a227"

SEED = 1037
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)

NREP = 8                    # replicate plants per cell
CYCLE = 28.0                # days from transplant to harvest
DENSITY = 25.0              # plants per m2
PPFD = 220.0                # photosynthetic photon flux density, umol m-2 s-1

# label, red %, blue %, green %, far-red %, colour
RECIPES = [
    ("R100", 100.0, 0.0, 0.0, 0.0, "#b23a48"),
    ("R90:B10", 90.0, 10.0, 0.0, 0.0, "#c0603f"),
    ("R80:B20", 80.0, 20.0, 0.0, 0.0, "#c9a227"),
    ("R70:B30", 70.0, 30.0, 0.0, 0.0, "#3d6b8e"),
    ("R75:B15:FR10", 75.0, 15.0, 0.0, 10.0, "#2a7f7a"),
    ("White + FR", 45.0, 20.0, 25.0, 10.0, "#0b3d63"),
]

DLI = [(12.0, "#9ccfae", "12 mol m$^{-2}$ d$^{-1}$"),
       (17.0, "#2e7d5b", "17 mol m$^{-2}$ d$^{-1}$")]

# centre nm, width nm, which recipe channel it belongs to
BANDS = [(450.0, 13.0, 2), (530.0, 34.0, 3), (660.0, 10.0, 1),
         (735.0, 12.0, 4)]


def fresh_weight(blue, green, farred, dli):
    """Mean shoot fresh weight in g per plant on the response model."""
    f_blue = 0.78 + 0.34 * np.exp(-0.5 * ((blue - 16.0) / 9.0) ** 2)
    f_fr = 1.0 + 0.011 * farred
    f_green = 1.0 - 0.0015 * green
    f_dli = (dli / 14.5) ** 0.62
    return 24.0 * f_blue * f_fr * f_green * f_dli


def dry_fraction(blue):
    return 0.048 + 0.0006 * blue


def trial(rng):
    """Replicate fresh weights, dry weights and light-use efficiencies."""
    fw = np.zeros((len(RECIPES), len(DLI), NREP))
    for i, (_lab, _r, b, g, fr, _c) in enumerate(RECIPES):
        for j, (dli, _c2, _l) in enumerate(DLI):
            mu = fresh_weight(b, g, fr, dli)
            fw[i, j] = mu * np.exp(rng.normal(0.0, 0.072, NREP))
    dm = np.array([dry_fraction(r[2]) for r in RECIPES])[:, None, None]
    dw = fw * dm
    photons = np.array([d[0] for d in DLI])[None, :] * CYCLE
    lue = dw.mean(axis=2) * DENSITY / photons
    return fw, dw, lue


def spectrum(nm, recipe):
    """Photon flux per nanometre, scaled so 400-700 nm integrates to PPFD."""
    _lab, red, blue, green, farred = recipe[0], recipe[1], recipe[2], \
        recipe[3], recipe[4]
    frac = {1: red, 2: blue, 3: green, 4: farred}
    s = np.zeros_like(nm)
    for (c, w, key) in BANDS:
        s += frac[key] / 100.0 * np.exp(-0.5 * ((nm - c) / w) ** 2) / (
            w * np.sqrt(2.0 * np.pi))
    par = (nm >= 400.0) & (nm <= 700.0)
    area = np.trapezoid(s[par], nm[par])
    return s * PPFD / area


def main_panel(ax, fw, best):
    x = np.arange(len(RECIPES), dtype=float)
    width = 0.34
    ctrl = fw[0, 1].mean()

    for j, (dli, col, lab) in enumerate(DLI):
        off = (j - 0.5) * width * 1.06
        mean = fw[:, j].mean(axis=1)
        sem = fw[:, j].std(axis=1, ddof=1) / np.sqrt(NREP)
        ax.bar(x + off, mean, width=width, color=col, edgecolor="white",
               linewidth=0.8, zorder=4, label=lab)
        ax.errorbar(x + off, mean, yerr=sem, fmt="none", ecolor="#3a3a3a",
                    elinewidth=0.9, capsize=2.4, capthick=0.9, zorder=7)
        jitter = np.linspace(-0.085, 0.085, NREP)
        for i in range(len(RECIPES)):
            ax.plot(x[i] + off + jitter, fw[i, j], ls="none", marker="o",
                    ms=2.4, mfc="white", mec="#5a5a5a", mew=0.5, alpha=0.85,
                    zorder=8)
        for i in range(len(RECIPES)):
            ax.text(x[i] + off, mean[i] + sem[i] + 0.9, "{:.1f}"
                    .format(mean[i]), ha="center", va="bottom", fontsize=6.0,
                    color=SUB, zorder=9)

    ax.axhline(ctrl, color=RED, lw=0.9, ls=(0, (4.5, 2.4)), zorder=3)

    i_b, j_b = best
    mean_b = fw[i_b, j_b].mean()
    ax.annotate("{:s} at {:.0f} mol m$^{{-2}}$ d$^{{-1}}$ is the best cell,\n"
                "{:.1f} g plant$^{{-1}}$, {:.0f} % over the red-only control"
                .format(RECIPES[i_b][0], DLI[j_b][0], mean_b,
                        100.0 * (mean_b / ctrl - 1.0)),
                xy=(i_b + 0.5 * 0.34 * 1.06, mean_b),
                xytext=(5.40, 45.0), fontsize=6.6, color=INK, ha="right",
                va="top", linespacing=1.34, zorder=10,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8, shrinkA=2,
                                shrinkB=6))

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in RECIPES], fontsize=7.0)
    for tick, r in zip(ax.get_xticklabels(), RECIPES):
        tick.set_color(r[5])
    ax.set_xlim(-0.62, len(RECIPES) - 0.38)
    ax.set_ylim(0.0, 48.0)
    ax.set_xlabel("spectral recipe, per cent of the photon flux in each band")
    ax.set_ylabel("shoot fresh weight (g plant$^{-1}$)")
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=c, edgecolor="white", lw=0.8,
                     label="daily light integral " + lab)
               for (_d, c, lab) in DLI]
    handles += [Line2D([], [], ls="none", marker="o", ms=2.8, mfc="white",
                       mec="#5a5a5a", label="{:d} replicate plants"
                       .format(NREP)),
                Line2D([], [], color="#3a3a3a", lw=0.9,
                       label="mean $\\pm$ standard error"),
                Line2D([], [], color=RED, lw=0.9, ls=(0, (4.5, 2.4)),
                       label="red-only control, {:.1f} g".format(ctrl))]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.5, ncol=2, columnspacing=0.9, borderaxespad=0.35,
              labelspacing=0.24)
    ax.set_title("Twenty per cent blue beats pure red at both light "
                 "integrals", fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def spectrum_panel(ax, nm, picks):
    ax.axvspan(400.0, 700.0, color="#f1f4f1", zorder=1)
    top = 0.0
    for i in picks:
        r = RECIPES[i]
        s = spectrum(nm, r)
        top = max(top, float(s.max()))
        ax.plot(nm, s, color=r[5], lw=1.5, zorder=5, label=r[0])
        ax.fill_between(nm, 0.0, s, color=r[5], alpha=0.14, zorder=3, lw=0)
    top = top * 1.24
    ax.text(548.0, top * 0.985, "PAR waveband 400-700 nm", ha="center",
            va="top", fontsize=6.0, color=SUB, zorder=6)
    ax.text(752.0, top * 0.985, "far red\noutside PAR", ha="center",
            va="top", fontsize=6.0, color="#8c3f3f", linespacing=1.3,
            zorder=6)

    ax.set_xlim(380.0, 790.0)
    ax.set_ylim(0.0, top)
    ax.set_xlabel("wavelength (nm)", fontsize=6.8)
    ax.set_ylabel("photon flux (" + r"$\mu mol\ m^{-2} s^{-1}$" + " nm$^{-1}$)",
                  fontsize=6.2)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.94, handlelength=1.3,
              borderaxespad=0.3, labelspacing=0.2)
    ax.set_title("Three recipes at the same PPFD of {:.0f} ".format(PPFD)
                 + r"$\mu mol\ m^{-2} s^{-1}$", fontsize=8.0,
                 fontweight="bold", color=INK)


def efficiency_panel(ax, lue):
    blue = np.array([r[2] for r in RECIPES])
    for j, (dli, col, lab) in enumerate(DLI):
        order = np.argsort(blue)
        ax.plot(blue[order], lue[order, j], color=col, lw=1.4, zorder=4)
        for i in range(len(RECIPES)):
            ax.plot([blue[i]], [lue[i, j]], marker="o" if j else "s", ms=4.4,
                    mfc=RECIPES[i][5], mec="white", mew=0.8, zorder=7)
    i_best = int(np.unravel_index(np.argmax(lue), lue.shape)[0])
    j_best = int(np.unravel_index(np.argmax(lue), lue.shape)[1])
    ax.annotate("peak {:.3f} g mol$^{{-1}}$\nat {:.0f} % blue"
                .format(lue[i_best, j_best], blue[i_best]),
                xy=(blue[i_best], lue[i_best, j_best]),
                xytext=(31.5, float(lue.max()) * 1.055),
                fontsize=6.1, color=INK, ha="right", va="top",
                linespacing=1.32, zorder=9,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7, shrinkA=2,
                                shrinkB=4))

    ax.set_xlim(-2.0, 33.0)
    ax.set_ylim(float(lue.min()) * 0.93, float(lue.max()) * 1.10)
    ax.set_xlabel("blue share of the photon flux (%)", fontsize=6.8)
    ax.set_ylabel("light-use efficiency (g mol$^{-1}$)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.42)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=DLI[0][1], lw=1.4, marker="s", ms=4.0,
                      mfc="#bbbbbb", mec="white", label="12 mol"),
               Line2D([], [], color=DLI[1][1], lw=1.4, marker="o", ms=4.0,
                      mfc="#bbbbbb", mec="white", label="17 mol")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.3, ncol=2, columnspacing=0.8, borderaxespad=0.3,
              labelspacing=0.2, title="daily light integral",
              title_fontsize=6.0)
    ax.set_title("Dry mass returned per mole of photons", fontsize=8.0,
                 fontweight="bold", color=INK)


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)

    for r in RECIPES:
        assert abs(r[1] + r[2] + r[3] + r[4] - 100.0) < 1e-9, \
            "every recipe must account for 100 % of the photon flux"

    fw, dw, lue = trial(rng)
    means = fw.mean(axis=2)
    best = np.unravel_index(int(np.argmax(means)), means.shape)
    ctrl = means[0, 1]

    nm = np.linspace(380.0, 790.0, 900)
    for r in RECIPES:
        s = spectrum(nm, r)
        par = (nm >= 400.0) & (nm <= 700.0)
        assert abs(float(np.trapezoid(s[par], nm[par])) - PPFD) < 0.5, \
            "each spectrum must carry the stated PPFD inside the PAR band"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_sp = axes_inches(fig, AUX_T)
    ax_lue = axes_inches(fig, AUX_B)

    main_panel(ax_main, fw, best)
    spectrum_panel(ax_sp, nm, [0, 2, 5])
    efficiency_panel(ax_lue, lue)

    gain = 100.0 * (means[best] / ctrl - 1.0)
    fr_gain = 100.0 * (means[4, 1] / means[2, 1] - 1.0)


    panel_labels_inches(fig, [ax_main, ax_sp, ax_lue], dx=-0.62, dy=0.14)
    savefig(fig, "1037_hrt_light_recipe.png")


if __name__ == "__main__":
    main()
