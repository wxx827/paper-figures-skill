"""Stacked XRD patterns of TiO2 calcined at increasing temperature.

Six vertically offset diffractograms show anatase crystallisation and the
anatase -> rutile transformation.  Dashed guide lines carry the (hkl) indices,
a two-row stick panel reproduces the reference PDF cards, and each trace is
labelled on the right with its Scherrer crystallite size.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, PALETTES, savefig

# --- reference cards: (2theta, relative intensity, hkl) --------------------
ANATASE = [(25.28, 100, "(101)"), (37.80, 20, "(004)"), (48.05, 35, "(200)"),
           (53.89, 20, "(105)"), (55.06, 20, "(211)"), (62.69, 14, "(204)"),
           (68.76, 6, "(116)"), (70.31, 6, "(220)"), (75.03, 10, "(215)")]
RUTILE = [(27.45, 100, "(110)"), (36.09, 50, "(101)"), (39.19, 8, "(200)"),
          (41.23, 25, "(111)"), (44.05, 10, "(210)"), (54.32, 60, "(211)"),
          (56.64, 20, "(220)"), (62.74, 10, "(002)"), (64.04, 10, "(310)"),
          (69.01, 20, "(301)"), (69.79, 12, "(112)")]

# guide lines that get an (hkl) caption (close pairs are merged)
GUIDES = [(25.28, "(101)"), (37.80, "(004)"), (48.05, "(200)"),
          (54.48, "(105)/(211)"), (62.69, "(204)"), (75.03, "(215)")]

# label, rutile weight fraction, peak FWHM (deg), crystallite size (nm)
SAMPLES = [("as-dried", None, 0.0, "amorphous"),
           ("400 °C", 0.00, 0.86, "D = 9 nm"),
           ("500 °C", 0.00, 0.55, "D = 14 nm"),
           ("600 °C", 0.09, 0.38, "D = 21 nm"),
           ("700 °C", 0.46, 0.27, "D = 30 nm"),
           ("800 °C", 0.93, 0.19, "D = 47 nm")]

C_ANA = "#2e5f8a"
C_RUT = "#b0562f"


def pseudo_voigt(x, x0, fwhm, eta=0.55):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - x0) / (0.5 * fwhm)) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def pattern(x, card, frac, fwhm):
    y = np.zeros_like(x)
    for pos, rel, _ in card:
        # Bragg peaks broaden slightly at high angle (instrumental)
        w = fwhm * (1.0 + 0.010 * (pos - 25.0))
        y += frac * (rel / 100.0) * pseudo_voigt(x, pos, w)
    return y


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(7)

    x = np.linspace(20.0, 80.0, 4200)
    cmap = LinearSegmentedColormap.from_list(
        "xrd", [PALETTES["cb_sequential_blue"][2], "#2e6f9e", "#3f6f5a",
                "#a8712f", PALETTES["earth"][3]])

    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[6.4, 1.25], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axp = fig.add_subplot(gs[1], sharex=ax)

    off = 1.0
    traces = []
    for i, (name, rfrac, fwhm, note) in enumerate(SAMPLES):
        if rfrac is None:                       # amorphous xerogel
            y = (0.62 * pseudo_voigt(x, 30.5, 13.0, eta=0.2)
                 + 0.26 * pseudo_voigt(x, 47.5, 11.0, eta=0.2))
        else:
            y = (pattern(x, ANATASE, 1.0 - rfrac, fwhm)
                 + pattern(x, RUTILE, rfrac, fwhm * 0.92))
        y = y + 0.11 * np.exp(-(x - 20.0) / 26.0) + 0.020
        y = y / y.max()
        y = y + rng.normal(0.0, 1.0, x.size) * 0.016 * np.sqrt(y + 0.05)
        traces.append(y)

    top_base = off * (len(SAMPLES) - 1)
    y_lab = top_base + 1.52
    y_top = top_base + 2.02

    # (hkl) guide lines spanning both panels
    for pos, txt in GUIDES:
        ax.plot([pos, pos], [-0.05, y_lab - 0.20], color="#c2c2c2", lw=0.55,
                ls=(0, (4.5, 3.0)), zorder=0)
        axp.axvline(pos, color="#c2c2c2", lw=0.55, ls=(0, (4.5, 3.0)),
                    zorder=0)
        ax.text(pos, y_lab, txt, ha="center", va="bottom", fontsize=6.8,
                color="#333333", fontweight="bold")

    for i, ((name, rfrac, fwhm, note), y) in enumerate(zip(SAMPLES, traces)):
        base = off * i
        col = cmap(i / (len(SAMPLES) - 1))
        ax.fill_between(x, base, base + y, color=col, alpha=0.10, lw=0,
                        zorder=2 + i)
        ax.plot(x, base + y, color=col, lw=0.85, zorder=10 + i)
        ax.text(79.4, base + 0.46, name, ha="right", va="bottom",
                fontsize=8, fontweight="bold", color=col, zorder=30)
        ax.text(79.4, base + 0.30, note, ha="right", va="bottom",
                fontsize=6.5, color="#5b5b5b", zorder=30)

    # rutile reflections flagged on the top-most (800 C) trace
    ytop = traces[-1] + top_base
    for pos, rel, _ in RUTILE:
        if rel < 25:
            continue
        j = np.argmin(np.abs(x - pos))
        ax.plot(pos, ytop[j] + 0.10, marker="v", ms=3.6, color=C_RUT,
                mec="white", mew=0.4, zorder=40, clip_on=False)

    ax.set_xlim(20.0, 80.0)
    ax.set_ylim(-0.06, y_top)
    ax.set_yticks([])
    ax.set_ylabel("Intensity (a.u.)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelbottom=False, length=2.5)

    ax.annotate("anatase $\\rightarrow$ rutile\ntransformation",
                xy=(27.45, top_base + 1.02), xytext=(31.6, top_base + 1.30),
                fontsize=7, color=C_RUT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_RUT, lw=0.7,
                                shrinkA=1.0, shrinkB=2.0,
                                connectionstyle="arc3,rad=-0.25"))

    # --- reference stick patterns -----------------------------------------
    for pos, rel, _ in ANATASE:
        axp.plot([pos, pos], [0.0, rel / 100.0], color=C_ANA, lw=1.0,
                 solid_capstyle="butt", zorder=3)
    for pos, rel, _ in RUTILE:
        axp.plot([pos, pos], [0.0, -rel / 100.0], color=C_RUT, lw=1.0,
                 solid_capstyle="butt", zorder=3)
    axp.axhline(0.0, color="#8c8c8c", lw=0.6, zorder=2)
    handles = [Line2D([], [], marker="v", ls="none", ms=3.6, color=C_RUT,
                      mec="white", mew=0.4, label="rutile reflection")]
    axp.legend(handles=handles, loc="upper left", fontsize=6.4,
               handletextpad=0.3, borderpad=0.25, borderaxespad=0.25)
    axp.text(79.4, 0.42, "anatase · JCPDS 21-1272", ha="right", va="bottom",
             fontsize=6.4, color=C_ANA)
    axp.text(79.4, -0.46, "rutile · JCPDS 21-1276", ha="right", va="top",
             fontsize=6.4, color=C_RUT)
    axp.set_ylim(-1.35, 1.35)
    axp.set_yticks([])
    axp.set_xlim(20.0, 80.0)
    axp.set_xticks(np.arange(20, 81, 10))
    axp.set_xticks(np.arange(20, 81, 2), minor=True)
    axp.set_xlabel("2$\\theta$ (°),  Cu K$\\alpha$  ($\\lambda$ = 1.5406 Å)")
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_visible(False)
    axp.spines["left"].set_visible(False)

    ax.set_title("Phase evolution of sol–gel TiO$_2$ upon calcination",
                 fontsize=10, pad=16)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.085)
    savefig(fig, "119_xrd_stacked.png")


if __name__ == "__main__":
    main()
