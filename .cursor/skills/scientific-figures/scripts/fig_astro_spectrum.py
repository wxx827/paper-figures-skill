"""Optical spectrum of an emission-line galaxy with identified transitions.

The synthetic spectrum is a reddened power-law continuum with a 4000 A break,
Gaussian emission lines for the usual nebular diagnostics and Gaussian
absorption for the stellar features.  Panel (a) carries the full range with
every identification; the three subordinate panels below re-plot the shaded
windows at their native resolution so that blended profiles separate, and the
bottom strip shows the fit residuals.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pubstyle import set_pub_style, savefig, add_panel_label

NAVY = "#1f3355"
INDIGO = "#5b4b8a"
GOLD = "#c9962f"
CYAN = "#3f7f95"
RUST = "#a8523a"
GREY = "#8b9099"

WMIN, WMAX = 3820.0, 7080.0

#            wl      amp   sigma  kind
LINES = [(3933.7, -0.34, 6.0, "abs"), (3968.5, -0.28, 6.0, "abs"),
         (4101.7, -0.16, 6.5, "abs"), (4340.5, 0.34, 3.6, "em"),
         (4861.3, 0.95, 3.8, "em"), (4958.9, 0.78, 3.5, "em"),
         (5006.8, 2.30, 3.5, "em"), (5175.0, -0.38, 9.0, "abs"),
         (5889.9, -0.40, 4.2, "abs"), (5895.9, -0.32, 4.2, "abs"),
         (6300.3, 0.26, 3.4, "em"), (6548.0, 0.52, 3.9, "em"),
         (6562.8, 3.30, 4.3, "em"), (6583.4, 1.55, 3.9, "em"),
         (6716.4, 0.58, 3.8, "em"), (6730.8, 0.46, 3.8, "em")]

GROUPS = [((3933.7, 3968.5), "Ca II H+K", "abs"),
          ((4101.7,), r"H$\delta$", "abs"),
          ((4340.5,), r"H$\gamma$", "em"),
          ((4861.3,), r"H$\beta$", "em"),
          ((4958.9, 5006.8), "[O III]", "em"),
          ((5175.0,), "Mg b", "abs"),
          ((5889.9, 5895.9), "Na I D", "abs"),
          ((6300.3,), "[O I]", "em"),
          ((6562.8,), r"H$\alpha$ + [N II]", "em"),
          ((6716.4, 6730.8), "[S II]", "em")]

WINDOWS = [
    (4820, 5060, r"H$\beta$ + [O III] doublet", "em",
     [(4861.3, r"H$\beta$"), (4958.9, "[O III] 4959"),
      (5006.8, "[O III] 5007")]),
    (5855, 5935, "Na I D absorption", "abs",
     [(5889.9, r"D$_2$ 5890"), (5895.9, r"D$_1$ 5896")]),
    (6500, 6760, r"H$\alpha$, [N II], [S II]", "em",
     [(6548.0, "[N II] 6548"), (6562.8, r"H$\alpha$"),
      (6583.4, "[N II] 6583"), (6716.4, "[S II]"), (6730.8, "")]),
]


def continuum(w):
    c = 2.35 * (w / 5500.0) ** -1.05
    c *= 0.58 + 0.42 / (1 + np.exp(-(w - 4060.0) / 95.0))     # 4000 A break
    c *= 1 + 0.030 * np.sin((w - 3800) / 640.0)
    return c


def spectrum(w):
    f = continuum(w)
    for wl, amp, sig, _ in LINES:
        f += amp * np.exp(-0.5 * ((w - wl) / sig) ** 2)
    return f


def panel_labels(pairs, dx=0.030, dy=0.011, size=10):
    """Stamp (a)(b)... with an identical *figure-space* offset everywhere."""
    for ax, lab in pairs:
        p = ax.get_position()
        add_panel_label(ax, lab, x=-dx / p.width, y=1.0 + dy / p.height,
                        size=size)


def small_frame(ax):
    ax.set_facecolor("#f6f7f9")
    ax.tick_params(labelsize=6.4, length=2.0)
    ax.grid(True, lw=0.35, color="#e6e8ec")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(53)

    w = np.arange(WMIN, WMAX, 0.55)
    model = spectrum(w)
    noise = rng.normal(0, 0.035 + 0.012 * np.sqrt(np.abs(model)), w.size)
    flux = model + noise
    resid = flux - model

    fig = plt.figure(figsize=(10.4, 8.3))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.46, 0.22],
                          hspace=0.42, wspace=0.16, left=0.062, right=0.985,
                          top=0.885, bottom=0.072)
    ax = fig.add_subplot(gs[0, :])
    axz = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_r = fig.add_subplot(gs[2, :])

    ymax = 1.34 * flux.max()
    ax.set_xlim(WMIN, WMAX)
    ax.set_ylim(-0.15, ymax)

    # shaded zoom windows, numbered
    for k, (a, b, _, _, _) in enumerate(WINDOWS, start=1):
        ax.add_patch(Rectangle((a, -0.15), b - a, ymax + 0.15, fc=GOLD,
                               alpha=0.10, ec="none", zorder=0))
        ax.text((a + b) / 2, -0.02, "%d" % k, ha="center", va="bottom",
                fontsize=6.6, color="#8a6b1f", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.20", fc="white",
                          ec="#d8c08a", lw=0.6))

    ax.plot(w, flux, color=NAVY, lw=0.65, zorder=3)
    ax.plot(w, continuum(w), color=RUST, lw=1.2, ls=(0, (5, 2.6)), zorder=4)

    # line identifications
    ybase = 0.795 * ymax
    for wls, name, kind in GROUPS:
        wc = float(np.mean(wls))
        col = CYAN if kind == "em" else INDIGO
        top = max(spectrum(np.array(wls)).max(), continuum(wc)) + 0.10 * ymax
        for wl in wls:
            ax.plot([wl, wl], [min(top, ybase - 0.02 * ymax), ybase],
                    color=col, lw=0.6, ls=(0, (2.2, 1.8)), zorder=2)
        ax.text(wc, ybase + 0.015 * ymax, name, rotation=90, ha="center",
                va="bottom", fontsize=6.8, color=col, zorder=6)

    ax.set_ylabel(r"$F_\lambda$  ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ "
                  r"$\mathrm{\AA}^{-1}$)", fontsize=8.2, labelpad=3)
    ax.set_xlabel(r"rest-frame wavelength  $\lambda$ ($\mathrm{\AA}$)",
                  fontsize=8.4, labelpad=2)
    ax.tick_params(labelsize=7.0, length=2.8)
    ax.grid(True, lw=0.4, color="#eceef1")
    ax.set_axisbelow(True)
    ax.set_title("Full optical spectrum with line identifications",
                 fontsize=8.8, fontweight="bold", pad=4)
    ax.annotate("4000 $\\mathrm{\\AA}$ break", xy=(4055, continuum(4055)),
                xytext=(4210, 0.60 * ymax), fontsize=7.0, color=RUST,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=RUST,
                                connectionstyle="arc3,rad=-0.25", shrinkB=3))

    # ---- zoom panels --------------------------------------------------------
    for i, (a, b, title, kind, marks) in enumerate(WINDOWS):
        az = axz[i]
        lcol = CYAN if kind == "em" else INDIGO
        m = (w >= a) & (w <= b)
        az.plot(w[m], flux[m], color=NAVY, lw=0.8, zorder=3)
        az.plot(w[m], model[m], color=GOLD, lw=1.3, alpha=0.95, zorder=4)
        az.plot(w[m], continuum(w[m]), color=RUST, lw=0.9, ls=(0, (4, 2.4)),
                zorder=2)
        lo, hi = flux[m].min(), flux[m].max()
        pad = 0.46 * (hi - lo)
        az.set_ylim(lo - 0.25 * pad, hi + pad)
        az.set_xlim(a, b)
        for wl, nm in marks:
            az.axvline(wl, color=lcol, lw=0.55, ls=(0, (2, 2)), zorder=1)
            if nm:
                az.text(wl, hi + 0.96 * pad, nm, rotation=90, ha="center",
                        va="top", fontsize=6.0, color=lcol, zorder=6)
        small_frame(az)
        az.set_title("%d  |  %s" % (i + 1, title), fontsize=7.2,
                     style="italic", color="#555555", pad=2.5)
        az.set_xlabel(r"$\lambda$ ($\mathrm{\AA}$)", fontsize=7.0, labelpad=1.2)
        if i == 0:
            az.set_ylabel("$F_\\lambda$", fontsize=7.0, labelpad=2)

    # ---- residual strip -----------------------------------------------------
    rms = float(np.std(resid))
    ax_r.set_facecolor("#f4f5f7")
    ax_r.axhspan(-rms, rms, color=CYAN, alpha=0.16, lw=0, zorder=0)
    ax_r.axhline(0, color=GREY, lw=0.7, zorder=1)
    ax_r.plot(w, resid, color=INDIGO, lw=0.45, alpha=0.85, zorder=2)
    ax_r.set_xlim(WMIN, WMAX)
    ax_r.set_ylim(-5.2 * rms, 5.2 * rms)
    ax_r.set_xlabel(r"rest-frame wavelength  $\lambda$ ($\mathrm{\AA}$)",
                    fontsize=8.0, labelpad=2)
    ax_r.set_ylabel("data $-$ model", fontsize=7.2, labelpad=2)
    ax_r.tick_params(labelsize=6.6, length=2.2)
    ax_r.grid(True, axis="y", lw=0.4, color="#e3e5e8")
    ax_r.set_axisbelow(True)
    for s in ax_r.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#9aa0a6")
    ax_r.set_title("Supporting statistic  |  residuals of panel (a),  "
                   "rms $= %.3f$ in the same flux units" % rms,
                   fontsize=7.2, style="italic", color="#555555", pad=2.5)

    panel_labels([(ax, "(a)"), (axz[0], "(b)"), (axz[1], "(c)"),
                  (axz[2], "(d)"), (ax_r, "(e)")])

    handles = [Line2D([], [], color=NAVY, lw=1.0, label="observed spectrum"),
               Line2D([], [], color=GOLD, lw=1.4, label="best-fit model"),
               Line2D([], [], color=RUST, lw=1.2, ls=(0, (5, 2.6)),
                      label="fitted continuum"),
               Line2D([], [], color=CYAN, lw=0.8, ls=(0, (2.2, 1.8)),
                      label="emission line"),
               Line2D([], [], color=INDIGO, lw=0.8, ls=(0, (2.2, 1.8)),
                      label="absorption line")]
    savefig(fig, "236_astro_spectrum.png")


if __name__ == "__main__":
    main()
