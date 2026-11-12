"""XPS core-level deconvolution: 2x2 panels with residual strips.

Each panel shows the raw spectrum (open circles), a Shirley inelastic
background (dashed), the chemically distinct components as filled
pseudo-Voigt peaks, the fitted envelope (red) and, underneath, the fit
residual.  The component sum plus background reproduces the envelope exactly,
so the residual contains only the synthetic counting noise.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, add_panel_label

COMP_COLORS = ["#4c78a8", "#5aa17f", "#dda15e", "#a86b8c", "#7f6bb0"]
C_ENV = "#b23a48"
C_DATA = "#3a3a3a"
C_BKG = "#8a8a8a"

# (centre eV, FWHM eV, amplitude, assignment)
PANELS = [
    dict(core="C 1s", lo=281.0, hi=294.0, scale=62.0, base=0.10, step=0.16,
         note="sp$^2$/sp$^3$ = 2.7", ticks=(282, 294, 2),
         comps=[(284.60, 1.00, 1.00, "sp$^2$ C=C"),
                (285.45, 1.25, 0.30, "sp$^3$ C–C"),
                (286.35, 1.35, 0.22, "C–N / C–O"),
                (288.05, 1.55, 0.11, "C=O"),
                (289.45, 1.75, 0.07, "O–C=O")]),
    dict(core="N 1s", lo=395.5, hi=406.5, scale=13.5, base=0.14, step=0.13,
         note="pyridinic N = 39 at.%", ticks=(396, 406, 2),
         comps=[(398.45, 1.35, 1.00, "pyridinic N"),
                (400.05, 1.45, 0.78, "pyrrolic N"),
                (401.30, 1.55, 0.46, "graphitic N"),
                (403.15, 1.90, 0.15, "oxidised N")]),
    dict(core="O 1s", lo=526.5, hi=537.5, scale=88.0, base=0.09, step=0.15,
         note="O$_V$/O$_L$ = 0.82", ticks=(527, 537, 2),
         comps=[(529.90, 1.10, 1.00, "lattice O (Ti–O)"),
                (531.30, 1.45, 0.62, "O vacancy"),
                (532.25, 1.55, 0.30, "–OH / C=O"),
                (533.45, 1.80, 0.11, "adsorbed H$_2$O")]),
    dict(core="Ti 2p", lo=452.5, hi=470.5, scale=118.0, base=0.08, step=0.14,
         note="Ti$^{3+}$/Ti$^{4+}$ = 0.22", ticks=(454, 470, 4),
         so=(458.65, 464.35, "$\\Delta_{SO}$ = 5.7 eV"),
         comps=[(458.65, 1.10, 1.00, "Ti$^{4+}$ 2p$_{3/2}$"),
                (464.35, 2.05, 0.27, "Ti$^{4+}$ 2p$_{1/2}$"),
                (457.35, 1.45, 0.17, "Ti$^{3+}$ 2p$_{3/2}$"),
                (463.05, 2.35, 0.05, "Ti$^{3+}$ 2p$_{1/2}$")]),
]


def pseudo_voigt(x, x0, fwhm, eta=0.30):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - x0) / (0.5 * fwhm)) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


def shirley(x, peaks, base, step):
    """Shirley-type background: the step builds up on the high-BE side."""
    dx = np.gradient(x)
    cum = np.cumsum(peaks * dx)
    tot = cum[-1] if cum[-1] > 0 else 1.0
    return base + step * cum / tot


def draw_panel(fig, sub, spec, rng, label):
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=sub,
                                    height_ratios=[4.3, 1.0], hspace=0.06)
    ax = fig.add_subplot(inner[0])
    axr = fig.add_subplot(inner[1], sharex=ax)

    xf = np.linspace(spec["lo"], spec["hi"], 1400)
    xd = np.arange(spec["lo"], spec["hi"] + 1e-9, 0.10)
    sc = spec["scale"]

    def build(x):
        comps = [a * pseudo_voigt(x, c, w) for c, w, a, _ in spec["comps"]]
        total = np.sum(comps, axis=0)
        bkg = shirley(x, total, spec["base"], spec["step"])
        return [c * sc for c in comps], bkg * sc, (total + bkg) * sc

    comps_f, bkg_f, env_f = build(xf)
    _, _, env_d = build(xd)
    noise = rng.normal(0.0, 1.0, xd.size) * 0.011 * np.sqrt(sc * env_d)
    data = env_d + noise

    for k, ((c, w, a, lab), yc) in enumerate(zip(spec["comps"], comps_f)):
        col = COMP_COLORS[k % len(COMP_COLORS)]
        ax.fill_between(xf, bkg_f, bkg_f + yc, color=col, alpha=0.30, lw=0,
                        zorder=3)
        ax.plot(xf, bkg_f + yc, color=col, lw=0.7, zorder=4)

    ax.plot(xd, data, ls="none", marker="o", ms=2.0, mfc="none",
            mec=C_DATA, mew=0.45, zorder=6)
    ax.plot(xf, env_f, color=C_ENV, lw=1.25, zorder=7)
    ax.plot(xf, bkg_f, color=C_BKG, lw=0.85, ls=(0, (4, 2.5)), zorder=8)

    ymax = env_f.max()
    ax.set_xlim(spec["hi"], spec["lo"])
    ax.set_ylim(0.0, ymax * 1.52)
    t0, t1, dt = spec["ticks"]
    ax.set_xticks(np.arange(t0, t1 + 0.1, dt))
    ax.tick_params(axis="x", labelbottom=False, length=2.5)
    ax.set_ylabel("Intensity (10$^3$ CPS)", fontsize=7.5, labelpad=2)
    ax.tick_params(axis="y", labelsize=6.5)

    handles = [Patch(facecolor=COMP_COLORS[k % len(COMP_COLORS)], alpha=0.55,
                     edgecolor=COMP_COLORS[k % len(COMP_COLORS)],
                     linewidth=0.6,
                     label=f"{lab}  {c:.1f} eV")
               for k, (c, w, a, lab) in enumerate(spec["comps"])]
    ax.legend(handles=handles, loc="upper left", fontsize=6.0,
              handlelength=1.0, handleheight=0.8, handletextpad=0.4,
              labelspacing=0.28, borderpad=0.3, borderaxespad=0.35,
              frameon=False)

    ax.text(0.975, 0.94, spec["core"], transform=ax.transAxes, ha="right",
            va="top", fontsize=9, fontweight="bold", color="#1a1a1a")
    ax.text(0.975, 0.845, spec["note"], transform=ax.transAxes, ha="right",
            va="top", fontsize=6.4, color="#4d4d4d")
    add_panel_label(ax, label, x=-0.155, y=1.01, size=10)

    if spec.get("so"):
        e1, e2, txt = spec["so"]
        ya = ymax * 1.20
        ax.annotate("", xy=(e1, ya), xytext=(e2, ya),
                    arrowprops=dict(arrowstyle="<|-|>", color="#4d4d4d",
                                    lw=0.7, mutation_scale=6))
        ax.text(0.5 * (e1 + e2), ya * 1.02, txt, ha="center", va="bottom",
                fontsize=6.2, color="#4d4d4d")

    res = data - env_d
    axr.axhline(0.0, color="#9a9a9a", lw=0.6)
    axr.plot(xd, res, color="#5b6b7a", lw=0.6)
    lim = max(np.abs(res).max() * 1.35, 1e-6)
    axr.set_ylim(-lim, lim)
    axr.set_yticks([0.0])
    axr.set_yticklabels(["0"], fontsize=6)
    axr.set_ylabel("res.", fontsize=6.2, labelpad=2)
    axr.set_xlabel("Binding energy (eV)", fontsize=7.8, labelpad=1.5)
    axr.tick_params(axis="x", labelsize=6.5, length=2.5)
    axr.spines["top"].set_visible(False)
    axr.spines["right"].set_visible(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def main():
    set_pub_style(base_font=9)
    rng = np.random.default_rng(3)

    fig = plt.figure(figsize=(7.4, 5.85))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24,
                          left=0.075, right=0.985, top=0.895, bottom=0.115)
    for i, spec in enumerate(PANELS):
        draw_panel(fig, gs[i // 2, i % 2], spec, rng, f"({'abcd'[i]})")

    shared = [
        Line2D([], [], ls="none", marker="o", ms=3.0, mfc="none", mec=C_DATA,
               mew=0.6, label="raw data"),
        Line2D([], [], color=C_ENV, lw=1.4, label="fitted envelope"),
        Line2D([], [], color=C_BKG, lw=0.9, ls=(0, (4, 2.5)),
               label="Shirley background"),
        Line2D([], [], color="#5b6b7a", lw=0.9, label="fit residual"),
    ]
    fig.legend(handles=shared, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.53, 0.005),
               handletextpad=0.5, columnspacing=1.8)
    savefig(fig, "121_xps_deconvolution.png")


if __name__ == "__main__":
    main()
