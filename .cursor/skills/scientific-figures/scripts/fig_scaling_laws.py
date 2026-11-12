"""Neural scaling laws: loss against parameters, tokens and compute.

Three log-log panels, one power law per model family.  Markers are the
"measured" runs, the solid line is the least-squares fit on log10 data and
the dashed continuation is the extrapolation into the shaded region beyond
the largest run.  Fitted exponents are printed next to each curve.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

OURS = "#3b5b92"
GREEN = "#2e7d5b"
GREY = "#8c8c8c"
GRID = "#eeeeee"

FAMILIES = ["Dense baseline", "MoE-16E", "Ours"]
FCOL = {"Dense baseline": GREY, "MoE-16E": GREEN, "Ours": OURS}
FLW = {"Dense baseline": 1.3, "MoE-16E": 1.3, "Ours": 2.0}
FMK = {"Dense baseline": "s", "MoE-16E": "^", "Ours": "o"}

# panel -> (symbol, unit label, pivot, x range measured, x range shown)
PANELS = [
    dict(sym="N", xlabel="Model parameters  $N$", unit="params",
         pivot=1e9, xmeas=(3e7, 2e10), xmax=4e11,
         law={"Dense baseline": (2.02, 0.068),
              "MoE-16E": (1.90, 0.075),
              "Ours": (1.80, 0.084)},
         ticks=[1e8, 1e9, 1e10, 1e11],
         tlab=[r"$10^{8}$", r"$10^{9}$", r"$10^{10}$", r"$10^{11}$"]),
    dict(sym="D", xlabel="Training tokens  $D$", unit="tokens",
         pivot=1e10, xmeas=(2e8, 1.5e11), xmax=3e12,
         law={"Dense baseline": (2.06, 0.086),
              "MoE-16E": (1.94, 0.092),
              "Ours": (1.83, 0.098)},
         ticks=[1e9, 1e10, 1e11, 1e12],
         tlab=[r"$10^{9}$", r"$10^{10}$", r"$10^{11}$", r"$10^{12}$"]),
    dict(sym="C", xlabel="Compute  $C$ (FLOPs)", unit="FLOPs",
         pivot=1e19, xmeas=(2e16, 8e19), xmax=5e21,
         law={"Dense baseline": (2.00, 0.045),
              "MoE-16E": (1.89, 0.049),
              "Ours": (1.79, 0.054)},
         ticks=[1e17, 1e19, 1e21],
         tlab=[r"$10^{17}$", r"$10^{19}$", r"$10^{21}$"]),
]

NPT = 7


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(4041)

    fig = plt.figure(figsize=(7.4, 3.05))
    gs = fig.add_gridspec(1, 3, left=0.062, right=0.988, top=0.795,
                          bottom=0.155, wspace=0.235)

    ylo, yhi = 0.95, 3.45
    for k, P in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_axisbelow(True)
        ax.grid(which="major", color=GRID, lw=0.7, zorder=0)
        ax.grid(which="minor", color="#f6f6f6", lw=0.5, zorder=0)

        x0, x1 = P["xmeas"]
        ax.axvspan(x1, P["xmax"], color="#f2f2f2", lw=0, zorder=0)
        ax.text(np.sqrt(x1 * P["xmax"]), yhi * 0.955, "extrapolated",
                ha="center", va="top", fontsize=6.3, color="#9a9a9a",
                style="italic")

        xm = np.logspace(np.log10(x0), np.log10(x1), NPT)
        xf = np.logspace(np.log10(x0), np.log10(x1), 60)
        xe = np.logspace(np.log10(x1), np.log10(P["xmax"]), 40)

        expo = {}
        for f in FAMILIES:
            a, alpha = P["law"][f]
            hot = f == "Ours"
            y = a * (xm / P["pivot"]) ** (-alpha)
            y = y * (1.0 + rng.normal(0, 0.011, NPT))
            b, c = np.polyfit(np.log10(xm), np.log10(y), 1)
            expo[f] = -b
            fit = lambda z: 10 ** (c + b * np.log10(z))

            ax.plot(xf, fit(xf), color=FCOL[f], lw=FLW[f],
                    zorder=6 if hot else 4)
            ax.plot(xe, fit(xe), color=FCOL[f], lw=FLW[f],
                    ls=(0, (3.4, 1.8)), zorder=6 if hot else 4)
            ax.plot(xm, y, ls="none", marker=FMK[f], ms=3.6 if hot else 3.0,
                    mfc="white", mec=FCOL[f], mew=1.15 if hot else 0.9,
                    zorder=7 if hot else 5)
            print(f"  panel {P['sym']:1s} {f:15s} fitted {-b:.4f} "
                  f"(target {alpha:.4f})")

        # fitted power laws, stacked in the empty lower-left corner
        s = P["sym"]
        for j, f in enumerate(FAMILIES[::-1]):
            hot = f == "Ours"
            ax.text(0.035, 0.035 + 0.088 * j,
                    rf"$L \propto {s}^{{-{expo[f]:.3f}}}$",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7.6 if hot else 6.9, color=FCOL[f],
                    fontweight="bold" if hot else "normal", zorder=9)

        ax.set_xlim(x0 * 0.55, P["xmax"])
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(P["ticks"])
        ax.set_xticklabels(P["tlab"])
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                               numticks=40))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_yticklabels(["1.0", "1.5", "2.0", "2.5", "3.0"])
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(11, 34) * 0.1,
                               numticks=40))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(which="minor", length=1.5)
        ax.set_xlabel(P["xlabel"], fontsize=8)
        if k == 0:
            ax.set_ylabel("Test cross-entropy  $L$ (nats/token)", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"$L$ vs. {P['unit']}", fontsize=8.5, pad=3.5)
        add_panel_label(ax, f"({'abc'[k]})", x=-0.055 if k else -0.155,
                        y=1.03, size=9.5)
        style_axes(ax)

    handles = [Line2D([], [], color=FCOL[f], lw=FLW[f], marker=FMK[f],
                      mfc="white", mec=FCOL[f], mew=1.0, ms=3.6, label=f)
               for f in FAMILIES]
    handles += [
        Line2D([], [], color="#4d4d4d", lw=1.1, ls=(0, (3.4, 1.8)),
               label="extrapolation beyond largest run"),
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.995),
                     ncol=min(len(handles), 5), frameon=False)
    leg.get_texts()[2].set_fontweight("bold")

    savefig(fig, "304_scaling_laws.png")


if __name__ == "__main__":
    main()
