"""Six-parameter water-quality dashboard for one river station.

Each panel carries the same furniture: a monthly mean line, the min-max envelope
of the individual samples, a dashed regulatory limit and a rust wash over the
months that breach it. Dissolved oxygen is judged against a minimum, pH against
a band and the remaining four against maxima, so all three kinds of compliance
test are shown with one visual grammar. A single legend at the foot serves all
six panels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"

MONTHS = np.arange(1, 13)
MLAB = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# name, unit, monthly means, sample spread, limit, kind, y-range, y-ticks
PARAMS = [
    dict(name="Dissolved oxygen", unit="mg L$^{-1}$",
         mean=[10.9, 10.4, 9.6, 8.7, 7.4, 6.1, 4.6, 4.2, 5.5, 7.2, 8.9, 10.2],
         sd=0.55, limit=5.0, kind="min", ylim=(2.5, 12.5),
         ticks=[4, 6, 8, 10, 12]),
    dict(name="pH", unit="",
         mean=[7.6, 7.7, 7.9, 8.1, 8.3, 8.6, 8.8, 8.7, 8.2, 7.8, 7.5, 7.4],
         sd=0.13, limit=(6.5, 8.5), kind="band", ylim=(6.2, 9.3),
         ticks=[6.5, 7.0, 7.5, 8.0, 8.5, 9.0]),
    dict(name="Turbidity", unit="NTU",
         mean=[18, 22, 41, 68, 52, 34, 26, 21, 47, 92, 45, 24],
         sd=6.5, limit=60.0, kind="max", ylim=(0, 125),
         ticks=[0, 30, 60, 90, 120]),
    dict(name="Total nitrogen", unit="mg L$^{-1}$",
         mean=[1.6, 1.8, 2.4, 3.1, 2.7, 2.1, 1.9, 1.7, 2.2, 2.9, 2.3, 1.7],
         sd=0.22, limit=2.0, kind="max", ylim=(0.8, 3.9),
         ticks=[1, 2, 3]),
    dict(name="Total phosphorus", unit="mg L$^{-1}$",
         mean=[0.09, 0.11, 0.17, 0.26, 0.21, 0.15, 0.12, 0.10, 0.19, 0.31,
               0.18, 0.10],
         sd=0.022, limit=0.20, kind="max", ylim=(0.02, 0.40),
         ticks=[0.1, 0.2, 0.3, 0.4]),
    dict(name="COD", unit="mg L$^{-1}$",
         mean=[11, 13, 17, 22, 19, 16, 14, 12, 18, 27, 20, 13],
         sd=1.9, limit=20.0, kind="max", ylim=(5, 33),
         ticks=[10, 20, 30]),
]


def panel_tag(fig, ax, text, dx=-0.046, dy=0.052, size=10.0):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def main():
    set_pub_style()
    rng = np.random.default_rng(41)

    fig = plt.figure(figsize=(9.9, 5.9))
    gs = fig.add_gridspec(2, 3, left=0.062, right=0.985, top=0.800,
                          bottom=0.145, wspace=0.235, hspace=0.470)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    for ax, p in zip(axes, PARAMS):
        mean = np.array(p["mean"], dtype=float)
        lo = mean - p["sd"] * (1.1 + 0.5 * rng.random(12))
        hi = mean + p["sd"] * (1.1 + 0.5 * rng.random(12))

        ax.fill_between(MONTHS, lo, hi, color=C_MAIN, alpha=0.16, lw=0,
                        zorder=2, label="Min\u2013max of samples")

        y0, y1 = p["ylim"]
        if p["kind"] == "band":
            blo, bhi = p["limit"]
            ax.axhspan(blo, bhi, color="#eef2f4", lw=0, zorder=1)
            for lv in (blo, bhi):
                ax.axhline(lv, color=C_ACC, lw=0.9, ls=(0, (4, 2.2)),
                           zorder=4)
            ax.fill_between(MONTHS, bhi, mean, where=mean > bhi,
                            color=C_ACC, alpha=0.30, lw=0, zorder=3,
                            interpolate=True)
            ax.fill_between(MONTHS, blo, mean, where=mean < blo,
                            color=C_ACC, alpha=0.30, lw=0, zorder=3,
                            interpolate=True)
            bad = int(np.sum((mean > bhi) | (mean < blo)))
            note = f"limit band {blo:g}\u2013{bhi:g}"
        else:
            lim = p["limit"]
            ax.axhline(lim, color=C_ACC, lw=0.9, ls=(0, (4, 2.2)), zorder=4)
            if p["kind"] == "max":
                ax.fill_between(MONTHS, lim, mean, where=mean > lim,
                                color=C_ACC, alpha=0.30, lw=0, zorder=3,
                                interpolate=True)
                bad = int(np.sum(mean > lim))
                note = f"limit {lim:g} max"
            else:
                ax.fill_between(MONTHS, lim, mean, where=mean < lim,
                                color=C_ACC, alpha=0.30, lw=0, zorder=3,
                                interpolate=True)
                bad = int(np.sum(mean < lim))
                note = f"limit {lim:g} min"

        ax.plot(MONTHS, mean, color=C_DEEP, lw=1.5, zorder=6,
                label="Monthly mean")
        ax.plot(MONTHS, mean, ls="none", marker="o", ms=2.8, mfc="white",
                mec=C_DEEP, mew=0.8, zorder=7)

        ax.set_xlim(0.4, 12.6)
        ax.set_ylim(y0, y1)
        ax.set_xticks(MONTHS)
        ax.set_xticklabels(MLAB, fontsize=7.0)
        ax.set_yticks(p["ticks"])
        ax.tick_params(labelsize=7.2, length=2.6)
        ax.grid(axis="y", color="#f0f0f0", lw=0.5)
        ax.set_axisbelow(True)
        style_axes(ax)

        unit = f" ({p['unit']})" if p["unit"] else ""
        ax.set_title(p["name"] + unit, fontsize=8.4, fontweight="bold",
                     loc="left", pad=13.0, color="#1a1a1a")
        ax.text(0.0, 1.028, note, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=6.9, color=C_ACC)
        ax.text(1.0, 1.028, f"{bad}/12 months exceed" if bad else
                "fully compliant", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6.9,
                color=C_ACC if bad else "#6b8f7a",
                fontweight="bold" if bad else "normal")

    for ax in axes[3:]:
        ax.set_xlabel("Month of 2024", fontsize=7.8, labelpad=2.5)

    handles = [
        plt.Line2D([], [], color=C_DEEP, lw=1.5, marker="o", ms=2.8,
                   mfc="white", mec=C_DEEP, mew=0.8),
        Patch(facecolor=C_MAIN, alpha=0.16, edgecolor="none"),
        plt.Line2D([], [], color=C_ACC, lw=0.9, ls=(0, (4, 2.2))),
        Patch(facecolor=C_ACC, alpha=0.30, edgecolor="none"),
    ]
    labels = ["Monthly mean", "Min\u2013max of samples",
              "Regulatory limit", "Exceedance"]
    leg = fig.legend(handles, labels, loc="lower center",
                     bbox_to_anchor=(0.52, 0.012), ncol=4, fontsize=7.8,
                     handlelength=1.9, columnspacing=2.4, handletextpad=0.55)
    leg.set_clip_on(False)


    for ax, lab in zip(axes, "abcdef"):
        panel_tag(fig, ax, f"({lab})")
    savefig(fig, "222_water_quality_panel.png")


if __name__ == "__main__":
    main()
