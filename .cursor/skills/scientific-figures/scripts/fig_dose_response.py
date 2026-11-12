"""Four-parameter Hill dose-response curves with fitted IC50 comparison."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

COMPOUNDS = [
    # label, true IC50 (nM), Hill slope, plateau (% of control)
    ("Compound A", 12.0, 1.35, 3.0, "#16324f"),
    ("Compound B", 85.0, 1.05, 7.0, "#3d7ab8"),
    ("Compound C", 430.0, 1.60, 5.0, "#6a51a3"),
    ("Compound D", 2100.0, 0.90, 12.0, "#b23a48"),
]
CONC = np.array([1., 3., 10., 30., 100., 300., 1e3, 3e3, 1e4, 3e4, 1e5])
N_REP = 3


def hill(x, log_ic50, slope, bottom, top):
    return bottom + (top - bottom) / (1.0 + 10 ** ((np.log10(x) - log_ic50) * slope))


def main():
    set_pub_style()
    rng = np.random.default_rng(11)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(7.4, 4.2), gridspec_kw={"width_ratios": [2.35, 1.0],
                                               "wspace": 0.42})
    fig.subplots_adjust(left=0.085, right=0.975, top=0.845, bottom=0.165)

    fits = []
    for label, ic50, slope, bottom, color in COMPOUNDS:
        truth = hill(CONC, np.log10(ic50), slope, bottom, 100.0)
        reps = truth[None, :] + rng.normal(0, 3.4, size=(N_REP, CONC.size))
        reps += rng.normal(0, 1.6, size=(N_REP, 1))          # plate effect
        mean = reps.mean(axis=0)
        sem = reps.std(axis=0, ddof=1) / np.sqrt(N_REP)

        p0 = [np.log10(ic50), 1.0, mean.min(), mean.max()]
        popt, pcov = curve_fit(hill, CONC, mean, p0=p0, sigma=sem,
                               absolute_sigma=True, maxfev=20000)
        se = np.sqrt(np.diag(pcov))
        lo, hi = popt[0] - 1.96 * se[0], popt[0] + 1.96 * se[0]
        fits.append(dict(label=label, color=color, ic50=10 ** popt[0],
                         lo=10 ** lo, hi=10 ** hi, slope=popt[1], popt=popt,
                         mean=mean, sem=sem))

    # ---- panel A: curves ---------------------------------------------------
    grid = np.logspace(np.log10(0.6), np.log10(2.0e5), 400)
    ax.axhline(50, color="#9a9a9a", lw=0.8, ls=(0, (5, 3)), zorder=1)
    ax.text(0.72, 52, "50% inhibition", fontsize=6.2, color="#7a7a7a",
            va="bottom", ha="left")

    for f in fits:
        ax.plot([f["ic50"]] * 2, [-8, 50], lw=0.8, ls=(0, (2, 2)),
                color=f["color"], alpha=0.85, zorder=2)
        ax.plot(grid, hill(grid, *f["popt"]), lw=1.5, color=f["color"],
                zorder=4,
                label=f"{f['label']}   IC$_{{50}}$ = {f['ic50']:,.0f} nM")
        ax.errorbar(CONC, f["mean"], yerr=f["sem"], fmt="o", ms=3.4,
                    mfc="white", mec=f["color"], mew=1.0, ecolor=f["color"],
                    elinewidth=0.8, capsize=2.0, zorder=5, ls="none")
        ax.plot(f["ic50"], 50, marker="o", ms=3.6, mfc=f["color"],
                mec="white", mew=0.7, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(0.6, 2.0e5)
    ax.set_ylim(-8, 127)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.set_xlabel("Compound concentration (nM)")
    ax.set_ylabel("Cell viability (% of vehicle control)")
    ax.grid(color="#eeeeee", lw=0.6, zorder=0)
    style_axes(ax)
    # reserve the band above 105% so the legend never sits on a curve
    leg = ax.legend(loc="upper left", bbox_to_anchor=(-0.005, 1.005), ncol=2,
                    frameon=False, fontsize=6.4, handlelength=1.6,
                    labelspacing=0.4, columnspacing=1.6)
    leg._legend_box.align = "left"
    ax.set_title("Concentration\u2013response, 72 h CellTiter-Glo",
                 fontsize=8.4, loc="left", pad=4)
    add_panel_label(ax, "(a)", x=-0.115, y=1.03, size=10)

    # ---- panel B: IC50 comparison -----------------------------------------
    order = np.argsort([f["ic50"] for f in fits])[::-1]
    y = np.arange(len(order))
    base = 1.0
    for k, idx in enumerate(order):
        f = fits[idx]
        axb.barh(k, f["ic50"] - base, left=base, height=0.6, color=f["color"],
                 edgecolor="white", linewidth=0.5, zorder=3)
        axb.plot([f["lo"], f["hi"]], [k, k], lw=1.0, color="#333333", zorder=4,
                 solid_capstyle="butt")
        for b in (f["lo"], f["hi"]):
            axb.plot([b, b], [k - 0.13, k + 0.13], lw=1.0, color="#333333",
                     zorder=4)
        axb.text(f["hi"] * 1.35, k, f"{f['ic50']:,.0f}", va="center",
                 ha="left", fontsize=6.5, fontweight="bold", color="#1a1a1a")

    axb.set_xscale("log")
    axb.set_xlim(base, 1.1e5)
    axb.set_ylim(-0.65, len(order) - 0.35)
    axb.set_yticks(y)
    axb.set_yticklabels([fits[i]["label"].replace("Compound ", "Cpd ")
                         for i in order], fontsize=7.0)
    axb.set_xticks([1, 10, 100, 1000, 10000])
    axb.set_xticklabels(["10$^0$", "10$^1$", "10$^2$", "10$^3$", "10$^4$"])
    axb.set_xlabel("Fitted IC$_{50}$ (nM)")
    axb.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
    axb.tick_params(axis="y", length=0)
    style_axes(axb)
    axb.spines["left"].set_visible(False)
    axb.set_title("Potency ranking (95% CI)", fontsize=8.4, loc="left", pad=4)
    add_panel_label(axb, "(b)", x=-0.30, y=1.03, size=10)


    savefig(fig, "115_dose_response.png")


if __name__ == "__main__":
    main()
