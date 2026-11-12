"""Scaling law for video world models: prediction loss against training compute.

Three model families on log-log axes.  Markers are the runs that were actually
trained, the solid line is the least-squares power-law fit on those runs, and
the dashed continuation crosses into a hatched wedge that marks the compute
range where nothing was trained.  The inset shows the fit residuals, so the
reader can see how much the extrapolation is worth.

Synthetic data lives in ``gen_runs`` only -- swap that one function for a real
loader and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

FAMILIES = ["Dense ViT-WM", "MoE-WM (16E)", "Ours (WM-8)"]
# cool baselines (neutral grey, house blue), one warm accent for Ours
COL = {"Dense ViT-WM": "#8c8c8c", "MoE-WM (16E)": "#3b5b92",
       "Ours (WM-8)": "#b23a48"}
LW = {"Dense ViT-WM": 1.5, "MoE-WM (16E)": 1.5, "Ours (WM-8)": 2.1}
MK = {"Dense ViT-WM": "s", "MoE-WM (16E)": "^", "Ours (WM-8)": "o"}
# family -> (loss at the pivot compute, power-law exponent)
LAW = {"Dense ViT-WM": (2.15, 0.082),
       "MoE-WM (16E)": (1.92, 0.099),
       "Ours (WM-8)": (1.62, 0.120)}

PIVOT = 1e21          # FLOPs
C_LO, C_HI = 1e18, 3e21     # compute range that was actually trained
C_MAX = 1e23                # right edge of the plot
N_RUN = 8
GRID = "#ededed"


def gen_runs(seed=1507):
    """{family: (compute, loss)} for the runs that were 'trained'."""
    rng = np.random.default_rng(seed)
    c = np.logspace(np.log10(C_LO), np.log10(C_HI), N_RUN)
    out = {}
    for f in FAMILIES:
        a, alpha = LAW[f]
        y = a * (c / PIVOT) ** (-alpha)
        out[f] = (c, y * (1.0 + rng.normal(0.0, 0.022, c.size)))
    return out


def main():
    set_pub_style(base_font=8)
    runs = gen_runs()

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_axes([0.093, 0.115, 0.885, 0.775])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.grid(which="major", color=GRID, lw=0.7, zorder=0)
    ax.grid(which="minor", color="#f7f7f7", lw=0.5, zorder=0)

    ylo, yhi = 0.30, 5.4
    xlo = C_LO * 0.5

    # ---- extrapolation wedge ------------------------------------------------
    ax.axvspan(C_HI, C_MAX, facecolor="#f7f7f7", edgecolor="#dcdcdc",
               hatch="//", lw=0.0, zorder=0)
    ax.axvline(C_HI, color="#9a9a9a", lw=0.9, ls=(0, (3.2, 2.0)), zorder=2)
    ax.text(C_HI * 0.82, yhi * 0.94, "largest trained run", ha="right",
            va="top", fontsize=6.8, color="#777777")
    ax.text(np.sqrt(C_HI * C_MAX), yhi * 0.94, "extrapolation", ha="center",
            va="top", fontsize=8.4, color="#8a8a8a", style="italic")

    xf = np.logspace(np.log10(C_LO), np.log10(C_HI), 80)
    xe = np.logspace(np.log10(C_HI), np.log10(C_MAX), 60)

    fits, resid = {}, {}
    for f in FAMILIES:
        hot = f == "Ours (WM-8)"
        c, y = runs[f]
        b, k = np.polyfit(np.log10(c), np.log10(y), 1)
        fit = lambda z, b=b, k=k: 10 ** (k + b * np.log10(z))
        fits[f] = -b
        resid[f] = (c, 100.0 * (y - fit(c)) / fit(c))

        ax.plot(xf, fit(xf), color=COL[f], lw=LW[f], zorder=6 if hot else 4)
        ax.plot(xe, fit(xe), color=COL[f], lw=LW[f], ls=(0, (3.6, 1.9)),
                zorder=6 if hot else 4)
        if hot:
            # widening uncertainty fan, only where nothing was trained
            spread = 0.055 * np.log10(xe / C_HI)
            ax.fill_between(xe, fit(xe) * (1 - spread), fit(xe) * (1 + spread),
                            color=COL[f], alpha=0.16, lw=0, zorder=3)
        ax.plot(c, y, ls="none", marker=MK[f], ms=4.4 if hot else 3.8,
                mfc="white", mec=COL[f], mew=1.25 if hot else 1.0,
                zorder=7 if hot else 5)

    ax.set_xlim(xlo, C_MAX)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([1e18, 1e19, 1e20, 1e21, 1e22, 1e23])
    ax.set_xticklabels([r"$10^{18}$", r"$10^{19}$", r"$10^{20}$",
                        r"$10^{21}$", r"$10^{22}$", r"$10^{23}$"])
    ax.set_yticks([0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(["0.5", "1.0", "2.0", "4.0"])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=40))
    ax.tick_params(which="minor", length=1.6)
    ax.set_xlabel("Pre-training compute  $C$ (FLOPs)", fontsize=8.8)
    ax.set_ylabel("Video prediction loss  $L$ (nats / frame)", fontsize=8.8)
    style_axes(ax)

    handles = [Line2D([], [], color=COL[f], lw=LW[f], marker=MK[f], ms=4.2,
                      mfc="white", mec=COL[f], mew=1.1,
                      label=f"{f}   $\\alpha = {fits[f]:.3f}$")
               for f in FAMILIES]
    handles.append(Line2D([], [], color="#4d4d4d", lw=1.1, ls=(0, (3.6, 1.9)),
                          label="extrapolated fit"))
    leg = ax.legend(handles=handles, loc="upper right",
                    bbox_to_anchor=(0.995, 0.865), fontsize=8.6,
                    handlelength=2.0, labelspacing=0.55, handletextpad=0.6,
                    frameon=True, facecolor="white", edgecolor="#e5e5e5",
                    framealpha=0.93)
    leg.get_texts()[2].set_fontweight("bold")

    # ---- residual inset, parked in the empty lower-right corner -------------
    axi = ax.inset_axes([0.325, 0.055, 0.385, 0.275])
    axi.set_zorder(8)
    axi.patch.set_visible(True)
    axi.patch.set_facecolor("white")
    axi.patch.set_alpha(0.96)
    axi.set_xscale("log")
    axi.axhline(0.0, color="#b3b3b3", lw=0.8, zorder=1)
    for f in FAMILIES:
        c, r = resid[f]
        axi.plot(c, r, ls="none", marker=MK[f], ms=2.9, mfc="white",
                 mec=COL[f], mew=0.9, zorder=3)
    axi.set_xlim(C_LO * 0.6, C_HI * 1.7)
    axi.set_ylim(-6.5, 6.5)
    axi.set_yticks([-5, 0, 5])
    axi.set_xticks([1e18, 1e20, 1e21 * 3])
    axi.set_xticklabels([r"$10^{18}$", r"$10^{20}$", r"$3{\times}10^{21}$"])
    axi.xaxis.set_minor_formatter(mticker.NullFormatter())
    axi.tick_params(labelsize=6.2, length=2.0, pad=1.6, which="major")
    axi.tick_params(which="minor", length=1.0)
    axi.set_title("fit residuals (%)", fontsize=6.8, color="#555555", pad=2.4)
    for s in ("top", "right"):
        axi.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axi.spines[s].set_linewidth(0.7)

    savefig(fig, "1095_wm_quant_scaling.png")


if __name__ == "__main__":
    main()
