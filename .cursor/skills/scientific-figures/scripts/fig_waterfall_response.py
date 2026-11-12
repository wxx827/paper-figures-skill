"""RECIST waterfall plot with a molecular-marker annotation track."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from pubstyle import set_pub_style, savefig, style_axes

RESP_COLOR = {"CR": "#16324f", "PR": "#4c78a8", "SD": "#b4b4b4", "PD": "#b23a48"}
RESP_LONG = {"CR": "complete response", "PR": "partial response",
             "SD": "stable disease", "PD": "progressive disease"}

PDL1_LEVELS = ["\u2265 50%", "1\u201349%", "< 1%"]
PDL1_COLOR = {"\u2265 50%": "#1f4e79", "1\u201349%": "#8fbcd9", "< 1%": "#ececec"}
DRIVER_LEVELS = ["EGFR", "KRAS", "BRAF", "None detected"]
DRIVER_COLOR = {"EGFR": "#6a51a3", "KRAS": "#d98032", "BRAF": "#2e7d5b",
                "None detected": "#e2e2e2"}


def simulate(rng, n=62):
    resp = rng.choice(["CR", "PR", "SD", "PD"], size=n,
                      p=[0.05, 0.29, 0.41, 0.25])
    change = np.empty(n)
    for i, r in enumerate(resp):
        if r == "CR":
            change[i] = -100.0
        elif r == "PR":
            change[i] = rng.uniform(-88, -32)
        elif r == "SD":
            change[i] = rng.uniform(-28, 17)
        else:
            change[i] = rng.uniform(22, 78)

    # two patients progressed on new lesions despite target-lesion shrinkage
    pd_idx = np.flatnonzero(resp == "PD")[:2]
    change[pd_idx] = rng.uniform(-24, -6, size=pd_idx.size)
    new_lesion = np.zeros(n, dtype=bool)
    new_lesion[pd_idx] = True

    # responders enriched for high PD-L1 and for EGFR/BRAF drivers
    pdl1, driver = [], []
    for i, r in enumerate(resp):
        good = r in ("CR", "PR")
        pdl1.append(rng.choice(PDL1_LEVELS,
                               p=[0.62, 0.28, 0.10] if good else
                                 [0.18, 0.34, 0.48]))
        driver.append(rng.choice(DRIVER_LEVELS,
                                 p=[0.46, 0.10, 0.16, 0.28] if good else
                                   [0.16, 0.34, 0.06, 0.44]))

    order = np.argsort(-change)
    return (change[order], resp[order], np.array(pdl1)[order],
            np.array(driver)[order], new_lesion[order])


def main():
    set_pub_style()
    rng = np.random.default_rng(7)
    change, resp, pdl1, driver, new_lesion = simulate(rng)
    n = change.size
    x = np.arange(n)

    fig, (axb, ax) = plt.subplots(
        2, 1, figsize=(7.4, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 5.4], "hspace": 0.06})
    fig.subplots_adjust(left=0.135, right=0.985, top=0.885, bottom=0.235)

    # ---- marker track ------------------------------------------------------
    for i in range(n):
        axb.add_patch(Rectangle((i - 0.42, 1.05), 0.84, 0.86,
                                facecolor=PDL1_COLOR[pdl1[i]],
                                edgecolor="white", linewidth=0.3))
        axb.add_patch(Rectangle((i - 0.42, 0.09), 0.84, 0.86,
                                facecolor=DRIVER_COLOR[driver[i]],
                                edgecolor="white", linewidth=0.3))
    axb.set_ylim(0, 2)
    axb.set_yticks([1.48, 0.52])
    axb.set_yticklabels(["PD-L1 TPS", "Driver alteration"], fontsize=6.8)
    axb.tick_params(axis="y", length=0)
    axb.tick_params(axis="x", length=0)
    for s in axb.spines.values():
        s.set_visible(False)

    # ---- waterfall ---------------------------------------------------------
    ax.axhline(0, color="#333333", lw=0.9, zorder=3)
    for thr, lab, col, xa, ha in (
            (20, "+20%  progressive disease", "#b23a48", n - 0.6, "right"),
            (-30, "\u221230%  partial response", "#2f5d8a", 0.2, "left")):
        ax.axhline(thr, color=col, lw=0.9, ls=(0, (5, 3)), zorder=2)
        ax.text(xa, thr + (2.5 if thr > 0 else -2.5), lab, ha=ha,
                va="bottom" if thr > 0 else "top", fontsize=6.4, color=col,
                fontweight="bold", zorder=5)

    ax.bar(x, change, width=0.82, color=[RESP_COLOR[r] for r in resp],
           edgecolor="white", linewidth=0.35, zorder=4)
    for i in np.flatnonzero(new_lesion):
        ax.text(i, change[i] - 3.5, "\u2020", ha="center", va="top",
                fontsize=7.0, color="#b23a48", fontweight="bold", zorder=5)

    ax.set_xlim(-0.9, n - 0.1)
    ax.set_ylim(-112, 96)
    ax.set_yticks(np.arange(-100, 101, 25))
    ax.set_xticks([])
    ax.set_ylabel("Best change from baseline in sum of\n"
                  "target-lesion diameters (%)")
    ax.set_xlabel(f"Individual patients ordered by best response  (n = {n})",
                  labelpad=6)
    ax.grid(axis="y", color="#ececec", lw=0.6, zorder=0)
    style_axes(ax)
    ax.spines["bottom"].set_visible(False)

    # ---- legends -----------------------------------------------------------
    counts = {k: int(np.sum(resp == k)) for k in RESP_COLOR}
    resp_h = [Patch(facecolor=RESP_COLOR[k], edgecolor="white",
                    label=f"{k} \u2014 {RESP_LONG[k]}  (n = {counts[k]})")
              for k in ("CR", "PR", "SD", "PD")]
    leg0 = ax.legend(handles=resp_h, title="Best overall response (RECIST 1.1)",
                     loc="upper right", frameon=True, fontsize=6.6,
                     handlelength=1.1, handleheight=0.9, labelspacing=0.38,
                     borderpad=0.5, framealpha=0.95, edgecolor="#d5d5d5")
    leg0.get_title().set_fontsize(7.0)
    leg0.get_title().set_fontweight("bold")
    leg0._legend_box.align = "left"
    ax.add_artist(leg0)
    leg0.set_clip_on(False)
    leg0.set_clip_path(None)

    h1 = [Patch(facecolor=PDL1_COLOR[k], edgecolor="#bbbbbb", linewidth=0.4,
                label=k) for k in PDL1_LEVELS]
    h2 = [Patch(facecolor=DRIVER_COLOR[k], edgecolor="#bbbbbb", linewidth=0.4,
                label=k) for k in DRIVER_LEVELS]
    leg1 = ax.legend(handles=h1, title="PD-L1 tumour proportion score",
                     loc="upper left", bbox_to_anchor=(0.0, -0.115), ncol=3,
                     frameon=False, fontsize=6.6, handlelength=1.0,
                     handleheight=0.9, columnspacing=1.0)
    leg1.get_title().set_fontsize(6.9)
    leg1.get_title().set_fontweight("bold")
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)
    leg1.set_clip_on(False)
    leg1.set_clip_path(None)

    leg2 = ax.legend(handles=h2, title="Driver alteration",
                     loc="upper left", bbox_to_anchor=(0.44, -0.115), ncol=4,
                     frameon=False, fontsize=6.6, handlelength=1.0,
                     handleheight=0.9, columnspacing=1.0)
    leg2.get_title().set_fontsize(6.9)
    leg2.get_title().set_fontweight("bold")
    leg2._legend_box.align = "left"

    orr = counts["CR"] + counts["PR"]
    dcr = orr + counts["SD"]
    ax.text(0.008, 0.045,
            f"Objective response rate  {orr}/{n} ({100 * orr / n:.0f}%)\n"
            f"Disease control rate  {dcr}/{n} ({100 * dcr / n:.0f}%)\n"
            f"\u2020 progression on new lesions despite target shrinkage",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
            linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#d5d5d5", linewidth=0.6, alpha=0.95))


    savefig(fig, "114_waterfall_response.png")


if __name__ == "__main__":
    main()
