"""Flow duration curves for four gauges with a subordinate summary panel.

Daily discharge records are ranked into exceedance-probability curves on a
semi-logarithmic canvas. Light bands separate the high-flow, mid-range and
low-flow segments; Q10, Q50 and Q90 are dimensioned on the reference gauge.
A deliberately smaller side panel (58 % of the main panel height, set on a grey
card) compares the same three characteristic flows across all four gauges.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pubstyle import set_pub_style, savefig, style_axes

C_DEEP = "#17384f"
C_MAIN = "#2c7a8c"
C_LIGHT = "#8fbdd0"
C_ACC = "#c0603f"
C_WARM = "#d9a441"

GAUGES = [
    dict(name="Upper Larch (natural)", color=C_DEEP, mu=3.05, sig=0.95,
         floor=1.8, ls="-"),
    dict(name="Mill Creek (natural)", color=C_MAIN, mu=2.62, sig=1.22,
         floor=0.55, ls="-"),
    dict(name="Weston (urbanised)", color=C_WARM, mu=2.30, sig=1.55,
         floor=0.16, ls="-"),
    dict(name="Ashby (reservoir-regulated)", color=C_LIGHT, mu=2.78,
         sig=0.48, floor=6.5, ls="-"),
]


def panel_tag(fig, ax, text, dx=-0.058, dy=0.012, size=10.5):
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, text, fontsize=size,
             fontweight="bold", ha="left", va="bottom", color="#1a1a1a")


def fdc(p, g, rng):
    """Discharge exceeded p per cent of the time (log-normal + noise)."""
    from scipy.stats import norm
    z = norm.ppf(1.0 - np.clip(p, 0.05, 99.95) / 100.0)
    q = np.exp(g["mu"] + g["sig"] * z) + g["floor"]
    wob = 1.0 + 0.018 * np.sin(p / 7.0) + 0.010 * rng.standard_normal(p.size)
    return q * np.maximum(wob, 0.9)


def main():
    set_pub_style()
    rng = np.random.default_rng(11)
    p = np.linspace(0.1, 99.9, 500)

    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_axes([0.068, 0.125, 0.545, 0.700])
    ax_s = fig.add_axes([0.762, 0.258, 0.205, 0.406])
    ax.set_zorder(2)
    ax_s.set_zorder(2)
    ax.patch.set_alpha(0.0)
    ax_s.patch.set_alpha(0.0)

    # ---- flow-regime bands ------------------------------------------------
    for lo, hi, lab in ((0, 10, "high flows"), (10, 90, "mid-range flows"),
                        (90, 100, "low flows")):
        if lab != "mid-range flows":
            ax.axvspan(lo, hi, color="#eef3f6", lw=0, zorder=0)
        ax.text((lo + hi) / 2.0, 1.028, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7.2, color="#6b6b6b")

    curves = {}
    for g in GAUGES:
        q = fdc(p, g, rng)
        curves[g["name"]] = q
        ax.plot(p, q, color=g["color"], lw=1.8, ls=g["ls"], label=g["name"],
                zorder=4, solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.7, 2000)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("Probability of exceedance (%)")
    ax.set_ylabel("Daily mean discharge (m$^{3}$ s$^{-1}$)")
    ax.grid(which="major", color="#e6e6e6", lw=0.55, zorder=1)
    ax.grid(which="minor", axis="y", color="#f2f2f2", lw=0.4, zorder=1)
    ax.set_axisbelow(True)
    style_axes(ax)

    # ---- Q10 / Q50 / Q90 on the reference gauge ---------------------------
    ref = GAUGES[1]
    qref = curves[ref["name"]]
    marks = ((10, "$Q_{10}$", 1.8, 1.32, "left", "bottom"),
             (50, "$Q_{50}$", 1.8, 0.62, "left", "top"),
             (90, "$Q_{90}$", -1.8, 1.32, "right", "bottom"))
    for pv, tag, dx, fy, ha, va in marks:
        qv = float(np.interp(pv, p, qref))
        ax.plot([pv, pv], [ax.get_ylim()[0], qv], color=C_ACC, lw=0.75,
                ls=(0, (3, 2)), zorder=3)
        ax.plot([pv], [qv], marker="o", ms=4.6, mfc="white", mec=C_ACC,
                mew=1.4, zorder=6)
        ax.text(pv + dx, qv * fy, f"{tag} = {qv:.0f}", fontsize=7.6,
                color=C_ACC, ha=ha, va=va, zorder=7,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.82))
    ax.text(0.135, 0.045,
            f"markers: {ref['name'].split(' (')[0]} gauge",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            color=C_ACC)

    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995),
                    fontsize=7.8, handlelength=1.9, labelspacing=0.45,
                    title="Gauging station")
    leg.get_title().set_fontsize(7.8)
    leg.get_title().set_fontweight("bold")
    leg.set_clip_on(False)

    # ---- subordinate summary card -----------------------------------------
    box = ax_s.get_position()
    card = FancyBboxPatch(
        (box.x0 - 0.088, box.y0 - 0.078), box.width + 0.108,
        box.height + 0.180, transform=fig.transFigure,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor="#f2f5f6", edgecolor="#dde3e6", linewidth=0.7, zorder=-5)
    fig.patches.append(card)

    ypos = np.arange(len(GAUGES))[::-1]
    for g, y in zip(GAUGES, ypos):
        q = curves[g["name"]]
        q10, q50, q90 = [float(np.interp(v, p, q)) for v in (10, 50, 90)]
        ax_s.plot([q90, q10], [y, y], color=g["color"], lw=2.2, alpha=0.45,
                  solid_capstyle="round", zorder=2)
        ax_s.plot([q90, q10], [y, y], ls="none", marker="|", ms=6,
                  mec=g["color"], mew=1.4, zorder=3)
        ax_s.plot([q50], [y], marker="o", ms=5.0, mfc="white",
                  mec=g["color"], mew=1.5, zorder=4)
        ax_s.text(q10 * 1.55, y, f"{q10 / q90:.0f}$\\times$", fontsize=7.0,
                  color="#5a5a5a", ha="left", va="center")

    ax_s.set_xscale("log")
    ax_s.set_xlim(0.5, 1600)
    ax_s.set_ylim(-0.75, len(GAUGES) - 0.25)
    ax_s.set_yticks(ypos)
    ax_s.set_yticklabels([g["name"].split(" (")[0] for g in GAUGES],
                         fontsize=7.2)
    ax_s.set_xticks([1, 10, 100, 1000])
    ax_s.set_xticklabels(["1", "10", "100", "1000"], fontsize=7.0)
    ax_s.set_xlabel("Discharge (m$^{3}$ s$^{-1}$), log scale", fontsize=7.4,
                    labelpad=2.0)
    ax_s.tick_params(length=2.4)
    ax_s.grid(axis="x", color="#e2e6e8", lw=0.5)
    ax_s.set_axisbelow(True)
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)

    ax_s.text(0.0, 1.16, "Flow variability by gauge",
              transform=ax_s.transAxes, fontsize=8.0, fontweight="bold",
              ha="left", va="bottom", color="#1a1a1a")
    ax_s.text(0.0, 1.045,
              "bar spans $Q_{90}$–$Q_{10}$, circle marks $Q_{50}$; "
              "label = $Q_{10}/Q_{90}$",
              transform=ax_s.transAxes, fontsize=6.9, ha="left", va="bottom",
              color="#6b6b6b")


    panel_tag(fig, ax, "(a)")
    panel_tag(fig, ax_s, "(b)", dx=-0.062, dy=0.098)
    savefig(fig, "214_flow_duration.png")


if __name__ == "__main__":
    main()
