"""Decomposition of the multi-objective UAV path-planning cost.

Left: stacked total cost per planner.  Middle: normalised radar profile.
Right: evolution of each cost term over the optimisation iterations.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

ALGOS = ["ISOA", "PSO", "GWO", "RRT*", "A*"]
ACOL = {"ISOA": "#b23a48", "PSO": "#1f6fb4", "GWO": "#2e8b57",
        "RRT*": "#7b3294", "A*": "#e07b1f"}

TERMS = ["Path length", "Altitude", "Threat exposure", "Smoothness",
         "Energy"]
TCOL = ["#3d6b8e", "#7fa8c9", "#c9a227", "#c0603f", "#7b5aa6"]

# cost components in physical units, converted to a common cost scale (-)
COMP = {                # length, altitude, threat, smooth, energy
    "ISOA": [24.5, 8.2, 11.4, 4.1, 15.8],
    "PSO":  [27.6, 11.9, 14.6, 9.4, 19.7],
    "GWO":  [28.8, 10.7, 13.1, 8.1, 20.9],
    "RRT*": [27.1, 13.4, 19.8, 17.6, 22.4],
    "A*":   [33.0, 12.2, 21.5, 12.9, 24.6],
}

RADAR = ["Path\noptimality", "Threat\nsafety", "Smoothness", "Energy\neff.",
         "Runtime", "Success\nrate"]
SCORE = {                       # 0..1, higher is better
    "ISOA": [0.94, 0.90, 0.95, 0.89, 0.72, 0.97],
    "PSO":  [0.78, 0.74, 0.72, 0.71, 0.80, 0.86],
    "GWO":  [0.75, 0.79, 0.76, 0.68, 0.77, 0.84],
    "RRT*": [0.79, 0.62, 0.44, 0.63, 0.55, 0.90],
    "A*":   [0.62, 0.58, 0.63, 0.58, 0.44, 0.99],
}


def convergence(n_iter=120, seed=3):
    """Monotone-decreasing cost traces for the three headline terms."""
    rng = np.random.default_rng(seed)
    it = np.arange(1, n_iter + 1)
    out = {}
    for name, (a0, ainf, tau) in {
            "Path length": (46.0, 24.5, 22.0),
            "Threat exposure": (38.0, 11.4, 15.0),
            "Smoothness": (21.0, 4.1, 30.0)}.items():
        base = ainf + (a0 - ainf) * np.exp(-it / tau)
        jit = np.abs(rng.normal(0, 0.35, n_iter)) * np.exp(-it / 45.0)
        y = np.minimum.accumulate(base + jit)     # enforce monotone decrease
        out[name] = y
    return it, out


def radar_axes(fig, rect):
    ax = fig.add_axes(rect, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    return ax


def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(7.4, 4.3))

    # ---- (a) stacked cost --------------------------------------------------
    axa = fig.add_axes([0.075, 0.175, 0.245, 0.70])
    xpos = np.arange(len(ALGOS))
    bottom = np.zeros(len(ALGOS))
    for k, term in enumerate(TERMS):
        vals = np.array([COMP[a][k] for a in ALGOS])
        axa.bar(xpos, vals, bottom=bottom, width=0.66, color=TCOL[k],
                edgecolor="white", linewidth=0.5, label=term, zorder=3)
        bottom += vals
    for i, a in enumerate(ALGOS):
        axa.text(i, bottom[i] + 1.6, f"{bottom[i]:.1f}", ha="center",
                 va="bottom", fontsize=6.8, zorder=4,
                 fontweight="bold" if a == "ISOA" else "normal",
                 color="#b23a48" if a == "ISOA" else "#1a1a1a")
    best = bottom.min()
    runner = float(np.sort(bottom)[1])
    axa.axhline(best, color="#b23a48", lw=0.8, ls=(0, (4, 2)), zorder=2)
    axa.text(4.42, best + 1.5, f"\u2212{100 * (runner - best) / runner:.1f} % "
             f"vs. best baseline", ha="right", va="bottom", fontsize=6.3,
             color="#a11417", zorder=5,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor="none", alpha=0.82))
    axa.set_xticks(xpos)
    axa.set_xticklabels(ALGOS, fontsize=7.5)
    axa.get_xticklabels()[0].set_fontweight("bold")
    axa.get_xticklabels()[0].set_color("#b23a48")
    axa.set_ylabel("Total weighted cost  $J_{tot}$ (\u2013)")
    axa.set_ylim(0, 118)
    axa.set_title("Cost decomposition", fontsize=8.5)
    axa.grid(axis="y", lw=0.4, alpha=0.5)
    style_axes(axa)
    axa.legend(fontsize=6.0, loc="upper left", frameon=True, framealpha=0.92,
               edgecolor="#cccccc", handlelength=1.2, labelspacing=0.25,
               borderpad=0.35).get_frame().set_linewidth(0.5)
    add_panel_label(axa, "(a)", x=-0.26, y=1.02, size=10)

    # ---- (b) radar ---------------------------------------------------------
    axb = radar_axes(fig, [0.385, 0.175, 0.245, 0.70])
    ang = np.linspace(0, 2 * np.pi, len(RADAR), endpoint=False)
    closed = np.r_[ang, ang[:1]]
    for a in ALGOS:
        v = np.array(SCORE[a] + SCORE[a][:1])
        hot = a == "ISOA"
        axb.plot(closed, v, color=ACOL[a], lw=2.0 if hot else 1.0,
                 ls="-" if hot else (0, (4, 1.6)), zorder=6 if hot else 4)
        if hot:
            axb.fill(closed, v, color=ACOL[a], alpha=0.16, zorder=3)
    axb.set_xticks(ang)
    axb.set_xticklabels(RADAR, fontsize=6.3)
    axb.tick_params(axis="x", pad=-2)
    axb.set_ylim(0, 1.0)
    axb.set_yticks([0.25, 0.5, 0.75, 1.0])
    axb.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.0,
                        color="#777777")
    axb.set_rlabel_position(30)
    axb.grid(lw=0.4, alpha=0.6)
    axb.spines["polar"].set_linewidth(0.7)
    axb.spines["polar"].set_color("#999999")
    axb.set_title("Normalised performance profile", fontsize=8.5, pad=11)
    fig.text(0.5075, 0.135, "higher is better (\u2013)", ha="center",
             fontsize=6.2, color="#666666")
    add_panel_label(axb, "(b)", x=-0.16, y=1.06, size=10)

    # ---- (c) convergence of each term -------------------------------------
    it, traces = convergence()
    names = list(traces)
    h = 0.203
    gap = 0.031
    axes_c = []
    for i, term in enumerate(names):
        y0 = 0.175 + (len(names) - 1 - i) * (h + gap)
        ax = fig.add_axes([0.735, y0, 0.245, h])
        axes_c.append(ax)
        col = TCOL[TERMS.index(term)]
        ax.plot(it, traces[term], color=col, lw=1.6, zorder=4)
        ax.axhline(traces[term][-1], color="#999999", lw=0.6,
                   ls=(0, (3, 2)), zorder=2)
        ax.fill_between(it, traces[term], traces[term][-1], color=col,
                        alpha=0.13, zorder=1)
        i95 = int(np.argmax(traces[term] <= traces[term][-1] * 1.05)) + 1
        ax.axvline(i95, color="#b23a48", lw=0.7, ls=(0, (2, 1.6)), zorder=3)
        ax.text(0.97, 0.86, term, transform=ax.transAxes, ha="right",
                va="top", fontsize=6.4, color=col, fontweight="bold")
        fx = i95 / it[-1]
        ax.text(fx + 0.02 if fx < 0.55 else fx - 0.02, 0.52,
                f"95 % @ it {i95}", transform=ax.transAxes, fontsize=6.0,
                color="#a11417", ha="left" if fx < 0.55 else "right",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85))
        ax.set_xlim(1, it[-1])
        ax.grid(lw=0.35, alpha=0.45)
        style_axes(ax)
        ax.tick_params(labelsize=6.4)
        if i < len(names) - 1:
            ax.tick_params(labelbottom=False)
    axes_c[-1].set_xlabel("ISOA iteration (\u2013)", fontsize=7.5)
    axes_c[1].set_ylabel("Cost term (\u2013)", fontsize=7.5)
    axes_c[0].set_title("Convergence of cost terms", fontsize=8.5)
    add_panel_label(axes_c[0], "(c)", x=-0.20, y=1.04, size=10)

    handles = [Line2D([], [], color=ACOL[a], lw=2.0 if a == "ISOA" else 1.0,
                      ls="-" if a == "ISOA" else (0, (4, 1.6)), label=a)
               for a in ALGOS]
    handles.append(Line2D([], [], color="#b23a48", lw=0.7, ls=(0, (2, 1.6)),
                          label="95 % convergence iteration"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=6,
                     bbox_to_anchor=(0.5, -0.012), frameon=False, fontsize=7.0,
                     handlelength=1.9, columnspacing=1.6)
    leg.get_texts()[0].set_fontweight("bold")

    tot = {a: sum(COMP[a]) for a in ALGOS}
    second = min(v for a, v in tot.items() if a != "ISOA")
    print("totals:", {a: round(v, 1) for a, v in tot.items()},
          f"  ISOA gain vs best baseline = "
          f"{100 * (second - tot['ISOA']) / second:.1f}%")
    savefig(fig, "157_uav_cost_breakdown.png")


if __name__ == "__main__":
    main()
