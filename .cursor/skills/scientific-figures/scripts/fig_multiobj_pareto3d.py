"""Three-objective Pareto fronts obtained by five multi-objective variants.

The large 3-D panel shows the non-dominated sets returned on a DTLZ2-style
problem, floating against the translucent analytical front. The middle column
projects the same sets onto the three objective planes, and the right column
compares the standard quality indicators: inverted generational distance,
hypervolume and spacing.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig

ALGOS = ["ISOA", "SOA", "PSO", "SSA", "GWO"]
COLORS = {"ISOA": "#b23a48", "SOA": "#3b5b92", "PSO": "#2e7d5b",
          "SSA": "#8e6bbf", "GWO": "#e0812f"}
MARKERS = {"ISOA": "o", "SOA": "s", "PSO": "^", "SSA": "D", "GWO": "v"}
SIZES = {"ISOA": 24, "SOA": 15, "PSO": 16, "SSA": 13, "GWO": 16}

# radial convergence error and angular scatter of every approximation set
QUALITY = {
    "ISOA": dict(rad=0.010, jit=0.012, drop=0.00),
    "SOA": dict(rad=0.034, jit=0.034, drop=0.06),
    "PSO": dict(rad=0.092, jit=0.085, drop=0.26),
    "SSA": dict(rad=0.052, jit=0.050, drop=0.14),
    "GWO": dict(rad=0.040, jit=0.038, drop=0.09),
}

METRICS = {
    "IGD": dict(better="lower", fmt="{:.4f}", vals={
        "ISOA": 0.0412, "SOA": 0.0687, "PSO": 0.1354, "SSA": 0.0913,
        "GWO": 0.0765}),
    "HV": dict(better="higher", fmt="{:.4f}", vals={
        "ISOA": 0.7412, "SOA": 0.6985, "PSO": 0.6031, "SSA": 0.6642,
        "GWO": 0.6873}),
    "Spacing": dict(better="lower", fmt="{:.4f}", vals={
        "ISOA": 0.0521, "SOA": 0.0743, "PSO": 0.1188, "SSA": 0.0902,
        "GWO": 0.0810}),
}

H = 7                                    # simplex lattice resolution


def simplex_lattice(h=H):
    """Evenly spaced reference directions on the 3-objective simplex."""
    pts = [(i / h, j / h, (h - i - j) / h)
           for i in range(h + 1) for j in range(h + 1 - i)]
    return np.array(pts)


def approximation_set(algo, seed):
    rng = np.random.default_rng(seed)
    base = simplex_lattice()
    q = QUALITY[algo]
    if q["drop"] > 0:                    # coverage gaps of the weaker methods
        w = 0.5 + 0.5 * np.cos(4.2 * base[:, 0] + 2.6 * base[:, 1])
        keep = rng.random(len(base)) > q["drop"] * (0.35 + 1.3 * w)
        base = base[keep]
    base = base + rng.normal(0, q["jit"], base.shape)
    base = np.clip(base, 1e-4, None)
    dirs = base / np.linalg.norm(base, axis=1, keepdims=True)
    radius = 1.0 + np.abs(rng.normal(0, q["rad"], (len(dirs), 1)))
    return dirs * radius


def true_front(n=44):
    a = np.linspace(0, np.pi / 2, n)
    b = np.linspace(0, np.pi / 2, n)
    A, B = np.meshgrid(a, b)
    return (np.cos(A) * np.cos(B), np.cos(A) * np.sin(B), np.sin(A))


def panel_3d(ax, sets):
    X, Y, Z = true_front()
    ax.plot_surface(X, Y, Z, color="#9fb4c7", alpha=0.16, linewidth=0,
                    antialiased=True, shade=False, zorder=1)
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color="#7b8ea1",
                      linewidth=0.35, alpha=0.55, zorder=2)
    for a in ALGOS:
        p = sets[a]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker=MARKERS[a],
                   s=SIZES[a], facecolor=COLORS[a],
                   edgecolor="#ffffff" if a == "ISOA" else "#33333355",
                   linewidths=0.5 if a == "ISOA" else 0.3,
                   alpha=0.95 if a == "ISOA" else 0.72,
                   depthshade=False, zorder=6 if a == "ISOA" else 4)

    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1.25)
    ax.set_zlim(0, 1.25)
    ax.view_init(elev=24, azim=38)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.18)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("#d5d5d5")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = "#e8e8e8"
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.set_xlabel("$f_1$", fontsize=9, labelpad=-1)
    ax.set_ylabel("$f_2$", fontsize=9, labelpad=-1)
    # the 3-D z-label is unreliable at this viewing angle, so anchor it in
    # axes coordinates instead
    ax.set_zlabel("")
    ax.text2D(0.075, 0.56, "$f_3$", transform=ax.transAxes, fontsize=9.5,
              rotation=90, ha="center", va="center", color="#1a1a1a")
    ax.set_xticks([0, 0.4, 0.8, 1.2])
    ax.set_yticks([0, 0.4, 0.8, 1.2])
    ax.set_zticks([0, 0.4, 0.8, 1.2])
    ax.tick_params(axis="x", labelsize=6.6, pad=-2.0)
    ax.tick_params(axis="y", labelsize=6.6, pad=-2.0)
    ax.tick_params(axis="z", labelsize=6.6, pad=0.6)
    ax.set_title("")
    ax.text2D(0.045, 0.965,
              "(a)   Non-dominated sets in the three-objective space",
              transform=ax.transAxes, fontsize=9.8, fontweight="bold",
              color="#1a1a1a")


def panel_proj(ax, sets, i, j, label):
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(th), np.sin(th), color="#555555", lw=1.0, ls=(0, (4, 2)),
            zorder=3, label="true front")
    for a in ALGOS:
        p = sets[a]
        ax.scatter(p[:, i], p[:, j], marker=MARKERS[a],
                   s=SIZES[a] * 0.80, facecolor=COLORS[a],
                   edgecolor="none", alpha=0.95 if a == "ISOA" else 0.68,
                   zorder=6 if a == "ISOA" else 4)
    ax.set_xlim(-0.03, 1.28)
    ax.set_ylim(-0.03, 1.28)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel(f"$f_{i + 1}$", fontsize=8, labelpad=1.0)
    ax.set_ylabel(f"$f_{j + 1}$", fontsize=8, labelpad=1.0)
    ax.tick_params(labelsize=6.6, length=2.2)
    ax.grid(True, lw=0.4, color="#f0f0f0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{label}  $f_{i + 1}$ - $f_{j + 1}$ projection",
                 fontsize=8, fontweight="bold", loc="left", pad=3)


def panel_metric(ax, name, spec, label):
    vals = spec["vals"]
    rev = spec["better"] == "higher"
    order = sorted(ALGOS, key=lambda a: vals[a], reverse=rev)
    y = np.arange(len(ALGOS))[::-1]
    top = max(vals.values())
    for yi, a in zip(y, order):
        ax.barh(yi, vals[a], height=0.62, color=COLORS[a],
                alpha=0.88 if a == "ISOA" else 0.60, edgecolor=COLORS[a],
                linewidth=1.5 if a == "ISOA" else 0.7, zorder=3)
        ax.text(vals[a] + top * 0.035, yi, spec["fmt"].format(vals[a]),
                va="center", ha="left", fontsize=6.9, color="#1a1a1a",
                fontweight="bold" if a == "ISOA" else "normal")
    ax.set_yticks(y, order, fontsize=7.2)
    for tick, a in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[a])
        if a == "ISOA":
            tick.set_fontweight("bold")
    ax.set_xlim(0, top * 1.38)
    ax.set_ylim(-0.62, len(ALGOS) - 0.38)
    arrow = "$\\downarrow$ lower is better" if not rev else \
            "$\\uparrow$ higher is better"
    ax.set_xlabel(f"{name}    {arrow}", fontsize=7.4, labelpad=1.5)
    ax.tick_params(axis="x", labelsize=6.4, length=2.2)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", lw=0.4, color="#f0f0f0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(f"{label}  {name}", fontsize=8, fontweight="bold",
                 loc="left", pad=3)


def main():
    set_pub_style()
    sets = {a: approximation_set(a, 5100 + 17 * i)
            for i, a in enumerate(ALGOS)}

    fig = plt.figure(figsize=(14.6, 8.6))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1.55, 0.82, 0.82],
                  left=0.010, right=0.972, top=0.872, bottom=0.072,
                  hspace=0.52, wspace=0.30)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    panel_3d(ax3, sets)

    projs = [(0, 1, "(b)"), (0, 2, "(c)"), (1, 2, "(d)")]
    for row, (i, j, lab) in enumerate(projs):
        panel_proj(fig.add_subplot(gs[row, 1]), sets, i, j, lab)
    for row, (name, lab) in enumerate(zip(["IGD", "HV", "Spacing"],
                                          ["(e)", "(f)", "(g)"])):
        panel_metric(fig.add_subplot(gs[row, 2]), name, METRICS[name], lab)

    handles = [Line2D([], [], marker=MARKERS[a], ls="none",
                      mfc=COLORS[a], mec=COLORS[a],
                      ms=6.5 if a == "ISOA" else 5, label=a) for a in ALGOS]
    handles.append(Line2D([], [], color="#555555", lw=1.1, ls=(0, (4, 2)),
                          label="true Pareto front"))

    savefig(fig, "151_multiobj_pareto3d.png")


if __name__ == "__main__":
    main()
