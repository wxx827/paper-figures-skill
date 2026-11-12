"""Causal DAG for an observational treatment-effect question.

The graph separates the causal machinery (X -> Y direct, X -> M -> Y) from
the non-causal backdoor paths that have to be blocked, flags the unobserved
variable with a dashed node, and marks the collider that must be left alone.
The side panel lists every backdoor path, the minimal sufficient adjustment
set and the resulting d-separation statement.

Everything is drawn with patches and FancyArrowPatch on an equal-aspect
axes; edges are trimmed to the node boundary so no arrowhead overlaps a node.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
from pubstyle import set_pub_style, savefig

CAUSAL = "#1f7a6f"
BACKDOOR = "#b23a48"
NEUTRAL = "#9aa0a6"
TREAT_FC = "#2a7f7a"
OUT_FC = "#3d5a8a"
R = 0.44

NODES = {
    "X":  dict(xy=(1.60, 2.90), label=r"$\bf{X}$", cap="treatment",
               fc=TREAT_FC, ec="#1c5c58", tc="white", ls="solid"),
    "Y":  dict(xy=(8.40, 2.90), label=r"$\bf{Y}$", cap="outcome",
               fc=OUT_FC, ec="#2b4166", tc="white", ls="solid"),
    "M":  dict(xy=(5.00, 4.05), label=r"$\bf{M}$", cap="mediator",
               fc="#dce8f2", ec="#5b7fa6", tc="#20374f", ls="solid"),
    "C":  dict(xy=(5.00, 1.15), label=r"$\bf{C}$", cap="collider",
               fc="#fbe6cd", ec="#c98b45", tc="#6b4212", ls="solid"),
    # captions of the top-row nodes sit above the circle: edges leave downward
    "Z1": dict(xy=(2.90, 6.20), label=r"$\bf{Z_1}$", cap="age",
               fc="white", ec="#5a5f66", tc="#1a1a1a", ls="solid",
               cap_above=True),
    "Z2": dict(xy=(7.10, 6.20), label=r"$\bf{Z_2}$", cap="clinic SES",
               fc="white", ec="#5a5f66", tc="#1a1a1a", ls="solid",
               cap_above=True),
    "U":  dict(xy=(10.30, 4.90), label=r"$\bf{U}$", cap="unobserved\nfrailty",
               fc="#f7f7f7", ec="#8a8f96", tc="#3a3a3a", ls="dashed",
               cap_above=True),
}

# (from, to, colour, linewidth, linestyle, curvature)
EDGES = [
    ("X", "Y", CAUSAL, 2.1, "solid", 0.0),
    ("X", "M", CAUSAL, 1.7, "solid", 0.0),
    ("M", "Y", CAUSAL, 1.7, "solid", 0.0),
    ("Z1", "X", BACKDOOR, 1.4, "solid", 0.0),
    ("Z1", "Y", BACKDOOR, 1.4, "solid", 0.0),
    ("Z2", "X", BACKDOOR, 1.4, "solid", 0.0),
    ("Z2", "Y", BACKDOOR, 1.4, "solid", 0.0),
    ("U", "Z2", BACKDOOR, 1.3, "dashed", 0.0),
    ("U", "Y", BACKDOOR, 1.3, "dashed", -0.16),
    ("X", "C", NEUTRAL, 1.2, "solid", 0.0),
    ("Y", "C", NEUTRAL, 1.2, "solid", 0.0),
]


def draw_node(ax, spec):
    x, y = spec["xy"]
    ax.add_patch(Circle((x, y), R, facecolor=spec["fc"],
                        edgecolor=spec["ec"], linewidth=1.3,
                        linestyle=spec["ls"], zorder=5))
    ax.text(x, y, spec["label"], ha="center", va="center", fontsize=10.5,
            color=spec["tc"], zorder=6)
    if spec.get("cap_above"):
        ax.text(x, y + R + 0.18, spec["cap"], ha="center", va="bottom",
                fontsize=6.4, color="#5a5f66", linespacing=1.25, zorder=6)
    else:
        ax.text(x, y - R - 0.20, spec["cap"], ha="center", va="top",
                fontsize=6.4, color="#5a5f66", linespacing=1.25, zorder=6)


def draw_edge(ax, a, b, color, lw, ls, rad):
    p0 = np.array(NODES[a]["xy"], dtype=float)
    p1 = np.array(NODES[b]["xy"], dtype=float)
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    start = p0 + u * (R + 0.02)
    end = p1 - u * (R + 0.10)
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9.5, lw=lw,
        color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=3,
        connectionstyle=f"arc3,rad={rad}", joinstyle="round"))


def main():
    set_pub_style(base_font=8)

    fig = plt.figure(figsize=(7.7, 3.95))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.60, 1.0], wspace=0.03,
                  left=0.012, right=0.988, top=0.855, bottom=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0.15, 11.35)
    ax.set_ylim(0.10, 7.10)
    ax.set_aspect("equal")
    ax.axis("off")

    for a, b, col, lw, ls, rad in EDGES:
        draw_edge(ax, a, b, col, lw, ls, rad)
    for spec in NODES.values():
        draw_node(ax, spec)

    ax.annotate("total effect\nof interest", xy=(5.0, 2.90),
                xytext=(5.0, 2.24), ha="center", va="top", fontsize=6.4,
                color=CAUSAL, zorder=7)
    ax.text(5.62, 1.15, "conditioning on $C$\nopens $X \\rightarrow C "
                        "\\leftarrow Y$", ha="left", va="center",
            fontsize=6.4, color="#a4661f", linespacing=1.35)

    handles = [
        plt.Line2D([], [], color=CAUSAL, lw=2.0, label="causal path"),
        plt.Line2D([], [], color=BACKDOOR, lw=1.4,
                   label="backdoor (confounding) path"),
        plt.Line2D([], [], color=BACKDOOR, lw=1.3, ls=(0, (4, 2)),
                   label="edge from an unobserved cause"),
        plt.Line2D([], [], color=NEUTRAL, lw=1.2, label="edge into a collider"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=False,
              handlelength=1.9, labelspacing=0.42,
              bbox_to_anchor=(-0.015, -0.02))

    # ---- side panel ---------------------------------------------------------
    axt = fig.add_subplot(gs[0, 1])
    axt.axis("off")
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.add_patch(plt.Rectangle((0.02, 0.03), 0.96, 0.94, transform=axt.transAxes,
                                facecolor="#fafbfc", edgecolor="#dcdfe3",
                                linewidth=0.7, zorder=0))

    lines = [
        ("Estimand", "head", None),
        (r"total causal effect of $X$ on $Y$", "body", "#1a1a1a"),
        ("", "gap", None),
        ("Backdoor paths", "head", None),
        (r"1.  $X \leftarrow Z_1 \rightarrow Y$", "body", BACKDOOR),
        (r"2.  $X \leftarrow Z_2 \rightarrow Y$", "body", BACKDOOR),
        (r"3.  $X \leftarrow Z_2 \leftarrow U \rightarrow Y$", "body",
         BACKDOOR),
        ("", "gap", None),
        ("Minimal sufficient adjustment set", "head", None),
        (r"$\{Z_1,\ Z_2\}$   (blocks all three paths)", "body", CAUSAL),
        ("", "gap", None),
        ("Never condition on", "head", None),
        (r"$M$  mediator: removes the indirect effect", "body", "#5a5f66"),
        (r"$C$  collider: creates spurious dependence", "body", "#5a5f66"),
        ("", "gap", None),
        ("d-separation after adjustment", "head", None),
        (r"$X \perp\!\!\!\perp Y \mid \{Z_1, Z_2\}$  in the graph", "body",
         "#1a1a1a"),
        (r"with all arrows out of $X$ deleted", "body", "#1a1a1a"),
    ]
    y = 0.935
    for txt, kind, col in lines:
        if kind == "gap":
            y -= 0.030
            continue
        if kind == "head":
            axt.text(0.065, y, txt.upper(), ha="left", va="top", fontsize=6.6,
                     color="#8a8f96", fontweight="bold")
            y -= 0.052
        else:
            axt.text(0.085, y, txt, ha="left", va="top", fontsize=7.1,
                     color=col)
            y -= 0.056


    savefig(fig, "134_causal_dag.png")


if __name__ == "__main__":
    main()
