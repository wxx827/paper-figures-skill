"""Onboarding funnel as a Sankey: who stays on the happy path and who leaves.

Six interface states are given a cohort size; everything else on the canvas
is arithmetic on that one vector.  The blue ribbon that runs along the top is
the surviving cohort, so its thickness at any point is the number of users
still in the flow, and each grey ribbon peels the difference between two
consecutive states down to a red exit stub whose height is the loss.  Because
the ribbons are top-aligned no two of them can cross, which is what makes a
hand-drawn Sankey readable.

The strip below reuses the same numbers: median dwell time for each
transition, and the exit interviews of the users in the stubs, grouped into
five reasons that sum back to the total loss.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, inch_axes, panel_labels_inches

W, H = 10.4, 6.4

BLUE = "#3b5b92"        # surviving cohort / main path
GREEN = "#2e7d5b"       # activated
RED = "#b23a48"         # loss
GREY = "#b3b3b3"        # drop-off ribbons
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

STAGES = [("Landing page", 10000), ("Sign-up form", 7400),
          ("Email verify", 5900), ("Profile setup", 4600),
          ("First key action", 3500), ("Activated", 2950)]
DWELL = [12.0, 96.0, 41.0, 138.0, 74.0]          # median seconds per step
REASONS = [("Form too long", 0.31), ("Verification mail late", 0.24),
           ("Pricing unclear", 0.18), ("Wanted more information", 0.16),
           ("Technical error", 0.11)]

Y_TOP = 2.28
Y_EXIT = 0.34
X0, DX, NODE_W = 0.62, 1.78, 0.15


def ribbon(ax, x0, x1, top0, bot0, top1, bot1, color, alpha, zorder=2):
    t = np.linspace(0.0, 1.0, 140)
    s = t * t * (3.0 - 2.0 * t)
    x = x0 + (x1 - x0) * t
    ax.fill_between(x, bot0 + (bot1 - bot0) * s, top0 + (top1 - top0) * s,
                    color=color, alpha=alpha, lw=0.0, zorder=zorder)


def draw_sankey(ax, scale):
    names = [n for n, _ in STAGES]
    users = np.array([u for _, u in STAGES], dtype=float)
    drops = users[:-1] - users[1:]

    for i, (name, u) in enumerate(zip(names, users)):
        x = X0 + i * DX
        last = i == len(users) - 1
        ax.add_patch(Rectangle((x, Y_TOP - u * scale), NODE_W, u * scale,
                               facecolor=GREEN if last else BLUE,
                               edgecolor="white", linewidth=0.7, zorder=5))
        ax.text(x + NODE_W / 2, Y_TOP + 0.34, name, ha="center", va="bottom",
                fontsize=7.4, color=INK, fontweight="bold")
        ax.text(x + NODE_W / 2, Y_TOP + 0.14,
                f"{u:,.0f} users ({100 * u / users[0]:.0f}%)", ha="center",
                va="bottom", fontsize=6.6, color=SUB)

    for i, d in enumerate(drops):
        xa = X0 + i * DX + NODE_W
        xb = X0 + (i + 1) * DX
        u_in, u_out = users[i], users[i + 1]
        ribbon(ax, xa, xb, Y_TOP, Y_TOP - u_out * scale,
               Y_TOP, Y_TOP - u_out * scale, BLUE, 0.42)
        ax.text(0.5 * (xa + xb), Y_TOP - 0.5 * u_out * scale,
                f"{100 * u_out / u_in:.0f}% continue", ha="center",
                va="center", fontsize=6.6, color="white", zorder=4)

        x_exit = xa + 0.86
        ribbon(ax, xa, x_exit, Y_TOP - u_out * scale, Y_TOP - u_in * scale,
               Y_EXIT + d * scale, Y_EXIT, GREY, 0.62, zorder=1)
        ax.add_patch(Rectangle((x_exit, Y_EXIT), NODE_W * 0.75, d * scale,
                               facecolor=RED, edgecolor="white",
                               linewidth=0.6, zorder=5))
        ax.text(x_exit + NODE_W * 0.375, Y_EXIT - 0.09,
                f"-{d:,.0f}\n({100 * d / u_in:.0f}% of step)", ha="center",
                va="top", fontsize=6.2, color=RED)

    ax.text(X0, Y_TOP + 0.62,
            "Main path (surviving cohort)", ha="left", va="bottom",
            fontsize=7.0, color=BLUE)
    ax.text(X0 + 3.05, Y_TOP + 0.62,
            "Drop-off is routed down to a red exit stub", ha="left",
            va="bottom", fontsize=7.0, color=RED)
    return users, drops


def panel_dwell(ax, users):
    steps = [f"{a}\nto {b}" for a, b in zip([n for n, _ in STAGES][:-1],
                                            [n for n, _ in STAGES][1:])]
    x = np.arange(len(DWELL))
    slow = np.array(DWELL) >= 90.0
    ax.bar(x[~slow], np.array(DWELL)[~slow], width=0.62, color=BLUE,
           edgecolor="white", linewidth=0.6)
    ax.bar(x[slow], np.array(DWELL)[slow], width=0.62, color=RED,
           edgecolor="white", linewidth=0.6)
    for xi, v in zip(x, DWELL):
        ax.text(xi, v + max(DWELL) * 0.035, f"{v:.0f} s", ha="center",
                va="bottom", fontsize=6.4, color=SUB)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=6.0, linespacing=1.25)
    ax.set_ylim(0.0, max(DWELL) * 1.28)
    ax.set_ylabel("median dwell (s)", fontsize=8.0)
    ax.set_title("Two steps dominate the time budget", fontsize=8.5,
                 fontweight="bold")
    ax.tick_params(axis="y", labelsize=6.6)
    ax.tick_params(axis="x", length=0.0)
    style_axes(ax)


def panel_reasons(ax, total_lost):
    labels = [r for r, _ in REASONS]
    share = np.array([s for _, s in REASONS])
    share = share / share.sum()
    counts = share * total_lost
    y = np.arange(len(labels))
    colors = [RED] + ["#8a8a8a"] * (len(labels) - 1)
    ax.barh(y, 100.0 * share, height=0.64, color=colors, edgecolor="white",
            linewidth=0.6)
    for yi, s, c in zip(y, share, counts):
        ax.text(100.0 * s + 0.9, yi, f"{100 * s:.0f}%  ({c:,.0f} users)",
                va="center", ha="left", fontsize=6.4, color=SUB)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 100.0 * share.max() * 1.75)
    ax.set_xlabel("share of all abandonments (%)", fontsize=8.0)
    ax.set_title(f"Exit reasons for {total_lost:,.0f} lost users",
                 fontsize=8.5, fontweight="bold")
    ax.tick_params(labelsize=6.6)
    style_axes(ax)
    return labels[int(np.argmax(share))], share.max()


def main():
    set_pub_style()
    users = np.array([u for _, u in STAGES], dtype=float)
    scale = 1.62 / users[0]

    fig = plt.figure(figsize=(W, H))
    ax_fl = inch_axes(fig, (0.10, 2.42, 10.20, 3.20), aspect=None)
    ax_dw = axes_inches(fig, (0.86, 0.66, 4.12, 1.46))
    ax_rs = axes_inches(fig, (6.30, 0.66, 3.05, 1.46))

    ax_fl.set_xlim(0.0, 10.20)
    ax_fl.set_ylim(0.0, 3.20)
    users, drops = draw_sankey(ax_fl, scale)
    panel_dwell(ax_dw, users)
    total_lost = float(drops.sum())
    worst, worst_share = panel_reasons(ax_rs, total_lost)

    names = [n for n, _ in STAGES]
    k = int(np.argmax(DWELL))

    panel_labels_inches(fig, [ax_fl], ["(a)"], dx=-0.02, dy=0.02)
    panel_labels_inches(fig, [ax_dw, ax_rs], ["(b)", "(c)"], dx=-0.62,
                        dy=0.20)
    savefig(fig, "677_hci_interaction_flow.png")


if __name__ == "__main__":
    main()
