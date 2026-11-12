"""Overall equipment effectiveness dashboard: KPI rings, six-big-losses
waterfall, shift trend and an equipment ranking."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes

INK = "#333333"
NAVY = "#2c4a6e"
BLUE = "#4c78a8"
AMBER = "#c9922f"
ALERT = "#b23a48"
PALE = "#c3d3e2"
GREY = "#8b949e"
TRACK = "#e4e9ee"
CARD_BG = "#f4f6f9"

TARGET = 85.0

# label, loss in percentage points, colour, loss family
LOSSES = [
    ("Breakdowns", 6.0, ALERT),
    ("Setup & adjustment", 5.3, ALERT),
    ("Idling & minor stops", 3.9, AMBER),
    ("Reduced speed", 3.2, AMBER),
    ("Defects & rework", 1.7, BLUE),
    ("Start-up losses", 0.8, BLUE),
]
EQUIPMENT = [
    ("Press 04", 88.6), ("CNC 11", 86.9), ("Laser 02", 84.1),
    ("Mill 07", 81.3), ("Grinder 03", 79.1), ("Robot cell 1", 74.8),
    ("Lathe 09", 71.2), ("Press 02", 66.4),
]


def ring(fig, ax, cx, cy, ry, frac, colour, track=TRACK, lw=5.6):
    """Draw an aspect-corrected progress ring in axes coordinates."""
    bb = ax.get_position()
    w_in, h_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    rx = ry * h_in / w_in
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(cx + rx * np.cos(th), cy + ry * np.sin(th), color=track, lw=lw,
            solid_capstyle="butt", zorder=3)
    a = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi * frac, 240)
    ax.plot(cx + rx * np.cos(a), cy + ry * np.sin(a), color=colour, lw=lw,
            solid_capstyle="round", zorder=4)


def kpi_card(fig, ax, name, value, note, hero=False):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bg = NAVY if hero else CARD_BG
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=bg, edgecolor=NAVY if hero else "#dfe4ea",
        linewidth=0.8, zorder=1))
    ink = "white" if hero else NAVY
    ax.text(0.5, 0.925, name, ha="center", va="center", fontsize=6.8,
            fontweight="bold", color="#b9cadd" if hero else GREY, zorder=5)
    ring(fig, ax, 0.5, 0.455, 0.285, value / 100.0,
         "#f0b400" if hero else BLUE, track="#3d5c80" if hero else TRACK,
         lw=6.4 if hero else 5.6)
    ax.text(0.5, 0.465, f"{value:.1f}", ha="center", va="center",
            fontsize=15.5 if hero else 13.0, fontweight="bold", color=ink,
            zorder=5)
    ax.text(0.5, 0.285, "%", ha="center", va="center", fontsize=6.5,
            color="#b9cadd" if hero else GREY, zorder=5)
    ax.text(0.5, 0.055, note, ha="center", va="center", fontsize=6.0,
            color="#9fb5cc" if hero else GREY, zorder=5)


def waterfall(ax, oee):
    rows = [("Planned production time", 0.0, 100.0, PALE, None)]
    top = 100.0
    for name, loss, col in LOSSES:
        rows.append((name, top - loss, top, col, loss))
        top -= loss
    rows.append(("Overall OEE", 0.0, oee, NAVY, None))

    n = len(rows)
    for k, (name, x0, x1, col, loss) in enumerate(rows):
        y = n - 1 - k
        ax.barh(y, x1 - x0, left=x0, height=0.60, color=col,
                edgecolor="white", linewidth=0.6, zorder=3)
        if loss is None:
            ax.text(1.6, y, name, ha="left", va="center", fontsize=6.6,
                    fontweight="bold", color="white" if col == NAVY else NAVY,
                    zorder=5)
            ax.text(x1 + 1.6, y, f"{x1:.1f}%", ha="left", va="center",
                    fontsize=6.8, fontweight="bold", color=NAVY, zorder=5)
        else:
            ax.text(x0 - 1.6, y, name, ha="right", va="center", fontsize=6.5,
                    color=INK, zorder=5)
            ax.text(x1 + 1.6, y, f"\u2212{loss:.1f} pp", ha="left",
                    va="center", fontsize=6.5, fontweight="bold", color=col,
                    zorder=5)
    for k in range(n - 1):
        x = rows[k][1] if k else rows[k][2]
        ax.plot([x, x], [n - 1 - k - 0.30, n - 1 - k - 0.70], color="#c8ced4",
                lw=0.8, ls=(0, (2, 2)), zorder=2)

    ax.set_xlim(0, 113)
    ax.set_ylim(-1.75, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Share of planned production time (%)", fontsize=7.5)
    ax.set_title("Six big losses", fontsize=7.8, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.xaxis.grid(True, color="#eef1f4", lw=0.5)
    ax.set_axisbelow(True)

    keys = [("Availability loss", ALERT), ("Performance loss", AMBER),
            ("Quality loss", BLUE)]
    for i, (lab, col) in enumerate(keys):
        ax.plot([2.0 + i * 30], [-1.15], "s", ms=4.4, color=col, zorder=5)
        ax.text(4.6 + i * 30, -1.15, lab, ha="left", va="center",
                fontsize=6.1, color=INK, zorder=5)


def shift_trend(ax, rng):
    days = np.arange(1, 8)
    base = np.array([78.4, 79.6, 77.1, 80.8, 82.3, 81.0, 83.4])
    series = [("Shift 1", NAVY, base + 1.8), ("Shift 2", BLUE, base - 0.6),
              ("Shift 3", AMBER, base - 4.4)]
    for name, col, v in series:
        v = v + rng.normal(0, 0.8, len(days))
        ax.plot(days, v, "-", color=col, lw=1.4, zorder=4)
        ax.plot(days, v, "o", ms=3.2, mfc="white", mec=col, mew=1.0, zorder=5)
    ax.axhline(TARGET, color=ALERT, lw=0.9, ls=(0, (5, 2.5)), zorder=3)
    ax.text(7.42, TARGET, f" target {TARGET:.0f}%", ha="right", va="bottom",
            fontsize=6.2, fontweight="bold", color=ALERT, zorder=5)
    ax.set_xlim(0.6, 7.4)
    ax.set_ylim(68, 90)
    ax.set_yticks([70, 75, 80, 85, 90])
    ax.set_xticks(days)
    ax.set_xticklabels([f"D{d}" for d in days])
    ax.set_xlabel("Production day", fontsize=7.5)
    ax.set_ylabel("OEE (%)", fontsize=7.5)
    ax.set_title("OEE by shift", fontsize=7.8, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.6)
    ax.yaxis.grid(True, color="#eef1f4", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)
    handles = [Line2D([], [], color=c, marker="o", ms=3.2, mfc="white",
                      mec=c, lw=1.4, label=n) for n, c, _ in series]
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, ncol=3,
              handlelength=1.4, columnspacing=1.0, borderaxespad=0.25,
              handletextpad=0.4)


def ranking(ax, oee):
    names = [e[0] for e in EQUIPMENT][::-1]
    vals = np.array([e[1] for e in EQUIPMENT][::-1])
    cols = [NAVY if v >= TARGET else (BLUE if v >= 75 else AMBER)
            for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="white",
            linewidth=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}%", ha="left", va="center",
                fontsize=6.4, fontweight="bold", color=INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.14", fc="white", ec="none",
                          alpha=0.85))
    ax.axvline(TARGET, color=ALERT, lw=0.9, ls=(0, (5, 2.5)), zorder=4)
    ax.text(TARGET + 0.6, len(vals) - 0.30, f"target {TARGET:.0f}%",
            ha="left", va="center", fontsize=6.2, fontweight="bold",
            color=ALERT, zorder=5)
    ax.axvline(oee, color=GREY, lw=0.9, ls=(0, (2, 2)), zorder=4)
    ax.text(oee - 0.6, len(vals) - 0.30, f"line mean {oee:.1f}%", ha="right",
            va="center", fontsize=6.2, color=GREY, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, len(vals) + 0.05)
    ax.set_xlabel("Overall equipment effectiveness (%)", fontsize=7.5)
    ax.set_title("Asset ranking", fontsize=7.8, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.6)
    ax.xaxis.grid(True, color="#eef1f4", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def main():
    set_pub_style()
    rng = np.random.default_rng(773)

    a_loss = sum(l for _, l, c in LOSSES if c == ALERT)
    p_loss = sum(l for _, l, c in LOSSES if c == AMBER)
    q_loss = sum(l for _, l, c in LOSSES if c == BLUE)
    avail = 100.0 - a_loss
    perf = (avail - p_loss) / avail * 100.0
    qual = (avail - p_loss - q_loss) / (avail - p_loss) * 100.0
    oee = avail * perf * qual / 1e4

    fig = plt.figure(figsize=(7.9, 6.7))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.62, 1.0, 0.62], hspace=0.44,
                          left=0.075, right=0.982, top=0.868, bottom=0.088)
    top = gs[0].subgridspec(1, 4, wspace=0.42)
    mid = gs[1].subgridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.26)
    cards = [fig.add_subplot(top[0, i]) for i in range(4)]
    ax_w = fig.add_subplot(mid[0, 0])
    ax_t = fig.add_subplot(mid[0, 1])
    ax_r = fig.add_subplot(gs[2])

    specs = [("AVAILABILITY", avail, "target 92%", False),
             ("PERFORMANCE", perf, "target 95%", False),
             ("QUALITY", qual, "target 99%", False),
             ("OEE", oee, f"world class {TARGET:.0f}%", True)]
    for ax, (name, val, note, hero) in zip(cards, specs):
        kpi_card(fig, ax, name, val, note, hero)

    for i, sym in enumerate(("\u00d7", "\u00d7", "=")):
        b0, b1 = cards[i].get_position(), cards[i + 1].get_position()
        fig.text((b0.x1 + b1.x0) / 2, (b0.y0 + b0.y1) / 2, sym,
                 ha="center", va="center", fontsize=11, color=GREY)

    waterfall(ax_w, oee)
    shift_trend(ax_t, rng)
    ranking(ax_r, oee)

    for ax, lab, dx in ((cards[0], "(a)", 0.062), (ax_w, "(b)", 0.062),
                        (ax_t, "(c)", 0.052), (ax_r, "(d)", 0.062)):
        bb = ax.get_position()
        fig.text(bb.x0 - dx, bb.y1 + 0.012, lab, fontsize=9,
                 fontweight="bold", ha="left", va="bottom")


    savefig(fig, "252_oee_dashboard.png")


if __name__ == "__main__":
    main()
