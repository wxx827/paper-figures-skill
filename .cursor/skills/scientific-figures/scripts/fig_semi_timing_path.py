"""Static timing analysis of one critical path: a cumulative delay waterfall
split into cell and net contributions, the arrival-versus-required comparison
that sets the slack, and the design-wide path-delay and slack histograms."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten, annotate_box
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"
NET = "#b3b3b3"
BASE = "#808080"

KIND_COLOR = {"seq": PURPLE, "logic": BLUE, "buf": lighten(BLUE, 0.38)}
KIND_NAME = {"seq": "sequential", "logic": "combinational", "buf": "buffer / inverter"}

PERIOD = 640.0        # clock period, ps
SETUP = 42.0          # capture flop setup time, ps
UNCERT = 25.0         # clock uncertainty, ps
SKEW = 18.0           # useful skew at the capture flop, ps

# instance, kind, cell delay (ps), net delay (ps), output load (fF)
STAGES = [
    ("DFF launch (CLK\u2192Q)", "seq", 63, 8, 5.4),
    ("NAND2_X2", "logic", 28, 14, 7.1),
    ("INV_X4", "buf", 19, 9, 4.6),
    ("AOI21_X2", "logic", 41, 22, 11.8),
    ("BUF_X8", "buf", 26, 31, 18.3),
    ("MUX2_X2", "logic", 47, 18, 9.7),
    ("XOR2_X1", "logic", 52, 24, 12.4),
    ("NAND3_X1", "logic", 34, 16, 8.2),
    ("BUF_X12", "buf", 24, 45, 26.5),
    ("OAI22_X2", "logic", 38, 19, 10.1),
    ("INV_X8", "buf", 17, 28, 15.9),
]


def path_totals():
    cell = sum(s[2] for s in STAGES)
    net = sum(s[3] for s in STAGES)
    arrival = cell + net
    required = PERIOD + SKEW - SETUP - UNCERT
    return cell, net, arrival, required


def population(rng, arrival, required, n=4200):
    """Endpoint delays for the whole design, the critical path included."""
    d = arrival - rng.gamma(2.3, 27.0, n)
    d = np.clip(d, 220.0, arrival)
    d = np.concatenate([[arrival], d])
    return d, required - d


def waterfall_panel(ax, arrival, required, wns, tns, nvio):
    n = len(STAGES)
    cum = 0.0
    for i, (name, kind, cell, net, load) in enumerate(STAGES):
        y = n - i
        ax.barh(y, cell, left=cum, height=0.60, color=KIND_COLOR[kind],
                edgecolor="white", linewidth=0.6, zorder=4)
        ax.barh(y, net, left=cum + cell, height=0.60, color=NET,
                edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(cum + cell + net + 5, y, f"{cell + net:.0f}", ha="left",
                va="center", fontsize=6.2, color=INK, zorder=5)
        ax.text(arrival * 1.30, y, f"{load:.1f}", ha="right", va="center",
                fontsize=6.2, color=MUTED, zorder=5)
        cum += cell + net
        if i < n - 1:
            ax.plot([cum, cum], [y - 0.30, y - 0.70], color="#9a9a9a",
                    lw=0.6, ls=(0, (1.8, 1.6)), zorder=3)
    ax.text(arrival * 1.30, n + 0.80, "load (fF)", ha="right", va="center",
            fontsize=6.3, color=SUB, fontweight="bold")

    ax.barh(0.0, arrival, height=0.60, color="#3a3a3a", edgecolor="white",
            linewidth=0.6, zorder=4)
    ax.text(arrival - 6, 0.0, f"data arrival {arrival:.0f} ps", ha="right",
            va="center", fontsize=6.5, color="white", fontweight="bold",
            zorder=5)
    ax.barh(-1.0, required, height=0.60, facecolor="white", edgecolor=BASE,
            linewidth=0.9, hatch="////", zorder=4)
    ax.text(required - 6, -1.0, f"required {required:.0f} ps", ha="right",
            va="center", fontsize=6.5, color=INK, zorder=5)

    bad = wns < 0
    col = RED if bad else GREEN
    ax.add_patch(Rectangle((min(arrival, required), -1.45),
                           abs(arrival - required), 2.0, facecolor=col,
                           alpha=0.20, edgecolor="none", zorder=2))
    ax.annotate("", xy=(arrival, -1.85), xytext=(required, -1.85),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=6)
    ax.text((arrival + required) / 2, -2.15,
            f"slack {wns:+.0f} ps", ha="center", va="top", fontsize=6.8,
            color=col, fontweight="bold")

    ax.set_yticks(list(range(1, n + 1))[::-1] + [0.0, -1.0])
    ax.set_yticklabels([s[0] for s in STAGES] + ["Data arrival",
                                                 "Required time"],
                       fontsize=6.4)
    ax.set_ylim(-2.9, n + 1.3)
    ax.set_xlim(0, arrival * 1.34)
    ax.set_xlabel("Cumulative delay along the path (ps)", fontsize=8.0)
    ax.set_title("Critical path delay waterfall", fontsize=9.0,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.8)
    ax.xaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)

    handles = [Rectangle((0, 0), 1, 1, facecolor=KIND_COLOR[k],
                         edgecolor="white", linewidth=0.6, label=KIND_NAME[k])
               for k in ("seq", "logic", "buf")]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=NET, edgecolor="white",
                             linewidth=0.6, label="net RC"))
    ax.legend(handles=handles, loc="lower right", fontsize=6.6, frameon=False,
              ncol=2, handlelength=1.1, columnspacing=1.2,
              bbox_to_anchor=(1.0, 0.06))

    annotate_box(ax, f"WNS  {wns:+.0f} ps\nTNS  {tns:.2f} ns\n"
                 f"violating endpoints  {nvio}\n"
                 f"clock {1000.0 / PERIOD:.2f} GHz, period {PERIOD:.0f} ps",
                 loc="center left", fontsize=6.6, linespacing=1.55)


def delay_hist(ax, d, arrival, required):
    bins = np.linspace(d.min(), d.max(), 46)
    counts, _e, _p = ax.hist(d, bins=bins, color=lighten(BLUE, 0.25),
                             edgecolor="white", linewidth=0.4, zorder=3)
    top = counts.max()
    ax.set_xlim(d.min() - 12, arrival + 42)
    ax.set_ylim(0, top * 1.10)
    ax.axvline(required, color=BASE, lw=1.0, ls=(0, (4, 2.5)), zorder=4,
               label=f"required time {required:.0f} ps")
    ax.axvline(arrival, color=RED, lw=1.1, zorder=5,
               label=f"critical path {arrival:.0f} ps")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.5,
              borderaxespad=0.25)
    ax.set_xlabel("Endpoint path delay (ps)", fontsize=7.5)
    ax.set_ylabel("Endpoints (count)", fontsize=7.5)
    ax.set_title(f"Path delay distribution, n = {d.size:,}", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def slack_hist(ax, s, wns, tns, nvio):
    bins = np.linspace(s.min(), s.max(), 46)
    counts, edges = np.histogram(s, bins=bins)
    mid = 0.5 * (edges[:-1] + edges[1:])
    cols = [RED if m < 0 else lighten(GREEN, 0.30) for m in mid]
    ax.bar(mid, counts, width=np.diff(edges), color=cols, edgecolor="white",
           linewidth=0.4, zorder=3)
    ax.axvline(0.0, color="#3a3a3a", lw=1.0, zorder=4)
    ax.axvline(wns, color=RED, lw=1.1, ls=(0, (3, 2.0)), zorder=5)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.text(wns + 4, counts.max() * 1.05, f"WNS {wns:+.0f} ps", ha="left",
            va="bottom", fontsize=6.4, color=RED, fontweight="bold")
    ax.text(0.0, -counts.max() * 0.055, "0", ha="center", va="top",
            fontsize=6.2, color=SUB)
    ax.set_xlabel("Setup slack (ps), negative = violating", fontsize=7.5)
    ax.set_ylabel("Endpoints (count)", fontsize=7.5)
    ax.set_title(f"Slack distribution, TNS = {tns:.2f} ns over {nvio} "
                 f"endpoints", fontsize=8.5, fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    style_axes(ax)


def main():
    set_pub_style()
    rng = np.random.default_rng(30517)
    cell, net, arrival, required = path_totals()
    d, s = population(rng, arrival, required)
    wns = float(s.min())
    tns = float(s[s < 0].sum()) / 1000.0
    nvio = int((s < 0).sum())

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.58], wspace=0.24,
                          hspace=0.52, left=0.115, right=0.985, top=0.885,
                          bottom=0.105)
    ax_wf = fig.add_subplot(gs[0, :])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[1, 1])

    waterfall_panel(ax_wf, arrival, required, wns, tns, nvio)
    delay_hist(ax_d, d, arrival, required)
    slack_hist(ax_s, s, wns, tns, nvio)

    verdict = "misses" if wns < 0 else "meets"

    panel_labels_inches(fig, [ax_wf, ax_d, ax_s], dx=-0.40, dy=0.08)
    savefig(fig, "503_semi_timing_path.png")


if __name__ == "__main__":
    main()
