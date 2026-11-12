"""Multi-touch attribution: the same conversions, five different stories.

Main panel: attributed conversions per channel under five attribution models,
grouped so all five sit on one shared y axis -- which is the whole point, since
every model is dividing the identical conversion pool and only their verdicts
differ.  The two narrow right-hand panels show where the disagreement comes
from: a thumbnail of how first touches hand over to last touches, and the
distribution of path lengths that makes single-touch models untenable.

``gen_attribution`` produces the model weights and ``gen_paths`` the journey
mix; every headline number in the figure is recomputed from those arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes, get_palette
from figkit import panel_labels_inches

RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ededed"

CHANNELS = ["Paid search", "Paid social", "Display", "Email",
            "Organic search", "Referral"]
# 0.0 = pure upper funnel, 1.0 = pure closer
FUNNEL = np.array([0.66, 0.24, 0.08, 0.88, 0.46, 0.36])
BASE = np.array([0.26, 0.16, 0.12, 0.14, 0.22, 0.10])

MODELS = ["First touch", "Last touch", "Linear", "Time decay", "Data driven"]
RECENCY = {"First touch": 0.0, "Last touch": 1.0, "Linear": 0.5,
           "Time decay": 0.80, "Data driven": 0.56}
MCOL = {"First touch": PURPLE, "Last touch": RED, "Linear": BLUE,
        "Time decay": GOLD, "Data driven": GREEN}
TOTAL = 12400.0
TILT = 2.4

PATH_STAGES = ["Social", "Display", "Organic", "Search"]
PATH_CLOSERS = ["Search", "Email", "Organic", "Direct"]


def gen_attribution(seed=640):
    """{model: attributed conversions per channel}; every model sums to TOTAL."""
    rng = np.random.default_rng(seed)
    quirk = np.exp(rng.normal(0.0, 0.13, len(CHANNELS)))   # data-driven only
    out = {}
    for m in MODELS:
        w = BASE * np.exp(TILT * (RECENCY[m] - 0.5) * (FUNNEL - 0.5) * 2.0)
        if m == "Data driven":
            w = w * quirk
        out[m] = TOTAL * w / w.sum()
    return out


def gen_paths(seed=640):
    """First-touch to last-touch flow matrix and the path-length histogram."""
    rng = np.random.default_rng(seed + 1)
    src = np.array([0.31, 0.24, 0.27, 0.18])
    aff = np.array([[0.34, 0.26, 0.18, 0.22],
                    [0.30, 0.22, 0.24, 0.24],
                    [0.26, 0.30, 0.26, 0.18],
                    [0.42, 0.20, 0.16, 0.22]])
    aff = aff * np.exp(rng.normal(0.0, 0.10, aff.shape))
    flow = src[:, None] * aff / aff.sum(axis=1, keepdims=True)
    flow = flow / flow.sum()

    lengths = np.arange(1, 9)
    share = np.exp(-0.42 * (lengths - 1)) * rng.normal(1.0, 0.05, 8)
    share[-1] += 0.045                       # the "8 or more" bucket
    share = share / share.sum()
    return flow, lengths, share


def ribbon(ax, x0, x1, y0, y1, h0, h1, color, alpha=0.42, n=64, zorder=3):
    """Smooth flow band from (x0, y0, h0) to (x1, y1, h1)."""
    t = np.linspace(0.0, 1.0, n)
    s = 0.5 - 0.5 * np.cos(np.pi * t)
    xs = x0 + (x1 - x0) * t
    lo = (y0 - h0 / 2) + ((y1 - h1 / 2) - (y0 - h0 / 2)) * s
    hi = (y0 + h0 / 2) + ((y1 + h1 / 2) - (y0 + h0 / 2)) * s
    ax.fill_between(xs, lo, hi, color=color, alpha=alpha, lw=0, zorder=zorder)


def main():
    set_pub_style(base_font=8)
    attr = gen_attribution()
    flow, lengths, share = gen_paths()

    spread = {c: (max(attr[m][i] for m in MODELS)
                  - min(attr[m][i] for m in MODELS))
              for i, c in enumerate(CHANNELS)}
    worst = max(CHANNELS, key=lambda c: spread[c] / np.mean(
        [attr[m][CHANNELS.index(c)] for m in MODELS]))
    wi = CHANNELS.index(worst)
    hi_m = max(MODELS, key=lambda m: attr[m][wi])
    lo_m = min(MODELS, key=lambda m: attr[m][wi])
    fold = attr[hi_m][wi] / attr[lo_m][wi]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          left=0.064, right=0.974, top=0.838, bottom=0.130,
                          hspace=0.46, wspace=0.215)
    ax = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    for a in (ax, ax_p):
        a.set_axisbelow(True)
        a.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        style_axes(a)

    # ---- main: grouped bars, one shared y axis ------------------------------
    xc = np.arange(len(CHANNELS))
    w = 0.155
    for k, m in enumerate(MODELS):
        off = (k - (len(MODELS) - 1) / 2.0) * w
        ax.bar(xc + off, attr[m], width=w * 0.92, color=MCOL[m],
               edgecolor="white", linewidth=0.4, zorder=4)
    ymax = max(attr[m].max() for m in MODELS)
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlim(-0.62, len(CHANNELS) - 0.38)
    ax.set_xticks(xc)
    ax.set_xticklabels(CHANNELS, fontsize=7.4)
    ax.set_xlabel("Marketing channel", fontsize=9.0)
    ax.set_ylabel("Attributed conversions (count)", fontsize=9.0)
    ax.set_title("Every model splits the same conversion pool", fontsize=9.0,
                 fontweight="bold", pad=5.0)

    # the widest disagreement, measured rather than asserted
    y_hi = attr[hi_m][wi]
    ax.annotate("", xy=(wi, y_hi * 1.045), xytext=(wi, attr[lo_m][wi]),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color=RED,
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text(wi + 0.34, (y_hi + attr[lo_m][wi]) / 2.0,
            f"{worst}: {fold:.1f}$\\times$ between\n{hi_m.lower()} and "
            f"{lo_m.lower()}", ha="left", va="center", fontsize=7.0,
            color=RED, linespacing=1.45, zorder=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#e0c3c8", linewidth=0.6, alpha=0.94))
    ax.text(0.985, 0.985,
            f"all five models allocate the same {TOTAL:,.0f} conversions;\n"
            "only the split differs, so the y axis is shared by construction",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=SUB, linespacing=1.5)

    # ---- aux (b): first touch to last touch ---------------------------------
    ax_s.axis("off")
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    src_tot = flow.sum(axis=1)
    dst_tot = flow.sum(axis=0)
    pad, top, bot = 0.030, 0.90, 0.10
    span = top - bot

    def stack(tot):
        gaps = pad * (tot.size - 1)
        hs = tot / tot.sum() * (span - gaps)
        cent, y = [], top
        for h in hs:
            cent.append(y - h / 2.0)
            y -= h + pad
        return np.array(cent), hs

    ysrc, hsrc = stack(src_tot)
    ydst, hdst = stack(dst_tot)
    ramp = get_palette("cb_sequential_blue", 7)[2:6]

    cursor_s = ysrc + hsrc / 2.0
    cursor_d = ydst + hdst / 2.0
    for i in range(4):
        for j in range(4):
            h = flow[i, j] / src_tot[i] * hsrc[i]
            hd = flow[i, j] / dst_tot[j] * hdst[j]
            ribbon(ax_s, 0.30, 0.70, cursor_s[i] - h / 2.0,
                   cursor_d[j] - hd / 2.0, h, hd, ramp[i])
            cursor_s[i] -= h
            cursor_d[j] -= hd

    for i in range(4):
        ax_s.add_patch(plt.Rectangle((0.265, ysrc[i] - hsrc[i] / 2), 0.035,
                                     hsrc[i], facecolor=ramp[i],
                                     edgecolor="none", zorder=5))
        ax_s.text(0.250, ysrc[i], f"{PATH_STAGES[i]}  {src_tot[i]:.0%}",
                  ha="right", va="center", fontsize=6.5, color=INK)
        ax_s.add_patch(plt.Rectangle((0.70, ydst[i] - hdst[i] / 2), 0.035,
                                     hdst[i], facecolor="#8a8a8a",
                                     edgecolor="none", zorder=5))
        ax_s.text(0.750, ydst[i], f"{PATH_CLOSERS[i]}  {dst_tot[i]:.0%}",
                  ha="left", va="center", fontsize=6.5, color=INK)
    ax_s.text(0.283, 0.965, "first touch", ha="center", va="bottom",
              fontsize=6.8, color=SUB, fontweight="bold")
    ax_s.text(0.718, 0.965, "last touch", ha="center", va="bottom",
              fontsize=6.8, color=SUB, fontweight="bold")
    ax_s.set_title("Journeys hand over between channels", fontsize=8.5,
                   fontweight="bold", pad=6.0)
    keep = float(sum(flow[i, j] for i, a in enumerate(PATH_STAGES)
                     for j, b in enumerate(PATH_CLOSERS) if a == b)
                 / flow.sum())
    ax_s.text(0.5, 0.012,
              f"only {keep:.0%} of conversions open and close on the "
              "same channel", ha="center", va="bottom", fontsize=6.4,
              color=MUTED)

    # ---- aux (c): path length ----------------------------------------------
    cum = np.cumsum(share)
    ax_p.bar(lengths, share * 100, width=0.68, color=BLUE, edgecolor="white",
             linewidth=0.5, zorder=4)
    ax_p.plot(lengths, cum * 100, color=RED, lw=1.6, marker="o", ms=3.4,
              mfc="white", mec=RED, mew=1.1, zorder=6)
    ax_p.set_xlim(0.4, 8.6)
    ax_p.set_ylim(0, 108)
    ax_p.set_xticks(lengths)
    ax_p.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8+"],
                         fontsize=6.8)
    ax_p.set_yticks([0, 25, 50, 75, 100])
    ax_p.tick_params(labelsize=6.8)
    ax_p.set_xlabel("touchpoints before conversion (count)", fontsize=8.0,
                    labelpad=1.5)
    ax_p.set_ylabel("share of conversions (%)", fontsize=8.0)
    ax_p.set_title("Most journeys are not single touch", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    multi = 1.0 - share[0]
    mean_len = float((lengths * share).sum())
    ax_p.axhline(50, color="#9a9a9a", lw=0.7, ls=(0, (2.4, 2.0)), zorder=3)
    ax_p.text(8.4, 56,
              f"{multi:.0%} of conversions touch\n"
              f"2 or more channels; mean\npath length {mean_len:.1f}",
              ha="right", va="bottom", fontsize=6.5, color=SUB,
              linespacing=1.5)
    ax_p.legend(handles=[
        Patch(facecolor=BLUE, label="share at this length"),
        Line2D([], [], color=RED, lw=1.6, marker="o", ms=3.4, mfc="white",
               mec=RED, label="cumulative share")],
        loc="center right", frameon=False, fontsize=6.4, handlelength=1.6,
        labelspacing=0.3, borderpad=0.25, bbox_to_anchor=(1.0, 0.30))

    panel_labels_inches(fig, [ax, ax_s, ax_p], dx=-0.44, dy=0.10)

    savefig(fig, "640_mkt_attribution.png")


if __name__ == "__main__":
    main()
