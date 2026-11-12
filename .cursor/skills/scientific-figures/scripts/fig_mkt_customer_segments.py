"""RFM customer segments: who they are, what they look like, what they are worth.

Main panel: every customer as a bubble on the frequency-monetary plane, sized
by recency and coloured by segment, with the segment centroid marked so the
five clouds can be read as five populations rather than one smear.  The two
narrow right-hand panels answer the two questions the scatter cannot: how the
segments differ across all five profile dimensions at once (radar), and how
badly customer count and revenue disagree (opposed bars).

Customers are drawn in ``gen_customers``; every share, centroid and profile
score below is computed from that table, so re-seeding moves the numbers.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches

GREEN = "#2e7d5b"
BLUE = "#3b5b92"
GOLD = "#c9a227"
RED = "#b23a48"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#ededed"

SEGMENTS = ["Champions", "Loyal", "Potential", "At risk", "Hibernating"]
COL = {"Champions": GREEN, "Loyal": BLUE, "Potential": GOLD,
       "At risk": RED, "Hibernating": GREY}
# segment -> (share, freq mean/sd, log-spend mean/sd, recency mean/sd days,
#             tenure months mean, gross margin mean %)
SPEC = {
    "Champions":   (0.13, 15.0, 3.4, 7.45, 0.42, 22, 12, 34, 41),
    "Loyal":       (0.24, 8.4, 2.2, 6.85, 0.44, 52, 22, 27, 36),
    "Potential":   (0.21, 4.1, 1.3, 6.20, 0.46, 34, 16, 9, 33),
    "At risk":     (0.17, 6.2, 2.0, 7.05, 0.50, 178, 46, 31, 30),
    "Hibernating": (0.25, 1.9, 0.8, 5.55, 0.52, 322, 96, 24, 26),
}
N_CUST = 1500
METRICS = ["R", "F", "M", "tenure", "margin"]


def gen_customers(seed=639):
    """Per-customer RFM table: frequency, spend, recency, tenure, margin."""
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in
            ("seg", "freq", "spend", "recency", "tenure", "margin")}
    for s in SEGMENTS:
        share, fm, fs, lm, ls, rm, rs, tm, gm = SPEC[s]
        k = int(round(N_CUST * share))
        rows["seg"] += [s] * k
        rows["freq"].append(np.maximum(rng.normal(fm, fs, k), 0.5))
        rows["spend"].append(np.exp(rng.normal(lm, ls, k)))
        rows["recency"].append(np.maximum(rng.normal(rm, rs, k), 2.0))
        rows["tenure"].append(np.maximum(rng.normal(tm, 9.0, k), 1.0))
        rows["margin"].append(np.clip(rng.normal(gm, 5.5, k), 5.0, 65.0))
    out = {k: np.concatenate(v) for k, v in rows.items() if k != "seg"}
    out["seg"] = np.array(rows["seg"])
    return out


def profile_scores(cust):
    """Segment medians min-max scaled across segments onto 0.12 - 1.0."""
    raw = {}
    for s in SEGMENTS:
        m = cust["seg"] == s
        raw[s] = np.array([
            -np.median(cust["recency"][m]),        # recent is good
            np.median(cust["freq"][m]),
            np.median(cust["spend"][m]),
            np.median(cust["tenure"][m]),
            np.median(cust["margin"][m]),
        ])
    stack = np.vstack([raw[s] for s in SEGMENTS])
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    scaled = 0.12 + 0.88 * (stack - lo) / np.maximum(hi - lo, 1e-9)
    return {s: scaled[i] for i, s in enumerate(SEGMENTS)}


def main():
    set_pub_style(base_font=8)
    cust = gen_customers()
    n = cust["freq"].size
    revenue = cust["spend"] * cust["freq"]

    share_cust = {s: float((cust["seg"] == s).mean()) for s in SEGMENTS}
    share_rev = {s: float(revenue[cust["seg"] == s].sum() / revenue.sum())
                 for s in SEGMENTS}
    scores = profile_scores(cust)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          left=0.062, right=0.972, top=0.838, bottom=0.130,
                          hspace=0.46, wspace=0.235)
    ax = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1], projection="polar")
    ax_b = fig.add_subplot(gs[1, 1])

    for a in (ax, ax_b):
        a.set_axisbelow(True)
        a.grid(color=GRID, lw=0.7, zorder=0)
        style_axes(a)

    # ---- main: frequency x monetary bubbles --------------------------------
    size = 6.0 + 62.0 * np.exp(-cust["recency"] / 150.0)
    for s in SEGMENTS:
        m = cust["seg"] == s
        ax.scatter(cust["freq"][m], cust["spend"][m], s=size[m], c=COL[s],
                   alpha=0.42, linewidths=0.3, edgecolors="white", zorder=4)
    # labels parked in the empty corners, each on a hairline leader, so no
    # callout box can drift over the y-axis label or bury its own cloud
    label_at = {"Champions": (21.0, 5400), "Loyal": (19.8, 470),
                "At risk": (3.6, 7600), "Potential": (10.6, 118),
                "Hibernating": (17.6, 88)}
    for s in SEGMENTS:
        m = cust["seg"] == s
        cx, cy = np.median(cust["freq"][m]), np.median(cust["spend"][m])
        ax.plot([cx], [cy], marker="X", ms=9.5, mfc=COL[s], mec="white",
                mew=1.3, zorder=8)
        ax.annotate(f"{s}\n{share_cust[s]:.0%} of base | "
                    f"{share_rev[s]:.0%} of revenue",
                    xy=(cx, cy), xytext=label_at[s], ha="center", va="center",
                    fontsize=6.8, color=COL[s], linespacing=1.4, zorder=9,
                    arrowprops=dict(arrowstyle="-", lw=0.7, color=COL[s],
                                    alpha=0.75, shrinkA=2, shrinkB=6),
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor=lighten(COL[s], 0.58), linewidth=0.6,
                              alpha=0.92))

    ax.set_yscale("log")
    ax.set_xlim(0, 26)
    ax.set_ylim(30, 2.2e4)
    ax.set_xlabel("Purchase frequency (orders per year)", fontsize=9.0)
    ax.set_ylabel("Average order value (USD, log scale)", fontsize=9.0)
    ax.set_title("Five RFM segments on the frequency-value plane",
                 fontsize=9.0, fontweight="bold", pad=5.0)

    # bubble-size key, built from the same recency-to-size mapping
    for k, days in enumerate([15, 90, 300]):
        ax.scatter([1.4 + 2.3 * k], [58.0],
                   s=6.0 + 62.0 * np.exp(-days / 150.0), c="#b3b3b3",
                   alpha=0.75, linewidths=0.3, edgecolors="white", zorder=6)
        ax.text(1.4 + 2.3 * k, 45.0, f"{days} d", ha="center", va="top",
                fontsize=6.2, color=MUTED)
    ax.text(0.6, 78.0, "bubble size = recency", ha="left", va="bottom",
            fontsize=6.6, color=SUB)

    # ---- aux (b): segment profile radar ------------------------------------
    ang = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    closed = np.concatenate([ang, ang[:1]])
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)
    for s in SEGMENTS:
        v = np.concatenate([scores[s], scores[s][:1]])
        ax_r.plot(closed, v, color=COL[s], lw=1.4, zorder=5)
        if s in ("Champions", "Hibernating"):
            ax_r.fill(closed, v, color=COL[s], alpha=0.10, lw=0, zorder=3)
    ax_r.set_xticks(ang)
    ax_r.set_xticklabels(METRICS, fontsize=6.8, color=INK)
    ax_r.set_ylim(0, 1.06)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels([])
    ax_r.tick_params(pad=0.5)
    ax_r.grid(color="#dddddd", lw=0.6)
    ax_r.spines["polar"].set_color("#cccccc")
    ax_r.spines["polar"].set_linewidth(0.7)
    ax_r.set_title("Profiles, min-max scaled across segments ($-$)",
                   fontsize=8.5, fontweight="bold", y=1.03)

    # ---- aux (c): customers versus revenue ---------------------------------
    order = sorted(SEGMENTS, key=lambda s: share_rev[s])
    ypos = np.arange(len(order))
    for k, s in enumerate(order):
        ax_b.barh(k, -share_cust[s] * 100, height=0.62, color=lighten(COL[s], 0.45),
                  edgecolor="white", linewidth=0.5, zorder=4)
        ax_b.barh(k, share_rev[s] * 100, height=0.62, color=COL[s],
                  edgecolor="white", linewidth=0.5, zorder=4)
        ax_b.text(-share_cust[s] * 100 - 1.6, k, f"{share_cust[s]:.0%}",
                  ha="right", va="center", fontsize=6.4, color=SUB)
        ax_b.text(share_rev[s] * 100 + 1.6, k, f"{share_rev[s]:.0%}",
                  ha="left", va="center", fontsize=6.4, color=COL[s])
    lim = max(max(share_cust.values()), max(share_rev.values())) * 100 * 1.42
    ax_b.set_xlim(-lim, lim)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels(order, fontsize=6.8)
    ax_b.set_xticks([-30, -15, 0, 15, 30, 45])
    ax_b.set_xticklabels(["30", "15", "0", "15", "30", "45"], fontsize=6.8)
    ax_b.axvline(0, color="#8a8a8a", lw=0.8, zorder=5)
    ax_b.grid(axis="y", visible=False)
    ax_b.set_xlabel("share of customers (%)   |   share of revenue (%)",
                    fontsize=8.0, labelpad=1.5)
    ax_b.set_title("Count and value pull in opposite directions",
                   fontsize=8.5, fontweight="bold", pad=4.0)
    top = max(SEGMENTS, key=lambda s: share_rev[s])
    ax_b.text(0.985, 0.20,
              f"{top}: {share_rev[top] / share_cust[top]:.1f}$\\times$ "
              "value index", transform=ax_b.transAxes, ha="right",
              va="center", fontsize=6.5, color=SUB)

    panel_labels_inches(fig, [ax, ax_b], labels=["(a)", "(c)"], dx=-0.44,
                        dy=0.10)
    # a polar axes shrinks its own box to a square, so its letter has to be
    # measured off the gridspec slot instead or it lands inside the title
    slot = gs[0, 1].get_position(fig)
    fig.text(slot.x0 - 0.44 / 10.4, slot.y1 + 0.10 / 6.4, "(b)", fontsize=11,
             fontweight="bold", color=INK, ha="left", va="bottom")


    savefig(fig, "639_mkt_customer_segments.png")


if __name__ == "__main__":
    main()
