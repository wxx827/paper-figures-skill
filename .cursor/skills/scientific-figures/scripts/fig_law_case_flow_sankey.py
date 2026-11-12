"""Where ten thousand civil filings actually end up: a five-stage alluvial.

Main panel: an alluvial of one annual civil docket through filing, screening,
first instance, appeal and enforcement.  Ribbon width is case count, node
bars are stacked top-aligned so the flow stays laminar, and every node is
labelled with its count and its share of the original filings.  The two
places the docket loses the most cases - rejection at the counter and
enforcement closed for want of assets - are called out.  Conservation is not
a drawing convention here: the script asserts that every non-terminal node
sends out exactly what it took in.

Auxiliary (both at 58 % of the main panel width): elapsed time per stage as
horizontal boxes, which is where the bottleneck shows up; and the reversal
rate on appeal by cause of action with binomial intervals.

The counts are generated top-down as multinomial splits of the parent node,
so conservation holds by construction and the asserts merely prove it.  The
appeal block of the alluvial is summed from the per-cause table in the third
panel, so the two panels cannot disagree.  Fictional court, fictional causes
of action, illustrative values only.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches, halo_text

BLUE = "#3b5b92"
DEEP = "#2b4470"
TEAL = "#2e7d5b"
RUST = "#c0603f"
ACCENT = "#b23a48"
GOLD = "#c9a227"
GREY = "#8f8f8f"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.46, 0.72, 5.90, 4.72)
AUX_T_RECT = (7.72, 3.48, 3.06, 2.00)
AUX_B_RECT = (7.72, 0.86, 3.06, 2.00)

TOTAL = 10000
# wide enough that a two-line node label fits in the gap above every bar
GAP = 620.0
NODE_W = 0.055

CAUSES = ["Contract", "Land tenancy", "Consumer", "Employment",
          "Construction", "Administrative"]
CAUSE_SHARE = [0.24, 0.17, 0.21, 0.14, 0.13, 0.11]
P_REVERSE = [0.185, 0.305, 0.165, 0.255, 0.215, 0.290]
P_REMAND = 0.125

STAGE_NAMES = ["1  Filing", "2  Screening", "3  First instance",
               "4  Appeal", "5  Enforcement"]
DURATION = [("Screening", 14.0, 0.55), ("First instance", 172.0, 0.62),
            ("Appeal", 214.0, 0.50), ("Handover", 33.0, 0.70),
            ("Enforcement", 262.0, 0.75)]


def build_docket(seed=925):
    """Counts for every node and flow, generated as splits of the parent."""
    rng = np.random.default_rng(seed)

    accepted, rejected = rng.multinomial(TOTAL, [0.862, 0.138])
    judgment, mediated, withdrawn = rng.multinomial(
        accepted, [0.608, 0.256, 0.136])
    appealed = int(rng.binomial(judgment, 0.320))
    no_appeal = int(judgment) - appealed

    n_cause = rng.multinomial(appealed, CAUSE_SHARE)
    cause = []
    for n, pr in zip(n_cause, P_REVERSE):
        rev, rem, aff = rng.multinomial(int(n), [pr, P_REMAND,
                                                 1.0 - pr - P_REMAND])
        cause.append(dict(n=int(n), rev=int(rev), rem=int(rem), aff=int(aff)))
    affirmed = sum(c["aff"] for c in cause)
    reversed_ = sum(c["rev"] for c in cause)
    remanded = sum(c["rem"] for c in cause)

    enf_src = [("no_appeal", no_appeal, 0.620), ("affirmed", affirmed, 0.648),
               ("reversed", reversed_, 0.596)]
    split = {}
    for key, n, rate in enf_src:
        done = int(rng.binomial(n, rate))
        split[key] = (done, int(n) - done)
    enforced = sum(v[0] for v in split.values())
    dead = sum(v[1] for v in split.values())

    stages = [
        [("Filed", TOTAL, DEEP, False)],
        [("Accepted for hearing", int(accepted), BLUE, False),
         ("Rejected at filing", int(rejected), RUST, True)],
        [("Judgment on merits", int(judgment), BLUE, False),
         ("Mediated settlement", int(mediated), TEAL, True),
         ("Withdrawn", int(withdrawn), GOLD, True)],
        [("No appeal filed", no_appeal, BLUE, False),
         ("Affirmed", affirmed, BLUE, False),
         ("Reversed", reversed_, GOLD, False),
         ("Remanded", remanded, GREY, True)],
        [("Enforced in full", enforced, TEAL, True),
         ("Closed, no assets", dead, RUST, True)],
    ]
    flows = [
        (0, 0, 1, 0, int(accepted)), (0, 0, 1, 1, int(rejected)),
        (1, 0, 2, 0, int(judgment)), (1, 0, 2, 1, int(mediated)),
        (1, 0, 2, 2, int(withdrawn)),
        (2, 0, 3, 0, no_appeal), (2, 0, 3, 1, affirmed),
        (2, 0, 3, 2, reversed_), (2, 0, 3, 3, remanded),
        (3, 0, 4, 0, split["no_appeal"][0]),
        (3, 0, 4, 1, split["no_appeal"][1]),
        (3, 1, 4, 0, split["affirmed"][0]),
        (3, 1, 4, 1, split["affirmed"][1]),
        (3, 2, 4, 0, split["reversed"][0]),
        (3, 2, 4, 1, split["reversed"][1]),
    ]
    return stages, flows, cause, rng


def check_conservation(stages, flows):
    """Every non-terminal node must emit exactly what it received."""
    for si, nodes in enumerate(stages):
        for ni, (name, size, _c, terminal) in enumerate(nodes):
            out = sum(f[4] for f in flows if f[0] == si and f[1] == ni)
            inn = sum(f[4] for f in flows if f[2] == si and f[3] == ni)
            if si > 0:
                assert inn == size, f"{name}: inflow {inn} != {size}"
            if terminal:
                assert out == 0, f"{name}: terminal node emits {out}"
            else:
                assert out == size, f"{name}: outflow {out} != {size}"
    assert sum(f[4] for f in flows if f[0] == 0) == TOTAL


def layout(stages, y_top=0.0):
    """Top-aligned (top, bottom) span per node, in case units."""
    pos = []
    for nodes in stages:
        y = y_top
        row = []
        for _n, size, _c, _t in nodes:
            row.append((y, y + size))
            y += size + GAP
        pos.append(row)
    return pos


def ribbon(ax, x0, x1, a0, b0, a1, b1, color, alpha=0.55, zorder=3):
    """Smoothstep flow band; a/b are the top and bottom edge at each end."""
    t = np.linspace(0.0, 1.0, 96)
    s = t * t * (3.0 - 2.0 * t)
    ax.fill_between(x0 + (x1 - x0) * t, a0 + (a1 - a0) * s,
                    b0 + (b1 - b0) * s, facecolor=color, alpha=alpha,
                    linewidth=0.0, zorder=zorder)


def box_row(ax, y, vals, color, height=0.46):
    """Horizontal box-and-whisker at row ``y`` from raw values."""
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo, hi = np.percentile(vals, [5, 95])
    ax.plot([lo, q1], [y, y], lw=0.8, color=color, zorder=4)
    ax.plot([q3, hi], [y, y], lw=0.8, color=color, zorder=4)
    for xv in (lo, hi):
        ax.plot([xv, xv], [y - height * 0.34, y + height * 0.34], lw=0.8,
                color=color, zorder=4)
    ax.add_patch(Rectangle((q1, y - height / 2), q3 - q1, height,
                           facecolor=lighten(color, 0.62), edgecolor=color,
                           linewidth=0.9, zorder=5))
    ax.plot([med, med], [y - height / 2, y + height / 2], lw=1.7, color=INK,
            zorder=6)
    return med


def main():
    set_pub_style(base_font=8)
    stages, flows, cause, rng = build_docket()
    check_conservation(stages, flows)
    pos = layout(stages)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_d = axes_inches(fig, AUX_T_RECT)
    ax_r = axes_inches(fig, AUX_B_RECT)
    for a in (ax_d, ax_r):
        a.set_axisbelow(True)
        style_axes(a)
        a.grid(axis="x", color=GRID, lw=0.5, alpha=0.7, zorder=0)

    # ---- main: the alluvial -------------------------------------------------
    ax.set_xlim(-0.10, 5.40)
    ax.set_ylim(11950.0, -1520.0)
    ax.axis("off")

    out_cur = {(si, ni): pos[si][ni][0]
               for si in range(len(stages)) for ni in range(len(stages[si]))}
    in_cur = dict(out_cur)
    for si, ni, sj, nj, n in sorted(flows, key=lambda f: (f[0], f[1], f[3])):
        a0 = out_cur[(si, ni)]
        out_cur[(si, ni)] = a0 + n
        a1 = in_cur[(sj, nj)]
        in_cur[(sj, nj)] = a1 + n
        tgt_col = stages[sj][nj][2]
        loud = stages[sj][nj][0].startswith(("Rejected", "Closed"))
        ribbon(ax, si + NODE_W, float(sj), a0, a0 + n, a1, a1 + n,
               lighten(tgt_col, 0.30 if loud else 0.46),
               alpha=0.80 if loud else 0.58, zorder=4 if loud else 3)

    for si, nodes in enumerate(stages):
        for ni, (name, size, col, _t) in enumerate(nodes):
            y0, y1 = pos[si][ni]
            ax.add_patch(Rectangle((si, y0), NODE_W, y1 - y0, facecolor=col,
                                   edgecolor="white", linewidth=0.6,
                                   zorder=7))
            halo_text(ax, si + NODE_W + 0.035, y0 - 120.0,
                      f"{name}\n{size:,} cases "
                      f"({100.0 * size / TOTAL:.1f} %)",
                      color=INK, halo="white", halo_lw=2.4, fontsize=6.3,
                      ha="left", va="bottom", zorder=9, linespacing=1.35)
        ax.text(si, -1180.0, STAGE_NAMES[si], ha="left", va="bottom",
                fontsize=7.6, fontweight="bold", color=SUB)

    rej = stages[1][1][1]
    dead = stages[4][1][1]
    enf_in = stages[4][0][1] + dead
    ax.annotate(f"{100.0 * rej / TOTAL:.1f} % of filings never reach\n"
                f"a hearing ({rej:,} cases)",
                xy=(1.0 + NODE_W, 0.5 * (pos[1][1][0] + pos[1][1][1])),
                xytext=(1.34, 11450.0), fontsize=6.7, color=ACCENT,
                ha="left", va="bottom", linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", lw=0.85, color=ACCENT,
                                connectionstyle="arc3,rad=-0.22",
                                shrinkA=2, shrinkB=2))
    ax.annotate(f"{100.0 * dead / enf_in:.1f} % of enforceable judgments\n"
                f"recover nothing ({dead:,} cases)",
                xy=(4.0, 0.5 * (pos[4][1][0] + pos[4][1][1])),
                xytext=(4.12, 7100.0), fontsize=6.7, color=ACCENT,
                ha="left", va="top", linespacing=1.5,
                arrowprops=dict(arrowstyle="-|>", lw=0.85, color=ACCENT,
                                connectionstyle="arc3,rad=0.25",
                                shrinkA=2, shrinkB=2))
    ax.set_title(f"One annual civil docket, Marrow District Court "
                 f"(n = {TOTAL:,} filings)",
                 fontsize=9.0, fontweight="bold", pad=6.0)

    # ---- aux (b): elapsed time per stage ------------------------------------
    meds = []
    for k, (label, med, sig) in enumerate(DURATION):
        vals = rng.lognormal(np.log(med), sig, 900)
        meds.append(box_row(ax_d, k, vals,
                            RUST if med == max(d[1] for d in DURATION)
                            else BLUE))
    slowest = int(np.argmax(meds))
    ax_d.set_yticks(range(len(DURATION)))
    ax_d.set_yticklabels([d[0] for d in DURATION], fontsize=6.6)
    ax_d.set_ylim(-0.75, len(DURATION) + 0.75)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0.0, 900.0)
    ax_d.set_xticks([0, 200, 400, 600, 800])
    ax_d.tick_params(axis="x", labelsize=6.6)
    ax_d.set_xlabel("elapsed time in stage (days)", fontsize=7.6, labelpad=1.5)
    ax_d.set_title("Enforcement is the bottleneck", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_d.text(880.0, len(DURATION) + 0.60,
              f"boxes are quartiles, whiskers the 5th-95th percentile\n"
              f"median {DURATION[slowest][0].lower()} takes "
              f"{meds[slowest]:.0f} days, "
              f"{meds[slowest] / min(meds):.0f}x the fastest stage",
              ha="right", va="bottom", fontsize=6.0, color=SUB,
              linespacing=1.5)

    # ---- aux (c): reversal rate by cause of action --------------------------
    rate = np.array([100.0 * c["rev"] / c["n"] for c in cause])
    nn = np.array([c["n"] for c in cause], dtype=float)
    ci = 196.0 * np.sqrt((rate / 100.0) * (1.0 - rate / 100.0) / nn)
    idx = np.argsort(rate)
    overall = 100.0 * sum(c["rev"] for c in cause) / sum(c["n"] for c in cause)

    ypos = np.arange(len(CAUSES))
    for k, c in enumerate(idx):
        col = ACCENT if rate[c] > overall else BLUE
        ax_r.barh(k, rate[c], height=0.62, color=lighten(col, 0.34),
                  edgecolor=col, linewidth=0.8, zorder=3)
        ax_r.plot([rate[c] - ci[c], rate[c] + ci[c]], [k, k], lw=0.9,
                  color=col, zorder=5)
        ax_r.text(rate[c] + ci[c] + 0.7, k, f"{rate[c]:.0f} %  n={nn[c]:.0f}",
                  ha="left", va="center", fontsize=6.0, color=SUB, zorder=5)
    ax_r.axvline(overall, color=FAINT, lw=0.9, ls=(0, (3.0, 2.2)), zorder=4)
    ax_r.set_yticks(ypos)
    ax_r.set_yticklabels([CAUSES[c] for c in idx], fontsize=6.6)
    ax_r.set_ylim(-0.72, len(CAUSES) + 0.55)
    ax_r.invert_yaxis()
    ax_r.set_xlim(0.0, 52.0)
    ax_r.set_xticks([0, 10, 20, 30, 40, 50])
    ax_r.tick_params(axis="x", labelsize=6.6)
    ax_r.set_xlabel("judgments reversed on appeal (%, +/- 95 % CI)",
                    fontsize=7.6, labelpad=1.5)
    ax_r.set_title("Reversal rate by cause of action", fontsize=8.4,
                   fontweight="bold", pad=4.0)
    ax_r.text(51.0, len(CAUSES) + 0.42,
              f"dashed rule is the docket-wide rate, {overall:.1f} % of "
              f"{sum(c['n'] for c in cause):,} appeals",
              ha="right", va="bottom", fontsize=6.0, color=SUB)

    panel_labels_inches(fig, [ax, ax_d, ax_r], dx=-0.34, dy=0.14)
    savefig(fig, "925_law_case_flow_sankey.png")


if __name__ == "__main__":
    main()
