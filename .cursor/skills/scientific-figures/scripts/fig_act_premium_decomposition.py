"""From expected loss to gross premium: the same six steps, three products.

Main panel: three horizontal waterfalls on one shared money axis, so the
loadings of a short-tail motor cover and a long-tail liability cover can be
read against each other rather than each against itself.  Every line starts
at the pure risk premium (frequency x severity), adds a cost-of-capital risk
margin, gives back an investment credit for the time the money is held, then
adds expenses, commission and premium tax to arrive at the gross premium.
The two narrow right-hand panels give the consequence and the driver: the
combined ratio against the 100% break-even rule, and the tariff relativities
that move an individual policy around its class average.

Commission and tax are quoted on the gross premium, so the gross is solved
for rather than accumulated; every printed amount and share is read back off
the same solution.  Values are illustrative.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig, style_axes
from figkit import axes_inches, panel_labels_inches

LOSS = "#b23a48"
CAPITAL = "#3b5b92"
CREDIT = "#2a7f7a"
EXP1 = "#555555"
EXP2 = "#8a8a8a"
EXP3 = "#cccccc"
GROSS = "#3a3a3a"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
GRID = "#d5d5d5"
BAND = "#f5f5f5"

FIG_W, FIG_H = 10.4, 6.4
MAIN_RECT = (1.16, 0.74, 4.94, 4.68)
AUX_T_RECT = (7.10, 3.47, 2.90, 1.95)
AUX_B_RECT = (7.10, 0.74, 2.90, 1.95)

# frequency (claims per policy-year), severity (USD), capital factor (x pure
# premium), payment duration (years), fixed expense (USD), commission and
# premium-tax rates (share of gross premium)
LINES = [
    ("Motor own damage", dict(freq=0.072, sev=5400.0, cap=0.90, dur=0.55,
                              expense=62.0, comm=0.110, tax=0.040)),
    ("Commercial property", dict(freq=0.031, sev=18500.0, cap=1.60, dur=0.85,
                                 expense=88.0, comm=0.145, tax=0.040)),
    ("General liability", dict(freq=0.017, sev=62000.0, cap=2.20, dur=4.80,
                               expense=110.0, comm=0.160, tax=0.040)),
]
COC = 0.060          # cost of capital, per year
YIELD = 0.034        # risk-free investment yield, per year

STEPS = [("Pure risk premium", LOSS), ("Risk margin (CoC)", CAPITAL),
         ("Investment credit", CREDIT), ("Expense loading", EXP1),
         ("Commission", EXP2), ("Taxes and levies", EXP3),
         ("Gross premium", GROSS)]

# log-linear tariff coefficients; relativities are exp(beta) against the base
TARIFF = [("Driver age 18-24", 0.62), ("Driver age 25-39", 0.14),
          ("Driver age 40-64 (base)", 0.0), ("Urban zone", 0.30),
          ("Rural zone", -0.22), ("No-claim discount 0 yr", 0.37),
          ("No-claim discount 5+ yr", -0.48)]


def decompose(freq, sev, cap, dur, expense, comm, tax):
    """Solve the gross premium and return the signed waterfall steps."""
    pure = freq * sev
    margin = COC * cap * pure
    credit = -YIELD * dur * pure
    base = pure + margin + credit + expense
    gross = base / (1.0 - comm - tax)
    parts = [pure, margin, credit, expense, comm * gross, tax * gross]
    return np.array(parts), gross


def main():
    set_pub_style(base_font=8)
    book = {}
    for name, kw in LINES:
        parts, gross = decompose(**kw)
        book[name] = dict(parts=parts, gross=gross, kw=kw)

    gmax = max(v["gross"] for v in book.values())

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_c = axes_inches(fig, AUX_T_RECT)
    ax_r = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_c, ax_r):
        a.set_axisbelow(True)
        style_axes(a)
    ax.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_c.grid(axis="y", color=GRID, lw=0.55, alpha=0.75, zorder=0)
    ax_r.grid(axis="x", color=GRID, lw=0.55, alpha=0.75, zorder=0)

    # ---- main: three horizontal waterfalls ---------------------------------
    ticks, labels = [], []
    for gi, (name, _) in enumerate(LINES):
        rec = book[name]
        parts, gross = rec["parts"], rec["gross"]
        top = gi * 8
        ax.axhspan(top - 0.62, top + 6.62, facecolor=BAND, lw=0, zorder=0)
        run = 0.0
        for k, (label, colour) in enumerate(STEPS):
            row = top + k
            ticks.append(row)
            labels.append(label)
            if k == 0:
                val, left = parts[0], 0.0
                run = parts[0]
            elif k == len(STEPS) - 1:
                val, left = gross, 0.0
            else:
                val = parts[k]
                left = run if val >= 0 else run + val
                val = abs(val)
                run = run + parts[k]
            ax.barh(row, val, left=left, height=0.60, color=colour,
                    edgecolor="white", linewidth=0.6, zorder=3)
            if k < len(STEPS) - 1:
                ax.plot([run, run], [row + 0.30, row + 0.70], color=FAINT,
                        lw=0.6, zorder=4)
            amount = parts[k] if k < len(STEPS) - 1 else gross
            share = 100.0 * amount / gross
            ax.text(left + val + 0.012 * gmax, row,
                    f"{amount:,.0f} USD  ({share:+.1f}%)"
                    if 0 < k < len(STEPS) - 1 else
                    f"{amount:,.0f} USD  ({share:.1f}%)",
                    ha="left", va="center", fontsize=6.2,
                    color=INK if k in (0, len(STEPS) - 1) else SUB)
        ax.text(gmax * 1.29, top, name, ha="right", va="center",
                fontsize=7.6, fontweight="bold", color=INK, zorder=6)

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_ylim(22.7, -0.7)
    ax.set_xlim(0, gmax * 1.30)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.set_xlabel("amount per policy-year (USD)", fontsize=8.5)
    ax.set_title("Six steps from expected loss to gross premium",
                 fontsize=9.0, fontweight="bold", pad=19.0)
    handles = [Patch(facecolor=c, edgecolor="white", label=t)
               for t, c in [("loss cost", LOSS), ("capital charge", CAPITAL),
                            ("investment credit (-)", CREDIT),
                            ("expense, commission, tax", EXP2),
                            ("gross premium", GROSS)]]
    ax.legend(handles=handles, loc="lower center", frameon=False, ncol=5,
              fontsize=6.4, borderpad=0.20, columnspacing=1.0,
              handlelength=1.2, handletextpad=0.4,
              bbox_to_anchor=(0.5, 1.003))

    # ---- aux (b): combined ratio -------------------------------------------
    names = [n for n, _ in LINES]
    lr = np.array([100.0 * book[n]["parts"][0] / book[n]["gross"]
                   for n in names])
    er = np.array([100.0 * book[n]["parts"][3:6].sum() / book[n]["gross"]
                   for n in names])
    idx = np.arange(len(names))
    ax_c.bar(idx, lr, width=0.58, color=LOSS, edgecolor="white",
             linewidth=0.6, zorder=3, label="loss ratio")
    ax_c.bar(idx, er, bottom=lr, width=0.58, color=EXP2, edgecolor="white",
             linewidth=0.6, zorder=3, label="expense ratio")
    ax_c.axhline(100.0, color=INK, lw=1.0, ls=(0, (3.0, 2.0)), zorder=5)
    for k in idx:
        cr = lr[k] + er[k]
        ax_c.text(k, cr + 1.6, f"{cr:.1f}%", ha="center", va="bottom",
                  fontsize=6.6, fontweight="bold",
                  color=LOSS if cr > 100.0 else CAPITAL, zorder=6)
        ax_c.text(k, lr[k] / 2.0, f"{lr[k]:.0f}", ha="center", va="center",
                  fontsize=6.2, color="white", zorder=6)
        ax_c.text(k, lr[k] + er[k] / 2.0, f"{er[k]:.0f}", ha="center",
                  va="center", fontsize=6.2, color="white", zorder=6)
    ax_c.set_xticks(idx)
    ax_c.set_xticklabels(["motor", "property", "liability"], fontsize=6.8)
    ax_c.set_ylim(0, 142)
    ax_c.tick_params(labelsize=6.8)
    ax_c.set_ylabel("share of gross premium (%)", fontsize=7.8)
    ax_c.set_xlabel("business line", fontsize=7.8, labelpad=1.5)
    ax_c.set_title("Combined ratio against break-even", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_c.text(0.03, 0.96,
              "lower block = loss ratio, upper = expense ratio;\n"
              "above the dashed 100% rule the line pays out\n"
              "more than it takes in",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=6.4,
              color=SUB, linespacing=1.45)

    # ---- aux (c): tariff relativities --------------------------------------
    rel = np.array([np.exp(b) for _, b in TARIFF])
    rows = np.arange(len(TARIFF))
    cols = [LOSS if r > 1.0 else CAPITAL if r < 1.0 else FAINT for r in rel]
    ax_r.barh(rows, rel - 1.0, left=1.0, height=0.58, color=cols,
              edgecolor="white", linewidth=0.6, zorder=3)
    ax_r.axvline(1.0, color=INK, lw=0.9, zorder=5)
    for k, r in enumerate(rel):
        ax_r.text(2.12, k, f"{r:.2f}x", ha="right", va="center",
                  fontsize=6.2, color=SUB)
    spread = float(np.exp(0.62 + 0.30 + 0.37) / np.exp(-0.22 - 0.48))
    ax_r.set_yticks(rows)
    ax_r.set_yticklabels([t for t, _ in TARIFF], fontsize=6.4)
    ax_r.set_ylim(len(TARIFF) - 0.4, -0.6)
    ax_r.set_xlim(0.45, 2.15)
    ax_r.tick_params(axis="y", length=0)
    ax_r.tick_params(labelsize=6.8)
    ax_r.set_xlabel("relative risk against the base cell (x)", fontsize=7.8,
                    labelpad=1.5)
    ax_r.set_title("Tariff relativities move the pure premium", fontsize=8.5,
                   fontweight="bold", pad=4.0)
    ax_r.text(0.97, 0.2, f"worst-to-best tariff\ncell spans {spread:.1f}x",
              ha="right", va="center", fontsize=6.4, color=MUTED,
              linespacing=1.5)

    panel_labels_inches(fig, [ax, ax_c, ax_r], dx=-0.90, dy=0.10)
    savefig(fig, "885_act_premium_decomposition.png")


if __name__ == "__main__":
    main()
