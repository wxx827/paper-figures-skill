"""Secure multi-party computation: the bill is written in bytes, not in flops.

Main panel: total communication for the same 100,000-multiplication circuit
as the number of participating parties grows, on log-log axes because the
pairwise protocols cost two decades more than the aggregation-shaped ones by
the time a few hundred parties are in the room.  The scaling exponents in
the legend are least-squares fits to the plotted curves, not the exponents
of the formulas they were generated from, and the ratio quoted at the right
edge is read off the two curves at the same party count.

Auxiliary (both 58 % of the main panel width): interaction rounds against
party count, which is a completely different ranking from bytes; and the
wall-clock split of one operating point on a wide-area link into local
computation, bytes on the wire and round-trip latency, every term computed
from the two panels above it plus one link model.

Illustrative synthetic protocol models; no real implementation is depicted.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

DEEP = "#2b4470"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
PURPLE = "#8e6bbf"
ACCENT = "#b23a48"
RUST = "#c0603f"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (0.98, 1.00, 5.50, 4.48)
AUX_T_RECT = (7.70, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.70, 0.86, 3.14, 2.00)

N_MUL = 100_000        # multiplication gates in the benchmark circuit
ELEM_B = 16.0          # bytes per field element on the wire
DEPTH = 40             # multiplicative depth of the circuit
N_OP = 16              # operating point carried into panel (c)
LAN_BW, LAN_RTT = 1250.0, 0.0001    # MB/s, seconds
WAN_BW, WAN_RTT = 12.5, 0.060       # MB/s, seconds

PROTOCOLS = [
    ("BGW pairwise (honest majority)", DEEP, 2.0,
     lambda n: n * (n - 1.0), lambda n: np.full_like(n, float(DEPTH)), 0.35),
    ("SPDZ online with MAC check", PURPLE, 1.8,
     lambda n: 2.1 * n * (n - 1.0),
     lambda n: np.full_like(n, DEPTH + 3.0), 0.90),
    ("tree-shaped aggregation", TEAL, 1.8,
     lambda n: n * np.log2(n), lambda n: np.log2(n), 0.12),
    ("server-aided additive shares", GOLD, 1.8,
     lambda n: 2.0 * n, lambda n: np.full_like(n, 2.0), 0.20),
]


def main():
    set_pub_style(base_font=8)

    n = np.logspace(np.log10(3.0), np.log10(1024.0), 260)
    mb, rounds, comp = [], [], []
    for _, _, _, bytes_fn, round_fn, t_comp in PROTOCOLS:
        mb.append(N_MUL * bytes_fn(n) * ELEM_B / 1e6)
        rounds.append(round_fn(n))
        comp.append(t_comp)
    mb = np.array(mb)
    rounds = np.array(rounds)
    slopes = [float(np.polyfit(np.log10(n), np.log10(y), 1)[0]) for y in mb]
    ratio_end = mb[0][-1] / mb[3][-1]

    # ---- operating point carried into panel (c) -----------------------------
    i_op = int(np.argmin(np.abs(n - N_OP)))
    n_op = n[i_op]
    mb_op = mb[:, i_op]
    rd_op = rounds[:, i_op]
    wan = np.array([comp, mb_op / WAN_BW, rd_op * WAN_RTT])
    lan = np.array([comp, mb_op / LAN_BW, rd_op * LAN_RTT])
    wan_tot, lan_tot = wan.sum(axis=0), lan.sum(axis=0)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(color=GRID, lw=0.5, alpha=0.55, which="major", zorder=0)
    ax.grid(color=GRID, lw=0.35, alpha=0.30, which="minor", zorder=0)
    ax.axvline(n_op, color=INK, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(n_op * 1.12, 1.0e5,
            f"{n_op:.0f} parties: the operating point of panel (c)",
            fontsize=6.5, color=INK, ha="left", va="bottom")

    for (name, col, lw, _, _, _), y, s in zip(PROTOCOLS, mb, slopes):
        ax.plot(n, y, color=col, lw=lw, zorder=5)
        ax.plot([n_op], [y[i_op]], marker="o", ms=4.2, mfc="white", mec=col,
                mew=1.3, zorder=8)
        ax.text(1080.0, y[-1], f"  {name.split(' (')[0].split(' with')[0]}",
                fontsize=6.4, color=col, ha="left", va="center")

    ax.annotate("", xy=(940.0, mb[0][-1]), xytext=(940.0, mb[3][-1]),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=ACCENT,
                                shrinkA=0, shrinkB=0))
    ax.text(880.0, np.sqrt(mb[0][-1] * mb[3][-1]),
            f"{ratio_end:,.0f}x more bytes\nat 1,024 parties",
            fontsize=6.8, color=ACCENT, ha="right", va="center",
            linespacing=1.5, fontweight="bold")

    ax.set_xlim(2.7, 3.0e3)
    ax.set_ylim(2.0, 4.0e6)
    ax.set_xticks([3, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(["3", "5", "10", "20", "50", "100", "200", "500",
                        "1,000"])
    ax.set_xlabel("participating parties, $n$ (parties, log scale)",
                  fontsize=8.5)
    ax.set_ylabel("total communication for the circuit (MB, log scale)",
                  fontsize=8.5)
    ax.set_title(f"Bytes on the wire for one {N_MUL:,}-multiplication "
                 "circuit", fontsize=9.0, fontweight="bold", pad=5.0)
    hm = [Line2D([], [], color=p[1], lw=p[2],
                 label=f"{p[0]}  -  fitted exponent {s:.2f}")
          for p, s in zip(PROTOCOLS, slopes)]
    leg = ax.legend(handles=hm, loc="upper left", frameon=False, fontsize=6.8,
                    labelspacing=0.45, handlelength=2.0, borderpad=0.2,
                    bbox_to_anchor=(-0.010, 1.012))
    for t in leg.get_texts():
        t.set_color(SUB)
    ax.text(18.5, 2.6,
            f"{ELEM_B:.0f} bytes per field element, "
            f"multiplicative depth {DEPTH} gates",
            fontsize=6.4, color=MUTED, ha="left", va="bottom")

    # ---- aux (b): rounds ----------------------------------------------------
    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    for (name, col, _, _, _, _), y in zip(PROTOCOLS, rounds):
        ax_t.semilogx(n, y, color=col, lw=1.7, zorder=5)
    ax_t.axvline(n_op, color=INK, lw=0.8, ls=(0, (2.4, 2.0)), zorder=3)
    ax_t.set_xlim(2.7, 1.2e3)
    ax_t.set_ylim(0.0, 52.0)
    ax_t.set_xticks([3, 10, 100, 1000])
    ax_t.set_xticklabels(["3", "10", "100", "1,000"])
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("participating parties (parties, log scale)",
                    fontsize=7.6, labelpad=1.5)
    ax_t.set_ylabel("interaction rounds\n(rounds)", fontsize=7.6)
    ax_t.set_title("Rounds rank the protocols the other way round",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    ax_t.text(3.4, 35.0, "depth-bound protocols: rounds do not\n"
                         "depend on how many parties join",
              fontsize=6.1, color=SUB, ha="left", va="top", linespacing=1.5)
    ax_t.text(1.05e3, 12.5, "$\\log_2 n$", fontsize=6.4, color=TEAL,
              ha="right", va="bottom")
    ax_t.text(1.05e3, 3.5, "2 rounds", fontsize=6.4, color=GOLD, ha="right",
              va="bottom")

    # ---- aux (c): wall clock at the operating point -------------------------
    labels = ["compute", "bytes on the wire", "round-trip latency"]
    seg_cols = ["#9cc3dd", "#2e74a8", GOLD]
    ypos = np.arange(len(PROTOCOLS), dtype=float)
    left = np.zeros(len(PROTOCOLS))
    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, axis="x", zorder=0)
    for row, col, lab in zip(wan, seg_cols, labels):
        ax_b.barh(ypos, row, left=left, height=0.62, color=col,
                  edgecolor="white", lw=0.6, zorder=4, label=lab)
        left = left + row
    for i, (tot, lt) in enumerate(zip(wan_tot, lan_tot)):
        ax_b.text(tot + 1.5, ypos[i], f"{tot:.0f} s WAN   /   {lt:.1f} s LAN",
                  fontsize=6.1, color=INK, ha="left", va="center", zorder=6)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([p[0].split(" (")[0].split(" with")[0]
                          for p in PROTOCOLS], fontsize=6.6)
    for tick, p in zip(ax_b.get_yticklabels(), PROTOCOLS):
        tick.set_color(p[1])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, 105.0)
    ax_b.set_ylim(len(PROTOCOLS) - 0.42, -0.60)
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("wall-clock time on the wide-area link (seconds)",
                    fontsize=7.6, labelpad=1.5)
    ax_b.set_title(f"Where the time goes at {n_op:.0f} parties",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    lb = ax_b.legend(loc="lower right", frameon=False, fontsize=6.1,
                     handlelength=1.3, labelspacing=0.28, borderpad=0.15,
                     bbox_to_anchor=(1.012, -0.020))
    for t in lb.get_texts():
        t.set_color(SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-0.70, dy=0.12)
    savefig(fig, "1017_ppc_mpc_communication.png")


if __name__ == "__main__":
    main()
