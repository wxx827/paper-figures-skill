"""What homomorphic encryption costs: six orders of magnitude, so the axis is
logarithmic.

Main panel: latency of five primitive operations under three encryption
schemes against the same operation in the clear, on a log time axis spanning
nanoseconds to seconds.  Every slowdown factor printed at the end of a bar is
the ratio of the two plotted numbers, not a quoted figure, and the
operations a scheme cannot perform natively are left blank rather than
padded.

Auxiliary (both 58 % of the main panel width): ciphertext expansion against
payload size, where batching lets a packed scheme amortise one fixed-size
ciphertext over thousands of slots while a bit-wise scheme pays the same
factor forever; and the noise budget of a levelled circuit draining one
multiplication at a time, next to the same circuit kept alive by
bootstrapping, with the wall-clock price of staying alive summed from the
latencies in the main panel.

Illustrative synthetic benchmark values; no real library or machine.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import axes_inches, panel_labels_inches

BLUE = "#3b5b92"
DEEP = "#2b4470"
TEAL = "#2a7f7a"
GREEN = "#2e7d5b"
PURPLE = "#8e6bbf"
ACCENT = "#b23a48"
RUST = "#c0603f"
GREY = "#808080"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRID = "#d5d5d5"

FIG_W, FIG_H = 11.0, 6.5
MAIN_RECT = (1.28, 0.86, 5.50, 4.62)
AUX_T_RECT = (7.90, 3.48, 3.14, 2.00)
AUX_B_RECT = (7.90, 0.86, 3.14, 2.00)

OPS = ["add", "multiply", "rotate / permute", "compare  a < b", "bootstrap"]
NAN = np.nan
# latency in milliseconds per operation on one 64-bit payload
LAT = {
    "plaintext (64-bit int)": (GREY, [1.2e-6, 1.8e-6, 0.9e-6, 1.0e-6, NAN]),
    "BFV (packed integers)": (BLUE, [0.11, 9.4, 6.2, 780.0, 5200.0]),
    "CKKS (packed reals)": (TEAL, [0.14, 12.5, 7.8, 1600.0, 6100.0]),
    "TFHE (bit-wise gates)": (PURPLE, [9.5, 21.0, 14.0, 34.0, 13.0]),
}

X_MIN, X_MAX = 1e-7, 2e8
SLOT_BYTES = 4096 * 8      # 4,096 packed slots of 8 bytes
CT_BYTES = 512 * 1024      # one packed ciphertext
TFHE_CT_BYTES = 8 * 1024   # one ciphertext per plaintext bit


def fmt_time(v):
    """Human-readable latency with an explicit unit."""
    if v < 1e-3:
        return f"{v * 1e6:.1f} ns"
    if v < 1.0:
        return f"{v * 1e3:.0f} us"
    if v < 100.0:
        return f"{v:.1f} ms"
    if v < 1000.0:
        return f"{v:.0f} ms"
    return f"{v / 1e3:.1f} s"


def fmt_factor(r):
    """Slowdown factor as mathtext, because it spans seven decades."""
    e = int(np.floor(np.log10(r)))
    if e < 3:
        return f"{r:,.0f}x"
    return rf"${r / 10.0 ** e:.1f}\times10^{{{e}}}$x"


def main():
    set_pub_style(base_font=8)

    names = list(LAT)
    lat = np.array([LAT[k][1] for k in names], dtype=float)
    cols = [LAT[k][0] for k in names]
    factor = lat / lat[0]           # slowdown against the plaintext row

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = axes_inches(fig, MAIN_RECT)
    ax_t = axes_inches(fig, AUX_T_RECT)
    ax_b = axes_inches(fig, AUX_B_RECT)
    for a in (ax, ax_t, ax_b):
        a.set_axisbelow(True)
        style_axes(a)

    # ---- main panel: grouped horizontal bars on a log time axis -------------
    ax.set_xscale("log")
    ax.grid(color=GRID, lw=0.5, alpha=0.6, axis="x", which="major", zorder=0)
    n_s = len(names)
    step = 0.19
    y0 = np.arange(len(OPS), dtype=float)
    for j, (name, col) in enumerate(zip(names, cols)):
        off = (j - (n_s - 1) / 2.0) * step
        for i, v in enumerate(lat[j]):
            if not np.isfinite(v):
                continue
            yy = y0[i] - off
            ax.barh(yy, v - X_MIN, left=X_MIN, height=step * 0.88,
                    color=lighten(col, 0.22), edgecolor=col, lw=0.7, zorder=4)
            lab = fmt_time(v)
            if j > 0 and np.isfinite(factor[j, i]):
                lab += f"   {fmt_factor(factor[j, i])}"
            ax.text(v * 1.45, yy, lab, fontsize=6.1, color=col if j else SUB,
                    ha="left", va="center", zorder=6)
    for i in range(len(OPS)):
        if i % 2 == 0:
            ax.axhspan(y0[i] - 0.46, y0[i] + 0.46, color="#f5f5f5", lw=0,
                       zorder=1)
    ax.text(X_MIN * 1.6, y0[-1] + 0.30,
            "no plaintext counterpart: bootstrapping exists only to undo\n"
            "the noise growth encryption itself introduces",
            fontsize=6.3, color=MUTED, ha="left", va="center",
            linespacing=1.5, zorder=6)

    ax.set_yticks(y0)
    ax.set_yticklabels(OPS, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_ylim(len(OPS) - 0.42, -0.62)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4])
    ax.set_xticklabels(["1 ns", "100 ns", "10 us", "1 ms", "100 ms", "10 s"])
    ax.set_xlabel("latency per operation on one 64-bit payload "
                  "(milliseconds, log scale)", fontsize=8.5)
    ax.set_title("Cost of one operation in the clear and under encryption",
                 fontsize=9.0, fontweight="bold", pad=5.0)
    hm = [Line2D([], [], color=c, lw=5.0, alpha=0.85, label=k)
          for k, c in zip(names, cols)]
    leg = ax.legend(handles=hm, loc="upper right", frameon=False, fontsize=6.9,
                    labelspacing=0.42, handlelength=1.5, borderpad=0.2,
                    bbox_to_anchor=(1.010, 1.012))
    for t in leg.get_texts():
        t.set_color(SUB)

    # ---- aux (b): ciphertext expansion --------------------------------------
    pay = np.logspace(0.0, 6.0, 400)
    exp_pack = np.ceil(pay / SLOT_BYTES) * CT_BYTES / pay
    exp_bits = np.full_like(pay, TFHE_CT_BYTES * 8.0)
    floor_pack = CT_BYTES / SLOT_BYTES
    at_1kb = float(np.interp(1024.0, pay, exp_pack))

    ax_t.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_t.loglog(pay, exp_pack, color=BLUE, lw=1.8, zorder=6,
                label="packed BFV / CKKS")
    ax_t.loglog(pay, exp_bits, color=PURPLE, lw=1.6, zorder=5,
                label="bit-wise TFHE")
    ax_t.axhline(floor_pack, color=GREY, lw=1.0, ls=(0, (3.2, 2.2)), zorder=4)
    ax_t.axhline(1.0, color=INK, lw=0.9, zorder=4)
    ax_t.text(1.3, 1.28, "no expansion (1x)", fontsize=6.0, color=INK,
              ha="left", va="bottom")
    ax_t.text(1.3, floor_pack * 1.15,
              f"packing floor {floor_pack:.0f}x", fontsize=6.0, color=SUB,
              ha="left", va="bottom")
    ax_t.plot([1024.0], [at_1kb], marker="o", ms=4.0, mfc="white", mec=BLUE,
              mew=1.3, zorder=8)
    ax_t.annotate(f"1 KB payload still\ncosts {at_1kb:,.0f}x",
                  xy=(1024.0, at_1kb), xytext=(1.1e4, 4.0e4), fontsize=6.1,
                  color=BLUE, ha="left", va="bottom", linespacing=1.5,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=BLUE,
                                  connectionstyle="arc3,rad=0.22"))
    ax_t.set_xlim(1.0, 1.1e6)
    ax_t.set_ylim(0.6, 4.0e6)
    ax_t.tick_params(labelsize=6.6)
    ax_t.set_xlabel("plaintext payload (bytes, log scale)", fontsize=7.6,
                    labelpad=1.5)
    ax_t.set_ylabel("ciphertext expansion\n(x, log scale)", fontsize=7.6)
    ax_t.set_title("Batching amortises the ciphertext, bits do not",
                   fontsize=8.4, fontweight="bold", pad=4.0)
    lt = ax_t.legend(loc="lower right", frameon=False, fontsize=6.1,
                     handlelength=1.7, labelspacing=0.30, borderpad=0.15,
                     bbox_to_anchor=(1.012, 0.075))
    for t in lt.get_texts():
        t.set_color(SUB)

    # ---- aux (c): noise budget and bootstrapping ----------------------------
    budget0, per_mul, refresh = 200.0, 33.0, 168.0
    n_mul = 20
    k = np.arange(0, n_mul + 1, dtype=float)
    lev = budget0 - per_mul * k
    depth = int(np.floor(budget0 / per_mul))          # levels before failure
    boot, b, n_boot = [budget0], budget0, 0
    for _ in range(n_mul):
        b -= per_mul
        if b < per_mul:                                # cannot afford the next
            b = refresh
            n_boot += 1
        boot.append(b)
    boot = np.array(boot)
    t_mul = LAT["BFV (packed integers)"][1][1]
    t_boot = LAT["BFV (packed integers)"][1][4]
    total_s = (n_mul * t_mul + n_boot * t_boot) / 1e3
    boot_share = 100.0 * n_boot * t_boot / (n_mul * t_mul + n_boot * t_boot)

    ax_b.grid(color=GRID, lw=0.5, alpha=0.7, zorder=0)
    ax_b.axhline(0.0, color=INK, lw=0.9, zorder=4)
    ax_b.plot(k[lev >= 0], lev[lev >= 0], color=RUST, lw=1.7, zorder=5)
    ax_b.plot(k[lev >= 0], lev[lev >= 0], marker="o", ls="none", ms=2.8,
              mfc="white", mec=RUST, mew=0.9, zorder=6)
    ax_b.plot(k, boot, color=TEAL, lw=1.7, zorder=7)
    jump = np.flatnonzero(np.diff(boot) > 0) + 1
    ax_b.plot(k[jump], boot[jump], marker="^", ls="none", ms=4.4, mfc="white",
              mec=TEAL, mew=1.1, zorder=8)
    ax_b.plot([depth + 1], [0.0], marker="x", ms=5.2, mec=ACCENT, mew=1.4,
              zorder=8)
    ax_b.text(7.6, 4.0,
              f"levelled circuit dies\nafter {depth} multiplications",
              fontsize=6.1, color=ACCENT, ha="left", va="bottom",
              linespacing=1.5)
    ax_b.text(19.6, 196.0,
              f"{n_boot} bootstraps keep it running:\n"
              f"{total_s:.1f} s total, {boot_share:.0f} % of it refreshing",
              fontsize=6.1, color=TEAL, ha="right", va="top", linespacing=1.5)
    ax_b.set_xlim(0.0, n_mul + 0.4)
    ax_b.set_ylim(-14.0, 232.0)
    ax_b.set_xticks([0, 4, 8, 12, 16, 20])
    ax_b.tick_params(labelsize=6.6)
    ax_b.set_xlabel("multiplications performed (count)", fontsize=7.6,
                    labelpad=1.5)
    ax_b.set_ylabel("noise budget left (bits)", fontsize=7.6)
    ax_b.set_title("Depth is a budget too", fontsize=8.4, fontweight="bold",
                   pad=4.0)
    hb = [Line2D([], [], color=RUST, lw=1.7, label="levelled, no bootstrap"),
          Line2D([], [], color=TEAL, lw=1.7, label="bootstrapped")]
    lb = ax_b.legend(handles=hb, loc="upper left", frameon=False, fontsize=6.1,
                     handlelength=1.7, labelspacing=0.30, borderpad=0.15,
                     bbox_to_anchor=(-0.012, 1.005))
    for t in lb.get_texts():
        t.set_color(SUB)

    panel_labels_inches(fig, [ax, ax_t, ax_b], dx=-1.00, dy=0.12)
    savefig(fig, "1016_ppc_he_overhead.png")


if __name__ == "__main__":
    main()
