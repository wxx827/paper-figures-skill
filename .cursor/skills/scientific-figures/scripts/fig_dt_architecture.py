"""Figure 666 - Digital twin architecture for a rotating-machinery asset.

The layout is deliberately mirror-symmetric: the left column is the physical
plant, the right column the virtual model, and every row is a matched pair, so
"what stands in for what" is readable without following a single arrow.  The
centre channel carries the two directions of the synchronisation loop with
their commit rate and measured latency; the strip along the bottom names the
four enabling layers both columns stand on.

Panel (b) is the latency distribution that the arrow annotations are computed
from - the numbers in the channel are percentiles of the very samples plotted
there, never hand-typed.

Drawn on an inch canvas (figkit.inch_canvas) so that block gaps, arrow
clearances and corner radii are all in the same physical unit and the rounded
corners stay circular (DESIGN_SYSTEM section 12.8).
"""
import numpy as np

from pubstyle import set_pub_style, savefig, lighten
from figkit import (inch_canvas, axes_inches, block, arrow, panel_card,
                    panel_label_inches)

INK, SUB, MUTED, HAIR = "#1a1a1a", "#555555", "#777777", "#cccccc"
CARD, GRID = "#f5f5f5", "#d9d9d9"

# Semantic colours (DESIGN_SYSTEM section 2.2: module-level, with meaning).
PHYS = "#6f6259"      # neutral warm grey - the physical plant
VIRT = "#3b5b92"      # house blue       - the virtual model
UP = "#2e7d5b"        # house green      - uplink telemetry
DOWN = "#c9a227"      # house gold       - downlink setpoints

# (physical block, its sub-label, virtual counterpart, its sub-label)
PAIRS = [
    ("Rotating compressor", "42 channels, 10 Hz",
     "Geometry twin", "CAD, 1.2 M cells"),
    ("Edge gateway", "OPC UA / MQTT broker",
     "Physics solver", "1D thermo-fluid"),
    ("PLC control loop", "100 ms scan cycle",
     "Residual learner", "gradient boosting"),
    ("Operator HMI", "shift dashboard",
     "Setpoint optimiser", "rolling horizon"),
]

LAYERS = [
    ("Data acquisition", "tags, buffering"),
    ("Model building", "calibrate, validate"),
    ("Simulation", "what-if replay"),
    ("Optimisation", "setpoint dispatch"),
]

W, H = 10.4, 6.4


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(666)

    # Round-trip latency of each direction of the sync loop, in ms.
    up_ms = rng.lognormal(np.log(38.0), 0.34, 4000)
    dn_ms = rng.lognormal(np.log(86.0), 0.27, 4000)
    up_p50, up_p95 = np.percentile(up_ms, [50, 95])
    dn_p50, dn_p95 = np.percentile(dn_ms, [50, 95])

    fig, ax = inch_canvas(W, H)

    # ---- (a) the two mirrored columns -------------------------------------
    bw, bh = 2.95, 0.58
    px, vx = 2.28, 8.12
    ys = [5.18, 4.48, 3.78, 3.08]

    ax.text(px, 5.80, "Physical entity", ha="center", va="bottom",
            fontsize=8.6, fontweight="bold", color=PHYS)
    ax.text(px, 5.66, "plant floor, real time", ha="center", va="bottom",
            fontsize=6.5, color=MUTED)
    ax.text(vx, 5.80, "Virtual model", ha="center", va="bottom",
            fontsize=8.6, fontweight="bold", color=VIRT)
    ax.text(vx, 5.66, "compute cluster, faster than real time", ha="center",
            va="bottom", fontsize=6.5, color=MUTED)

    for y, (pl, ps, vl, vs) in zip(ys, PAIRS):
        # Pairing rule, drawn under the channel card so only the stubs show.
        ax.plot([px + bw / 2, vx - bw / 2], [y, y], color=HAIR, lw=0.8,
                ls=(0, (2.2, 2.0)), zorder=1)
        block(ax, px, y, bw, bh, pl, color=PHYS, sub=ps, fontsize=8.0,
              sub_color="#e6e0da", zorder=3)
        block(ax, vx, y, bw, bh, vl, color=VIRT, sub=vs, fontsize=8.0,
              sub_color="#dbe3f0", zorder=3)

    # ---- the bidirectional sync channel ------------------------------------
    panel_card(ax, 5.20, 4.20, 2.56, 2.72, fc=CARD, ec=HAIR, zorder=2,
               rounding=0.09)
    ax.text(5.20, 5.40, "Bidirectional sync", ha="center", va="top",
            fontsize=7.6, fontweight="bold", color=INK, zorder=4)

    arrow(ax, (4.10, 4.80), (6.30, 4.80), color=UP, lw=1.7, ms=13, zorder=5)
    ax.text(5.20, 5.02, "Uplink telemetry", ha="center", va="bottom",
            fontsize=7.0, color=UP, fontweight="bold", zorder=5)
    ax.text(5.20, 4.62, "10 Hz aggregate", ha="center", va="top",
            fontsize=6.5, color=SUB, zorder=5)
    ax.text(5.20, 4.47, f"p50 {up_p50:.0f} ms, p95 {up_p95:.0f} ms",
            ha="center", va="top", fontsize=6.5, color=SUB, zorder=5)

    arrow(ax, (6.30, 3.62), (4.10, 3.62), color=DOWN, lw=1.7, ms=13, zorder=5)
    ax.text(5.20, 3.84, "Downlink setpoints", ha="center", va="bottom",
            fontsize=7.0, color="#8a6f12", fontweight="bold", zorder=5)
    ax.text(5.20, 3.44, "1 Hz commit", ha="center", va="top", fontsize=6.5,
            color=SUB, zorder=5)
    ax.text(5.20, 3.29, f"p50 {dn_p50:.0f} ms, p95 {dn_p95:.0f} ms",
            ha="center", va="top", fontsize=6.5, color=SUB, zorder=5)

    ax.text(5.20, 3.02, f"loop budget {up_p95 + dn_p95:.0f} ms", ha="center",
            va="top", fontsize=6.5, color=MUTED, zorder=5)

    # ---- the enabling layers -----------------------------------------------
    panel_card(ax, 3.52, 1.53, 6.46, 1.92, fc="#f0f0f0", ec=HAIR, zorder=1,
               rounding=0.09)
    ax.text(0.40, 2.38, "Enabling layers", ha="left", va="top", fontsize=7.6,
            fontweight="bold", color=INK, zorder=4)

    lw_, lh = 1.42, 1.00
    lxs = [1.18, 2.74, 4.30, 5.86]
    shades = [lighten(VIRT, 0.62), lighten(VIRT, 0.46),
              lighten(VIRT, 0.30), lighten(VIRT, 0.14)]
    for i, (x, (name, note), col) in enumerate(zip(lxs, LAYERS, shades)):
        block(ax, x, 1.32, lw_, lh, name, color=col, sub=note, fontsize=7.4,
              text_color=INK if i < 3 else "white",
              sub_color=SUB if i < 3 else "#dbe3f0", zorder=3)
        if i:
            arrow(ax, (lxs[i - 1] + lw_ / 2, 1.32), (x - lw_ / 2, 1.32),
                  color="#8a8a8a", lw=0.9, ms=8, gap=0.045, zorder=4)

    # ---- (b) sync latency distribution -------------------------------------
    axl = axes_inches(fig, (7.34, 0.94, 2.46, 1.48))
    bins = np.linspace(0, 200, 41)
    for s, col, name in ((up_ms, UP, "uplink"), (dn_ms, DOWN, "downlink")):
        share = 100.0 * np.histogram(s, bins=bins)[0] / s.size
        axl.step(bins[:-1], share, where="post", lw=1.2, color=col,
                 label=name)
        axl.fill_between(bins[:-1], 0.0, share, step="post",
                         facecolor=col, alpha=0.22, edgecolor="none")
    for v, col in ((up_p50, UP), (dn_p50, DOWN)):
        axl.axvline(v, color=col, lw=0.9, ls=(0, (2.6, 1.8)))
    axl.set_xlim(0, 200)
    axl.set_ylim(0, None)
    axl.set_xlabel("sync latency (ms)", fontsize=7.0)
    axl.set_ylabel("updates (%)", fontsize=7.0)
    axl.tick_params(labelsize=6.5, length=2.2)
    axl.grid(axis="y", color=GRID, lw=0.6)
    axl.legend(fontsize=6.5, loc="upper right", frameon=False,
               handlelength=1.2, borderaxespad=0.15)
    axl.set_title("Latency of one sync round trip", fontsize=8.0,
                  fontweight="bold", pad=2.5)
    for s in ("top", "right"):
        axl.spines[s].set_visible(False)

    panel_label_inches(fig, ax, "(a)", dx=0.10, dy=-0.44)
    panel_label_inches(fig, axl, "(b)", dx=-0.52, dy=0.10)
    savefig(fig, "666_dt_architecture.png")


if __name__ == "__main__":
    main()
