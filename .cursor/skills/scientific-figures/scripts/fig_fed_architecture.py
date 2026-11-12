"""Figure 486 - Federated learning system architecture.

A star topology drawn on an inch canvas: one central aggregator at the top,
four participating clients along the middle row, and a fan of paired arrows
carrying the global weights down and the local updates back up.  Each client
card keeps its own class histogram inside the card, which is the whole point
of the setting - the data never crosses the arrow.  The aggregation rule sits
inside the server block so the reader meets the weighting scheme at the place
where it is applied, and the strip along the bottom unrolls one communication
round into its four sequential stages.

Geometry is expressed in inches on an equal-aspect canvas, so corner radii,
arrow heads and edge clearances match the rest of the architecture batch.
"""
import numpy as np
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, lighten
from figkit import (inch_canvas, anchors, block, panel_card, text_card, arrow,
                    INK, SUB, MUTED, HAIR)

BLUE = "#3b5b92"     # server / global model
GREEN = "#2e7d5b"    # client 1
GOLD = "#c9a227"     # client 2
PURPLE = "#8e6bbf"   # client 3
TEAL = "#2a7f7a"     # client K
UP = "#555555"       # uplink stroke

W, H = 11.4, 6.8
SERVER = dict(x=5.70, y=5.05, w=3.60, h=0.95)
CLIENT_Y, CLIENT_W, CLIENT_H = 2.80, 2.20, 1.50
CLIENT_X = [1.80, 4.40, 7.00, 9.60]
CLIENT_COL = [GREEN, GOLD, PURPLE, TEAL]
CLIENT_NAME = ["Client 1", "Client 2", "Client 3", "Client K"]

N_CLASSES = 6
PARAMS = 3.10e6                       # weights exchanged per direction
CONC = [4.0, 0.9, 0.35, 1.8]          # Dirichlet skew per client card
SIZES = [1240, 860, 2110, 1475]


def bar(ax, x, y, w, h, color):
    """Histogram bar for the miniature class distribution inside a card."""
    return ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                                  edgecolor="white", linewidth=0.5, zorder=3))


def client_card(ax, cx, color, name, counts):
    """Rounded client card carrying a local class histogram."""
    panel_card(ax, cx, CLIENT_Y, CLIENT_W, CLIENT_H,
               fc=lighten(color, 0.90), ec=lighten(color, 0.42), lw=0.8,
               rounding=0.07, zorder=2)
    ax.text(cx, CLIENT_Y + CLIENT_H / 2 - 0.19, name, ha="center",
            va="center", fontsize=7.8, fontweight="bold", color=color,
            zorder=4)

    base = CLIENT_Y - CLIENT_H / 2 + 0.25
    bw, gap = 0.19, 0.075
    span = N_CLASSES * bw + (N_CLASSES - 1) * gap
    x0 = cx - span / 2
    top = 0.62 * counts / counts.max()
    for j, hgt in enumerate(top):
        bar(ax, x0 + j * (bw + gap), base, bw, hgt, color)
    ax.plot([x0 - 0.05, x0 + span + 0.05], [base, base], lw=0.7, color=MUTED,
            zorder=3)
    ax.text(cx, CLIENT_Y - CLIENT_H / 2 + 0.13,
            f"$D_k$ on device · $n_k$ = {int(counts.sum()):,}", ha="center",
            va="center", fontsize=6.4, color=SUB, zorder=4)
    return anchors(cx, CLIENT_Y, CLIENT_W, CLIENT_H)


def main():
    set_pub_style()
    rng = np.random.default_rng(486)
    fig, ax = inch_canvas(W, H)

    counts = [np.maximum(rng.dirichlet(np.full(N_CLASSES, a)) * n, 8.0)
              for a, n in zip(CONC, SIZES)]
    payload = PARAMS * 4.0 / 1.0e6        # fp32 weights, in MB

    # ---- titles ----------------------------------------------------------
    ax.text(0.45, 6.42, "Federated averaging keeps the data on the device "
            "and moves only the weights", fontsize=13.5, fontweight="bold",
            color=INK, ha="left", va="center")
    ax.text(0.45, 6.10, "One synchronous round: broadcast, local SGD, upload, "
            "sample-weighted aggregation", fontsize=8.6, color=MUTED,
            ha="left", va="center")

    # ---- central server --------------------------------------------------
    srv = block(ax, SERVER["x"], SERVER["y"], SERVER["w"], SERVER["h"],
                color=BLUE, lw=0.9, rounding=0.075)
    ax.text(SERVER["x"], SERVER["y"] + 0.29, "Central server · aggregator",
            ha="center", va="center", fontsize=8.6, fontweight="bold",
            color="white", zorder=6)
    ax.text(SERVER["x"], SERVER["y"] - 0.02,
            r"$w_{t+1}=\sum_{k \in S_t} \frac{n_k}{n}\, w^{k}_{t+1}$",
            ha="center", va="center", fontsize=10.0, color="white", zorder=6)
    ax.text(SERVER["x"], SERVER["y"] - 0.31,
            "FedAvg · weighted mean over the sampled cohort $S_t$",
            ha="center", va="center", fontsize=6.5, color="white",
            alpha=0.90, zorder=6)

    # ---- side cards ------------------------------------------------------
    panel_card(ax, 2.10, 5.15, 3.30, 1.20, fc="#f5f5f5", ec=HAIR, lw=0.8,
               rounding=0.07, zorder=2)
    ax.text(0.62, 5.62, "Arrow vocabulary", ha="left", va="center",
            fontsize=7.6, fontweight="bold", color=INK, zorder=4)
    vocab = [(BLUE, "-", r"Downlink: global weights $w_t$"),
             (UP, (0, (4, 2.4)), r"Uplink: update $\Delta w^{k}_{t}$ and $n_k$"),
             (None, None, r"Local data $D_k$ never leaves the client")]
    for i, (col, ls, txt) in enumerate(vocab):
        yy = 5.32 - 0.27 * i
        if col is None:
            bar(ax, 0.66, yy - 0.055, 0.30, 0.11, GREEN)
        else:
            arrow(ax, (0.66, yy), (1.02, yy), color=col, ls=ls, lw=1.15,
                  ms=9, zorder=4)
        ax.text(1.14, yy, txt, ha="left", va="center", fontsize=6.8,
                color=SUB, zorder=4)

    panel_card(ax, 9.30, 5.15, 3.30, 1.20, fc="#f5f5f5", ec=HAIR, lw=0.8,
               rounding=0.07, zorder=2)
    ax.text(7.82, 5.62, "Federated setting", ha="left", va="center",
            fontsize=7.6, fontweight="bold", color=INK, zorder=4)
    setting = [r"$K$ = 100 clients, fraction $C$ = 0.10 per round",
               r"$E$ = 2 local epochs, batch $B$ = 32, lr 0.01",
               f"payload {payload:.1f} MB down / {payload:.1f} MB up"]
    for i, txt in enumerate(setting):
        ax.text(7.82, 5.32 - 0.27 * i, txt, ha="left", va="center",
                fontsize=6.8, color=SUB, zorder=4)

    # ---- clients and the arrow fan ---------------------------------------
    cards = [client_card(ax, cx, col, nm, c) for cx, col, nm, c
             in zip(CLIENT_X, CLIENT_COL, CLIENT_NAME, counts)]

    y_srv = SERVER["y"] - SERVER["h"] / 2
    y_cli = CLIENT_Y + CLIENT_H / 2
    for i, card in enumerate(cards):
        hub = 4.30 + 0.72 * i
        arrow(ax, (hub - 0.15, y_srv), (card["x"] - 0.38, y_cli),
              color=BLUE, lw=1.05, rad=0.10, ms=10, gap=0.05, zorder=5)
        arrow(ax, (card["x"] + 0.38, y_cli), (hub + 0.15, y_srv),
              color=UP, lw=1.05, ls=(0, (4, 2.4)), rad=0.10, ms=10, gap=0.05,
              zorder=5)

    # ---- one round, unrolled --------------------------------------------
    panel_card(ax, 5.70, 1.05, 10.50, 1.40, fc="#f5f5f5", ec=HAIR, lw=0.8,
               rounding=0.07, zorder=1)
    ax.text(0.62, 1.60, r"One communication round  $t \rightarrow t+1$",
            ha="left", va="center", fontsize=7.6, fontweight="bold",
            color=INK, zorder=4)

    stages = [("1 · Broadcast", "server sends $w_t$\nto the cohort $S_t$", BLUE),
              ("2 · Local training", "$E$ epochs of SGD\non private $D_k$", GREEN),
              ("3 · Upload", "compressed update\n$\\Delta w^{k}_{t}$ and $n_k$", UP),
              ("4 · Aggregate", "sample-weighted mean\ngives $w_{t+1}$", BLUE)]
    sx = [0.925, 3.425, 5.925, 8.425]
    for (title, body, col), x0 in zip(stages, sx):
        text_card(ax, x0, 0.50, 2.05, 0.76, body, fc="white", ec=HAIR,
                  fs=6.4, color=SUB, ha="center", va="center", pad=0.09,
                  rounding=0.055, zorder=3, title=title, title_fs=7.0,
                  title_color=col, linespacing=1.30)
    for x0 in sx[:-1]:
        arrow(ax, (x0 + 2.05, 0.88), (x0 + 2.50, 0.88), color=MUTED, lw=1.0,
              ms=9, gap=0.07, zorder=4)

    savefig(fig, "486_fed_architecture.png")


if __name__ == "__main__":
    main()
