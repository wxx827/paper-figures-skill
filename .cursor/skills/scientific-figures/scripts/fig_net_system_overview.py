"""End-to-end system overview: abstract flow on top, three zones below.

The top strip is the five-step story a reader can follow in one second;
underneath, the same system is opened into three tinted zones - data plane,
learning core, serving and feedback - each with its own internal chain.
Traffic between zones runs in the gutters between the zones and is coded by
line style: solid for live data, dash-dot for artifacts, dashed red for the
labelling feedback loop that closes the system.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pubstyle import set_pub_style, PALETTES, savefig

ENC = "#3b5b92"
DEC = "#2e7d5b"
ATT = "#c9a227"
LOSS = "#b23a48"
DATA = "#8c8c8c"
AUX = "#8e6bbf"
INK = "#24303c"
MUTED = "#6f7783"
SOFT = "#f2f4f7"
LINE = "#dde2e8"

R = 0.055
LW = 0.9
DASH = (0, (4, 2.6))
DASHDOT = (0, (5, 2, 1, 2))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def anchors(x, y, w, h):
    return dict(x=x, y=y, w=w, h=h, c=(x, y), l=(x - w / 2, y),
                r=(x + w / 2, y), t=(x, y + h / 2), b=(x, y - h / 2))


def block(ax, x, y, w, h, label="", color=ENC, sub=None, ec="white",
          text_color="white", fontsize=7.6, sub_color=None, lw=LW, ls="-",
          alpha=1.0, zorder=4, rounding=R):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder))
    if label:
        ax.text(x, y + (0.095 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.105, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.3, alpha=0.94,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0,
                        shrinkB=0, zorder=zorder)
    ax.add_patch(a)
    return a


def route(ax, pts, color=INK, lw=1.2, ls="-", ms=12, zorder=6):
    pts = [tuple(p) for p in pts]
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color, lw=lw, ls=ls, solid_capstyle="round",
                zorder=zorder)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, ms=ms,
          zorder=zorder)


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.10):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    return anchors(x, y, w, h)


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kw):
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kw)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- figure ----------------------------------------------------------------
W, H = 13.6, 8.2
ZTOP, ZBOT = 6.30, 1.93
ZY, ZH = (ZTOP + ZBOT) / 2, ZTOP - ZBOT
ZONES = [dict(cx=2.50, w=4.10, tag="A", name="Data plane",
              fc="#f5f8fb", ec="#cfdbe6", col=ENC),
         dict(cx=7.15, w=4.80, tag="B", name="Learning core",
              fc="#fdf8f2", ec="#ecd8bf", col=ATT),
         dict(cx=11.45, w=3.40, tag="C", name="Serving & feedback",
              fc="#f5faf6", ec="#cfe2d4", col=DEC)]

TOP = [(1.55, 1.60, "Sensor input", DATA), (3.45, 1.60, "Pre-processing", ENC),
       (7.15, 1.90, "Model core", ATT), (10.35, 1.60, "Post-processing", DEC),
       (12.35, 1.60, "Output products", DATA)]


def top_flow(ax):
    boxes = []
    for x, w, name, col in TOP:
        boxes.append(block(ax, x, 7.02, w, 0.56, name, col, fontsize=8.0))
    labels = ["raw frames", "mini-batches", "logits", "products"]
    for (a, b), lab in zip(zip(boxes[:-1], boxes[1:]), labels):
        arrow(ax, (a["r"][0] + 0.05, 7.02), (b["l"][0] - 0.05, 7.02),
              color="#8c97a3", lw=1.4, ms=13)
        if b["l"][0] - a["r"][0] > 0.9:
            ax.text((a["r"][0] + b["l"][0]) / 2, 7.20, lab, ha="center",
                    va="center", fontsize=6.5, color=MUTED)
    for x, *_ in TOP:
        ax.plot([x, x], [6.72, ZTOP + 0.02], color="#b9c4ce", lw=0.9,
                ls=(0, (2, 2.4)), zorder=2)


def zone_frames(ax):
    for z in ZONES:
        panel(ax, z["cx"], ZY, z["w"], ZH, fc=z["fc"], ec=z["ec"], lw=1.0)
        x0 = z["cx"] - z["w"] / 2 + 0.18
        ax.add_patch(FancyBboxPatch(
            (x0, 5.96), 0.28, 0.24,
            boxstyle="round,pad=0,rounding_size=0.05", facecolor=z["col"],
            edgecolor="none", zorder=4))
        ax.text(x0 + 0.14, 6.08, z["tag"], ha="center", va="center",
                color="white", fontsize=7.4, fontweight="bold", zorder=5)
        ax.text(x0 + 0.38, 6.08, z["name"], ha="left", va="center",
                fontsize=8.4, fontweight="bold", color=INK)


def zone_a(ax):
    cx = 2.50
    src = [block(ax, x, 5.42, 1.22, 0.44, n, DATA, fontsize=7.2)
           for x, n in [(1.13, "Camera"), (2.50, "IMU + GPS"),
                        (3.87, "HD map")]]
    chain = [block(ax, cx, 4.62, 3.10, 0.52, "Sync + calibrate", ENC,
                   fontsize=7.6),
             block(ax, cx, 3.85, 3.10, 0.52, "Augment + normalise", ENC,
                   fontsize=7.6),
             block(ax, cx, 3.08, 3.10, 0.52, "Shard writer", AUX,
                   fontsize=7.6),
             block(ax, cx, 2.31, 3.10, 0.52, "Object store", DATA,
                   sub="42 TB · sharded by flight", fontsize=7.4)]
    for s, xt in zip(src, [1.62, 2.50, 3.38]):
        arrow(ax, (s["x"], s["b"][1] - 0.03), (xt, chain[0]["t"][1] + 0.03),
              color=MUTED, lw=1.0, ms=10)
    for a, b in zip(chain[:-1], chain[1:]):
        arrow(ax, (cx, a["b"][1] - 0.03), (cx, b["t"][1] + 0.03), lw=1.2,
              ms=11)
    return chain


def zone_b(ax):
    cx = 7.15
    row = [block(ax, x, 5.42, 1.24, 0.62, n, c, fontsize=7.6)
           for x, n, c in [(5.62, "Encoder", ENC), (7.15, "Fusion", ATT),
                           (8.68, "Decoder", DEC)]]
    for a, b in zip(row[:-1], row[1:]):
        arrow(ax, (a["r"][0] + 0.04, 5.42), (b["l"][0] - 0.04, 5.42), ms=11)
    loss = block(ax, cx, 4.55, 3.10, 0.52, "Loss + AdamW", LOSS,
                 fontsize=7.6)
    ema = block(ax, cx, 3.80, 3.10, 0.52, "EMA teacher", AUX, fontsize=7.6)
    ckpt = block(ax, cx, 3.05, 3.10, 0.52, "Checkpoint store", DATA,
                 sub="every 2 000 steps", fontsize=7.4)
    ev = block(ax, cx, 2.30, 3.10, 0.52, "Eval harness", AUX,
               sub="6 benchmarks, nightly", fontsize=7.4)
    arrow(ax, (8.68, row[2]["b"][1] - 0.03), (8.68, loss["t"][1] + 0.03),
          lw=1.2, ms=11)
    for a, b in [(loss, ema), (ema, ckpt), (ckpt, ev)]:
        arrow(ax, (cx, a["b"][1] - 0.03), (cx, b["t"][1] + 0.03), lw=1.2,
              ms=11)
    route(ax, [(loss["l"][0] - 0.03, 4.55), (5.32, 4.55),
               (5.32, row[0]["b"][1] - 0.03)], color=LOSS, lw=1.1, ls=DASH,
          ms=11)
    ax.text(5.14, 4.98, "grads", fontsize=6.3, color=LOSS, ha="center",
            va="center", rotation=90)
    return row, ckpt


def zone_c(ax):
    cx = 11.45
    chain = [block(ax, cx, 5.42, 2.90, 0.52, "Quantise + export", AUX,
                   fontsize=7.6),
             block(ax, cx, 4.64, 2.90, 0.52, "Runtime engine", DEC,
                   sub="INT8 · 41 ms per frame", fontsize=7.4),
             block(ax, cx, 3.86, 2.90, 0.52, "Deployed fleet", DATA,
                   sub="240 airframes", fontsize=7.4),
             block(ax, cx, 3.08, 2.90, 0.52, "Drift monitor", ATT,
                   fontsize=7.6),
             block(ax, cx, 2.30, 2.90, 0.52, "Active-learning queue", LOSS,
                   fontsize=7.6)]
    for a, b in zip(chain[:-1], chain[1:]):
        arrow(ax, (cx, a["b"][1] - 0.03), (cx, b["t"][1] + 0.03), lw=1.2,
              ms=11)
    return chain


def cross_links(ax, a_chain, b_row, b_ckpt, c_chain):
    ay = a_chain[-1]["y"]
    route(ax, [(a_chain[-1]["r"][0] + 0.03, ay), (4.65, ay), (4.65, 5.42),
               (b_row[0]["l"][0] - 0.04, 5.42)], color=INK, lw=1.4, ms=13)
    ax.text(4.42, 3.95, "training shards", rotation=90, fontsize=6.5,
            color=INK, ha="center", va="center")

    cy = b_ckpt["y"]
    route(ax, [(b_ckpt["r"][0] + 0.03, cy), (9.65, cy), (9.65, 5.42),
               (c_chain[0]["l"][0] - 0.04, 5.42)], color=MUTED, lw=1.2,
          ls=DASHDOT, ms=12)
    ax.text(9.50, 4.30, "exported weights", rotation=90, fontsize=6.5,
            color=MUTED, ha="center", va="center")

    route(ax, [(11.45, c_chain[-1]["b"][1] - 0.03), (11.45, 1.72),
               (2.50, 1.72), (2.50, a_chain[-1]["b"][1] - 0.03)],
          color=LOSS, lw=1.2, ls=DASH, ms=12)
    ax.text(7.00, 1.60, "active-learning feedback  ·  newly labelled frames "
            "return to the store", fontsize=7.0, color=LOSS, ha="center",
            va="center")


def legend(ax):
    panel(ax, 10.18, 1.00, 5.95, 1.12, rounding=0.08)
    ax.text(7.42, 1.42, "Legend", fontsize=8.0, fontweight="bold", color=INK,
            ha="left", va="center", zorder=5)
    items = [(DATA, "data / storage"), (ENC, "encode · preprocess"),
             (ATT, "fusion · monitor"), (DEC, "decode · serve"),
             (AUX, "auxiliary tooling"), (LOSS, "loss · feedback"),
             ("solid", "live data flow"), ("dashdot", "artifact transfer"),
             ("dash", "feedback loop")]
    for k, (kind, txt) in enumerate(items):
        x = 7.42 + 2.00 * (k % 3)
        y = 1.16 - 0.28 * (k // 3)
        if kind == "solid":
            arrow(ax, (x - 0.19, y), (x + 0.19, y), color=INK, lw=1.4,
                  ms=11, zorder=5)
        elif kind == "dashdot":
            arrow(ax, (x - 0.19, y), (x + 0.19, y), color=MUTED, lw=1.2,
                  ls=DASHDOT, ms=11, zorder=5)
        elif kind == "dash":
            arrow(ax, (x - 0.19, y), (x + 0.19, y), color=LOSS, lw=1.2,
                  ls=DASH, ms=11, zorder=5)
        else:
            block(ax, x, y, 0.38, 0.24, "", kind, zorder=5)
        ax.text(x + 0.28, y, txt, ha="left", va="center", fontsize=7.0,
                color=INK, zorder=5)


def notes(ax):
    ax.text(0.50, 1.42, "Deployment notes", fontsize=8.0, fontweight="bold",
            color=INK, ha="left", va="center")
    lines = ["8 × A100 for the learning core; one full cycle takes 19 h.",
             "Runtime engine is pinned to INT8; FP16 kept as a fallback.",
             "Drift above 0.15 PSI pushes frames into the labelling queue."]
    for k, t in enumerate(lines):
        ax.text(0.50, 1.16 - 0.28 * k, "•  " + t, fontsize=7.0, color=INK,
                ha="left", va="center")


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.45, 7.92,
           "System overview: from sensors to a monitored, self-improving "
           "deployment",
           "Top strip is the abstract five-step flow; the three tinted zones "
           "below open the same system up into its running parts")
    top_flow(ax)
    zone_frames(ax)
    a_chain = zone_a(ax)
    b_row, b_ckpt = zone_b(ax)
    c_chain = zone_c(ax)
    cross_links(ax, a_chain, b_row, b_ckpt, c_chain)
    legend(ax)
    notes(ax)
    savefig(fig, "272_net_system_overview.png")


if __name__ == "__main__":
    main()
