"""Transformer encoder block: whole-stack overview plus a zoomed-in block.

The left column is the N-times repeated stack from tokens to task head; a
dashed callout lifts one block into the wide detail panel on the right,
where the pre-norm ordering is drawn explicitly: LayerNorm, the Q/K/V
projections feeding h parallel scaled-dot-product heads, the concat + output
projection, the residual add, then the second LayerNorm and the FFN with its
own residual.  Two auxiliary strips at the bottom carry the attention
equation, the hyper-parameters and four small attention maps.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
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
CMAP_BLUE = LinearSegmentedColormap.from_list("b", PALETTES["cb_sequential_blue"])


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
          text_color="white", fontsize=8, sub_color=None, lw=LW, ls="-",
          alpha=1.0, zorder=3, rounding=R, hatch=None, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=color,
        edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, hatch=hatch,
        zorder=zorder))
    if label:
        ax.text(x, y + (0.105 if sub else 0.0), label, ha="center",
                va="center", color=text_color, fontsize=fontsize,
                fontweight=weight, zorder=zorder + 2)
    if sub:
        ax.text(x, y - 0.115, sub, ha="center", va="center",
                color=sub_color or text_color, fontsize=6.5, alpha=0.92,
                zorder=zorder + 2)
    return anchors(x, y, w, h)


def stack_block(ax, x, y, w, h, label="", color=ATT, n=3, off=0.07,
                fontsize=7.4, zorder=3, ec="white"):
    """Several offset rounded boxes: 'this module runs in parallel/repeats'."""
    for i in range(n - 1, 0, -1):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + i * off, y - h / 2 + i * off), w, h,
            boxstyle=f"round,pad=0,rounding_size={R}", facecolor=color,
            edgecolor=ec, linewidth=0.7, alpha=0.45 + 0.15 * (n - 1 - i),
            zorder=zorder + (n - i)))
    return block(ax, x, y, w, h, label, color, fontsize=fontsize,
                 zorder=zorder + n)


def arrow(ax, p0, p1, color=INK, rad=0.0, lw=1.2, ls="-", ms=12, zorder=6,
          alpha=1.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, linestyle=ls, color=color, alpha=alpha,
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


def op_circle(ax, x, y, sym="+", r=0.135, color=INK, fc="white", fontsize=9,
              zorder=7):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=color, lw=1.0,
                        zorder=zorder))
    ax.text(x, y + 0.004, sym, ha="center", va="center", fontsize=fontsize,
            color=color, zorder=zorder + 1)
    return dict(x=x, y=y, c=(x, y), l=(x - r, y), r=(x + r, y),
                t=(x, y + r), b=(x, y - r))


def tensor_label(ax, x, y, text, color=MUTED, fontsize=6.5, ha="center",
                 va="center", **kw):
    return ax.text(x, y, text, ha=ha, va=va, color=color, fontsize=fontsize,
                   **kw)


def panel(ax, x, y, w, h, fc=SOFT, ec=LINE, lw=0.8, ls="-", zorder=1,
          rounding=0.08):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=fc,
        edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zorder))
    return anchors(x, y, w, h)


def titles(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=13.5, fontweight="bold", color=INK,
            ha="left", va="center")
    if sub:
        ax.text(x, y - 0.32, sub, fontsize=8.6, color=MUTED, ha="left",
                va="center")


# ---- figure ----------------------------------------------------------------
W, H = 13.4, 7.7
SPINE = 4.65          # y of the detail flow
CX = 1.45             # x of the overview column


def overview(ax):
    ax.text(CX, 6.62, "Full stack", fontsize=8.6, fontweight="bold",
            color=INK, ha="center", va="center")
    inp = block(ax, CX, 1.20, 2.05, 0.44, "Input tokens", DATA, fontsize=7.6)
    emb = block(ax, CX, 2.05, 2.05, 0.44, "Embed + positional enc.", AUX,
                fontsize=7.6)
    blocks = [block(ax, CX, 2.95 + 0.52 * k, 2.05, 0.44,
                    "Transformer block", ENC, fontsize=7.6) for k in range(3)]
    nrm = block(ax, CX, 4.95, 2.05, 0.44, "Final LayerNorm", DATA,
                fontsize=7.6)
    head = block(ax, CX, 5.80, 2.05, 0.44, "Task head", DEC, fontsize=7.6)

    chain = [inp, emb] + blocks + [nrm, head]
    for a, b in zip(chain[:-1], chain[1:]):
        arrow(ax, (CX, a["y"] + 0.22 + 0.02), (CX, b["y"] - 0.22 - 0.02),
              color=MUTED, lw=1.0, ms=10)

    ax.text(0.24, 3.47, "× N = 12", rotation=90, fontsize=7.6, color=ENC,
            fontweight="bold", ha="center", va="center")
    ax.plot([0.42, 0.42], [2.70, 4.24], color=ENC, lw=1.0)
    for yy in (2.70, 4.24):
        ax.plot([0.42, 0.52], [yy, yy], color=ENC, lw=1.0)

    top = blocks[-1]
    ax.add_patch(FancyBboxPatch(
        (CX - 1.14, top["y"] - 0.34), 2.28, 0.68,
        boxstyle=f"round,pad=0,rounding_size=0.07", facecolor="none",
        edgecolor=ATT, linewidth=1.1, linestyle=DASH, zorder=8))
    for y0, y1 in [(top["y"] + 0.34, 6.48), (top["y"] - 0.34, 2.82)]:
        ax.plot([CX + 1.14, 3.18], [y0, y1], color=ATT, lw=0.9, ls=DASH,
                zorder=2)


def detail(ax):
    panel(ax, 8.06, 4.65, 9.76, 3.66, fc="#fbfcfd", ec="#cfd6de")
    ax.text(3.30, 6.28, "One block, expanded  ·  pre-norm ordering",
            fontsize=8.6, fontweight="bold", color=INK, ha="left",
            va="center")

    # entry / exit
    ax.text(3.42, SPINE + 0.20, r"$z_{l-1}$", fontsize=8.5, color=INK,
            ha="center", va="center")
    ax.plot([3.42], [SPINE], marker="o", ms=4, color=INK, zorder=7)
    ln1 = block(ax, 4.02, SPINE, 0.50, 0.90, "Layer\nNorm", DATA,
                fontsize=6.8)
    arrow(ax, (3.46, SPINE), (ln1["l"][0] - 0.03, SPINE))

    # multi-head attention group
    panel(ax, 6.72, SPINE - 0.03, 4.36, 2.46, fc="#fdf6ee", ec=ATT, ls=DASH,
          lw=1.1, zorder=1)
    ax.text(4.60, SPINE + 1.01, "Multi-Head Self-Attention", fontsize=7.8,
            fontweight="bold", color=ATT, ha="left", va="center")

    xs = 4.72
    arrow(ax, (ln1["r"][0] + 0.03, SPINE), (xs, SPINE), ms=10)
    ax.plot([xs, xs], [4.00, 5.30], color=INK, lw=1.1, zorder=6)
    ax.plot([xs], [SPINE], marker="o", ms=3.6, color=INK, zorder=7)
    qkv = []
    for name, yy in [("$W^Q$", 5.30), ("$W^K$", SPINE), ("$W^V$", 4.00)]:
        b = block(ax, 5.40, yy, 0.64, 0.38, name, ATT, fontsize=7.6)
        arrow(ax, (xs, yy), (b["l"][0] - 0.03, yy), ms=10)
        qkv.append(b)
    tensor_label(ax, 5.40, 4.00 - 0.36, "L × $d_k$,  $d_k$ = d/h")

    heads = stack_block(ax, 6.86, SPINE, 1.20, 1.10,
                        "Scaled dot-\nproduct\nattention", ATT, n=4,
                        fontsize=7.0)
    for b in qkv:
        arrow(ax, (b["r"][0] + 0.03, b["y"]), (heads["l"][0] - 0.03, SPINE),
              ms=10)
    tensor_label(ax, 6.86, SPINE - 0.78, "h = 8 heads in parallel",
                 color=ATT)

    cat = block(ax, 8.42, SPINE, 0.66, 1.10, "Concat\n+ $W^O$", ATT,
                fontsize=7.2)
    arrow(ax, (heads["r"][0] + 0.10, SPINE), (cat["l"][0] - 0.03, SPINE),
          ms=10)
    tensor_label(ax, 8.42, SPINE - 0.78, "L × d")

    add1 = op_circle(ax, 9.38, SPINE, "+")
    arrow(ax, (cat["r"][0] + 0.03, SPINE), (add1["l"][0] - 0.03, SPINE))

    ln2 = block(ax, 10.02, SPINE, 0.50, 0.90, "Layer\nNorm", DATA,
                fontsize=6.8)
    arrow(ax, (add1["r"][0] + 0.03, SPINE), (ln2["l"][0] - 0.03, SPINE),
          ms=10)

    ffn = block(ax, 11.10, SPINE, 1.32, 1.10, "", ATT)
    ax.text(11.10, SPINE + 0.34, "Feed-Forward", ha="center", va="center",
            color="white", fontsize=7.6, fontweight="bold", zorder=6)
    for k, t in enumerate(["Linear  d → 4d", "GELU", "Linear  4d → d"]):
        ax.text(11.10, SPINE + 0.10 - 0.20 * k, t, ha="center", va="center",
                color="white", fontsize=6.2, alpha=0.95, zorder=6)
    arrow(ax, (ln2["r"][0] + 0.03, SPINE), (ffn["l"][0] - 0.03, SPINE),
          ms=10)

    add2 = op_circle(ax, 12.06, SPINE, "+")
    arrow(ax, (ffn["r"][0] + 0.03, SPINE), (add2["l"][0] - 0.03, SPINE))
    arrow(ax, (add2["r"][0] + 0.03, SPINE), (12.66, SPINE))
    ax.text(12.72, SPINE + 0.20, r"$z_{l}$", fontsize=8.5, color=INK,
            ha="center", va="center")
    ax.plot([12.72], [SPINE], marker="o", ms=4, color=INK, zorder=7)

    # residual paths
    route(ax, [(3.64, SPINE), (3.64, 5.92), (9.38, 5.92),
               (9.38, add1["t"][1] + 0.03)], color=ENC, lw=1.1, ms=11)
    ax.text(6.55, 6.00, "residual", fontsize=6.5, color=ENC, ha="center",
            va="bottom")
    route(ax, [(9.70, SPINE), (9.70, 5.50), (12.06, 5.50),
               (12.06, add2["t"][1] + 0.03)], color=ENC, lw=1.1, ms=11)
    ax.text(10.88, 5.58, "residual", fontsize=6.5, color=ENC, ha="center",
            va="bottom")
    for x0 in (3.64, 9.70):
        ax.plot([x0], [SPINE], marker="o", ms=3.6, color=ENC, zorder=7)


def bottom(fig, ax):
    panel(ax, 5.35, 1.66, 4.40, 2.16)
    ax.text(3.32, 2.56, "Definitions", fontsize=8.0, fontweight="bold",
            color=INK, ha="left", va="center", zorder=4)
    ax.text(5.35, 2.16,
            r"$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left("
            r"\frac{QK^{\top}}{\sqrt{d_k}}\right)V$",
            fontsize=10.5, color=INK, ha="center", va="center", zorder=4)
    specs = ["d = 768,   h = 8,   $d_k$ = 96,   L = 197 tokens",
             "FFN expansion 4×, GELU, dropout 0.1",
             "Pre-norm: LayerNorm before every sub-layer",
             "7.1 M parameters per block,  85 M total"]
    for k, s in enumerate(specs):
        ax.text(3.32, 1.72 - 0.24 * k, "•  " + s, fontsize=7.0, color=INK,
                ha="left", va="center", zorder=4)

    panel(ax, 10.42, 1.66, 5.20, 2.16)
    ax.text(7.94, 2.56, "Attention maps, layer 6", fontsize=8.0,
            fontweight="bold", color=INK, ha="left", va="center", zorder=4)
    tensor_label(ax, 7.94, 0.78, "rows = query token, columns = key token",
                 ha="left")

    rng = np.random.default_rng(11)
    n = 24
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    maps = [np.exp(-((i - j) ** 2) / 6.0),
            np.exp(-j / 2.2) + 0.05,
            np.exp(-((i // 6 - j // 6) ** 2) / 0.5) * 0.9 + 0.04,
            0.3 + 0.7 * np.exp(-((i - j - 4) ** 2) / 40.0)]
    for k, m in enumerate(maps):
        m = m + 0.05 * rng.random((n, n))
        m = m / m.sum(axis=1, keepdims=True)
        axm = fig.add_axes([(8.10 + 1.20 * k) / W, 1.02 / H,
                            1.00 / W, 1.00 / H])
        axm.imshow(m, cmap=CMAP_BLUE, aspect="equal",
                   vmax=np.percentile(m, 99))
        axm.set_xticks([])
        axm.set_yticks([])
        for sp in axm.spines.values():
            sp.set_linewidth(0.6)
            sp.set_edgecolor("#b9c2cc")
        axm.set_title(f"head {k + 1}", fontsize=6.2, color=MUTED, pad=2)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    titles(ax, 0.30, 7.32,
           "Transformer encoder block: stack overview and expanded internals",
           "Each block is pre-norm; the dashed callout expands the top block "
           "into its attention and feed-forward sub-layers")
    overview(ax)
    detail(ax)
    bottom(fig, ax)
    savefig(fig, "264_net_transformer_block.png")


if __name__ == "__main__":
    main()
