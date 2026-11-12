"""Attribute co-occurrence network on a ring, edge weight = pair frequency.

Counts come from a small latent-factor model over three attribute families
(colour, material, appearance) plus a handful of deliberately strong
cross-family pairs, so the graph has block structure instead of noise.  Only
the strongest pairs are drawn; the rest would turn the disc into felt.

Edges are quadratic Beziers pulled toward the centre, so the ring stays
readable where many chords cross.  Stroke width and opacity both encode the
pair count, node area encodes the attribute's own frequency, and the legend
in the bottom-right corner is built from real values taken out of the drawn
data -- the thin / medium / thick samples are the lightest, the median and
the heaviest edge actually on the canvas.

Equal-aspect inch canvas (DESIGN_SYSTEM section 1.2 exception E3: a circular
main panel cannot be squeezed into the 0.55-0.80 band).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from pubstyle import set_pub_style, savefig, lighten

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#d5d5d5"
CARD = "#f5f5f5"
EDGE = "#8a8a8a"

FAMILIES = [("colour", "#3b5b92",
             ["white", "black", "red", "green", "blue", "brown"]),
            ("material", "#2e7d5b",
             ["wooden", "metal", "plastic", "glass", "ceramic"]),
            ("appearance", "#c9a227",
             ["shiny", "dirty", "empty", "open", "round", "tall", "small"])]
# pairs the corpus really does favour across families
BRIDGES = {("wooden", "brown"): 3.1, ("metal", "shiny"): 2.9,
           ("glass", "empty"): 2.4, ("ceramic", "white"): 2.6,
           ("plastic", "small"): 1.9, ("black", "shiny"): 2.2,
           ("open", "empty"): 2.0, ("round", "ceramic"): 1.7,
           ("tall", "glass"): 1.6, ("dirty", "brown"): 1.5}

SEED = 1538  # original id; figure number is 1133
W, H = 7.6, 6.8
CX, CY, RING = 3.80, 3.60, 1.98
TOP_K = 40


def stroke(w, lo, hi):
    """Log-scaled width and opacity: the counts span half an order of
    magnitude, so a linear ramp would leave everything but the top pair
    invisible."""
    t = np.log(w / lo) / np.log(hi / lo)
    return 0.60 + 3.50 * t, 0.28 + 0.52 * t


def build_counts(rng):
    """Symmetric co-occurrence counts with family blocks and bridges."""
    words, fam, col = [], [], []
    for fi, (_, c, ws) in enumerate(FAMILIES):
        words += ws
        fam += [fi] * len(ws)
        col += [c] * len(ws)
    n = len(words)
    fam = np.asarray(fam)
    base = rng.gamma(2.2, 26.0, size=(n, n))
    base = (base + base.T) / 2.0
    within = (fam[:, None] == fam[None, :]).astype(float)
    weight = 0.55 + 1.05 * within
    idx = {w: i for i, w in enumerate(words)}
    for (a, b), boost in BRIDGES.items():
        weight[idx[a], idx[b]] *= boost
        weight[idx[b], idx[a]] *= boost
    popular = rng.gamma(4.0, 0.30, size=n)
    counts = base * weight * np.sqrt(np.outer(popular, popular))
    counts = np.round(counts).astype(int)
    np.fill_diagonal(counts, 0)
    return words, np.asarray(col), counts


def ring_positions(n):
    ang = np.pi / 2 - np.arange(n) * 2 * np.pi / n
    return np.column_stack([CX + RING * np.cos(ang), CY + RING * np.sin(ang)]),\
        ang


def chord(ax, p, q, lw, alpha):
    mid = (p + q) / 2.0
    ctrl = np.array([CX, CY]) + (mid - np.array([CX, CY])) * 0.28
    path = Path([tuple(p), tuple(ctrl), tuple(q)],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=EDGE,
                           linewidth=lw, alpha=alpha, capstyle="round",
                           zorder=2))


def main():
    set_pub_style()
    rng = np.random.default_rng(SEED)
    words, colors, counts = build_counts(rng)
    n = len(words)
    pos, ang = ring_positions(n)

    iu, ju = np.triu_indices(n, k=1)
    wts = counts[iu, ju]
    order = np.argsort(wts)[::-1][:TOP_K]
    sel_i, sel_j, sel_w = iu[order], ju[order], wts[order]
    freq = counts.sum(axis=1)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    lo, hi = float(sel_w.min()), float(sel_w.max())
    for i, j, w in zip(sel_i, sel_j, sel_w):
        chord(ax, pos[i], pos[j], *stroke(float(w), lo, hi))

    rmin, rmax = 0.062, 0.145
    fnorm = (freq - freq.min()) / max(np.ptp(freq), 1)
    for k in range(n):
        r = rmin + (rmax - rmin) * np.sqrt(fnorm[k])
        ax.add_patch(Circle(tuple(pos[k]), r, facecolor=lighten(colors[k],
                                                                0.55),
                            edgecolor=colors[k], linewidth=1.0, zorder=5))
        a = ang[k]
        lx = CX + (RING + r + 0.12) * np.cos(a)
        ly = CY + (RING + r + 0.12) * np.sin(a)
        ha = "left" if np.cos(a) > 0.08 else \
             "right" if np.cos(a) < -0.08 else "center"
        va = "bottom" if np.sin(a) > 0.30 else \
             "top" if np.sin(a) < -0.30 else "center"
        ax.text(lx, ly, words[k], ha=ha, va=va, fontsize=7.4,
                color=INK, zorder=6,
                path_effects=[pe.withStroke(linewidth=2.6,
                                            foreground="white")])

    # ---- headings ----------------------------------------------------------
    ax.text(0.42, 6.52, "Attribute co-occurrence in the region descriptions",
            ha="left", va="center", fontsize=11.4, fontweight="bold",
            color=INK)
    ax.text(0.42, 6.28, f"the {TOP_K} strongest of "
                        f"{n * (n - 1) // 2} attribute pairs; chords bend "
                        f"toward the centre, stroke weight and opacity both "
                        f"grow with the pair count",
            ha="left", va="center", fontsize=7.6, color=MUTED)

    # family key, bottom-left corner, where the ring leaves the canvas empty
    ax.text(0.46, 1.32, "Attribute family", ha="left", va="center",
            fontsize=7.0, fontweight="bold", color=RULE, zorder=6)
    for k, (name, color, ws) in enumerate(FAMILIES):
        y = 1.08 - k * 0.24
        ax.add_patch(Circle((0.54, y), 0.070, facecolor=lighten(color, 0.55),
                            edgecolor=color, linewidth=1.0, zorder=6))
        ax.text(0.70, y, f"{name}  ({len(ws)} words)", ha="left", va="center",
                fontsize=7.0, color=SUB, zorder=6)

    # ---- legend built from values that are actually on the canvas ----------
    lx0, ly0, lw_, lh = 5.28, 0.30, 2.02, 1.42
    ax.add_patch(FancyBboxPatch((lx0, ly0), lw_, lh,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=CARD, edgecolor=HAIR, linewidth=0.8,
                                zorder=7))
    ax.text(lx0 + 0.14, ly0 + lh - 0.18, "Pair count (edge weight)",
            ha="left", va="center", fontsize=7.0, fontweight="bold",
            color=RULE, zorder=8)
    med = int(sel_w[np.argmin(np.abs(sel_w - np.median(sel_w)))])
    for k, value in enumerate([int(lo), med, int(hi)]):
        y = ly0 + lh - 0.42 - k * 0.24
        lw_k, alpha_k = stroke(float(value), lo, hi)
        ax.plot([lx0 + 0.16, lx0 + 0.86], [y, y], color=EDGE, lw=lw_k,
                alpha=alpha_k, solid_capstyle="round", zorder=8)
        ax.text(lx0 + 0.96, y, f"{value:,} mentions", ha="left", va="center",
                fontsize=6.8, color=SUB, zorder=8)
    ax.plot([lx0 + 0.14, lx0 + lw_ - 0.14], [ly0 + 0.42] * 2, color=HAIR,
            lw=0.7, zorder=8)
    ax.text(lx0 + 0.14, ly0 + 0.28, "node area  ~  attribute frequency",
            ha="left", va="center", fontsize=6.8, color=SUB, zorder=8)
    ax.text(lx0 + 0.14, ly0 + 0.13,
            f"{freq.min():,} to {freq.max():,} mentions per word", ha="left",
            va="center", fontsize=6.8, color=FAINT, zorder=8)

    savefig(fig, "1133_wm_data_cooccurrence.png")


if __name__ == "__main__":
    main()
