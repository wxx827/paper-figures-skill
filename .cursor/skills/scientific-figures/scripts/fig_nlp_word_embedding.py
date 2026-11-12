"""Figure 465 - Word embedding space with an analogy pair.

Main panel: 80 word vectors generated in 64 dimensions and projected with PCA,
coloured by semantic category and wrapped in convex hulls.  Eight
representative words carry elbow leader labels whose vertical order is settled
by two monotone sweeps, so the leaders can never cross.  The four analogy
words sit on chips, joined by two arrows that are parallel *by construction*:
``v(king) - v(man)`` and ``v(queen) - v(woman)`` are the same 64-d vector, and
PCA is linear, so the projection preserves the equality exactly.

Right column: nearest-neighbour cards, ranked by true cosine similarity in the
original 64-d space rather than in the projected plane.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig, lighten, darken, annotate_box
from figkit import inch_axes, panel_label_inches

# ---------------------------------------------------------------------------
# Shared NLP-batch vocabulary (464-473)
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
FAINT = "#8a8a8a"
HAIR = "#cccccc"
TRACK = "#ececec"
ACCENT = "#b23a48"

MONO = {"fontfamily": "monospace"}
CHAR_W = 0.6022 / 72.0        # DejaVu Sans Mono advance, inches per pt
CHIP_H = 0.205                # batch-wide chip height, inches
CHIP_PAD = 0.058              # batch-wide horizontal padding, inches
CHIP_R = 0.045                # batch-wide corner radius, inches

CATS = {
    "animals": "#b23a48",
    "places": "#3b5b92",
    "motion verbs": "#2e7d5b",
    "food": "#c9a227",
    "kinship": "#8e6bbf",
}
WORDS = {
    "animals": ["tiger", "falcon", "otter", "lynx", "heron", "ibex",
                "marten", "gannet", "badger", "osprey", "jackal", "weasel",
                "egret", "bison", "stoat", "cormorant"],
    "places": ["lisbon", "norway", "kyoto", "andorra", "helsinki", "morocco",
               "quebec", "bruges", "iceland", "seville", "kerala", "tunis",
               "oslo", "bergen", "malta", "cairo"],
    "motion verbs": ["sprinting", "drifted", "hurtled", "ambled", "glided",
                     "lunged", "swerved", "trudged", "vaulted", "scurried",
                     "coasted", "darted", "waded", "loped", "careened",
                     "sauntered"],
    "food": ["risotto", "paprika", "brioche", "tamarind", "gnocchi",
             "saffron", "chorizo", "focaccia", "tahini", "halloumi",
             "polenta", "harissa", "quinoa", "pesto", "miso", "ricotta"],
    "kinship": ["king", "man", "woman", "queen", "prince", "duchess", "heir",
                "monarch", "uncle", "niece", "widow", "regent", "empress",
                "baron", "consort", "kin"],
}
LABELLED = ["tiger", "osprey", "kyoto", "iceland", "sprinting", "trudged",
            "risotto", "saffron"]
ANALOGY = ["man", "king", "woman", "queen"]
QUERIES = ["king", "risotto", "sprinting"]
DIM = 64


def chip_width(text, fs=7.0, pad=CHIP_PAD):
    """Exact chip width in inches for monospaced text of ``fs`` points."""
    return 2.0 * pad + len(text) * CHAR_W * fs


def chip(ax, x, y, text, fs=7.0, h=CHIP_H, fc="#f0f0f0", ec=HAIR, color=INK,
         weight="normal", pad=CHIP_PAD, lw=0.7, zorder=6):
    """Rounded token chip with its lower-left corner at ``(x, y)`` in inches."""
    w = chip_width(text, fs, pad)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={CHIP_R}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        mutation_aspect=1.0))
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fs, color=color, fontweight=weight, zorder=zorder + 2,
            **MONO)
    return w


def build_space(rng):
    """64-d vectors with a planted analogy, plus their PCA projection."""
    centres = rng.standard_normal((len(CATS), DIM)) * 0.92
    vecs, names, cats = [], [], []
    for ci, (cat, words) in enumerate(WORDS.items()):
        v = centres[ci] + 0.60 * rng.standard_normal((len(words), DIM))
        vecs.append(v)
        names += words
        cats += [cat] * len(words)
    V = np.vstack(vecs)
    idx = {n: i for i, n in enumerate(names)}

    # plant the analogy: king - man == queen - woman, exactly
    royal = 0.62 * rng.standard_normal(DIM)
    V[idx["king"]] = V[idx["man"]] + royal
    V[idx["queen"]] = V[idx["woman"]] + royal

    Vc = V - V.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Vc, full_matrices=False)
    P = Vc @ vt[:2].T
    return V, P, names, np.array(cats), idx


def cosine_neighbours(V, names, query, k=5):
    """Top-``k`` cosine neighbours of ``query`` in the original 64-d space."""
    N = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    qi = names.index(query)
    sim = N @ N[qi]
    sim[qi] = -np.inf
    order = np.argsort(sim)[::-1][:k]
    return [(names[i], float(sim[i])) for i in order]


def place_cluster_name(P, poly, xlim, ylim, span_x, span_y, off=0.055):
    """Pick the side of a hull with the most free space for its name.

    Four candidates (above, below, left, right of the hull bounding box) are
    scored by their normalised distance to the nearest word of *any* cluster
    and rejected if they would fall outside the view; the winner is the one
    that sits furthest from every point, which keeps names off the clouds no
    matter how the projection comes out.
    """
    bx0, bx1 = poly[:, 0].min(), poly[:, 0].max()
    by0, by1 = poly[:, 1].min(), poly[:, 1].max()
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    cands = [((cx, by1 + off * span_y), "center", 0.13, 0.05),
             ((cx, by0 - off * span_y), "center", 0.13, 0.05),
             ((bx0 - off * span_x, cy), "right", 0.02, 0.03),
             ((bx1 + off * span_x, cy), "left", 0.02, 0.03)]
    best, best_s = None, -np.inf
    for (px, py), ha, mx, my in cands:
        inside = (xlim[0] + mx * (xlim[1] - xlim[0]) <= px <=
                  xlim[1] - mx * (xlim[1] - xlim[0]) and
                  ylim[0] + my * (ylim[1] - ylim[0]) <= py <=
                  ylim[1] - my * (ylim[1] - ylim[0]))
        s = np.hypot((P[:, 0] - px) / span_x, (P[:, 1] - py) / span_y).min()
        s -= 0.0 if inside else 10.0
        if s > best_s:
            best, best_s = (px, py, ha), s
    return best


def leader_labels(ax, items, dx=0.62, min_gap=0.44, lw=0.7, fs=7.0):
    """Elbow labels beside a scatter, de-overlapped by two monotone sweeps.

    Same ordering argument as ``figkit.place_outside_labels`` - both passes
    only ever preserve the original y ordering within a side, so two leaders
    on the same side cannot swap and therefore cannot cross - but the anchor
    is the data point itself rather than a wedge angle, which keeps the
    leaders short enough to sit inside a scatter panel.
    """
    anchors = []
    for side in (+1, -1):
        grp = [it for it in items if it[4] == side]
        if not grp:
            continue
        grp.sort(key=lambda it: -it[1])
        ys = [it[1] for it in grp]
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - min_gap)
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = max(ys[i], ys[i + 1] + min_gap)
        for (x, y, text, col, _s), yl in zip(grp, ys):
            xk, xt = x + side * dx * 0.52, x + side * dx
            ax.plot([x, xk, xt], [y, yl, yl], lw=lw, color="#9a9a9a",
                    zorder=4, solid_capstyle="round")
            ax.plot([x], [y], marker="o", ms=3.4, mfc=col, mec="white",
                    mew=0.6, zorder=5)
            ax.text(xt + side * 0.07, yl, text, ha="left" if side > 0
                    else "right", va="center", fontsize=fs, color=INK,
                    zorder=5, **MONO)
            anchors.append((xt + side * (0.07 + 0.5 * dx), yl))
    return anchors


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(465)
    V, P, names, cats, idx = build_space(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = fig.add_axes([0.055, 0.085, 0.605, 0.825])

    span_x = float(P[:, 0].max() - P[:, 0].min())
    span_y = float(P[:, 1].max() - P[:, 1].min())
    ax.set_xlim(P[:, 0].min() - 0.16 * span_x, P[:, 0].max() + 0.16 * span_x)
    ax.set_ylim(P[:, 1].min() - 0.10 * span_y, P[:, 1].max() + 0.16 * span_y)

    # ---- hulls and points -------------------------------------------------
    polys = {}
    for cat, col in CATS.items():
        pts = P[cats == cat]
        polys[cat] = pts[ConvexHull(pts).vertices]
        ax.add_patch(Polygon(polys[cat], closed=True,
                             facecolor=lighten(col, 0.86),
                             edgecolor=lighten(col, 0.35), linewidth=0.9,
                             linestyle=(0, (4.0, 2.0)), zorder=1))
        ax.scatter(pts[:, 0], pts[:, 1], s=17, color=col, alpha=0.88,
                   edgecolor="white", linewidth=0.35, zorder=3)

    # ---- leader-labelled representative words -----------------------------
    xmid = float(np.median(P[:, 0]))
    items = []
    for w in LABELLED:
        i = idx[w]
        side = 1 if P[i, 0] >= xmid else -1
        items.append((P[i, 0], P[i, 1], w, CATS[cats[i]], side))
    anchors = leader_labels(ax, items, dx=0.070 * span_x,
                            min_gap=0.055 * span_y)

    # ---- cluster names, kept clear of both points and leader labels -------
    for cat, col in CATS.items():
        # the cluster's own words sit at a fixed offset from every candidate,
        # so only *other* clusters and the leader labels can break the tie
        occupied = np.vstack([P[cats != cat], np.asarray(anchors)])
        lx, ly, lha = place_cluster_name(occupied, polys[cat], ax.get_xlim(),
                                         ax.get_ylim(), span_x, span_y)
        ax.text(lx, ly, cat, ha=lha, va="center", fontsize=7.6,
                fontweight="bold", color=darken(col, 0.18), zorder=6)

    # ---- the analogy ------------------------------------------------------
    kin = CATS["kinship"]
    for a, b in (("man", "king"), ("woman", "queen")):
        p0, p1 = P[idx[a]], P[idx[b]]
        fa = FancyArrowPatch(
            tuple(p0), tuple(p1), arrowstyle="-|>", mutation_scale=12,
            linewidth=1.6, color=ACCENT, shrinkA=4, shrinkB=4,
            connectionstyle="arc3,rad=0.0", zorder=7)
        fa.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white",
                                           alpha=0.85)])
        ax.add_patch(fa)
    quad = np.array([P[idx[w]] for w in ANALOGY])
    qc = quad.mean(axis=0)
    for w, p in zip(ANALOGY, quad):
        d = p - qc
        d = d / (np.linalg.norm(d) + 1e-9)
        ax.plot([p[0]], [p[1]], marker="o", ms=5.0, mfc=kin, mec="white",
                mew=0.8, zorder=8)
        ax.annotate(w, xy=(p[0], p[1]),
                    xytext=(19.0 * d[0], 16.0 * d[1]),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=7.2, fontweight="bold",
                    color=darken(kin, 0.22), zorder=9,
                    bbox=dict(boxstyle="round,pad=0.24",
                              facecolor=lighten(kin, 0.88),
                              edgecolor=lighten(kin, 0.42), linewidth=0.7),
                    **MONO)

    # the explanation box goes wherever the scatter is emptiest, so it can
    # never land on a cluster if the seed changes
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    best, best_n = "upper left", None
    for loc, (fx, fy) in (("upper left", (0, 1)), ("upper right", (1, 1)),
                          ("lower left", (0, 0)), ("lower right", (1, 0))):
        cx = x0 + (0.30 if fx == 0 else 0.70) * (x1 - x0)
        cy = y0 + (0.30 if fy == 0 else 0.70) * (y1 - y0)
        n = int(((np.abs(P[:, 0] - cx) < 0.30 * (x1 - x0)) &
                 (np.abs(P[:, 1] - cy) < 0.30 * (y1 - y0))).sum())
        if best_n is None or n < best_n:
            best, best_n = loc, n
    resid = float(np.linalg.norm((P[idx["king"]] - P[idx["man"]]) -
                                 (P[idx["queen"]] - P[idx["woman"]])))
    annotate_box(ax, "king $-$ man $=$ queen $-$ woman\n"
                     f"parallel offsets, residual {resid:.1e}",
                 loc=best, fontsize=6.8, color=ACCENT,
                 edgecolor=lighten(ACCENT, 0.55))

    ax.set_xlabel("PCA component 1 (-)", fontsize=8.0)
    ax.set_ylabel("PCA component 2 (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.set_title(f"Embedding space of {len(names)} words, "
                 f"{DIM}-d projected to 2-d", fontsize=9.0,
                 fontweight="bold")

    # ---- nearest-neighbour cards -----------------------------------------
    axc = inch_axes(fig, (7.10, 0.52, 3.10, 5.30))
    CW, CH = axc.get_xlim()[1], axc.get_ylim()[1]
    axc.text(0.06, CH - 0.10, "Nearest neighbours", fontsize=9.0,
             fontweight="bold", ha="left", va="top", color=INK)
    axc.text(CW - 0.02, CH - 0.12, f"cosine, {DIM}-d", fontsize=6.8,
             ha="right", va="top", color=FAINT)

    card_h, gap = 1.44, 0.24
    top = CH - 0.38
    for qi, q in enumerate(QUERIES):
        y0 = top - (qi + 1) * card_h - qi * gap
        col = CATS[cats[names.index(q)]]
        axc.add_patch(FancyBboxPatch(
            (0.04, y0), CW - 0.08, card_h,
            boxstyle="round,pad=0,rounding_size=0.06", facecolor="#f7f7f7",
            edgecolor=HAIR, linewidth=0.7, zorder=2, mutation_aspect=1.0))
        chip(axc, 0.16, y0 + card_h - 0.34, q, fs=7.6,
             fc=lighten(col, 0.86), ec=lighten(col, 0.40),
             color=darken(col, 0.22), weight="bold")
        axc.text(CW - 0.18, y0 + card_h - 0.34 + CHIP_H / 2.0, "top-5",
                 fontsize=6.4, ha="right", va="center", color=FAINT)

        nb = cosine_neighbours(V, names, q, k=5)
        bx0, bx1 = 1.28, CW - 0.60
        for ri, (w, s) in enumerate(nb):
            ry = y0 + card_h - 0.60 - ri * 0.215
            axc.text(0.18, ry, f"{ri + 1}", fontsize=6.2, ha="left",
                     va="center", color=FAINT, **MONO)
            axc.text(0.36, ry, w, fontsize=6.8, ha="left", va="center",
                     color=SUB, **MONO)
            axc.add_patch(FancyBboxPatch(
                (bx0, ry - 0.043), bx1 - bx0, 0.086,
                boxstyle="round,pad=0,rounding_size=0.043",
                facecolor=TRACK, edgecolor="none", zorder=3,
                mutation_aspect=1.0))
            axc.add_patch(FancyBboxPatch(
                (bx0, ry - 0.043), max((bx1 - bx0) * s, 0.09), 0.086,
                boxstyle="round,pad=0,rounding_size=0.043", facecolor=col,
                edgecolor="none", zorder=4, mutation_aspect=1.0))
            axc.text(CW - 0.14, ry, f"{s:.3f}", fontsize=6.2, ha="right",
                     va="center", color=SUB, **MONO)

    panel_label_inches(fig, ax, "(a)", dx=-0.50, dy=0.10)
    panel_label_inches(fig, axc, "(b)", dx=-0.24, dy=0.02)
    savefig(fig, "465_nlp_word_embedding.png")


if __name__ == "__main__":
    main()
