"""Figure 1174 - Text-to-image retrieval: three queries, five candidates each.

Every row ranks the top five of a 360-image gallery for one caption.  A
candidate counts as a hit when the frame really was rendered from the scene
family the caption describes, so the bold border is a fact about the pixels,
not a decoration: the row's Recall@5 is the number of bold frames divided by
five, and the mean underneath is the same twelve hits over the same fifteen
slots.  The arithmetic is spelled out in the summary card and asserted in the
script, because a reader who divides will do it whether or not it closes.

The violin panel is the *same* similarity pool the ranking came from: the
markers riding on each violin are the five frames shown above it, and the
short vertical rule is the fifth-highest score, i.e. the cut that produced
that row.  Hits are deep blue discs, misses hollow grey ones - no red/green
pair, and no warm hue anywhere in the ranking.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig
from _wm_scenes import build, _EGO_ROOMS

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DEEP = "#2f4a75"
STEEL = "#3d6b8e"
PALE = "#cfdbe8"
MISS = "#9a9a9a"

# Cameras that keep every scene family inside the batch's warm-area budget:
# fields, wood and tungsten all drift orange if left on the presets.
CAMS = {
    "ego": {},
    "aerial": dict(sat=0.55, wb=(0.965, 1.000, 1.060), contrast=1.10),
    "shelf": dict(sat=0.58, wb=(0.950, 1.000, 1.070), contrast=1.10),
    "tabletop": dict(sat=0.70, wb=(0.960, 1.000, 1.058), contrast=1.12),
    "road": dict(sat=0.85, wb=(0.975, 1.000, 1.045)),
    "indoor": dict(sat=0.72, wb=(0.955, 1.000, 1.065), contrast=1.12),
}

# (caption, relevant scene, relevant kwargs, distractor pool, relevant mean)
QUERIES = [
    ("a kettle standing on a tiled kitchen counter",
     ("ego", dict(room="kitchen")),
     [("tabletop", {}), ("shelf", dict(shelves=3, plan=8)),
      ("road", dict(horizon=0.45))], 0.630),
    ("an aerial view of a road curving through farmland",
     ("aerial", dict(plan=19, pan=30.0)),
     [("indoor", {}), ("ego", dict(room="desk")),
      ("shelf", dict(shelves=4, plan=12))], 0.742),
    ("books packed tightly on a wooden shelf",
     ("shelf", dict(shelves=3, plan=23)),
     [("ego", dict(room="living")), ("road", dict(horizon=0.43)),
      ("tabletop", {})], 0.618),
]

N_POOL, N_REL, K = 360, 34, 5
W, H = 10.4, 6.4
PH, PW = 74, 106                        # candidate thumbnail raster

QX0, QWD = 0.34, 2.40
TX0, TWD, TGAP = 2.86, 1.17, 0.10
RX0 = TX0 + K * TWD + (K - 1) * TGAP + 0.16
ROW_TOP, ROW_H, ROW_GAP = 5.82, 1.02, 0.16
TH = 0.82


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, w, h, fc="white", ec=HAIR, lw=0.8, r=0.06, zorder=2):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)
    return p


def tick_mark(ax, x, y, s=0.048, color="white", lw=1.5, zorder=12):
    ax.plot([x - s, x - s * 0.22, x + s], [y + 0.02, y - s * 0.80,
                                           y + s * 0.92],
            color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder)


def rank_pool(rng, mu_rel):
    """One query's gallery: similarity per item plus its relevance flag."""
    rel = np.zeros(N_POOL, dtype=bool)
    rel[:N_REL] = True
    sim = np.where(rel, rng.normal(mu_rel, 0.082, N_POOL),
                   rng.normal(0.446, 0.112, N_POOL))
    sim = np.clip(sim, 0.02, 0.98)
    order = np.argsort(-sim)
    return sim, rel, order


def vary(kind, prm, j):
    """A different frame per rank, not the same frame with a new seed.

    Two candidates for one caption are two *photographs*, so the props move,
    the horizon shifts and the exposure changes; only the room stays put,
    because the caption is what pins it there.
    """
    p = dict(prm)
    r = np.random.default_rng(7000 + 61 * j + len(kind))
    if kind == "ego":
        room = _EGO_ROOMS[p["room"]]
        p["props"] = [(n, float(np.clip(u + r.uniform(-0.13, 0.13),
                                        0.07, 0.94)),
                       float(np.clip(d + r.uniform(-0.10, 0.10), 0.20, 0.90)),
                       f, c, float(s * r.uniform(0.86, 1.16)))
                      for (n, u, d, f, c, s) in room["props"]]
        p["horizon"] = float(room["horizon"] + r.uniform(-0.045, 0.045))
    elif kind == "aerial":
        p["plan"] = int(5 + 9 * j + 3 * (j % 2))
        p["pan"] = float(8.0 + 21.0 * j)
    elif kind == "shelf":
        p["plan"] = int(p.get("plan", 3) + 7 * j)
        p["shelves"] = 3 + (j % 2)
    elif kind == "road":
        p["horizon"] = float(0.415 + 0.018 * j)
        p["vpoint"] = float(0.44 + 0.05 * j)
        p["block"] = 11 + 2 * j
    elif kind in ("tabletop", "indoor"):
        p["horizon"] = float((0.34 if kind == "tabletop" else 0.54)
                             + r.uniform(-0.05, 0.05))
    cam = dict(CAMS[kind])
    cam["exposure"] = cam.get("exposure", 1.0) * float(r.uniform(0.95, 1.06))
    cam["contrast"] = cam.get("contrast", 1.08) * float(r.uniform(0.96, 1.07))
    return p, cam


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    rng = np.random.default_rng(1579)

    pools, rows = [], []
    for qi, (cap, relscene, distract, mu) in enumerate(QUERIES):
        sim, rel, order = rank_pool(rng, mu)
        top = order[:K]
        pools.append((sim, rel, top))
        hits = int(rel[top].sum())
        rows.append(hits)

    total = sum(rows)
    mean = total / float(K * len(QUERIES))
    assert abs(mean - sum(h / K for h in rows) / len(QUERIES)) < 1e-12

    # ---- ranked rows ------------------------------------------------------
    for qi, (cap, relscene, distract, mu) in enumerate(QUERIES):
        sim, rel, top = pools[qi]
        ry = ROW_TOP - (qi + 1) * ROW_H - qi * ROW_GAP
        box(ax, QX0 - 0.06, ry, RX0 + 0.86 - QX0 + 0.06, ROW_H, fc="#f7f7f7",
            ec="#ececec", lw=0.7, r=0.07, zorder=1)

        box(ax, QX0, ry + 0.12, QWD, ROW_H - 0.24, fc="white", ec=HAIR,
            lw=0.8, r=0.06, zorder=2)
        ax.text(QX0 + 0.14, ry + ROW_H - 0.30, f"query {qi + 1}", ha="left",
                va="center", fontsize=6.6, color=MUTED, zorder=3)
        ax.text(QX0 + 0.14, ry + 0.34,
                "\u201c" + textwrap.fill(cap, 34) + "\u201d", ha="left",
                va="center", fontsize=7.6, color=INK, style="italic",
                linespacing=1.44, zorder=3)

        ty = ry + ROW_H - 0.10 - TH
        for j, idx in enumerate(top):
            hit = bool(rel[idx])
            kind, prm = relscene if hit else distract[j % len(distract)]
            prm2, cam2 = vary(kind, prm, j + (0 if hit else 5))
            sc = build(kind, seed=4100 + 31 * qi + 7 * j, shape=(PH, PW),
                       cam=cam2, **prm2)
            tx = TX0 + j * (TWD + TGAP)
            ax.imshow(np.clip(sc["rgb"], 0, 1),
                      extent=[tx, tx + TWD, ty, ty + TH], zorder=4,
                      interpolation="bilinear")
            ax.add_patch(Rectangle(
                (tx, ty), TWD, TH, facecolor="none",
                edgecolor=DEEP if hit else "#b8b8b8",
                linewidth=2.1 if hit else 0.8, zorder=8))
            if hit:
                ax.add_patch(Rectangle((tx + TWD - 0.20, ty + TH - 0.20),
                                       0.20, 0.20, facecolor=DEEP,
                                       edgecolor="none", zorder=10))
                tick_mark(ax, tx + TWD - 0.10, ty + TH - 0.10)
            ax.text(tx + TWD / 2, ty - 0.12, f"{sim[idx]:.3f}", ha="center",
                    va="center", fontsize=6.4,
                    color=DEEP if hit else MUTED,
                    fontweight="bold" if hit else "normal", zorder=5)
            if qi == 0:
                ax.text(tx + TWD / 2, ROW_TOP + 0.12, f"rank {j + 1}",
                        ha="center", va="center", fontsize=6.6, color=SUB)

        hits = rows[qi]
        box(ax, RX0, ry + 0.12, 0.86, ROW_H - 0.24, fc="white", ec=HAIR,
            lw=0.8, r=0.06, zorder=2)
        ax.text(RX0 + 0.43, ry + ROW_H - 0.34, f"{hits} / {K}", ha="center",
                va="center", fontsize=8.6, fontweight="bold", color=DEEP,
                zorder=3)
        ax.text(RX0 + 0.43, ry + 0.34, f"{hits / K:.3f}", ha="center",
                va="center", fontsize=7.6, color=SUB, zorder=3)

    ax.text(RX0 + 0.43, ROW_TOP + 0.12, "Recall@5", ha="center", va="center",
            fontsize=6.9, fontweight="bold", color=INK)
    ax.text(QX0 + 0.06, ROW_TOP + 0.12,
            f"gallery of {N_POOL} frames, ranked by cosine similarity",
            ha="left", va="center", fontsize=6.9, color=MUTED)

    # ---- violin panel: the same pools the ranking came from ---------------
    axv = fig.add_axes([QX0 / W, 0.50 / H, 5.70 / W, 1.60 / H])
    parts = axv.violinplot([pools[q][0] for q in range(len(QUERIES))],
                           positions=[3, 2, 1], vert=False, widths=0.80,
                           showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor(PALE)
        body.set_edgecolor(STEEL)
        body.set_linewidth(0.8)
        body.set_alpha(0.95)
    for qi in range(len(QUERIES)):
        sim, rel, top = pools[qi]
        y = 3 - qi
        cut = float(sim[top].min())
        axv.plot([cut, cut], [y - 0.34, y + 0.34], color=DEEP, lw=1.1,
                 zorder=5)
        axv.text(cut, y + 0.40, f"cut {cut:.3f}", ha="center", va="bottom",
                 fontsize=6.2, color=DEEP, zorder=6,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.92))
        for idx in top:
            if rel[idx]:
                axv.plot([sim[idx]], [y - 0.16], marker="o", ms=4.4,
                         mfc=DEEP, mec="white", mew=0.7, zorder=7)
            else:
                axv.plot([sim[idx]], [y - 0.16], marker="o", ms=4.4,
                         mfc="white", mec=MISS, mew=1.0, zorder=7)
    axv.set_yticks([3, 2, 1])
    axv.set_yticklabels([f"query {i + 1}" for i in range(len(QUERIES))],
                        fontsize=7.0)
    axv.set_xlabel("cosine similarity to the query text (-)", fontsize=8.0)
    axv.tick_params(axis="x", labelsize=7.0)
    axv.set_xlim(0.10, 0.95)
    axv.set_ylim(0.42, 3.72)
    axv.spines["left"].set_visible(False)
    axv.tick_params(axis="y", length=0)
    axv.text(0.0, 1.04, "(a)", transform=axv.transAxes, ha="left",
             va="bottom", fontsize=8.0, color=SUB)
    axv.text(0.145, 1.04, "similarity over the whole gallery; markers are "
             "the five frames shown above", transform=axv.transAxes,
             ha="left", va="bottom", fontsize=6.6, color=MUTED)

    # ---- summary card: the division, written out --------------------------
    sx, sy, sw, sh = 6.40, 0.62, 3.70, 1.44
    box(ax, sx, sy, sw, sh, fc="#f7f7f7", ec="#ececec", lw=0.8, r=0.08,
        zorder=1)
    ax.text(sx + 0.18, sy + sh - 0.20, "Recall@5 = hits in the top 5, / 5",
            ha="left", va="center", fontsize=7.6, fontweight="bold",
            color=INK, zorder=3)
    for qi, hits in enumerate(rows):
        ax.text(sx + 0.18, sy + sh - 0.48 - qi * 0.22,
                f"query {qi + 1}      {hits} / {K}  =  {hits / K:.3f}",
                ha="left", va="center", fontsize=7.2, color=SUB, zorder=3)
    ax.plot([sx + 0.18, sx + 2.06],
            [sy + sh - 0.48 - len(rows) * 0.22 + 0.11] * 2,
            color=HAIR, lw=0.8, zorder=3)
    ax.text(sx + 0.18, sy + 0.22,
            f"mean       {total} / {K * len(QUERIES)}  =  {mean:.3f}",
            ha="left", va="center", fontsize=7.6, fontweight="bold",
            color=DEEP, zorder=3)
    ax.add_patch(Rectangle((sx + 2.36, sy + sh - 0.55), 0.20, 0.20,
                           facecolor=DEEP, edgecolor="none", zorder=3))
    tick_mark(ax, sx + 2.46, sy + sh - 0.45, zorder=4)
    ax.text(sx + 2.64, sy + sh - 0.45, "hit  ·  bold border", ha="left",
            va="center", fontsize=6.8, color=SUB, zorder=3)
    ax.add_patch(Rectangle((sx + 2.36, sy + sh - 0.90), 0.20, 0.20,
                           facecolor="white", edgecolor=MISS, lw=1.0,
                           zorder=3))
    ax.text(sx + 2.64, sy + sh - 0.80, "miss  ·  hairline", ha="left",
            va="center", fontsize=6.8, color=SUB, zorder=3)
    ax.text(sx, sy + sh + 0.13, "(b)", ha="left", va="center", fontsize=8.0,
            color=SUB, zorder=3)

    best = int(np.argmax(rows)) + 1
    worst = int(np.argmin(rows)) + 1

    savefig(fig, "1174_wm_vlm_cross_retrieval.png")


if __name__ == "__main__":
    main()
