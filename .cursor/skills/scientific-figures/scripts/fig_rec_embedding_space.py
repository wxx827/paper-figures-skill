"""Figure 476 - Joint user-item embedding space after t-SNE projection.

Main panel: users as circles, items as triangles, both coloured by category,
with a sample of "user -> recommended item" links drawn as thin arcs.  Shape
carries the entity type and colour carries the category, so neither channel is
overloaded and the links stay readable where the two clouds overlap.

Right column: the category-by-category centroid similarity that the projection
is meant to reflect, and the share of each category's recommendations that stay
inside its own category.  Both are computed from the same embedding sample as
the scatter, so the matrix and the picture cannot drift apart.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig, get_cmap, lighten, darken
from figkit import (axes_inches, panel_label_inches, safe_colorbar,
                    unclip_legend)

INK, SUB, MUTED, FAINT = "#1a1a1a", "#555555", "#777777", "#8a8a8a"
HAIR, GRID = "#cccccc", "#d9d9d9"

CATS = ["Electronics", "Home", "Apparel", "Grocery", "Media"]
CAT_C = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]
CENTRES = np.array([[-3.4, 2.3], [2.9, 3.1], [3.6, -2.2],
                    [-3.0, -3.0], [0.1, 0.2]])
N_USER, N_ITEM = 90, 150      # per category
N_LINK = 26                   # user -> item links drawn
DIM = 32                      # latent width the projection came from


def sample(rng):
    """Latent users and items per category, plus their 2-D projection."""
    users, items = [], []
    for c, ctr in enumerate(CENTRES):
        spread = 0.70 + 0.05 * c
        users.append(rng.normal(ctr, spread, size=(N_USER, 2)))
        # Items sit in the same neighbourhood but scatter wider: the catalogue
        # covers more of the space than the audience does.
        items.append(rng.normal(ctr, spread * 1.38, size=(N_ITEM, 2)))
    return np.array(users), np.array(items)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(476)
    users, items = sample(rng)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax = axes_inches(fig, (0.66, 0.66, 5.30, 4.82))

    # ---- (a) projection ----------------------------------------------------
    for c, col in enumerate(CAT_C):
        ax.scatter(items[c][:, 0], items[c][:, 1], s=11, marker="^",
                   facecolor=lighten(col, 0.52), edgecolor=col,
                   linewidth=0.35, alpha=0.90, zorder=3)
    for c, col in enumerate(CAT_C):
        ax.scatter(users[c][:, 0], users[c][:, 1], s=13, marker="o",
                   facecolor=col, edgecolor="white", linewidth=0.35,
                   alpha=0.95, zorder=4)

    # Recommendation links: nearest item in the joint space, so a link that
    # crosses into another category really is a cross-category recommendation.
    flat_items = items.reshape(-1, 2)
    item_cat = np.repeat(np.arange(len(CATS)), N_ITEM)
    cross = 0
    for i in range(N_LINK):
        c = int(rng.integers(len(CATS)))
        u = users[c][rng.integers(N_USER)]
        d = np.linalg.norm(flat_items - u, axis=1)
        # A recommender does not always take the nearest neighbour; sampling
        # from further down the candidate list is what sends a few links out
        # of the home cluster, which is the point of drawing them at all.
        j = int(rng.choice(np.argsort(d)[8:120]))
        cross += int(item_cat[j] != c)
        ax.add_patch(FancyArrowPatch(
            tuple(u), tuple(flat_items[j]), arrowstyle="-|>",
            mutation_scale=6.5, linewidth=0.75, color="#3a3a3a", alpha=0.80,
            connectionstyle=f"arc3,rad={0.24 if i % 2 else -0.24}",
            shrinkA=1.5, shrinkB=1.5, zorder=6))

    ax.set_xlabel("t-SNE dimension 1 (-)", fontsize=8.0)
    ax.set_ylabel("t-SNE dimension 2 (-)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_title(f"Joint projection of {DIM}-D user and item embeddings",
                 fontsize=9.0, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    cat_h = [Line2D([], [], marker="s", ls="none", ms=4.6, mfc=c,
                    mec="white", mew=0.4, label=n)
             for n, c in zip(CATS, CAT_C)]
    typ_h = [
        Line2D([], [], marker="o", ls="none", ms=4.2, mfc="#5a5a5a",
               mec="white", mew=0.4, label=f"user (n = {N_USER * len(CATS)})"),
        Line2D([], [], marker="^", ls="none", ms=4.6, mfc="#d5d5d5",
               mec="#5a5a5a", mew=0.5, label=f"item (n = {N_ITEM * len(CATS)})"),
        Line2D([], [], color="#5a5a5a", lw=0.8, alpha=0.8,
               label=f"recommendation ({N_LINK} shown)"),
    ]
    leg_cat = ax.legend(handles=cat_h, loc="upper left", fontsize=7.0,
                        frameon=True, ncol=2, handletextpad=0.4,
                        columnspacing=0.9, borderaxespad=0.4)
    leg_typ = ax.legend(handles=typ_h, loc="lower right", fontsize=7.0,
                        frameon=True, handletextpad=0.5, borderaxespad=0.4)
    ax.add_artist(leg_cat)
    unclip_legend(leg_cat)

    # ---- (b) category similarity ------------------------------------------
    axm = axes_inches(fig, (7.24, 3.28, 2.34, 2.34))
    cent = np.array([np.vstack([users[c], items[c]]).mean(axis=0)
                     for c in range(len(CATS))])
    # Gaussian affinity on centroid distance: monotone, bounded and unit on
    # the diagonal, so the matrix reads the same way as the picture does.
    dd = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)
    sim = np.exp(-(dd ** 2) / (2.0 * 3.4 ** 2))

    im = axm.imshow(sim, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                    vmax=1.0)
    axm.set_aspect("equal")
    axm.set_xticks(range(len(CATS)))
    axm.set_yticks(range(len(CATS)))
    axm.set_xticklabels([c[:4] for c in CATS], rotation=45, ha="right")
    axm.set_yticklabels([c[:4] for c in CATS])
    axm.tick_params(labelsize=6.5, length=2.0)
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            axm.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                     fontsize=6.0,
                     color="white" if sim[i, j] > 0.60 else INK)
    axm.set_title("Category affinity", fontsize=8.5, fontweight="bold",
                  pad=3.0)
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    safe_colorbar(fig, im, axm, label="embedding affinity (-)", width=0.10,
                  pad=0.06, labelsize=6.5, ticks=[0.0, 0.5, 1.0])

    # ---- (c) in-category recommendation share ------------------------------
    axb = axes_inches(fig, (7.24, 0.66, 2.34, 1.62))
    share = []
    for c in range(len(CATS)):
        hit = 0
        for u in users[c]:
            d = np.linalg.norm(flat_items - u, axis=1)
            hit += int(item_cat[int(np.argmin(d))] == c)
        share.append(100.0 * hit / N_USER)
    y = np.arange(len(CATS))[::-1]
    axb.barh(y, share, height=0.62, color=CAT_C, edgecolor="white",
             linewidth=0.5, zorder=3)
    for yy, v in zip(y, share):
        axb.text(v + 1.4, yy, f"{v:.0f}", va="center", ha="left",
                 fontsize=6.2, color=SUB)
    axb.set_yticks(y)
    axb.set_yticklabels([c[:4] for c in CATS])
    axb.set_xlim(0, 108)
    axb.set_xticks([0, 50, 100])
    axb.set_xlabel("in-category top-1 hits (%)", fontsize=7.0)
    axb.tick_params(labelsize=6.5, length=2.2)
    axb.grid(axis="x", color=GRID, linewidth=0.6)
    axb.set_title("Nearest item stays in category", fontsize=8.5,
                  fontweight="bold", pad=3.0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    off = np.mean([sim[i, j] for i in range(len(CATS))
                   for j in range(len(CATS)) if i != j])
    panel_label_inches(fig, ax, "(a)", dx=-0.50, dy=0.10)
    panel_label_inches(fig, axm, "(b)", dx=-0.50, dy=0.10)
    panel_label_inches(fig, axb, "(c)", dx=-0.50, dy=0.10)
    savefig(fig, "476_rec_embedding_space.png")


if __name__ == "__main__":
    main()
