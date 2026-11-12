"""Figure 1168 - Video tokenisation: a space-time cube becomes a token string.

Everything on the page is one arithmetic chain and it closes exactly:

    clip      T=4 frames, 96 x 144 px, 3 channels  = 165 888 B
    patches   96/24 = 4 rows, 144/24 = 6 cols      = 24 per frame
    tokens    4 frames x 24                        = 96 tokens
    payload   96 x log2(512) = 96 x 9 bit = 864 bit = 108 B
    ratio     165 888 / 108                        = 1536x   (exact)

The codebook indices are not invented: each patch is reduced to a five-number
feature vector and assigned its nearest of 512 random codewords, so the strip
in (c), the usage bars in (b) and the perplexity in the stats card are three
readings of one ``codes`` array.  A schematic like this runs sparse, so the
cube and the stats sit on the deeper of the two house card greys.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from pubstyle import set_pub_style, savefig, get_cmap, add_panel_label
from _wm_scenes import build

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# Two card greys, on either side of the 0.94 ink threshold, never on it.
# CARD (0.91) is for the two panels that really are too sparse and is meant to
# count as ink; LAYOUT (0.96) only groups panels that already carry their own
# weight.  Putting the deep grey behind all four pushed the loose ink ratio to
# 64 % while the strict one stayed at 18 %, i.e. a washed-out figure.
HAIR, CARD, LAYOUT = "#d5d5d5", "#e6e9ec", "#f5f5f5"
DEEP, MID = "#134a76", "#5f9dc6"
CODE_CMAP = get_cmap("cb_sequential_blue")
SEED = 1573  # original id; figure number is 1168

T, IH, IW, P = 4, 96, 144, 24
PAN = 5                                        # px of dolly per frame
ROWS, COLS = IH // P, IW // P                  # 4 x 6 patches per frame
NTOK = T * ROWS * COLS                         # 96 tokens
K = 512                                        # codebook size, 9 bit
HL_R, HL_C = 2, 2                              # the patch that gets traced
CAM = dict(wb=(0.955, 1.000, 1.062), sat=0.70, contrast=1.12, exposure=1.06,
           dof=1.2, sharpen=1.15)


def clip():
    sc = build("indoor", seed=SEED, shape=(IH, IW + PAN * (T - 1)), cam=CAM)
    return [sc["rgb"][:, k * PAN:k * PAN + IW] for k in range(T)]


def tokenise(frames):
    """Five features per patch, nearest of K random codewords."""
    feats = []
    for f in frames:
        for r in range(ROWS):
            for c in range(COLS):
                q = f[r * P:(r + 1) * P, c * P:(c + 1) * P]
                lum = q @ np.array([0.30, 0.59, 0.11])
                gy, gx = np.gradient(lum)
                feats.append([q[..., 0].mean(), q[..., 1].mean(),
                              q[..., 2].mean(), lum.std(),
                              float(np.hypot(gx, gy).mean())])
    F = np.asarray(feats)
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-9)
    rng = np.random.default_rng(SEED)
    # A trained codebook sits *on* the data manifold, not uniformly in the
    # unit cube; seeding it from jittered observations is what keeps the
    # usage histogram from collapsing onto a handful of entries.
    book = np.clip(Fn[rng.integers(0, Fn.shape[0], K)]
                   + rng.normal(0.0, 0.045, (K, Fn.shape[1])), 0.0, 1.0)
    d = ((Fn[:, None, :] - book[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def main():
    set_pub_style(base_font=8.5)
    frames = clip()
    codes = tokenise(frames)
    used, counts = np.unique(codes, return_counts=True)
    order = np.argsort(counts)[::-1]
    used, counts = used[order], counts[order]
    p = counts / counts.sum()
    perp = float(np.exp(-(p * np.log(p)).sum()))

    raw_b = T * IH * IW * 3
    bits = int(np.log2(K))
    tok_b = NTOK * bits // 8
    ratio = raw_b / tok_b
    hl = HL_R * COLS + HL_C                    # token index of the traced patch

    fig = plt.figure(figsize=(10.4, 6.15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.42, 1.0],
                          height_ratios=[1.0, 0.60], left=0.045, right=0.975,
                          top=0.840, bottom=0.078, wspace=0.155, hspace=0.335)
    axc = fig.add_subplot(gs[0, 0])
    axk = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, 0])
    axz = fig.add_subplot(gs[1, 1])

    # ---- (a) the space-time cube ------------------------------------------
    AX_W = 1.86                                # on-paper aspect of this axes
    axc.set_facecolor(CARD)
    fw, fh = 0.780, 0.780 * IH / IW
    ox, oy = 0.100, 0.112
    x0, y0 = 0.058, 0.072
    for k in range(T - 1, -1, -1):
        x, y = x0 + k * ox, y0 + k * oy
        axc.imshow(frames[k], extent=(x, x + fw, y, y + fh), aspect="auto",
                   interpolation="bilinear", zorder=2 + (T - k))
        axc.add_patch(Rectangle((x, y), fw, fh, facecolor="none",
                                edgecolor="#6b7480" if k else "#3a3a3a",
                                lw=0.6 if k else 0.9, zorder=2 + (T - k) + 0.1))
        axc.text(x + fw + 0.012, y + fh - 0.012, f"$t_{{{k + 1}}}$",
                 fontsize=7.2, color=SUB if k else INK, ha="left", va="top",
                 zorder=9)
    # patch grid on the front frame
    for r in range(1, ROWS):
        axc.plot([x0, x0 + fw], [y0 + fh * r / ROWS] * 2, color="white",
                 lw=0.55, alpha=0.80, zorder=7)
    for c in range(1, COLS):
        axc.plot([x0 + fw * c / COLS] * 2, [y0, y0 + fh], color="white",
                 lw=0.55, alpha=0.80, zorder=7)

    # the traced tubelet: the same spatial patch on all four frames
    pw, ph = fw / COLS, fh / ROWS
    px = x0 + HL_C * pw
    py = y0 + fh - (HL_R + 1) * ph
    for k in range(T):
        axc.add_patch(Rectangle((px + k * ox, py + k * oy), pw, ph,
                                facecolor="none", edgecolor=DEEP,
                                lw=1.15, zorder=9))
    for cx, cy in ((0, 0), (pw, 0), (0, ph), (pw, ph)):
        axc.plot([px + cx, px + cx + (T - 1) * ox],
                 [py + cy, py + cy + (T - 1) * oy], color=DEEP, lw=0.55,
                 ls=(0, (2, 2)), zorder=8)
    axc.annotate(f"one spatial patch, {P}x{P} px,\nfollowed through all "
                 f"{T} frames", xy=(px + pw + 0.5 * (T - 1) * ox,
                                    py + ph + 0.5 * (T - 1) * oy),
                 xytext=(1.845, 0.815), ha="right", va="center", fontsize=6.9,
                 color=DEEP, linespacing=1.5, zorder=10,
                 bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9d8",
                           lw=0.6, alpha=0.95),
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=DEEP,
                                 shrinkA=3, shrinkB=3))
    axc.text(1.845, 0.045, f"{ROWS} x {COLS} = {ROWS * COLS} patches\n"
             f"per frame, {T} frames\n= {NTOK} tokens", ha="right",
             va="bottom", fontsize=7.0, color=INK, linespacing=1.55, zorder=10,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=HAIR, lw=0.6,
                       alpha=0.95))
    axc.set_xlim(0, AX_W)
    axc.set_ylim(0, 1.0)
    axc.set_xticks([])
    axc.set_yticks([])
    for s in axc.spines.values():
        s.set_visible(False)
    axc.set_title(f"A {T}-frame {IW}x{IH} clip cut into "
                  f"{P}x{P} space-time patches", fontsize=9.0,
                  fontweight="bold", pad=4.0, color=INK, loc="left")
    add_panel_label(axc, "(a)", x=-0.058, y=1.005, size=9.5)

    # ---- (b) codebook usage -----------------------------------------------
    rank = np.arange(1, used.size + 1)
    axk.bar(rank, counts, width=0.86,
            color=[CODE_CMAP(0.20 + 0.72 * c / K) for c in used],
            edgecolor="white", linewidth=0.35, zorder=3)
    axk.set_xlim(0.2, used.size + 0.8)
    axk.set_ylim(0, counts.max() + 1.25)
    ystep = max(1, int(np.ceil(counts.max() / 6.0)))
    axk.set_yticks(np.arange(0, counts.max() + 1, ystep))
    axk.set_xlabel(f"codebook entry, ranked by use "
                   f"(only the {used.size} that fire)", fontsize=7.6)
    axk.set_ylabel("times used", fontsize=7.6)
    axk.tick_params(labelsize=7.0, length=2.4)
    axk.grid(True, axis="y", lw=0.45, color="#ececec")
    axk.set_axisbelow(True)
    axk.set_facecolor(LAYOUT)
    axk.text(0.975, 0.955, f"$K$ = {K} entries, {used.size} distinct fire "
             f"({100 * used.size / K:.1f} %)\n"
             f"perplexity {perp:.1f}   |   busiest entry fires "
             f"{counts.max()}x", transform=axk.transAxes, ha="right",
             va="top", fontsize=6.9, color=SUB, linespacing=1.6, zorder=7,
             bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=HAIR, lw=0.6,
                       alpha=0.95))
    axk.set_title("Codebook usage over the 96 tokens below", fontsize=9.0,
                  fontweight="bold", pad=4.0, color=INK, loc="left")
    add_panel_label(axk, "(b)", x=-0.120, y=1.005, size=9.5)

    # ---- (c) the token string ----------------------------------------------
    axs.set_facecolor(CARD)
    for i, c in enumerate(codes):
        axs.add_patch(Rectangle((i, 0.0), 0.92, 1.0,
                                facecolor=CODE_CMAP(0.16 + 0.76 * c / K),
                                edgecolor="none", zorder=3))
    for k in range(1, T):
        axs.plot([k * ROWS * COLS - 0.10] * 2, [-0.16, 1.12], color="#3a3a3a",
                 lw=0.9, zorder=5)
    for k in range(T):
        axs.text((k + 0.5) * ROWS * COLS, 1.40, f"$t_{{{k + 1}}}$: patches "
                 f"1-{ROWS * COLS}", ha="center", va="bottom", fontsize=7.0,
                 color=INK if k == 0 else SUB)
    for k in range(T):
        i = k * ROWS * COLS + hl
        axs.add_patch(Rectangle((i - 0.10, -0.11), 1.12, 1.20,
                                facecolor="none", edgecolor=DEEP, lw=1.1,
                                zorder=6))
        axs.text(i + 0.46, -0.22, f"{i + 1}", ha="center", va="top",
                 fontsize=6.4, color=DEEP)
    axs.set_xlim(-1.6, NTOK + 1.6)
    axs.set_ylim(-1.05, 1.88)
    axs.set_xticks([])
    axs.set_yticks([])
    for s in axs.spines.values():
        s.set_visible(False)
    axs.text(-1.6, -0.82, f"token index (1-{NTOK}), time-major order; "
             f"fill = codebook entry, light to dark with the entry number",
             ha="left", va="center", fontsize=6.8, color=MUTED)
    axs.set_title(f"{NTOK} discrete tokens, {bits} bit each",
                  fontsize=9.0, fontweight="bold", pad=13.0, color=INK,
                  loc="left")
    add_panel_label(axs, "(c)", x=-0.058, y=1.05, size=9.5)

    con = ConnectionPatch(xyA=(px + 0.5 * pw, py), coordsA=axc.transData,
                          xyB=(hl + 0.46, 1.10), coordsB=axs.transData,
                          arrowstyle="-|>", mutation_scale=11, lw=1.0,
                          color=DEEP, shrinkA=3, shrinkB=3, zorder=11)
    fig.add_artist(con)

    # ---- (d) the compression arithmetic ------------------------------------
    axz.set_facecolor(LAYOUT)
    axz.set_xlim(0, 1)
    axz.set_ylim(0, 1)
    axz.set_xticks([])
    axz.set_yticks([])
    for s in axz.spines.values():
        s.set_visible(False)
    lines = [("raw clip", f"{T} x {IH} x {IW} x 3 ch", f"{raw_b:,} B"),
             ("tokens", f"{NTOK} x {bits} bit = {NTOK * bits} bit",
              f"{tok_b} B"),
             ("compression", f"{raw_b:,} / {tok_b}", f"{ratio:.0f}x")]
    for j, (a, b, c) in enumerate(lines):
        y = 0.78 - 0.29 * j
        bold = j == 2
        axz.text(0.045, y, a, fontsize=7.6, color=INK if bold else SUB,
                 ha="left", va="center",
                 fontweight="bold" if bold else "normal")
        axz.text(0.360, y, b, fontsize=7.4, color=SUB, ha="left", va="center")
        axz.text(0.960, y, c, fontsize=8.6 if bold else 7.6,
                 color=DEEP if bold else INK, ha="right", va="center",
                 fontweight="bold" if bold else "normal")
        if j < 2:
            axz.plot([0.045, 0.960], [y - 0.145] * 2, color=HAIR, lw=0.6)
    axz.set_title("What the tokeniser buys", fontsize=9.0, fontweight="bold",
                  pad=4.0, color=INK, loc="left")
    add_panel_label(axz, "(d)", x=-0.105, y=1.01, size=9.5)

    # ---- headline ----------------------------------------------------------
    traced = [hl + 1 + k * ROWS * COLS for k in range(T)]
    savefig(fig, "1168_wm_vid_tokenization.png")


if __name__ == "__main__":
    main()
