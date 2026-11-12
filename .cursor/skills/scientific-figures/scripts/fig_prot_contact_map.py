"""Residue-residue contact map with predicted contacts on the lower triangle.

The upper triangle holds the reference contacts shaded by C-alpha separation:
the diagonal band is local backbone contact, the short anti-diagonal ribbons
are beta-sheet pairings and the two blocks are the structural domains.  The
lower triangle holds the top-L predicted contacts, split into hits and misses,
and the companion panel tracks precision as more predictions are accepted.

The distance matrix is synthesised directly from a polymer scaling law plus
explicit strand registers, helix-packing blobs and per-domain tertiary
contacts, so every feature the caption names is present in the drawn array.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_label_inches, safe_colorbar, axes_inches

SEED = 90210

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
H_RED, H_BLUE, H_GOLD = "#b23a48", "#3b5b92", "#c9a227"
SS_COLOR = {"H": H_RED, "E": H_BLUE, "C": "#c4c4c4"}

L = 140
CUTOFF = 8.0                     # contact definition, angstrom
MIN_SEP = 3                      # ignore trivial |i-j| < 3 pairs
LONG_SEP = 24                    # "long range" in the CASP sense

ELEMENTS = [("E", 6, 13), ("H", 20, 35), ("E", 42, 49), ("E", 55, 62),
            ("H", 70, 84), ("E", 92, 99), ("H", 106, 120), ("E", 126, 133)]
STRANDS = [(6, 13), (42, 49), (55, 62), (92, 99), (126, 133)]
PAIRS = [(0, 1, "anti"), (1, 2, "anti"), (3, 4, "anti"), (0, 3, "para")]
DOMAINS = [("domain I", 1, 68), ("domain II", 69, 140)]


def ss_track():
    track = np.array(["C"] * L, dtype=object)
    for kind, a, b in ELEMENTS:
        track[a - 1:b] = kind
    return track


def _pull_down(d, i, j, value, spread=1):
    for di in range(-spread, spread + 1):
        for dj in range(-spread, spread + 1):
            ii, jj = i + di, j + dj
            if 0 <= ii < L and 0 <= jj < L:
                v = value + 1.4 * (abs(di) + abs(dj))
                d[ii, jj] = min(d[ii, jj], v)
                d[jj, ii] = d[ii, jj]


def distance_matrix(rng):
    """Polymer scaling, then explicit sheet, packing and domain contacts."""
    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :]).astype(float)
    d = 3.6 * np.sqrt(np.maximum(sep, 1e-9))

    same = ((idx[:, None] < 68) == (idx[None, :] < 68))
    d = np.minimum(d, np.where(same, 24.0, 32.0))

    noise = rng.normal(0.0, 1.0, (L, L))
    noise = 0.5 * (noise + noise.T)
    for _ in range(5):
        noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0) +
                 np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5.0
    d = d + 2.2 * noise / (np.abs(noise).max() + 1e-9)

    for a, b, mode in PAIRS:                      # beta-sheet registers
        (a0, a1), (b0, b1) = STRANDS[a], STRANDS[b]
        n = min(a1 - a0, b1 - b0) + 1
        for k in range(n):
            i = a0 - 1 + k
            j = (b1 - 1 - k) if mode == "anti" else (b0 - 1 + k)
            _pull_down(d, i, j, 4.9, spread=1)

    for ci, cj in ((27, 77), (113, 60), (112, 30)):   # helix packing patches
        for _ in range(26):
            i = int(np.clip(rng.normal(ci, 4.0), 0, L - 1))
            j = int(np.clip(rng.normal(cj, 4.0), 0, L - 1))
            _pull_down(d, i, j, rng.uniform(5.2, 7.4), spread=0)

    for _, a, b in DOMAINS:                       # diffuse tertiary packing
        for _ in range(150):
            i, j = rng.integers(a - 1, b, size=2)
            if abs(i - j) < 6:
                continue
            _pull_down(d, int(i), int(j), rng.uniform(5.4, 7.8), spread=0)

    d = np.clip(0.5 * (d + d.T), 3.6, 40.0)
    np.fill_diagonal(d, 0.0)
    return d


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    dist = distance_matrix(rng)
    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :])
    valid = sep >= MIN_SEP
    true_c = (dist < CUTOFF) & valid

    # predicted score: monotone in distance, but the uncertainty grows with
    # sequence separation, which is exactly where coevolution signal thins out
    score = 1.0 / (1.0 + np.exp((dist - CUTOFF) / 1.15))
    jitter = rng.normal(0.0, 1.0, (L, L))
    jitter = 0.5 * (jitter + jitter.T)
    score = score + jitter * (0.16 + 0.62 * sep / L)
    score[~valid] = -np.inf

    iu, ju = np.triu_indices(L, k=MIN_SEP)
    order = np.argsort(-score[iu, ju])
    top = order[:L]
    pred_i, pred_j = iu[top], ju[top]
    hit = true_c[pred_i, pred_j]
    prec_L = float(hit.mean())

    ks = np.unique(np.clip((np.linspace(0.08, 2.0, 44) * L).astype(int),
                           4, order.size))
    prec_k = np.array([true_c[iu[order[:k]], ju[order[:k]]].mean()
                       for k in ks])

    long_mask = sep[pred_i, pred_j] >= LONG_SEP
    prec_long = float(hit[long_mask].mean()) if long_mask.any() else 0.0
    prec_short = float(hit[~long_mask].mean()) if (~long_mask).any() else 0.0
    n_true = int(np.triu(true_c, MIN_SEP).sum())

    # ---- RGBA canvas: reference above the diagonal, prediction below -------
    cmap = get_cmap("cb_sequential_blue", reverse=True)
    norm = plt.Normalize(4.0, CUTOFF)
    img = np.ones((L, L, 4), dtype=float)
    ti, tj = np.where(np.triu(true_c, k=MIN_SEP))
    img[ti, tj] = cmap(norm(dist[ti, tj]))
    img[pred_j, pred_i] = np.where(hit[:, None],
                                   np.array(mpl.colors.to_rgba(H_BLUE)),
                                   np.array(mpl.colors.to_rgba(H_RED)))

    # ---- layout in inches: the matrix cell is square, so the SS tracks sit
    # flush against it instead of drifting away from an equal-aspect axes ----
    FW, FH = 10.4, 6.4
    S, X0, Y0, TR, GAP = 4.62, 0.98, 1.16, 0.13, 0.045
    fig = plt.figure(figsize=(FW, FH))
    axm = axes_inches(fig, (X0, Y0, S, S))
    axt = axes_inches(fig, (X0, Y0 + S + GAP, S, TR))
    axl = axes_inches(fig, (X0 - GAP - TR, Y0, TR, S))
    aux = 0.58 * S
    axk = axes_inches(fig, (X0 + S + 1.02, Y0 + S - aux, aux * 1.14, aux))

    axm.imshow(img, interpolation="nearest", origin="upper",
               extent=[0.5, L + 0.5, L + 0.5, 0.5], zorder=2)
    axm.plot([0.5, L + 0.5], [0.5, L + 0.5], color="#8a8a8a", lw=0.7,
             zorder=4)
    axm.set_aspect("equal")
    axm.set_xlim(0.5, L + 0.5)
    axm.set_ylim(L + 0.5, 0.5)

    for name, a, b in DOMAINS:
        axm.add_patch(Rectangle((a - 0.5, a - 0.5), b - a + 1, b - a + 1,
                                facecolor="none", edgecolor=H_GOLD, lw=0.9,
                                ls=(0, (3.4, 2.2)), zorder=5))
        axm.text(a + 2.5, b - 2.5, name, fontsize=6.6, color="#8c6a12",
                 ha="left", va="bottom", zorder=6)

    axm.annotate("local backbone band", xy=(20, 26), xytext=(44, 13),
                 fontsize=6.6, color=SUB, ha="left", va="center", zorder=7,
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                 color=SUB, shrinkA=1, shrinkB=2))
    axm.annotate(r"antiparallel $\beta$ pairing", xy=(45, 59),
                 xytext=(72, 44), fontsize=6.6, color=H_BLUE, ha="left",
                 va="center", zorder=7,
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                 color=H_BLUE, shrinkA=1, shrinkB=2))
    axm.text(L - 3, 6, "reference", fontsize=7.4, color=SUB, ha="right",
             va="top", style="italic", zorder=7)
    axm.text(5, L - 3, "prediction", fontsize=7.4, color=SUB, ha="left",
             va="bottom", style="italic", zorder=7)

    axm.set_xlabel("residue index i (-)", fontsize=8.0)
    axm.tick_params(labelsize=6.8, length=2.4, labelleft=False)
    axm.set_xticks(np.arange(20, L + 1, 20))
    axm.set_yticks(np.arange(20, L + 1, 20))
    for s in axm.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.7)
        s.set_color("#8a8a8a")

    safe_colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=cmap), axm,
                  side="bottom", width=0.12, pad=0.40, shrink=0.58,
                  label=r"C$_\alpha$-C$_\alpha$ distance ($\mathrm{\AA}$)",
                  ticks=[4, 5, 6, 7, 8])

    # ---- secondary-structure tracks ---------------------------------------
    track = ss_track()
    for ax, horizontal in ((axt, True), (axl, False)):
        for i, k in enumerate(track):
            xy = (i + 0.5, 0.0) if horizontal else (0.0, i + 0.5)
            w, h = (1.0, 1.0)
            ax.add_patch(Rectangle(xy, w, h, facecolor=SS_COLOR[k], lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axt.set_xlim(0.5, L + 0.5)
    axt.set_ylim(0.0, 1.0)
    axl.set_ylim(L + 0.5, 0.5)
    axl.set_xlim(0.0, 1.0)
    # the j axis lives on the outside of the left track so the track can stay
    # flush against the matrix without the tick labels landing underneath it
    axl.set_yticks(np.arange(20, L + 1, 20))
    axl.tick_params(axis="y", labelsize=6.8, length=2.4, left=True,
                    labelleft=True)
    axl.set_ylabel("residue index j (-)", fontsize=8.0, labelpad=2.0)
    fig.text((X0 + S / 2) / FW, (Y0 + S + GAP + TR + 0.09) / FH,
             "Contact matrix with secondary-structure tracks",
             fontsize=9.0, fontweight="bold", ha="center", va="bottom")

    # ---- companion: precision as more predictions are accepted ------------
    axk.plot(ks / L, prec_k, color=RULE, lw=1.3, zorder=4)
    axk.fill_between(ks / L, 0.0, prec_k, color=H_BLUE, alpha=0.13, lw=0,
                     zorder=2)
    axk.axvline(1.0, color=H_RED, lw=0.9, ls=(0, (3.2, 2.0)), zorder=3)
    axk.plot([1.0], [prec_L], marker="o", ms=4.2, mfc="white", mec=H_RED,
             mew=1.2, zorder=6)
    axk.annotate(f"precision@L = {prec_L:.2f}", xy=(1.0, prec_L),
                 xytext=(1.16, prec_L + 0.16), fontsize=6.9, color=H_RED,
                 arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                 color=H_RED, shrinkA=1, shrinkB=3))
    axk.set_xlim(0.0, 2.0)
    axk.set_ylim(0.0, 1.05)
    axk.set_xlabel("accepted predictions (multiples of L)", fontsize=7.6)
    axk.set_ylabel("precision (-)", fontsize=7.6)
    axk.tick_params(labelsize=6.6, length=2.4)
    axk.grid(lw=0.45, color="#ececec")
    axk.set_axisbelow(True)
    axk.set_title("Precision versus depth of the list", fontsize=8.5,
                  fontweight="bold", pad=4.0)

    handles = [
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=cmap(0.15),
               mec="none", label="reference contact (close)"),
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=cmap(0.80),
               mec="none", label="reference contact (far)"),
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=H_BLUE,
               mec="none", label="predicted hit"),
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=H_RED,
               mec="none", label="predicted miss"),
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=SS_COLOR["H"],
               mec="none", label=r"$\alpha$ helix track"),
        Line2D([], [], marker="s", ls="none", ms=5.0, mfc=SS_COLOR["E"],
               mec="none", label=r"$\beta$ strand track"),
    ]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=((X0 + S + 1.02) / FW, (Y0 + S - aux - 0.34) / FH),
               ncol=2, fontsize=6.9, frameon=False, handletextpad=0.5,
               labelspacing=0.42, columnspacing=1.2, borderaxespad=0.0)

    fig.text((X0 + S + 1.02) / FW, (Y0 + 0.62) / FH,
             f"precision@L (short range, |i - j| < {LONG_SEP})   "
             f"{prec_short:.2f}\n"
             f"precision@L (long range,  |i - j| " + r"$\geq$" +
             f" {LONG_SEP})   {prec_long:.2f}\n"
             f"reference contacts in the upper triangle   {n_true}",
             fontsize=6.8, color=SUB, ha="left", va="top", linespacing=1.7)


    panel_label_inches(fig, axt, "(a)", dx=-0.34, dy=0.20)
    panel_label_inches(fig, axk, "(b)", dx=-0.34, dy=0.20)
    savefig(fig, "524_prot_contact_map.png")


if __name__ == "__main__":
    main()
