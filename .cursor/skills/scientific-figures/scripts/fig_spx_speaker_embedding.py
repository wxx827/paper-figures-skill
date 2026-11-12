"""457 - Speaker embedding space: t-SNE map, centroid similarity, EER.

Embeddings are drawn from a speaker mean plus an utterance offset plus segment
noise, then L2-normalised, so cosine scores and the resulting error rates are
all computed from the same points that appear in the map.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE

from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label,
                      get_cmap, lighten)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
ACCENT = "#b23a48"
SPEAKERS = ["F01", "F02", "M01", "M02", "M03"]
SPK_COLOR = ["#b23a48", "#3b5b92", "#2e7d5b", "#c9a227", "#8e6bbf"]
DIM, N_UTT, N_SEG = 64, 4, 15
FOCUS = 0                       # the speaker whose utterances get called out


def make_embeddings(seed=11):
    """Speaker mean (sharing a channel component) + utterance + segment noise.

    A few segments are marked as short / noisy and get five times the segment
    noise; they are what puts a tail on the genuine score distribution and so
    give the equal-error rate something to be non-zero about.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1.0, DIM)
    common /= np.linalg.norm(common)
    spk_mu = 0.62 * common + rng.normal(0, 1.0 / np.sqrt(DIM),
                                        (len(SPEAKERS), DIM))
    spk_mu /= np.linalg.norm(spk_mu, axis=1, keepdims=True)
    X, spk, utt, hard = [], [], [], []
    for s in range(len(SPEAKERS)):
        for u in range(N_UTT):
            off = rng.normal(0, 0.085, DIM)
            noise = rng.normal(0, 0.075, (N_SEG, DIM))
            bad = rng.random(N_SEG) < 0.055
            noise[bad] *= 4.8
            X.append(spk_mu[s] + off + noise)
            spk += [s] * N_SEG
            utt += [u] * N_SEG
            hard += list(bad)
    X = np.vstack(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.asarray(spk), np.asarray(utt), np.asarray(hard)


def conf_ellipse(ax, pts, color, n_std=2.0):
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    ang = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(val, 1e-12))
    ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=lighten(color, 0.86),
                         edgecolor=color, lw=0.9, ls="--", alpha=0.85,
                         zorder=2))
    return mu


def main():
    set_pub_style(base_font=8)
    X, spk, utt, hard = make_embeddings()
    n_spk = len(SPEAKERS)

    Z = TSNE(n_components=2, perplexity=22, init="pca", learning_rate="auto",
             random_state=4).fit_transform(X)

    # ---- verification scores from the same embeddings ----------------------
    S = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    same_spk = spk[iu[0]] == spk[iu[1]]
    same_utt = utt[iu[0]] == utt[iu[1]]
    gen = S[iu][same_spk & ~same_utt]          # same speaker, other utterance
    imp = S[iu][~same_spk]
    thr = np.linspace(min(imp.min(), gen.min()), max(imp.max(), gen.max()), 600)
    far = np.array([(imp >= th).mean() for th in thr]) * 100.0
    frr = np.array([(gen < th).mean() for th in thr]) * 100.0
    k = int(np.argmin(np.abs(far - frr)))
    eer, eer_thr = 0.5 * (far[k] + frr[k]), thr[k]

    centro = np.array([X[spk == s].mean(axis=0) for s in range(n_spk)])
    centro /= np.linalg.norm(centro, axis=1, keepdims=True)
    C = centro @ centro.T

    fig = plt.figure(figsize=(7.8, 4.9))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.56],
                  height_ratios=[1.0, 0.92], hspace=0.52, wspace=0.30,
                  left=0.070, right=0.905, top=0.870, bottom=0.158)

    # ---- (a) t-SNE map -----------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    for s in range(n_spk):
        m = spk == s
        conf_ellipse(ax, Z[m & ~hard], SPK_COLOR[s])
        ax.scatter(Z[m & ~hard, 0], Z[m & ~hard, 1], s=13, color=SPK_COLOR[s],
                   alpha=0.88, linewidths=0.35, edgecolors="white", zorder=4)
        ax.scatter(Z[m & hard, 0], Z[m & hard, 1], s=15, marker="D",
                   facecolor="white", edgecolors=SPK_COLOR[s], linewidths=0.9,
                   zorder=5)
    for s in range(n_spk):
        mu = Z[spk == s].mean(axis=0)
        ax.text(mu[0], mu[1], SPEAKERS[s], ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor=SPK_COLOR[s], linewidth=0.7, alpha=0.92))

    # utterance sub-clusters of one speaker
    fm = (spk == FOCUS) & ~hard
    ucen = np.array([Z[fm & (utt == u)].mean(axis=0) for u in range(N_UTT)])
    for u, c in enumerate(ucen):
        ax.plot([c[0]], [c[1]], marker="x", ms=4.4, mew=1.1, color=INK,
                zorder=7)
        ax.text(c[0], c[1] + 1.4, f"u{u + 1}", ha="center", va="bottom",
                fontsize=6.4, color=SUB, zorder=7)
    spread = float(np.linalg.norm(ucen - ucen.mean(axis=0), axis=1).mean())
    far_pt = ucen[int(np.argmax(ucen[:, 1]))]
    ax.annotate(f"{N_UTT} utterances of {SPEAKERS[FOCUS]} share one cluster\n"
                f"(centroid spread {spread:.1f} t-SNE units)",
                xy=(far_pt[0], far_pt[1] + 2.0),
                xytext=(0.03, 0.965), textcoords="axes fraction",
                fontsize=6.6, color=INK, va="top", ha="left", linespacing=1.3,
                arrowprops=dict(arrowstyle="-|>", color=RULE, lw=0.8,
                                shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_xlabel("t-SNE dimension 1 (-)")
    ax.set_ylabel("t-SNE dimension 2 (-)")
    ax.set_title(f"{X.shape[0]} segment embeddings, {DIM}-D, L2-normalised",
                 fontsize=8.5, fontweight="bold")
    style_axes(ax)
    ax.grid(lw=0.4, alpha=0.45)
    add_panel_label(ax, "(a)", x=-0.095, y=1.03)
    marks = [Line2D([], [], marker="o", ls="none", ms=3.6, color="#8a8a8a",
                    label="clean segment"),
             Line2D([], [], marker="D", ls="none", ms=3.6, mfc="white",
                    mec="#8a8a8a", mew=0.9,
                    label=f"short / noisy segment (n = {int(hard.sum())})"),
             Line2D([], [], marker="x", ls="none", ms=4.2, mew=1.1, color=INK,
                    label="utterance centroid")]
    ax.legend(handles=marks, loc="lower left", frameon=False, fontsize=6.6,
              handletextpad=0.4, borderaxespad=0.4, labelspacing=0.30)

    # ---- (b) centroid cosine similarity (subordinate) ----------------------
    ax_m = fig.add_subplot(gs[0, 1])
    im = ax_m.imshow(C, cmap=get_cmap("cb_sequential_blue"), vmin=0.0,
                     vmax=1.0, origin="upper")
    ax_m.set_xticks(range(n_spk))
    ax_m.set_yticks(range(n_spk))
    ax_m.set_xticklabels(SPEAKERS, fontsize=6.5, rotation=40, ha="right")
    ax_m.set_yticklabels(SPEAKERS, fontsize=6.5)
    for i in range(n_spk):
        for j in range(n_spk):
            ax_m.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                      fontsize=6.0,
                      color="white" if C[i, j] > 0.55 else INK)
    for s in ax_m.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor("#8a8a8a")
    ax_m.set_title("Centroid cosine similarity", fontsize=8.0,
                   fontweight="bold")
    ax_m.tick_params(length=0)
    add_panel_label(ax_m, "(b)", x=-0.24, y=1.06)
    cax = ax_m.inset_axes([1.035, 0.0, 0.045, 1.0])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1.0])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.2, width=0.7, length=2.5, color=RULE)
    cb.set_label("cosine similarity (-)", fontsize=6.6, color=INK, labelpad=3)

    # ---- (c) FAR / FRR and the equal-error point (subordinate) -------------
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot(thr, far, color="#3b5b92", lw=1.3, label="FAR (impostor)")
    ax_e.plot(thr, frr, color="#c0603f", lw=1.3, label="FRR (genuine)")
    ax_e.plot([eer_thr], [eer], marker="o", ms=4.2, mfc="white", mec=ACCENT,
              mew=1.3, zorder=5)
    ax_e.axvline(eer_thr, color=ACCENT, lw=0.7, ls="--", alpha=0.8, zorder=2)
    ax_e.annotate(f"EER = {eer:.1f} %\n@ cos = {eer_thr:.2f}",
                  xy=(eer_thr, eer), xytext=(0.98, 0.96),
                  textcoords="axes fraction", ha="right", va="top",
                  fontsize=6.6, color=ACCENT, fontweight="bold",
                  linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.7))
    ax_e.set_xlabel("Decision threshold, cosine score (-)", fontsize=7.5)
    ax_e.set_ylabel("Error rate (%)", fontsize=7.5)
    ax_e.set_ylim(0, 102)
    ax_e.set_xlim(thr[0], thr[-1])
    ax_e.tick_params(labelsize=6.5)
    ax_e.grid(lw=0.4, alpha=0.45)
    style_axes(ax_e)
    ax_e.legend(loc="center left", frameon=False, fontsize=6.6,
                handlelength=1.2, borderaxespad=0.3)
    ax_e.set_title(f"{gen.size} genuine / {imp.size} impostor trials",
                   fontsize=8.0, fontweight="bold")
    add_panel_label(ax_e, "(c)", x=-0.24, y=1.06)

    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).

    savefig(fig, "457_spx_speaker_embedding.png")


if __name__ == "__main__":
    main()
