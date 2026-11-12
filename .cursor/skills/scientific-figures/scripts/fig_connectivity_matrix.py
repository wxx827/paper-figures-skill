"""Functional connectivity: module-sorted matrix + circular connectogram."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Arc, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, style_axes, add_panel_label

MODULES = [
    ("VIS", ["V1", "V2", "V3", "V4", "MT", "LOC"], "#7b5aa6"),
    ("SM", ["M1", "S1", "PMd", "PMv", "SMA", "A1"], "#3d7fa6"),
    ("DAN", ["FEF", "IPS", "SPL", "MT+", "PCv"], "#2e8b6f"),
    ("FPN", ["dlPFC", "IFJ", "aINS", "IPL", "dACC", "FPl"], "#d0913c"),
    ("DMN", ["mPFC", "PCC", "AG", "Hip", "PHC", "TPJ", "TPol"], "#b8503f"),
]

# 4-D latent coordinates; DMN deliberately opposes DAN to give the classic
# task-positive / task-negative anticorrelation.
CENTROIDS = np.array([
    [1.00, 0.10, 0.00, 0.00],
    [0.05, 1.00, 0.10, 0.00],
    [0.25, 0.25, 1.00, 0.10],
    [0.00, 0.05, 0.70, 0.75],
    [0.00, 0.00, -0.75, 0.55],
])

THRESH = 0.45


def build_matrix(rng):
    labels, mod_id, colors = [], [], []
    for mi, (_, regs, col) in enumerate(MODULES):
        labels += regs
        mod_id += [mi] * len(regs)
        colors += [col] * len(regs)
    mod_id = np.asarray(mod_id)
    n = len(labels)

    cent = CENTROIDS / np.linalg.norm(CENTROIDS, axis=1, keepdims=True)
    emb = cent[mod_id] + rng.normal(0, 0.24, (n, CENTROIDS.shape[1]))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    C = 0.88 * (emb @ emb.T)
    C = (C + C.T) / 2.0
    np.fill_diagonal(C, 1.0)
    return C, labels, mod_id, colors


def draw_matrix(ax, C, mod_id, vlim):
    n = C.shape[0]
    M = C.copy()
    np.fill_diagonal(M, np.nan)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e9e9e9")
    im = ax.imshow(M, cmap=cmap, vmin=-vlim, vmax=vlim,
                   interpolation="nearest", origin="upper")

    bounds = np.flatnonzero(np.diff(mod_id)) + 1
    for b in bounds:
        ax.axhline(b - 0.5, color="#ffffff", lw=1.1)
        ax.axvline(b - 0.5, color="#ffffff", lw=1.1)

    edges = np.concatenate([[0], bounds, [n]])
    for mi, (name, _, col) in enumerate(MODULES):
        lo, hi = edges[mi], edges[mi + 1]
        ax.add_patch(Rectangle((lo - 0.5, -1.85), hi - lo, 1.1,
                               facecolor=col, edgecolor="none",
                               clip_on=False, zorder=4))
        ax.add_patch(Rectangle((-1.85, lo - 0.5), 1.1, hi - lo,
                               facecolor=col, edgecolor="none",
                               clip_on=False, zorder=4))
        mid = (lo + hi) / 2.0 - 0.5
        ax.text(mid, -2.35, name, ha="center", va="bottom", fontsize=7.4,
                fontweight="bold", color=col, clip_on=False)
        ax.text(-2.35, mid, name, ha="right", va="center", fontsize=7.4,
                fontweight="bold", color=col, rotation=90, clip_on=False)

    ax.set_xlim(-1.9, n - 0.5)
    ax.set_ylim(n - 0.5, -1.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def bezier_chord(ax, a1, a2, color, lw, alpha, r=1.0):
    p0 = np.array([np.cos(a1), np.sin(a1)]) * r
    p2 = np.array([np.cos(a2), np.sin(a2)]) * r
    d = abs(np.angle(np.exp(1j * (a2 - a1))))
    mid = np.angle(np.exp(1j * a1) + np.exp(1j * a2))
    ctrl_r = r * max(0.02, 1.0 - d / np.pi) * 0.92
    p1 = np.array([np.cos(mid), np.sin(mid)]) * ctrl_r
    path = Path([p0, p1, p2], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color, lw=lw,
                           alpha=alpha, capstyle="round", zorder=2))


def draw_circle(ax, C, labels, mod_id, colors):
    n = len(labels)
    gap = np.deg2rad(7.0)
    span = 2 * np.pi - gap * len(MODULES)
    ang = np.empty(n)
    pos = np.pi / 2                       # start at the top, go clockwise
    step = span / n
    prev = mod_id[0]
    mod_ranges = []
    start = pos
    for i in range(n):
        if mod_id[i] != prev:
            mod_ranges.append((prev, start, pos + step * 0.0))
            pos -= gap
            start = pos
            prev = mod_id[i]
        ang[i] = pos - step / 2.0
        pos -= step
    mod_ranges.append((prev, start, pos))

    iu = np.triu_indices(n, 1)
    w = C[iu]
    sel = w > THRESH
    order = np.argsort(w[sel])
    ii, jj, ww = iu[0][sel][order], iu[1][sel][order], w[sel][order]
    within = mod_id[ii] == mod_id[jj]
    # between-module chords underneath, within-module chords on top
    for keep in (~within, within):
        for i, j, val in zip(ii[keep], jj[keep], ww[keep]):
            s = float(np.clip((val - THRESH) / (0.95 - THRESH), 0, 1))
            if mod_id[i] == mod_id[j]:
                bezier_chord(ax, ang[i], ang[j], colors[i],
                             lw=0.35 + 2.1 * s, alpha=0.30 + 0.50 * s)
            else:
                bezier_chord(ax, ang[i], ang[j], "#8d8d8d",
                             lw=0.25 + 1.3 * s, alpha=0.11 + 0.32 * s)

    ax.scatter(np.cos(ang), np.sin(ang), s=17, c=colors, zorder=5,
               edgecolors="white", linewidths=0.5)

    for i, lab in enumerate(labels):
        deg = np.rad2deg(ang[i]) % 360
        flip = 90 < deg < 270
        ax.text(1.085 * np.cos(ang[i]), 1.085 * np.sin(ang[i]), lab,
                rotation=deg + 180 if flip else deg,
                rotation_mode="anchor", ha="right" if flip else "left",
                va="center", fontsize=6.2, color="#2b2b2b")

    for mi, a0, a1 in mod_ranges:
        name, _, col = MODULES[mi]
        t0, t1 = np.rad2deg(min(a0, a1)), np.rad2deg(max(a0, a1))
        ax.add_patch(Arc((0, 0), 2 * 1.40, 2 * 1.40, theta1=t0, theta2=t1,
                         lw=3.2, color=col, zorder=4, capstyle="butt"))
        am = (a0 + a1) / 2.0
        deg = np.rad2deg(am) % 360
        flip = 90 < deg < 270
        ax.text(1.50 * np.cos(am), 1.50 * np.sin(am), name,
                rotation=deg + 180 if flip else deg, rotation_mode="anchor",
                ha="right" if flip else "left", va="center", fontsize=7.4,
                fontweight="bold", color=col)

    ax.set_xlim(-1.72, 1.72)
    ax.set_ylim(-1.72, 1.72)
    ax.set_aspect("equal")
    ax.axis("off")
    return int(sel.sum()), n * (n - 1) // 2


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(21)
    C, labels, mod_id, colors = build_matrix(rng)
    vlim = 0.8

    fig = plt.figure(figsize=(7.2, 4.35))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 0.035, 1.22],
                  wspace=0.17, left=0.085, right=0.975, top=0.885,
                  bottom=0.135)

    ax_m = fig.add_subplot(gs[0, 0])
    im = draw_matrix(ax_m, C, mod_id, vlim)
    ax_m.set_title("Module-sorted connectivity", fontsize=8.5, pad=16)
    add_panel_label(ax_m, "(a)", x=-0.115, y=1.035, size=10)

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=6.8, width=0.7, length=2.2, color="#333333")
    cb.set_label("Pearson $r$  (Fisher-$z$ averaged)", fontsize=7, labelpad=4)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + pos.height * 0.14, pos.width,
                      pos.height * 0.66])

    ax_c = fig.add_subplot(gs[0, 2])
    n_edge, n_pair = draw_circle(ax_c, C, labels, mod_id, colors)
    ax_c.set_title("Suprathreshold connectome", fontsize=8.5, pad=2)
    add_panel_label(ax_c, "(b)", x=-0.035, y=0.965, size=10)
    ax_c.text(0.5, -0.045, f"$r$ > {THRESH:.2f}   \u2022   {n_edge} of "
              f"{n_pair} edges  (density {100 * n_edge / n_pair:.0f}%)",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.8,
              color="#5c5c5c")

    handles = [Line2D([], [], marker="o", ls="none", ms=5, mfc=col,
                      mec="white", mew=0.5, label=f"{name}")
               for name, _, col in MODULES]
    handles.append(Line2D([], [], color="#8d8d8d", lw=1.6,
                          label="between-module edge"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=7.2, handletextpad=0.5, columnspacing=1.5,
               bbox_to_anchor=(0.53, 0.005))

    savefig(fig, "138_connectivity_matrix.png")


if __name__ == "__main__":
    main()
