"""Figure 538 - Modular architecture of a resting-state network.

Main panel: a circular connectogram, parcels placed around the ring in module
order with a coloured band naming each module and chords for the strongest
edges.  Within-module chords take the module colour, between-module chords
stay grey, so the block structure is legible without reading a single label.

Subordinate panels: how modularity Q depends on the edge density you happen to
threshold at (the number everyone quotes and nobody qualifies), and the
Guimera-Amaral cartography that separates provincial from connector hubs.
Q, participation and degree are all recomputed from the same matrix that is
drawn, at the density marked in panel (b).

The module palette matches figures 536 and 539.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, COLORS
from figkit import axes_inches, panel_labels_inches

# (name, parcels, colour) - identical order and colour to figure 536
NETWORKS = [
    ("Visual", 11, "#b23a48"),
    ("Somatomotor", 12, "#3b5b92"),
    ("Dorsal attention", 8, "#2e7d5b"),
    ("Salience", 7, "#c9a227"),
    ("Frontoparietal", 9, "#8e6bbf"),
    ("Default mode", 13, "#c0603f"),
]
NET_COV = np.array([
    [1.00, 0.30, 0.22, 0.10, 0.06, -0.10],
    [0.30, 1.00, 0.28, 0.18, 0.08, -0.16],
    [0.22, 0.28, 1.00, 0.34, 0.40, -0.42],
    [0.10, 0.18, 0.34, 1.00, 0.38, -0.24],
    [0.06, 0.08, 0.40, 0.38, 1.00, -0.18],
    [-0.10, -0.16, -0.42, -0.24, -0.18, 1.00],
])
N_VOL = 320
TR = 2.0
DENSITY = 0.12                 # working edge density for panels (a) and (c)
GAP_DEG = 4.5                  # blank arc between modules


def simulate(rng):
    """Network factor model with a handful of genuine cross-module hubs."""
    n_net = len(NETWORKS)
    sizes = np.array([n for _, n, _ in NETWORKS])
    labels = np.repeat(np.arange(n_net), sizes)
    L = np.linalg.cholesky(NET_COV + 1e-9 * np.eye(n_net))
    F = L @ rng.standard_normal((n_net, N_VOL))
    g = rng.standard_normal(N_VOL)

    # two parcels per module listen to a second network: without them every
    # surviving edge is within-module and the cartography panel is empty.
    partner = {}
    for m in range(n_net):
        idx = np.where(labels == m)[0]
        picks = rng.choice(idx, 2, replace=False)
        partner[int(picks[0])] = (m + 1) % n_net
        partner[int(picks[1])] = 4 if m != 4 else 1

    ts = np.empty((labels.size, N_VOL))
    for i, m in enumerate(labels):
        if i in partner:
            ts[i] = (0.72 * F[m] + 0.78 * F[partner[i]] + 0.18 * g +
                     0.42 * rng.standard_normal(N_VOL))
        else:
            ts[i] = (rng.uniform(0.60, 1.05) * F[m] + 0.20 * g +
                     rng.uniform(0.55, 0.95) * rng.standard_normal(N_VOL))
    return np.corrcoef(ts), labels


def binarise(R, density):
    """Keep the strongest positive edges up to the requested density."""
    n = R.shape[0]
    iu = np.triu_indices(n, 1)
    w = R[iu]
    k = max(int(round(density * w.size)), 1)
    thr = np.sort(w)[-k]
    A = np.zeros_like(R)
    A[iu] = (w >= thr).astype(float)
    return A + A.T, thr


def modularity(A, labels):
    """Newman Q for a fixed partition."""
    m2 = A.sum()
    if m2 <= 0:
        return 0.0
    k = A.sum(axis=0)
    q = 0.0
    for c in np.unique(labels):
        idx = labels == c
        q += A[np.ix_(idx, idx)].sum() / m2 - (k[idx].sum() / m2) ** 2
    return float(q)


def cartography(A, labels):
    """Participation coefficient and within-module degree z-score."""
    k = A.sum(axis=0)
    part = np.ones_like(k)
    zscore = np.zeros_like(k)
    for c in np.unique(labels):
        idx = labels == c
        kin = A[:, idx].sum(axis=1)
        part -= np.divide(kin, k, out=np.zeros_like(k), where=k > 0) ** 2
        ki = A[np.ix_(idx, idx)].sum(axis=1)
        s = ki.std(ddof=0)
        zscore[idx] = (ki - ki.mean()) / s if s > 1e-9 else 0.0
    return part, zscore


def ring_angles(labels):
    n = labels.size
    span = 360.0 - GAP_DEG * len(NETWORKS)
    step = span / n
    ang, a = np.empty(n), 90.0
    for i in range(n):
        if i and labels[i] != labels[i - 1]:
            a -= GAP_DEG
        ang[i] = a - step / 2.0
        a -= step
    return np.deg2rad(ang)


def bezier_chord(p0, p1, n=48):
    """Chord whose sag scales with the angular gap it spans.

    A fixed control point near the origin drags even neighbouring nodes on a
    dive through the centre, and the ring turns into a starburst.  Pulling the
    control point back toward the midpoint for short spans keeps local edges
    hugging the rim while long-range edges still cross the disc.
    """
    t = np.linspace(0, 1, n)[:, None]
    d = np.arccos(np.clip(np.dot(p0, p1) /
                          (np.linalg.norm(p0) * np.linalg.norm(p1)), -1, 1))
    c = (p0 + p1) / 2.0 * (1.0 - d / np.pi) ** 0.85
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(21)
    R, labels = simulate(rng)
    n = labels.size
    colors = [c for _, _, c in NETWORKS]
    names = [s for s, _, _ in NETWORKS]
    node_c = [colors[m] for m in labels]

    A, thr = binarise(R, DENSITY)
    deg = A.sum(axis=0)
    part, zsc = cartography(A, labels)
    q_here = modularity(A, labels)

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) circular connectogram ----------------------------------------
    side = 5.00
    ax = axes_inches(fig, (0.34, 0.62, side, side))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.40, 1.40)
    ax.set_ylim(-1.40, 1.40)

    th = ring_angles(labels)
    xy = np.column_stack([np.cos(th), np.sin(th)])
    iu = np.triu_indices(n, 1)
    order = np.argsort(R[iu])
    keep = [(iu[0][k], iu[1][k]) for k in order if A[iu[0][k], iu[1][k]] > 0]
    for i, j in keep:
        same = labels[i] == labels[j]
        w = (R[i, j] - thr) / (R.max() - thr + 1e-9)
        pts = bezier_chord(xy[i], xy[j])
        ax.plot(pts[:, 0], pts[:, 1],
                color=colors[labels[i]] if same else "#b3b3b3",
                lw=0.35 + 1.15 * w, alpha=0.72 if same else 0.42,
                solid_capstyle="round", zorder=3 if same else 2)

    ax.scatter(xy[:, 0], xy[:, 1], s=8 + 34 * deg / deg.max(), c=node_c,
               edgecolors="white", linewidths=0.4, zorder=5)

    for k, (name, cnt, col) in enumerate(NETWORKS):
        sel = np.where(labels == k)[0]
        a0, a1 = th[sel[0]], th[sel[-1]]
        arc = np.linspace(a0, a1, 60)
        ax.plot(1.09 * np.cos(arc), 1.09 * np.sin(arc), color=col, lw=3.4,
                solid_capstyle="butt", zorder=4)
        am = (a0 + a1) / 2.0
        ax.text(1.19 * np.cos(am), 1.19 * np.sin(am),
                f"{name}\n{cnt} parcels", color=col, fontsize=6.8,
                ha="left" if np.cos(am) > 0.05 else
                   ("right" if np.cos(am) < -0.05 else "center"),
                va="center", linespacing=1.3, zorder=6)

    ax.set_title(f"Strongest {DENSITY:.0%} of positive edges "
                 f"(r \u2265 {thr:.2f})", fontsize=8.8, fontweight="bold",
                 pad=2)
    ax.text(0.0, -1.34, f"node size = degree (2\u2013{int(deg.max())} edges)",
            fontsize=6.6, color=COLORS["muted"], ha="center", va="center")
    handles = [Line2D([], [], color="#8a8a8a", lw=1.4, label="within module"),
               Line2D([], [], color="#c9c9c9", lw=1.0,
                      label="between modules")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 0.0),
              frameon=False, fontsize=6.8, handlelength=1.6,
              labelspacing=0.25)

    # ---- (b) modularity against threshold ---------------------------------
    axb = axes_inches(fig, (6.95, 3.52, 2.90, 1.86))
    dens = np.linspace(0.02, 0.40, 40)
    qs = np.array([modularity(binarise(R, d)[0], labels) for d in dens])
    axb.plot(dens * 100, qs, color="#3b5b92", lw=1.6)
    axb.axvline(DENSITY * 100, color="#b23a48", lw=0.9, ls="--", dashes=(3, 2))
    axb.plot([DENSITY * 100], [q_here], marker="o", ms=4.0, color="#b23a48",
             zorder=5)
    axb.annotate(f"Q = {q_here:.3f} at {DENSITY:.0%}",
                 xy=(DENSITY * 100, q_here), xytext=(14, 6),
                 textcoords="offset points", fontsize=6.8, color="#b23a48",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#b23a48"))
    axb.set_xlabel("Edge density retained (%)", fontsize=7.6)
    axb.set_ylabel("Modularity $Q$ (-)", fontsize=7.6)
    axb.set_xlim(0, 40)
    axb.set_ylim(0, max(qs) * 1.22)
    axb.tick_params(labelsize=6.8, length=2.2)
    axb.set_title("Modularity depends on where you threshold", fontsize=8.2,
                  fontweight="bold", pad=4)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    # ---- (c) hub cartography ----------------------------------------------
    axc = axes_inches(fig, (6.95, 0.62, 2.90, 1.86))
    axc.axhline(2.5, color=COLORS["baseline"], lw=0.7, ls="--", dashes=(3, 2))
    axc.axvline(0.62, color=COLORS["baseline"], lw=0.7, ls="--", dashes=(3, 2))
    axc.scatter(part, zsc, s=16, c=node_c, edgecolors="white", linewidths=0.4,
                zorder=4)
    axc.set_ylim(-3.2, 3.4)
    axc.text(0.01, 3.28, f"provincial: {int((part < 0.62).sum())} parcels",
             fontsize=6.5, color=COLORS["sub"], ha="left", va="top")
    axc.text(0.845, 3.28, f"connector: {int((part >= 0.62).sum())}",
             fontsize=6.5, color=COLORS["sub"], ha="right", va="top")
    axc.text(0.01, 2.60, "hub threshold $z$ = 2.5", fontsize=6.2,
             color=COLORS["muted"], ha="left", va="bottom")
    axc.set_xlabel("Participation coefficient $P$ (-)", fontsize=7.6)
    axc.set_ylabel("Within-module degree $z$ (-)", fontsize=7.6)
    axc.set_xlim(-0.02, 0.86)
    axc.tick_params(labelsize=6.8, length=2.2)
    axc.set_title(f"Hub cartography at {DENSITY:.0%} density", fontsize=8.2,
                  fontweight="bold", pad=4)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)

    panel_labels_inches(fig, [ax, axb, axc], dx=-0.30, dy=0.26)

    savefig(fig, "538_fmri_network_modules.png")


if __name__ == "__main__":
    main()
