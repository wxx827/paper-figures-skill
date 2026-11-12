"""Backbone flexibility rendered as a B-factor putty tube.

Main panel: the C-alpha trace drawn as a tube whose colour and thickness both
follow the B factor, so the rigid core reads thin and dark while the mobile
loops read thick and pale.  Companions: the B factor along the sequence with
the three most flexible stretches marked, and the distribution of B by
secondary-structure type.  All three panels share one B-factor scale.

The backbone is synthesised from parametric secondary-structure segments
(helix = 3.6 residues per turn on a 2.3 A radius, strand = extended zig-zag,
loop = smoothed random walk) with a restoring pull toward the centre so the
chain folds into a globule instead of a filament.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_label_inches, axes_inches

SEED = 60613

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
H_RED, H_BLUE = "#b23a48", "#3b5b92"
SS_COLOR = {"H": H_RED, "E": H_BLUE, "C": "#9a9a9a"}
SS_NAME = {"H": r"$\alpha$ helix", "E": r"$\beta$ strand", "C": "coil"}

TOPOLOGY = [("C", 5), ("H", 15), ("C", 6), ("E", 8), ("C", 7), ("H", 13),
            ("C", 5), ("E", 7), ("C", 8), ("H", 17), ("C", 6), ("E", 7),
            ("C", 9), ("H", 12), ("C", 5), ("E", 8), ("C", 6)]


def _rot_to(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))


def seg_helix(n, rng):
    t = np.arange(n, dtype=float)
    a = np.deg2rad(100.0) * t
    return np.column_stack([2.3 * np.cos(a), 2.3 * np.sin(a), 1.5 * t])


def seg_strand(n, rng):
    t = np.arange(n, dtype=float)
    return np.column_stack([0.9 * (-1.0) ** t,
                            0.25 * rng.normal(0.0, 1.0, n), 3.3 * t])


def seg_coil(n, rng):
    pts = [np.zeros(3)]
    v = np.array([0.0, 0.0, 1.0])
    for _ in range(n - 1):
        v = v + rng.normal(0.0, 0.62, 3)
        v /= np.linalg.norm(v)
        pts.append(pts[-1] + 3.8 * v)
    return np.asarray(pts)


SEGMENTS = {"H": seg_helix, "E": seg_strand, "C": seg_coil}


def build_chain(topology, rng, radius=12.0):
    chunks, labels = [], []
    cur = np.zeros(3)
    d = np.array([1.0, 0.3, 0.15])
    d /= np.linalg.norm(d)
    for kind, n in topology:
        seg = SEGMENTS[kind](n, rng)
        seg = seg @ _rot_to([0.0, 0.0, 1.0], d).T
        seg = seg - seg[0] + cur
        chunks.append(seg)
        labels += [kind] * n
        cur = seg[-1] + 3.8 * d
        tail = seg[-1] - seg[max(n - 4, 0)]
        if np.linalg.norm(tail) > 1e-6:
            d = tail / np.linalg.norm(tail)
        r = float(np.linalg.norm(cur))
        pull = -cur / (r + 1e-9) * float(np.clip(r / radius, 0.0, 3.0))
        d = 0.40 * d + rng.normal(0.0, 0.42, 3) + 1.35 * pull
        d /= np.linalg.norm(d)
    xyz = np.vstack(chunks)
    return xyz - xyz.mean(axis=0), np.array(labels)


def _smooth(x, w):
    k = np.exp(-0.5 * (np.arange(-3 * w, 3 * w + 1) / w) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, 3 * w, mode="reflect"), k,
                       mode="same")[3 * w:-3 * w]


def flexible_runs(b, res, n_keep=3, q=76.0):
    """Contiguous stretches above a high percentile, ranked by peak B."""
    thr = np.percentile(b, q)
    runs, start = [], None
    for i, hot in enumerate(b > thr):
        if hot and start is None:
            start = i
        elif not hot and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(b) - 1))
    runs = [r for r in runs if r[1] - r[0] >= 3]
    runs.sort(key=lambda r: -b[r[0]:r[1] + 1].max())
    return runs[:n_keep], thr


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    xyz, ss = build_chain(TOPOLOGY, rng)
    n = len(xyz)
    res = np.arange(1, n + 1)

    # B factor: quiet inside secondary structure, loud in loops and termini
    raw = 0.45 + 0.9 * np.abs(_smooth(rng.normal(0.0, 1.0, n), 5))
    raw[np.array([c == "C" for c in ss])] += 0.85
    raw += 1.35 * np.exp(-res / 7.0) + 1.35 * np.exp(-(n + 1 - res) / 7.0)
    radial = np.linalg.norm(xyz, axis=1)
    raw += 0.9 * (radial - radial.mean()) / (radial.std() + 1e-9) * 0.5
    raw = _smooth(raw, 2)
    b = 9.0 + 72.0 * (raw - raw.min()) / (raw.max() - raw.min())

    runs, thr = flexible_runs(b, res)
    blo = 0.0
    bhi = float(np.ceil(b.max() / 10.0) * 10.0) + 12.0   # headroom for labels
    norm = plt.Normalize(blo, bhi)
    base = get_cmap("cb_sequential_warm", reverse=True)
    cmap = ListedColormap(base(np.linspace(0.0, 0.74, 256)))

    FW, FH = 10.4, 6.4
    fig = plt.figure(figsize=(FW, FH))
    ax3 = fig.add_axes([0.045 / FW, 2.62 / FH, 6.05 / FW, 3.20 / FH],
                       projection="3d", computed_zorder=False)
    axb = axes_inches(fig, (0.86, 0.62, 9.05, 1.72))
    axv = axes_inches(fig, (7.15, 2.66, 2.75, 1.86))

    # ---- main: putty tube --------------------------------------------------
    segs = np.stack([xyz[:-1], xyz[1:]], axis=1)
    bmid = 0.5 * (b[:-1] + b[1:])
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, capstyle="round",
                          zorder=4)
    lc.set_array(bmid)
    lc.set_linewidth(1.0 + 4.2 * norm(bmid))
    ax3.add_collection3d(lc)

    for k, (i0, i1) in enumerate(runs):
        mid = (i0 + i1) // 2
        p = xyz[mid]
        ax3.text(p[0], p[1], p[2], f"  L{k + 1}", fontsize=7.4,
                 fontweight="bold", color=H_RED, ha="left", va="center",
                 zorder=9,
                 bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                           edgecolor="none", alpha=0.78))
    for idx, txt in ((0, "N"), (n - 1, "C")):
        p = xyz[idx]
        ax3.text(p[0], p[1], p[2], txt, fontsize=8.2, fontweight="bold",
                 color=INK, ha="center", va="center", zorder=9)

    span = np.ptp(xyz, axis=0)
    ax3.set_box_aspect(tuple(span / span.max()))
    ax3.view_init(elev=18, azim=-60)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e2e2e2")
        axis.line.set_color("#bbbbbb")
    ax3.grid(True, color="#ececec", linewidth=0.4)
    ax3.set_xlabel(r"x ($\mathrm{\AA}$)", fontsize=7.4, labelpad=-4.0)
    ax3.set_ylabel(r"y ($\mathrm{\AA}$)", fontsize=7.4, labelpad=-4.0)
    ax3.set_zlabel(r"z ($\mathrm{\AA}$)", fontsize=7.4, labelpad=-3.0)
    ax3.tick_params(labelsize=6.2, pad=-1.5)
    fig.text((0.045 + 6.05 / 2) / FW, 5.88 / FH,
             "Backbone putty sized and coloured by B factor",
             fontsize=9.0, fontweight="bold", ha="center", va="bottom")

    cax = fig.add_axes([7.15 / FW, 5.58 / FH, 1.70 / FW, 0.135 / FH])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color=RULE)
    cb.set_label(r"B factor ($\mathrm{\AA}^2$)", fontsize=6.8, color=INK)

    handles = [
        Line2D([], [], color=cmap(0.08), lw=1.2,
               label=r"rigid core (thin, dark)"),
        Line2D([], [], color=cmap(0.92), lw=4.2,
               label=r"mobile loop (thick, pale)"),
    ]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=(7.15 / FW, 5.24 / FH), ncol=1, fontsize=6.9,
               frameon=False, handlelength=2.2, handletextpad=0.6,
               labelspacing=0.42)

    # ---- companion: B factor along the sequence ----------------------------
    for i0, i1 in runs:
        axb.axvspan(res[i0], res[i1], color=H_RED, alpha=0.10, lw=0, zorder=0)
    points = np.stack([np.column_stack([res[:-1], b[:-1]]),
                       np.column_stack([res[1:], b[1:]])], axis=1)
    lc2 = LineCollection(points, cmap=cmap, norm=norm, linewidths=1.7,
                         zorder=5)
    lc2.set_array(bmid)
    axb.add_collection(lc2)
    axb.fill_between(res, blo, b, color=cmap(0.55), alpha=0.16, lw=0,
                     zorder=2)
    axb.axhline(thr, color=RULE, lw=0.7, ls=(0, (3.2, 2.0)), zorder=4)
    axb.text(n * 0.998, thr + 1.5, "76th-percentile cut-off "
             f"({thr:.0f} " + r"$\mathrm{\AA}^2$)", fontsize=6.3, color=SUB,
             ha="right", va="bottom")
    for k, (i0, i1) in enumerate(runs):
        pk = i0 + int(np.argmax(b[i0:i1 + 1]))
        frac = res[pk] / n
        ha = "left" if frac < 0.15 else ("right" if frac > 0.85 else "center")
        dx = {"left": 5, "right": -5, "center": 0}[ha]
        axb.annotate(f"L{k + 1}: {res[i0]}-{res[i1]}", xy=(res[pk], b[pk]),
                     xytext=(dx, 12), textcoords="offset points",
                     fontsize=6.6, color=H_RED, ha=ha, va="bottom",
                     arrowprops=dict(arrowstyle="-", lw=0.55, color=H_RED,
                                     shrinkA=1, shrinkB=2))
    axb.set_xlim(1, n)
    axb.set_ylim(blo, bhi)
    axb.set_xlabel("residue index (-)", fontsize=7.8)
    axb.set_ylabel(r"B factor ($\mathrm{\AA}^2$)", fontsize=7.8)
    axb.tick_params(labelsize=6.8, length=2.4)
    axb.grid(axis="y", lw=0.45, color="#ececec")
    axb.set_axisbelow(True)
    axb.set_title("B factor along the sequence, with the three most flexible "
                  "stretches marked", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    # ---- companion: distribution by secondary-structure type ---------------
    kinds = ["H", "E", "C"]
    groups = [b[np.array([c == k for c in ss])] for k in kinds]
    bp = axv.boxplot(groups, widths=0.52, patch_artist=True,
                     medianprops=dict(color="white", lw=1.1),
                     whiskerprops=dict(color=RULE, lw=0.8),
                     capprops=dict(color=RULE, lw=0.8),
                     flierprops=dict(marker="o", ms=2.4, mfc="none",
                                     mec=MUTED, mew=0.6))
    for patch, k in zip(bp["boxes"], kinds):
        patch.set_facecolor(SS_COLOR[k])
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
    for i, (k, g) in enumerate(zip(kinds, groups), start=1):
        axv.text(i, bhi * 0.965, f"{g.mean():.0f}", fontsize=6.5, color=SUB,
                 ha="center", va="top")
    axv.set_xticks([1, 2, 3])
    axv.set_xticklabels([SS_NAME[k] for k in kinds], fontsize=6.8)
    axv.set_ylim(blo, bhi)
    axv.set_ylabel(r"B factor ($\mathrm{\AA}^2$)", fontsize=7.6)
    axv.tick_params(labelsize=6.6, length=2.4)
    axv.grid(axis="y", lw=0.45, color="#ececec")
    axv.set_axisbelow(True)
    axv.set_title("B factor by structure type", fontsize=8.5,
                  fontweight="bold", pad=4.0)
    axv.text(0.025, 0.030, "mean B printed above each box",
             transform=axv.transAxes, fontsize=6.3, color=MUTED,
             ha="left", va="bottom")


    panel_label_inches(fig, ax3, "(a)", dx=0.02, dy=-0.24)
    panel_label_inches(fig, axv, "(b)", dx=-0.42, dy=0.10)
    panel_label_inches(fig, axb, "(c)", dx=-0.42, dy=0.10)
    savefig(fig, "527_prot_bfactor_flexibility.png")


if __name__ == "__main__":
    main()
