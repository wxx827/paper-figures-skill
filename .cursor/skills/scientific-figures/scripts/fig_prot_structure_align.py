"""Superposition of a predicted model onto a reference protein backbone.

Main panel: the two C-alpha traces in 3-D, the reference drawn translucent and
the model coloured by its local deviation.  Companions: per-residue deviation
along the sequence and the mean deviation of each secondary-structure element,
both on one shared deviation scale.

Backbones are synthesised from parametric secondary-structure segments (helix
= 3.6 residues per turn on a 2.3 A radius, strand = extended zig-zag, loop =
smoothed random walk), so the trace has a believable fold topology.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import panel_labels_inches, shared_limits

SEED = 4471

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
H_RED, H_BLUE, H_GREEN = "#b23a48", "#3b5b92", "#2e7d5b"
SS_COLOR = {"H": H_RED, "E": H_BLUE, "C": "#9a9a9a"}

# (type, length) along the chain; H = alpha helix, E = beta strand, C = coil
TOPOLOGY = [("C", 4), ("E", 7), ("C", 5), ("H", 14), ("C", 5), ("E", 8),
            ("C", 4), ("H", 12), ("C", 6), ("E", 7), ("C", 5), ("H", 16),
            ("C", 4), ("E", 6), ("C", 7), ("H", 10), ("C", 5), ("E", 7),
            ("C", 4)]


# ---------------------------------------------------------------- geometry --
def _rot_to(a, b):
    """Rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
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
    """Right-handed alpha helix: radius 2.3 A, rise 1.5 A, 100 deg per residue."""
    t = np.arange(n, dtype=float)
    a = np.deg2rad(100.0) * t
    return np.column_stack([2.3 * np.cos(a), 2.3 * np.sin(a), 1.5 * t])


def seg_strand(n, rng):
    """Extended beta strand: 3.3 A rise with a 0.9 A pleat."""
    t = np.arange(n, dtype=float)
    return np.column_stack([0.9 * (-1.0) ** t,
                            0.25 * rng.normal(0.0, 1.0, n), 3.3 * t])


def seg_coil(n, rng):
    """Random coil: smoothed random walk with a 3.8 A virtual bond."""
    pts = [np.zeros(3)]
    v = np.array([0.0, 0.0, 1.0])
    for _ in range(n - 1):
        v = v + rng.normal(0.0, 0.62, 3)
        v /= np.linalg.norm(v)
        pts.append(pts[-1] + 3.8 * v)
    return np.asarray(pts)


SEGMENTS = {"H": seg_helix, "E": seg_strand, "C": seg_coil}


def build_chain(topology, rng, radius=17.0):
    """Concatenate secondary-structure segments into one globular chain.

    Each new segment points along the previous one plus noise, plus a restoring
    pull toward the centre that grows with the distance already travelled --
    without that term the chain simply walks away and looks like a filament.
    """
    chunks, labels, bounds = [], [], []
    cur = np.zeros(3)
    d = np.array([1.0, 0.25, 0.1])
    d /= np.linalg.norm(d)
    i0 = 0
    for kind, n in topology:
        seg = SEGMENTS[kind](n, rng)
        seg = seg @ _rot_to([0.0, 0.0, 1.0], d).T
        seg = seg - seg[0] + cur
        chunks.append(seg)
        labels += [kind] * n
        bounds.append((kind, i0, i0 + n - 1))
        i0 += n
        cur = seg[-1] + 3.8 * d
        tail = seg[-1] - seg[max(n - 4, 0)]
        if np.linalg.norm(tail) > 1e-6:
            d = tail / np.linalg.norm(tail)
        r = float(np.linalg.norm(cur))
        pull = -cur / (r + 1e-9) * float(np.clip(r / radius, 0.0, 2.6))
        d = 0.45 * d + rng.normal(0.0, 0.45, 3) + 1.15 * pull
        d /= np.linalg.norm(d)
    xyz = np.vstack(chunks)
    return xyz - xyz.mean(axis=0), np.array(labels), bounds


def _smooth(x, w):
    k = np.exp(-0.5 * (np.arange(-3 * w, 3 * w + 1) / w) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, 3 * w, mode="reflect"), k,
                       mode="same")[3 * w:-3 * w]


# -------------------------------------------------------------------- draw --
def draw_3d(fig, ax, ref, mod, dev, norm, cmap, ss, bounds):
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="#8a8a8a", lw=2.9,
            alpha=0.42, solid_capstyle="round", zorder=1)
    segs = np.stack([mod[:-1], mod[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=2.6,
                          capstyle="round", zorder=4)
    lc.set_array(0.5 * (dev[:-1] + dev[1:]))
    ax.add_collection3d(lc)

    worst = int(np.argmax(dev))
    smoothed = _smooth(dev, 6)
    structured = np.array([c != "C" for c in ss])
    far = np.arange(len(dev)) > worst + 45
    cand = np.where(structured & far)[0]
    if cand.size == 0:
        cand = np.where(structured)[0]
    core = int(cand[np.argmin(smoothed[cand])])
    for idx, txt, col, ha in ((core, "aligned core", SUB, "right"),
                              (worst, "divergent loop", H_RED, "left")):
        p = mod[idx]
        pad = "  " if ha == "left" else ""
        tail = "" if ha == "left" else "  "
        ax.text(p[0], p[1], p[2], f"{pad}{txt}\n{pad}residue {idx + 1}{tail}",
                fontsize=6.8, color=col, zorder=9, ha=ha, va="center",
                linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                          edgecolor="none", alpha=0.78))
        ax.scatter([p[0]], [p[1]], [p[2]], s=16, facecolor="white",
                   edgecolor=col, linewidth=0.9, depthshade=False, zorder=8)
    for idx, txt in ((0, "N"), (len(mod) - 1, "C")):
        p = mod[idx]
        ax.text(p[0], p[1], p[2], txt, fontsize=8.0, fontweight="bold",
                color=INK, ha="center", va="center", zorder=9)

    span = np.ptp(np.vstack([ref, mod]), axis=0)
    ax.set_box_aspect(tuple(span / span.max()))
    ax.view_init(elev=18, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#e2e2e2")
        axis.line.set_color("#bbbbbb")
    ax.grid(True, color="#ececec", linewidth=0.4)
    ax.set_xlabel(r"x ($\mathrm{\AA}$)", fontsize=7.4, labelpad=-4.0)
    ax.set_ylabel(r"y ($\mathrm{\AA}$)", fontsize=7.4, labelpad=-4.0)
    ax.set_zlabel(r"z ($\mathrm{\AA}$)", fontsize=7.4, labelpad=-4.0)
    ax.tick_params(labelsize=6.2, pad=-1.5)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)

    ref, ss, bounds = build_chain(TOPOLOGY, rng)
    n = len(ref)
    res = np.arange(1, n + 1)

    # deviation field: quiet in the secondary structure, loud in two loops
    base = 0.28 + 0.55 * np.abs(_smooth(rng.normal(0.0, 1.0, n), 5))
    for centre, amp, width in ((0.30 * n, 4.6, 7.0), (0.74 * n, 3.1, 5.5)):
        base += amp * np.exp(-0.5 * ((res - centre) / width) ** 2)
    base[np.array([c == "C" for c in ss])] *= 1.35
    dev = _smooth(base, 2)

    dirs = np.column_stack([_smooth(rng.normal(0.0, 1.0, n), 6)
                            for _ in range(3)])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mod = ref + dirs * dev[:, None]

    core_mask = dev < 3.0
    rmsd_all = float(np.sqrt(np.mean(dev ** 2)))
    rmsd_core = float(np.sqrt(np.mean(dev[core_mask] ** 2)))

    cmap = get_cmap("cb_sequential_warm")
    lo, hi = 0.0, float(np.ceil(dev.max() * 2) / 2)
    norm = plt.Normalize(lo, hi)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.015, right=0.965, bottom=0.095, top=0.905,
                          wspace=0.13, hspace=0.52)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axd = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, 1])

    draw_3d(fig, ax3, ref, mod, dev, norm, cmap, ss, bounds)
    ax3.set_title("Model superposed on the reference backbone",
                  fontsize=9.0, fontweight="bold", pad=-6.0)

    cax = fig.add_axes([0.048, 0.760, 0.150, 0.020])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(RULE)
    cb.ax.tick_params(labelsize=6.4, width=0.7, length=2.5, color=RULE)
    cb.set_label(r"local deviation ($\mathrm{\AA}$)", fontsize=6.8, color=INK)

    # ---- companion 1: deviation along the sequence -------------------------
    elem = [(k, a, b) for k, a, b in bounds if k in ("H", "E")]
    means = np.array([dev[a:b + 1].mean() for _, a, b in elem])
    ylo, yhi = shared_limits([dev, means, [0.0]], pad=0.06)

    for k, a, b in bounds:
        if k == "C":
            continue
        axd.axvspan(a + 1, b + 1, color=SS_COLOR[k], alpha=0.11, lw=0,
                    zorder=0)
    axd.plot(res, dev, color=RULE, lw=1.1, zorder=4)
    axd.fill_between(res, 0.0, dev, color=cmap(0.45), alpha=0.32, lw=0,
                     zorder=3)
    axd.axhline(3.0, color=H_RED, lw=0.8, ls=(0, (3.2, 2.0)), zorder=5)
    axd.text(n * 0.985, 3.18, r"3 $\mathrm{\AA}$ cut-off", fontsize=6.3,
             color=H_RED, ha="right", va="bottom")
    worst = int(np.argmax(dev))
    axd.annotate(f"peak {dev[worst]:.1f} " + r"$\mathrm{\AA}$",
                 xy=(worst + 1, dev[worst]),
                 xytext=(worst + 1 + 24, dev[worst] - 0.15),
                 fontsize=6.5, color=H_RED, va="center",

                 arrowprops=dict(arrowstyle="-|>", lw=0.7, mutation_scale=6,
                                 color=H_RED, shrinkA=1, shrinkB=2))
    axd.set_xlim(1, n)
    axd.set_ylim(ylo, yhi)
    axd.set_xlabel("residue index (-)", fontsize=7.6)
    axd.set_ylabel(r"C$_\alpha$ deviation ($\mathrm{\AA}$)", fontsize=7.6)
    axd.tick_params(labelsize=6.5, length=2.4)
    axd.grid(axis="y", lw=0.45, color="#ececec")
    axd.set_axisbelow(True)
    axd.set_title("Deviation along the sequence", fontsize=8.5,
                  fontweight="bold", pad=4.0)

    # ---- companion 2: per-element mean deviation ---------------------------
    xs = np.arange(len(elem))
    axb.bar(xs, means, width=0.66,
            color=[SS_COLOR[k] for k, _, _ in elem], edgecolor="white",
            linewidth=0.6, zorder=3)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"{'H' if k == 'H' else 'S'}{i + 1}"
                         for i, (k, _, _) in enumerate(elem)], fontsize=6.4)
    axb.set_ylim(ylo, yhi)
    axb.set_ylabel(r"mean dev. ($\mathrm{\AA}$)", fontsize=7.6)
    axb.set_xlabel("secondary-structure element (-)", fontsize=7.6)
    axb.tick_params(labelsize=6.5, length=2.4)
    axb.grid(axis="y", lw=0.45, color="#ececec")
    axb.set_axisbelow(True)
    axb.set_title("Element-wise agreement", fontsize=8.5, fontweight="bold",
                  pad=4.0)

    handles = [
        Line2D([], [], color="#8a8a8a", lw=2.6, alpha=0.55,
               label="reference backbone"),
        Line2D([], [], color=cmap(0.75), lw=2.6, label="model backbone"),
        Line2D([], [], color=H_RED, lw=6, alpha=0.18, label=r"$\alpha$ helix"),
        Line2D([], [], color=H_BLUE, lw=6, alpha=0.18,
               label=r"$\beta$ strand"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.062), ncol=4, fontsize=7.0,
               frameon=False, handlelength=1.8, columnspacing=1.6)

    fig.text(0.048, 0.700,
             f"global RMSD  {rmsd_all:.2f} " + r"$\mathrm{\AA}$" + "\n"
             f"core RMSD    {rmsd_core:.2f} " + r"$\mathrm{\AA}$" + "\n"
             f"aligned      {int(core_mask.sum())} / {n} residues",
             fontsize=7.0, color=SUB, ha="left", va="top", linespacing=1.55,
             family="DejaVu Sans Mono")

    panel_labels_inches(fig, [ax3, axd, axb], dx=-0.10, dy=0.10)
    savefig(fig, "523_prot_structure_align.png")


if __name__ == "__main__":
    main()
