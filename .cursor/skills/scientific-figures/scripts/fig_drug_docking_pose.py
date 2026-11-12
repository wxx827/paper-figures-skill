"""Docked pose of a kinase inhibitor: a schematic cross-section of the binding
pocket with the ligand drawn as a skeletal formula, hydrogen bonds and
hydrophobic contacts annotated, beside the decomposed docking score and the
pose metrics it implies."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from pubstyle import set_pub_style, savefig, style_axes, lighten
from figkit import panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
RED = "#b23a48"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

PROTEIN = "#dfe6ee"
PROTEIN_EDGE = "#9aa7b5"
RT = 0.5925              # kcal/mol at 298 K, for the Kd conversion

CX, CY = 13.0, 13.2      # pocket centre, in arbitrary Angstrom-like units

# scoring term, contribution per pose (kcal/mol), colour
TERMS = [("van der Waals", [-6.8, -6.1, -5.2], BLUE),
         ("Hydrogen bonds", [-2.4, -1.5, -0.9], GREEN),
         ("Electrostatics", [-1.6, -1.2, -1.4], lighten(BLUE, 0.45)),
         ("Desolvation", [-1.1, -0.9, -0.7], GOLD),
         ("Torsional penalty", [2.1, 2.1, 2.1], RED)]

POSES = ["Pose 1\n(top ranked)", "Pose 2", "Pose 3"]
RMSD = [0.8, 1.9, 3.4]          # A, against the co-crystal ligand
BURIED = [486.0, 402.0, 351.0]  # A^2 of buried solvent-accessible surface

# residue label, angle on the cavity rim (deg), contact type
RESIDUES = [("Leu 83", 78.0, "hbond"), ("Glu 81", 128.0, "hbond"),
            ("Asp 145", 262.0, "hbond"), ("Val 18", 168.0, "phobic"),
            ("Ala 31", 205.0, "phobic"), ("Phe 80", 22.0, "phobic"),
            ("Lys 33", 318.0, "polar"), ("Thr 47", 232.0, "polar")]


def blob(cx, cy, r0, harmonics, n=400):
    """Closed irregular contour: a base radius plus a few sine harmonics."""
    th = np.linspace(0.0, 2 * np.pi, n)
    r = np.full_like(th, r0)
    for k, amp, ph in harmonics:
        r = r + amp * np.sin(k * th + ph)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]), th, r


def cavity_radius(a):
    """Radius of the cavity rim at angle ``a`` (radians)."""
    return 4.55 + 0.62 * np.sin(2 * a + 1.15) + 0.34 * np.cos(4 * a + 0.4)


def hexagon(cx, cy, r=1.05, rot=0.0):
    a = np.deg2rad(rot) + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def draw_ring(ax, verts, aromatic=True, lw=1.5):
    ax.add_patch(Polygon(verts, closed=True, facecolor="none", edgecolor=INK,
                         linewidth=lw, joinstyle="round", zorder=6))
    if not aromatic:
        return
    c = verts.mean(axis=0)
    for i in range(0, 6, 2):
        p, q = verts[i], verts[(i + 1) % 6]
        pi = p + (c - p) * 0.20
        qi = q + (c - q) * 0.20
        ax.plot([pi[0], qi[0]], [pi[1], qi[1]], color=INK, lw=1.1, zorder=6,
                solid_capstyle="round")


def bond(ax, p, q, lw=1.5, double=False, off=0.13):
    p, q = np.asarray(p, float), np.asarray(q, float)
    ax.plot([p[0], q[0]], [p[1], q[1]], color=INK, lw=lw, zorder=6,
            solid_capstyle="round")
    if double:
        d = q - p
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * off
        ax.plot([p[0] + n[0], q[0] + n[0]], [p[1] + n[1], q[1] + n[1]],
                color=INK, lw=1.1, zorder=6, solid_capstyle="round")


def atom(ax, p, sym, color):
    ax.add_patch(Circle(p, 0.30, facecolor="white", edgecolor="none",
                        zorder=7))
    ax.text(p[0], p[1], sym, ha="center", va="center", fontsize=7.0,
            color=color, zorder=8)


def hydrophobic_arc(ax, centre, target, span=52.0, r=0.72, n_spoke=5):
    """LigPlot-style eyelash arc facing the contacting residue."""
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    a0 = np.degrees(np.arctan2(*(target - centre)[::-1]))
    a = np.deg2rad(np.linspace(a0 - span / 2, a0 + span / 2, 60))
    ax.plot(centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), color=GOLD,
            lw=1.1, zorder=6)
    for ang in np.linspace(a[0], a[-1], n_spoke):
        ax.plot([centre[0] + r * np.cos(ang), centre[0] + 1.28 * r
                 * np.cos(ang)],
                [centre[1] + r * np.sin(ang), centre[1] + 1.28 * r
                 * np.sin(ang)], color=GOLD, lw=0.9, zorder=6)


def build_ligand():
    """Skeletal coordinates of the docked inhibitor, in pocket units."""
    ring_a = hexagon(CX - 2.35, CY + 0.95, rot=0.0)          # phenol
    ring_b = hexagon(CX + 1.05, CY + 0.95, rot=0.0)          # aminopyrimidine
    link = (ring_a[0], ring_b[3])
    oh = ring_a[3] + np.array([-1.05, 0.30])                 # -OH
    nh2 = ring_b[1] + np.array([0.55, 0.95])                 # -NH2
    c_carbonyl = ring_b[5] + np.array([0.95, -0.62])
    o_carbonyl = c_carbonyl + np.array([0.92, 0.42])
    n_amide = c_carbonyl + np.array([0.20, -1.12])
    ring_c = hexagon(n_amide[0] + 0.75, n_amide[1] - 1.20, r=0.95, rot=12.0)
    o_ring = ring_c[2]
    lig = dict(ring_a=ring_a, ring_b=ring_b, link=link, oh=oh, nh2=nh2,
               c_carbonyl=c_carbonyl, o_carbonyl=o_carbonyl, n_amide=n_amide,
               ring_c=ring_c, o_ring=o_ring)

    # keep the whole skeleton clear of the cavity rim
    ctr, scale, shift = np.array([CX, CY]), 0.90, np.array([-0.15, 0.45])

    def T(p):
        return ctr + (np.asarray(p, float) - ctr) * scale + shift

    return {k: (tuple(T(q) for q in v) if isinstance(v, tuple) else T(v))
            for k, v in lig.items()}


def pocket_panel(ax, lig):
    outer, _th, _r = blob(CX, CY - 0.4, 10.6,
                          [(3, 1.15, 0.7), (5, 0.72, 2.1), (7, 0.42, 0.3)])
    ax.add_patch(Polygon(outer, closed=True, facecolor=PROTEIN,
                         edgecolor=PROTEIN_EDGE, linewidth=1.1, zorder=1))
    inner = 0.90 * (outer - np.array([CX, CY - 0.4])) + np.array([CX, CY - 0.4])
    ax.add_patch(Polygon(inner, closed=True, facecolor=lighten(PROTEIN, 0.35),
                         edgecolor="none", zorder=2))

    th = np.linspace(0.0, 2 * np.pi, 400)
    rc = cavity_radius(th)
    cav = np.column_stack([CX + rc * np.cos(th), CY + rc * np.sin(th)])
    ax.add_patch(Polygon(cav, closed=True, facecolor="white",
                         edgecolor=PROTEIN_EDGE, linewidth=1.0, zorder=3))

    # residues sit on the rim, labels are pushed out into the protein body
    pos = {}
    for name, ang, kind in RESIDUES:
        a = np.deg2rad(ang)
        rr = cavity_radius(a)
        p = np.array([CX + rr * np.cos(a), CY + rr * np.sin(a)])
        pos[name] = p
        col = {"hbond": GREEN, "phobic": GOLD, "polar": SUB}[kind]
        ax.add_patch(Circle(p, 0.26, facecolor="white", edgecolor=col,
                            linewidth=1.0, zorder=5))
        lp = np.array([CX + (rr + 1.35) * np.cos(a),
                       CY + (rr + 1.35) * np.sin(a)])
        ax.text(lp[0], lp[1], name, ha="center", va="center", fontsize=6.8,
                color=INK, zorder=6)

    draw_ring(ax, lig["ring_a"])
    draw_ring(ax, lig["ring_b"])
    draw_ring(ax, lig["ring_c"], aromatic=False)
    bond(ax, *lig["link"])
    bond(ax, lig["ring_a"][3], lig["oh"])
    bond(ax, lig["ring_b"][1], lig["nh2"])
    bond(ax, lig["ring_b"][5], lig["c_carbonyl"])
    bond(ax, lig["c_carbonyl"], lig["o_carbonyl"], double=True)
    bond(ax, lig["c_carbonyl"], lig["n_amide"])
    bond(ax, lig["n_amide"], lig["ring_c"][0])
    atom(ax, lig["oh"], "OH", RED)
    atom(ax, lig["nh2"], "NH$_2$", BLUE)
    atom(ax, lig["o_carbonyl"], "O", RED)
    atom(ax, lig["n_amide"], "N", BLUE)
    atom(ax, lig["o_ring"], "O", RED)

    hb = [("Leu 83", lig["nh2"]), ("Glu 81", lig["oh"]),
          ("Asp 145", lig["n_amide"])]
    dists = []
    for name, p in hb:
        q = pos[name]
        d = float(np.linalg.norm(q - p))
        dists.append(d)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GREEN, lw=1.2,
                ls=(0, (2.4, 1.8)), zorder=5)
        u = (q - p) / max(d, 1e-9)
        nrm = np.array([-u[1], u[0]])
        m = p + u * d * 0.52 + nrm * 0.55
        ax.text(m[0], m[1], f"{2.55 + 0.11 * d:.2f} \u00c5", ha="center",
                va="center", fontsize=6.2, color=GREEN, zorder=8)

    for name in ("Val 18", "Ala 31", "Phe 80"):
        anchor = (lig["ring_a"].mean(axis=0) if name != "Phe 80"
                  else lig["ring_b"].mean(axis=0))
        hydrophobic_arc(ax, anchor + 0.62 * (pos[name] - anchor)
                        / np.linalg.norm(pos[name] - anchor) * 1.9,
                        pos[name])

    ax.plot([], [], color=GREEN, lw=1.2, ls=(0, (2.4, 1.8)),
            label="hydrogen bond")
    ax.plot([], [], color=GOLD, lw=1.1, label="hydrophobic contact")
    ax.plot([], [], ls="none", marker="o", ms=4.0, mfc="white", mec=SUB,
            mew=1.0, label="pocket-lining residue")
    ax.legend(loc="lower left", fontsize=6.4, frameon=True, framealpha=0.94,
              facecolor="white", edgecolor="#cccccc", borderpad=0.4,
              handlelength=1.5, labelspacing=0.32,
              bbox_to_anchor=(0.005, 0.005))

    ax.set_xlim(1.4, 24.6)
    ax.set_ylim(1.0, 24.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Binding-pocket cross-section, top-ranked pose",
                 fontsize=9.0, fontweight="bold", pad=6)
    return dists


def score_panel(ax, totals):
    x = np.arange(len(POSES))
    pos_base = np.zeros(len(POSES))
    neg_base = np.zeros(len(POSES))
    for name, vals, col in TERMS:
        v = np.array(vals, dtype=float)
        base = np.where(v >= 0, pos_base, neg_base)
        ax.bar(x, v, bottom=base, width=0.56, color=col, edgecolor="white",
               linewidth=0.6, zorder=3, label=name)
        pos_base = pos_base + np.where(v >= 0, v, 0.0)
        neg_base = neg_base + np.where(v < 0, v, 0.0)

    ax.plot(x, totals, ls="none", marker="D", ms=4.6, mfc="white", mec=INK,
            mew=1.1, zorder=6)
    for xi, t in zip(x, totals):
        ax.text(xi + 0.34, t, f"{t:.1f}", ha="left", va="center",
                fontsize=6.4, color=INK, zorder=6)
    ax.axhline(0.0, color=BASE, lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("\n", " ") for p in POSES], fontsize=6.4)
    ax.set_xlim(-0.6, len(POSES) - 0.15)
    ax.set_ylim(min(neg_base) * 1.52, max(pos_base) * 1.9)
    ax.set_ylabel("Score contribution\n$\\Delta G$ (kcal/mol)", fontsize=7.5)
    ax.set_title("Docking score decomposition", fontsize=8.5,
                 fontweight="bold", pad=4)
    ax.tick_params(labelsize=6.5)
    ax.yaxis.grid(True, color="#e6e9ec", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=6.2, frameon=False, ncol=2,
              handlelength=1.2, columnspacing=1.0, labelspacing=0.28,
              borderaxespad=0.25, bbox_to_anchor=(0.005, -0.02))
    style_axes(ax)
    ax.plot([], [], ls="none", marker="D", ms=4.6, mfc="white", mec=INK,
            mew=1.1)
    ax.text(0.99, 0.965, "$\\diamond$  total $\\Delta G$", transform=
            ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK)


def table_panel(ax, totals, n_hb):
    cols = [("$\\Delta G$\n(kcal/mol)", lambda i: f"{totals[i]:.1f}"),
            ("$K_d$\n(nM)", lambda i: f"{1e9 * np.exp(totals[i] / RT):.0f}"),
            ("RMSD\n(\u00c5)", lambda i: f"{RMSD[i]:.1f}"),
            ("Buried SASA\n(\u00c5$^2$)", lambda i: f"{BURIED[i]:.0f}"),
            ("H-bonds\n(n)", lambda i: f"{n_hb[i]}")]
    nrow, ncol = len(POSES), len(cols)
    ax.set_xlim(-0.05, ncol)
    ax.set_ylim(-0.85, nrow + 0.18)
    ax.axis("off")
    for j, (head, _f) in enumerate(cols):
        ax.text(j + 0.5, nrow + 0.10, head, ha="center", va="bottom",
                fontsize=6.1, color=INK, linespacing=1.30)
    for i in range(nrow):
        y = nrow - 1 - i
        ax.text(-0.12, y + 0.5, POSES[i].replace("\n", " "), ha="right",
                va="center", fontsize=6.5, color=INK)
        for j, (_h, fn) in enumerate(cols):
            top = i == 0
            ax.add_patch(Rectangle(
                (j + 0.05, y + 0.12), 0.90, 0.76,
                facecolor=lighten(BLUE, 0.82 if top else 0.92),
                edgecolor="white", linewidth=0.8, zorder=2))
            ax.text(j + 0.5, y + 0.50, fn(i), ha="center", va="center",
                    fontsize=6.4, color=INK,
                    fontweight="bold" if top else "normal", zorder=3)
    ax.text(0.0, -0.34, f"$K_d$ from $\\Delta G$ = RT ln $K_d$ with "
                        f"RT = {RT} kcal/mol; RMSD against the co-crystal "
                        "ligand", fontsize=6.0, color=MUTED, ha="left",
            va="top")
    ax.set_title("Pose metrics", fontsize=8.5, fontweight="bold", pad=16)


def main():
    set_pub_style()
    rng = np.random.default_rng(4520)
    lig = build_ligand()
    totals = [float(sum(vals[i] for _n, vals, _c in TERMS))
              for i in range(len(POSES))]
    n_hb = [3, 2, 1]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.72], wspace=0.28, hspace=0.55,
                          left=0.030, right=0.965, top=0.885, bottom=0.105)
    ax_p = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])

    dists = pocket_panel(ax_p, lig)
    score_panel(ax_s, totals)
    table_panel(ax_t, totals, n_hb)

    kd = 1e9 * np.exp(totals[0] / RT)

    panel_labels_inches(fig, [ax_p, ax_s, ax_t], dx=-0.30, dy=0.08)
    savefig(fig, "520_drug_docking_pose.png")


if __name__ == "__main__":
    main()
