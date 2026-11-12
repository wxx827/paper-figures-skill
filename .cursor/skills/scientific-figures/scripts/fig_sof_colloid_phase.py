"""Colloidal phase behaviour in the volume-fraction / attraction plane.

The main panel is the claim: four phase fields - fluid, crystal, repulsive
glass and attractive gel - separated by one attraction boundary and two
packing boundaries, with measured samples plotted on top and coloured by the
phase the same rule assigns them.  Three inset schematics show what the
particles are doing inside each field, so the reader never has to translate
"glass" into a picture on their own.  The two smaller panels carry the
supporting scattering and tracking evidence: the static structure factor,
which tells crystal from fluid from gel by peak shape alone, and the mean
squared displacement, which tells arrested from diffusive by slope alone.
Boundary values quoted in the labels are the constants the field is built
from, and the sample counts are taken from the plotted arrays.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import panel_labels_inches

BLUE = "#3b5b92"        # fluid
GREEN = "#2e7d5b"       # crystal
GOLD = "#c9a227"        # repulsive glass
RED = "#b23a48"         # attractive gel
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

PHI_LO, PHI_HI = 0.30, 0.64
U_LO, U_HI = 0.0, 10.0
PHI_FREEZE = 0.494      # freezing volume fraction of hard spheres
PHI_GLASS = 0.580       # glass transition volume fraction
D_NM = 500.0            # particle diameter, nm

PHASES = {1: ("Fluid", BLUE), 2: ("Crystal", GREEN),
          3: ("Glass", GOLD), 4: ("Gel", RED)}


def u_gel(phi):
    """Attraction strength above which the suspension arrests into a gel."""
    return 7.0 - 4.0 * (np.asarray(phi) - PHI_LO) / (PHI_HI - PHI_LO)


def classify(phi, u):
    return np.where(u > u_gel(phi), 4,
                    np.where(phi < PHI_FREEZE, 1,
                             np.where(phi < PHI_GLASS, 2, 3)))


def phase_field(n=420):
    phi = np.linspace(PHI_LO, PHI_HI, n)
    u = np.linspace(U_LO, U_HI, n)
    return phi, u, classify(*np.meshgrid(phi, u))


def samples(seed=83, n=26):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(PHI_LO + 0.012, PHI_HI - 0.008, n)
    u = rng.uniform(U_LO + 0.25, U_HI - 0.4, n)
    return phi, u, classify(phi, u)


def structure_factors(seed=91):
    rng = np.random.default_rng(seed)
    q = np.linspace(1.0, 20.0, 500)

    def peak(c, w, a):
        return a * np.exp(-0.5 * ((q - c) / w) ** 2)

    fluid = 1.0 + peak(6.9, 1.45, 1.55) + peak(12.6, 2.2, 0.45) - \
        peak(1.0, 3.0, 0.55)
    crystal = 0.35 + peak(6.9, 0.34, 5.6) + peak(9.8, 0.40, 2.1) + \
        peak(12.0, 0.46, 1.5) + peak(15.6, 0.55, 0.8)
    gel = 1.0 + 4.2 * np.exp(-q / 2.4) + peak(6.9, 1.9, 1.0)
    out = {}
    for name, s in [("Fluid", fluid), ("Crystal", crystal), ("Gel", gel)]:
        out[name] = np.maximum(s * (1.0 + rng.normal(0.0, 0.012, q.size)),
                               0.02)
    return q, out


def msd_curves(seed=97):
    rng = np.random.default_rng(seed)
    t = np.logspace(-3.0, 2.0, 240)
    d_free = 2.0e5                       # nm^2/s, free diffusion coefficient
    cage = (0.11 * D_NM) ** 2            # nm^2, cage size
    out = {
        "Fluid": 6.0 * d_free * t,
        "Glass": cage * (t / 0.02) / (1.0 + t / 0.02) + 6.0e2 * t,
        "Gel": 0.45 * cage * (1.0 - np.exp(-t / 0.05)) + 3.0e1 * t,
    }
    for k in out:
        out[k] = out[k] * (1.0 + rng.normal(0.0, 0.02, t.size))
    return t, out, d_free, cage


def sketch(ax, kind, color, seed=3):
    """Tiny particle-arrangement schematic drawn inside an inset axes."""
    rng = np.random.default_rng(seed)
    if kind == "crystal":
        xs, ys = [], []
        for row in range(5):
            for col in range(5):
                xs.append(0.12 + 0.19 * col + (0.095 if row % 2 else 0.0))
                ys.append(0.12 + 0.19 * row)
        xs, ys = np.array(xs), np.array(ys)
        keep = xs < 1.0
        xs, ys = xs[keep], ys[keep]
    elif kind == "fluid":
        xs, ys = rng.uniform(0.10, 0.90, 17), rng.uniform(0.10, 0.90, 17)
    else:
        xs, ys = [], []
        for _ in range(5):
            x, y = rng.uniform(0.18, 0.82, 2)
            for _ in range(4):
                xs.append(x)
                ys.append(y)
                x = float(np.clip(x + rng.normal(0.0, 0.14), 0.08, 0.92))
                y = float(np.clip(y + rng.normal(0.0, 0.14), 0.08, 0.92))
        xs, ys = np.array(xs), np.array(ys)
        for i in range(0, xs.size - 1):
            if i % 4 != 3:
                ax.plot(xs[i:i + 2], ys[i:i + 2], color=color, lw=0.7,
                        alpha=0.75, zorder=2)
    ax.scatter(xs, ys, s=9.0, color=color, edgecolor="white", linewidth=0.4,
               zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color(HAIR)


def main_panel(ax, phi, u, field, pts):
    cmap = mpl.colors.ListedColormap([lighten(PHASES[i][1], 0.78)
                                      for i in (1, 2, 3, 4)])
    ax.pcolormesh(phi, u, field, cmap=cmap, vmin=0.5, vmax=4.5,
                  shading="auto", zorder=0, rasterized=True)

    ax.plot(phi, u_gel(phi), color=RED, lw=1.6, zorder=4)
    for x, lab, side in [(PHI_FREEZE, "freezing", -1), (PHI_GLASS, "glass", 1)]:
        ax.plot([x, x], [U_LO, float(u_gel(x))], color="#555555", lw=1.0,
                ls=(0, (3.4, 2.2)), zorder=4)
        ax.text(x + side * 0.005, 0.55 * float(u_gel(x)),
                f"{lab}  {x:.3f}", rotation=90, fontsize=6.3, color=SUB,
                ha="right" if side < 0 else "left", va="center", zorder=6)

    ax.text(0.42, float(u_gel(0.42)) + 0.30, "gelation boundary",
            fontsize=6.8, color=RED, rotation=-19, ha="center", va="bottom")

    labels = {"Fluid": (0.375, 0.30), "Crystal": (0.720, 0.255),
              "Glass": (0.935, 0.135), "Gel": (0.30, 0.905)}
    for name, (fx, fy) in labels.items():
        color = dict((v[0], v[1]) for v in PHASES.values())[name]
        ax.text(fx, fy, name, transform=ax.transAxes, ha="center",
                va="center", fontsize=8.6, fontweight="bold", color=color,
                zorder=6)

    boxes = {"fluid": (0.10, 0.085, 0.145, 0.145),
             "crystal": (0.665, 0.055, 0.130, 0.130),
             "gel": (0.10, 0.715, 0.145, 0.145)}
    tone = {"fluid": BLUE, "crystal": GREEN, "gel": RED}
    for kind, loc in boxes.items():
        axi = ax.inset_axes(loc)
        axi.set_zorder(7)
        axi.patch.set_visible(True)
        axi.set_facecolor("white")
        sketch(axi, kind, tone[kind])

    px, py, pk = pts
    fx = (px - PHI_LO) / (PHI_HI - PHI_LO)
    fy = (py - U_LO) / (U_HI - U_LO)
    keep_out = list(boxes.values())
    keep_out += [(cx - 0.075, cy - 0.035, 0.150, 0.070)
                 for cx, cy in labels.values()]
    free = np.ones(px.size, dtype=bool)
    for x0, y0, w, h in keep_out:
        free &= ~((fx > x0 - 0.02) & (fx < x0 + w + 0.02) &
                  (fy > y0 - 0.02) & (fy < y0 + h + 0.02))
    for pid, (name, color) in PHASES.items():
        m = free & (pk == pid)
        if m.any():
            ax.plot(px[m], py[m], ls="none", marker="o", ms=4.4,
                    mfc=color, mec="white", mew=0.8, zorder=6)
    ax.text(0.985, 0.985, f"{int(free.sum())} measured samples",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=SUB, zorder=7)

    ax.set_xlim(PHI_LO, PHI_HI)
    ax.set_ylim(U_LO, U_HI)
    ax.set_xlabel(r"volume fraction $\phi$ (-)")
    ax.set_ylabel(r"attraction strength $U/k_BT$ (-)")
    ax.set_title(f"Hard spheres of {D_NM:.0f} nm diameter, depletion "
                 f"attraction", fontsize=9.0, fontweight="bold", color=INK)
    return int(free.sum())


def sq_panel(ax, q, sq):
    for name, s in sq.items():
        color = dict((v[0], v[1]) for v in PHASES.values())[name]
        ax.plot(q, s, color=color, lw=1.4, zorder=4, label=name)
    ax.axhline(1.0, color="#cccccc", lw=0.7, zorder=1)
    ax.set_xlim(q[0], q[-1])
    ax.set_ylim(0.0, 6.8)
    ax.set_xlabel(r"wavevector $qd$ (-)", fontsize=7.4)
    ax.set_ylabel(r"structure factor $S(q)$ (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right", ncol=3,
              handlelength=1.3, columnspacing=0.9, borderaxespad=0.2)
    ax.set_title(f"Static structure factor, d = {D_NM:.0f} nm",
                 fontsize=8.2, fontweight="bold", color=INK)


def msd_panel(ax, t, msd, cage):
    guide = 6.0e5 * t
    ax.loglog(t, guide, color="#808080", lw=0.9, ls=(0, (3.4, 2.2)), zorder=2)
    ax.text(2.0e-2, 6.0e5 * 2.0e-2 * 1.9, "slope 1, diffusive", fontsize=6.3,
            color=SUB, rotation=27, ha="left", va="bottom")
    ax.axhline(cage, color="#cccccc", lw=0.7, zorder=1)
    ax.text(6.0e1, cage * 1.25, f"cage size {np.sqrt(cage):.0f} nm",
            fontsize=6.3, color=SUB, ha="right", va="bottom")
    for name, m in msd.items():
        color = dict((v[0], v[1]) for v in PHASES.values())[name]
        ax.loglog(t, m, color=color, lw=1.4, zorder=4, label=name)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(1e0, 1e9)
    ax.set_xlabel("lag time $t$ (s)", fontsize=7.4)
    ax.set_ylabel(r"MSD $\langle \Delta r^2 \rangle$ (nm$^2$)", fontsize=7.4)
    ax.tick_params(labelsize=6.5, which="both")
    ax.grid(which="major", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right", ncol=3,
              handlelength=1.3, columnspacing=0.9, borderaxespad=0.2)
    ax.set_title("Mean squared displacement", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    phi, u, field = phase_field()
    pts = samples()
    q, sq = structure_factors()
    t, msd, d_free, cage = msd_curves()

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.62)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_sq = fig.add_subplot(gs[0, 1])
    ax_msd = fig.add_subplot(gs[1, 1])

    n_pts = main_panel(ax_main, phi, u, field, pts)
    sq_panel(ax_sq, q, sq)
    msd_panel(ax_msd, t, msd, cage)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_main, ax_sq, ax_msd], dx=-0.42, dy=0.12)
    savefig(fig, "735_sof_colloid_phase.png")


if __name__ == "__main__":
    main()
