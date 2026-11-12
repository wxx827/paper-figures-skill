"""MEP coordination section with the clash register that comes out of it.

Every service run is a box with a discipline tag, so the clash test is a real
geometric test rather than a drawing convention: boxes that interpenetrate are
hard clashes, boxes separated by less than the installation clearance or less
than the maintenance access zone are the two softer grades.  The section drawn
in the main panel is the worst of twenty-four stations along the corridor; the
bar chart and the discipline matrix aggregate the same test over all of them,
so the counts and the red circles can never disagree.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

from pubstyle import set_pub_style, get_cmap, savefig
from figkit import panel_labels_inches, safe_colorbar

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"
CONCRETE = "#8a8a8a"

DISC = ["HVAC", "Plumbing", "Electrical", "Fire", "Structure"]
DCOL = {"HVAC": "#3b5b92", "Plumbing": "#2e7d5b", "Electrical": "#c9a227",
        "Fire": "#8e6bbf", "Structure": "#8a8a8a"}

CLEARANCE = 0.050        # m, minimum installation clearance
ACCESS = 0.150           # m, maintenance access zone
N_STATIONS = 24
STATION_STEP = 1.5       # m along the corridor

CORRIDOR = 2.60          # m clear width
CEILING = 2.68           # m, top of the finished ceiling
SOFFIT = 3.95            # m, underside of the slab
BEAM_DEPTH = 0.34        # m

# name, discipline, shape, x centre, z centre, width, height (m)
SERVICES = [
    ("Supply duct", "HVAC", "rect", 0.55, 3.30, 0.70, 0.35),
    ("Return duct", "HVAC", "rect", 1.45, 3.34, 0.60, 0.32),
    ("Extract duct", "HVAC", "rect", 2.25, 3.10, 0.42, 0.26),
    ("CHW flow", "Plumbing", "pipe", 0.30, 3.66, 0.18, 0.18),
    ("CHW return", "Plumbing", "pipe", 0.54, 3.66, 0.18, 0.18),
    ("Soil stack", "Plumbing", "pipe", 2.30, 3.70, 0.24, 0.24),
    ("Cold water", "Plumbing", "pipe", 1.95, 3.62, 0.12, 0.12),
    ("Cable tray A", "Electrical", "tray", 1.05, 2.90, 0.50, 0.08),
    ("Cable tray B", "Electrical", "tray", 2.05, 2.88, 0.40, 0.07),
    ("Busbar", "Electrical", "rect", 0.20, 2.92, 0.22, 0.10),
    ("Sprinkler main", "Fire", "pipe", 1.60, 2.92, 0.11, 0.11),
    ("Sprinkler branch", "Fire", "pipe", 0.95, 3.02, 0.07, 0.07),
    ("Transfer beam", "Structure", "rect", CORRIDOR / 2,
     SOFFIT - BEAM_DEPTH / 2, CORRIDOR, BEAM_DEPTH),
]


def station(rng, jitter=True):
    """One cross-section: services nudged by routing and installation slack."""
    out = []
    for name, disc, shape, x, z, w, h in SERVICES:
        if jitter:
            sx, sz = (0.008, 0.008) if disc == "Structure" else (0.045, 0.030)
            x = x + rng.normal(0.0, sx)
            z = z + rng.normal(0.0, sz)
        out.append(dict(name=name, disc=disc, shape=shape, x=x, z=z, w=w,
                        h=h, x0=x - w / 2, x1=x + w / 2,
                        z0=z - h / 2, z1=z + h / 2))
    return out


def test_clashes(items):
    """Signed gap between every pair; negative means interpenetration."""
    found = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            dx = max(a["x0"] - b["x1"], b["x0"] - a["x1"])
            dz = max(a["z0"] - b["z1"], b["z0"] - a["z1"])
            if dx < 0.0 and dz < 0.0:
                gap, kind = max(dx, dz), "Hard clash"
            else:
                gap = float(np.hypot(max(dx, 0.0), max(dz, 0.0)))
                if gap < CLEARANCE:
                    kind = "Clearance breach"
                elif gap < ACCESS:
                    kind = "Access breach"
                else:
                    continue
            found.append(dict(a=a, b=b, gap=gap, kind=kind,
                              x=0.5 * (max(a["x0"], b["x0"]) +
                                       min(a["x1"], b["x1"])),
                              z=0.5 * (max(a["z0"], b["z0"]) +
                                       min(a["z1"], b["z1"]))))
    return found


def survey(seed=41):
    rng = np.random.default_rng(seed)
    kinds = {"Hard clash": 0, "Clearance breach": 0, "Access breach": 0}
    matrix = np.zeros((len(DISC), len(DISC)))
    worst, worst_n, hard_struct = None, -1, 0
    for _ in range(N_STATIONS):
        items = station(rng)
        cl = test_clashes(items)
        hard = sum(1 for c in cl if c["kind"] == "Hard clash")
        if hard > worst_n:
            worst, worst_n = (items, cl), hard
        for c in cl:
            kinds[c["kind"]] += 1
            i, j = DISC.index(c["a"]["disc"]), DISC.index(c["b"]["disc"])
            matrix[i, j] += 1
            if i != j:
                matrix[j, i] += 1
            if c["kind"] == "Hard clash" and "Structure" in (
                    c["a"]["disc"], c["b"]["disc"]):
                hard_struct += 1
    return worst[0], worst[1], kinds, matrix, hard_struct


def draw_service(ax, it):
    col = DCOL[it["disc"]]
    if it["shape"] == "pipe":
        ax.add_patch(Circle((it["x"], it["z"]), it["w"] / 2, facecolor="white",
                            edgecolor=col, linewidth=1.1, zorder=4))
        ax.add_patch(Circle((it["x"], it["z"]), it["w"] / 2 * 0.55,
                            facecolor=col, edgecolor="none", alpha=0.30,
                            zorder=4))
    elif it["shape"] == "tray":
        ax.add_patch(Rectangle((it["x0"], it["z0"]), it["w"], it["h"],
                               facecolor="white", edgecolor=col,
                               linewidth=1.0, zorder=4))
        for xr in np.linspace(it["x0"] + 0.03, it["x1"] - 0.03, 6):
            ax.plot([xr, xr], [it["z0"], it["z1"]], color=col, lw=0.5,
                    zorder=5)
    else:
        ec = CONCRETE if it["disc"] == "Structure" else col
        fc = "#e8e8e8" if it["disc"] == "Structure" else "white"
        ax.add_patch(Rectangle((it["x0"], it["z0"]), it["w"], it["h"],
                               facecolor=fc, edgecolor=ec, linewidth=1.0,
                               hatch="//" if it["disc"] == "Structure" else
                               None, zorder=3))
        if it["disc"] == "HVAC":
            ax.plot([it["x0"] + 0.02, it["x1"] - 0.02],
                    [it["z"], it["z"]], color=col, lw=0.5, ls=(0, (3, 2)),
                    zorder=4)


def section_panel(ax, items, clashes):
    ax.add_patch(Rectangle((-0.22, -0.30), CORRIDOR + 0.44, 0.30,
                           facecolor="#e8e8e8", edgecolor=CONCRETE,
                           linewidth=0.8, hatch="//", zorder=2))
    ax.add_patch(Rectangle((-0.22, SOFFIT), CORRIDOR + 0.44, 0.30,
                           facecolor="#e8e8e8", edgecolor=CONCRETE,
                           linewidth=0.8, hatch="//", zorder=2))
    for x0 in (-0.22, CORRIDOR):
        ax.add_patch(Rectangle((x0, 0.0), 0.22, SOFFIT, facecolor="#e8e8e8",
                               edgecolor=CONCRETE, linewidth=0.8, hatch="//",
                               zorder=2))
    ax.add_patch(Rectangle((0.0, CEILING), CORRIDOR, 0.06,
                           facecolor="#f0f0f0", edgecolor=CONCRETE,
                           linewidth=0.7, zorder=3))
    ax.text(CORRIDOR / 2, 0.55, "occupied zone", ha="center", va="center",
            fontsize=6.6, color=MUTED)

    for it in items:
        draw_service(ax, it)

    # Only the two actionable grades are circled; marking every access breach
    # as well buries the drawing under overlapping rings.
    order = sorted([c for c in clashes if c["kind"] != "Access breach"],
                   key=lambda c: (c["x"], -c["z"]))
    for k, c in enumerate(order):
        ax.add_patch(Circle((c["x"], c["z"]), 0.085, facecolor="none",
                            edgecolor=ACCENT, linewidth=1.2, zorder=8))
        ax.text(c["x"] + 0.10, c["z"] + 0.10, f"{k + 1}", ha="left",
                va="bottom", fontsize=6.2, color=ACCENT, zorder=9)

    ax.text(2.95, SOFFIT + 0.14, "Hard and clearance items", ha="left",
            va="top", fontsize=7.0, color=INK)
    for k, c in enumerate(order):
        y = SOFFIT - 0.26 - k * 0.30
        ax.text(2.95, y, f"{k + 1}  {c['a']['name']} / {c['b']['name']}",
                ha="left", va="center", fontsize=6.2, color=INK)
        ax.text(3.13, y - 0.15, f"{c['kind']}, {c['gap'] * 1000:+.0f} mm",
                ha="left", va="center", fontsize=6.0,
                color=ACCENT if c["kind"] == "Hard clash" else SUB)

    for z0, z1, label in ((0.0, CEILING, f"{CEILING * 1000:.0f} mm clear"),
                          (CEILING + 0.06, SOFFIT - BEAM_DEPTH,
                           f"{(SOFFIT - BEAM_DEPTH - CEILING - 0.06) * 1000:.0f}"
                           f" mm void")):
        ax.annotate("", xy=(-0.52, z0), xytext=(-0.52, z1),
                    arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.7,
                                    mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.text(-0.60, (z0 + z1) / 2, label, rotation=90, ha="right",
                va="center", fontsize=6.4, color=SUB)

    handles = [Patch(facecolor="white", edgecolor=DCOL[d], linewidth=1.0,
                     label=d) for d in DISC]
    handles.append(Patch(facecolor="none", edgecolor=ACCENT, linewidth=1.2,
                         label="clash"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.21, 0.44),
              frameon=False, fontsize=6.6, ncol=2, handlelength=1.4,
              columnspacing=1.0, borderaxespad=0.0)

    ax.set_xlim(-1.05, 4.45)
    ax.set_ylim(-0.75, 4.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Coordination section at the worst station", fontsize=9.0,
                 fontweight="bold", color=INK)


def kind_panel(ax, kinds):
    names = ["Hard clash", "Clearance breach", "Access breach"]
    vals = [kinds[k] for k in names]
    colors = [ACCENT, "#c0603f", "#c9a227"]
    ax.barh(np.arange(3)[::-1], vals, height=0.6, color=colors,
            edgecolor="white", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, 2 - i, f"{v:.0f}", ha="left",
                va="center", fontsize=6.6, color=SUB)
    ax.set_yticks(np.arange(3)[::-1])
    ax.set_yticklabels([f"{n}\n({t})" for n, t in zip(
        names, ["overlap < 0 mm", f"gap < {CLEARANCE * 1000:.0f} mm",
                f"gap < {ACCESS * 1000:.0f} mm"])], fontsize=6.4)
    ax.set_xlim(0.0, max(vals) * 1.18)
    ax.set_xlabel("clashes over the whole run (count)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.set_title("Clash register by severity", fontsize=8.2,
                 fontweight="bold", color=INK)


def matrix_panel(ax, matrix):
    shown = np.where(matrix > 0, matrix, np.nan)
    im = ax.imshow(shown, cmap=get_cmap("cb_sequential_blue"), origin="upper",
                   vmin=0.0, vmax=float(np.nanmax(shown)))
    for i in range(len(DISC)):
        for j in range(len(DISC)):
            if matrix[i, j] > 0:
                dark = matrix[i, j] > 0.55 * matrix.max()
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.2, color="white" if dark else INK)
    ax.set_xticks(np.arange(len(DISC)))
    ax.set_yticks(np.arange(len(DISC)))
    ax.set_xticklabels(DISC, fontsize=6.4, rotation=32, ha="right")
    ax.set_yticklabels(DISC, fontsize=6.4)
    ax.set_xticks(np.arange(len(DISC) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(DISC) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Clashes by discipline pair", fontsize=8.2,
                 fontweight="bold", color=INK)
    return im


def main():
    set_pub_style()
    items, clashes, kinds, matrix, hard_struct = survey()
    total = sum(kinds.values())
    run = N_STATIONS * STATION_STEP

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.66)
    ax_sec = fig.add_subplot(gs[:, 0])
    ax_kind = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 1])

    section_panel(ax_sec, items, clashes)
    kind_panel(ax_kind, kinds)
    im = matrix_panel(ax_mat, matrix)

    fig.tight_layout(rect=[0, 0.045, 1, 0.945], h_pad=2.6, w_pad=2.6)
    ax_sec.apply_aspect()
    safe_colorbar(fig, im, ax_mat, label="clash count (-)", width=0.09,
                  pad=0.07, labelsize=6.5)
    panel_labels_inches(fig, [ax_sec, ax_kind, ax_mat], dx=-0.34, dy=0.12)
    savefig(fig, "593_bim_clash_detection.png")


if __name__ == "__main__":
    main()
