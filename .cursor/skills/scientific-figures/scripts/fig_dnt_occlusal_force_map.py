"""Occlusal contact map: where the bite force lands and in what order.

Every contact carries a force in newtons and a footprint in square
millimetres, so its pressure is a derived quantity rather than a second
independent number, and the marker encodes both: area for force, colour for
pressure.  The centre of force is the force-weighted centroid of exactly the
points drawn on the arch, the left/right and anterior/posterior shares are
sums over the same table, and the contact-timing bars in panel (c) are ordered
by each tooth's mean pressure - the harder a tooth is loaded at full
intercuspation, the earlier it is shown meeting its antagonist.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pubstyle import set_pub_style, savefig, get_cmap
from figkit import axes_inches, panel_labels_inches, safe_colorbar

FW, FH = 10.4, 6.4
MAIN = (0.86, 0.86, 5.10, 4.56)
AUX_T = (7.12, 3.52, 3.00, 1.90)
AUX_B = (7.12, 0.86, 3.00, 1.76)

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
HAIR = "#d5d5d5"
RED, BLUE, GREEN, GOLD = "#b23a48", "#3b5b92", "#2e7d5b", "#c9a227"
TOOTH_FC, TOOTH_EC = "#f4f0e6", "#a89f8f"
PRESS_CMAP = get_cmap("cb_sequential_blue")

# Occlusal-view arch, patient's left half, in millimetres from the midline.
# (name, x, y, mesiodistal width, buccolingual width, contacts, load weight)
TEETH = [("1", 4.0, 33.0, 6.3, 6.5, 1, 0.12),
         ("2", 10.5, 31.0, 6.4, 6.0, 1, 0.12),
         ("3", 16.5, 26.5, 7.0, 8.0, 1, 0.22),
         ("4", 20.5, 19.5, 7.2, 9.0, 2, 0.55),
         ("5", 23.0, 12.0, 7.6, 9.2, 2, 0.75),
         ("6", 25.0, 3.0, 8.6, 10.5, 4, 1.00),
         ("7", 26.0, -6.5, 8.6, 10.0, 3, 0.85)]
TOTAL_FORCE = 620.0          # N at full intercuspation
CLOSE_MS = 130.0             # duration of the closing stroke

FOOT = ("Synthetic occlusal contact map for the maxillary arch, viewed from "
        "below, so the patient's right side is on the left of the panel. "
        "Marker area encodes contact force and colour encodes pressure = "
        "force / contact footprint; the cross is the force-weighted centroid "
        "of the plotted contacts. Contact order in panel (c) is derived from "
        "each tooth's mean pressure. Illustrative data only, not a T-Scan "
        "recording.")


def fit_limits(ax, xlim, ylim, rect):
    """Pad the shorter data axis so an equal-aspect panel fills its box."""
    want = rect[2] / rect[3]
    (x0, x1), (y0, y1) = xlim, ylim
    if (x1 - x0) / (y1 - y0) < want:
        e = (y1 - y0) * want - (x1 - x0)
        x0, x1 = x0 - e / 2.0, x1 + e / 2.0
    else:
        e = (x1 - x0) / want - (y1 - y0)
        y0, y1 = y0 - e / 2.0, y1 + e / 2.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


def squircle(cx, cy, w, h, ang_deg, n=4.0, k=90):
    """Rounded occlusal footprint, rotated to sit tangent to the arch."""
    t = np.linspace(0.0, 2.0 * np.pi, k)
    x = (w / 2.0) * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / n)
    y = (h / 2.0) * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / n)
    a = np.deg2rad(ang_deg)
    return np.column_stack([cx + x * np.cos(a) - y * np.sin(a),
                            cy + x * np.sin(a) + y * np.cos(a)])


def arch_layout():
    """Full arch: mirror the half arch and give every tooth a tangent angle."""
    half = [(nm, x, y, w, h, nc, wt) for nm, x, y, w, h, nc, wt in TEETH]
    out = []
    for side, sgn in (("R", -1.0), ("L", +1.0)):
        for i, (nm, x, y, w, h, nc, wt) in enumerate(half):
            prev = half[max(i - 1, 0)]
            nxt = half[min(i + 1, len(half) - 1)]
            # Mesiodistal axis tangent to the arch; mirroring flips only x.
            dx, dy = nxt[1] - prev[1], nxt[2] - prev[2]
            ang = np.degrees(np.arctan2(dy, sgn * dx))
            out.append(dict(name=f"{side}{nm}", side=side, idx=i,
                            x=sgn * x, y=y, w=w, h=h, nc=nc, wt=wt, ang=ang))
    return out


def make_contacts(rng, teeth):
    """One row per contact: position, force (N), footprint (mm2), pressure."""
    rows = []
    for t in teeth:
        bias = 1.16 if t["side"] == "L" else 0.88      # habitual chewing side
        for j in range(t["nc"]):
            u = rng.uniform(-0.26, 0.26)
            v = rng.uniform(-0.26, 0.26)
            a = np.deg2rad(t["ang"])
            px = t["x"] + u * t["w"] * np.cos(a) - v * t["h"] * np.sin(a)
            py = t["y"] + u * t["w"] * np.sin(a) + v * t["h"] * np.cos(a)
            f = t["wt"] * bias * rng.uniform(0.62, 1.38) / max(t["nc"], 1)
            area = rng.uniform(0.50, 1.10) * (0.90 + 0.13 * t["idx"])
            rows.append(dict(tooth=t["name"], side=t["side"], idx=t["idx"],
                             x=px, y=py, raw=f, area=area))
    scale = TOTAL_FORCE / sum(r["raw"] for r in rows)
    for r in rows:
        r["force"] = r["raw"] * scale
        r["press"] = r["force"] / r["area"]
    return rows


def main():
    set_pub_style()
    rng = np.random.default_rng(820)
    teeth = arch_layout()
    rows = make_contacts(rng, teeth)

    px = np.array([r["x"] for r in rows])
    py = np.array([r["y"] for r in rows])
    pf = np.array([r["force"] for r in rows])
    pp = np.array([r["press"] for r in rows])
    is_left = np.array([r["side"] == "L" for r in rows])
    is_ant = np.array([r["idx"] <= 2 for r in rows])

    cof = np.array([np.sum(pf * px), np.sum(pf * py)]) / pf.sum()
    share_l = pf[is_left].sum() / pf.sum() * 100.0
    share_r = 100.0 - share_l
    share_a = pf[is_ant].sum() / pf.sum() * 100.0
    share_p = 100.0 - share_a

    fig = plt.figure(figsize=(FW, FH))

    # ---- (a) arch map --------------------------------------------------------
    ax = axes_inches(fig, MAIN)
    ax.axis("off")
    for t in teeth:
        n = 3.0 if t["idx"] <= 2 else 4.0
        ax.add_patch(Polygon(squircle(t["x"], t["y"], t["w"], t["h"],
                                      t["ang"], n=n), closed=True,
                             facecolor=TOOTH_FC, edgecolor=TOOTH_EC, lw=0.6,
                             zorder=2))
        # Labels sit radially outside the arch, so none of them lands on a
        # crown however tightly the anterior teeth are packed.
        v = np.array([t["x"], t["y"] - 14.0])
        v = v / np.hypot(*v)
        lx, ly = np.array([t["x"], t["y"]]) + v * (0.5 * max(t["w"], t["h"])
                                                   + 3.6)
        ax.text(lx, ly, t["name"], ha="center", va="center", fontsize=6.0,
                color=MUTED, zorder=3)

    ax.plot([0, 0], [-7.0, 40.0], color=RULE, lw=0.7, ls=(0, (3.0, 2.4)),
            zorder=1)
    ax.text(0.0, 40.8, "midline", ha="center", va="bottom", fontsize=6.2,
            color=RULE)
    sc = ax.scatter(px, py, s=10.0 + 118.0 * pf / pf.max(), c=pp,
                    cmap=PRESS_CMAP, vmin=0.0, vmax=pp.max(),
                    edgecolors="white", linewidths=0.5, zorder=6)

    ax.plot([cof[0] - 3.4, cof[0] + 3.4], [cof[1]] * 2, color=RED, lw=1.4,
            zorder=8)
    ax.plot([cof[0]] * 2, [cof[1] - 3.4, cof[1] + 3.4], color=RED, lw=1.4,
            zorder=8)
    ax.annotate(f"centre of force\n{abs(cof[0]):.1f} mm "
                f"{'left' if cof[0] > 0 else 'right'} of the midline",
                xy=(cof[0], cof[1]), xytext=(cof[0] - 15.0, cof[1] - 12.0),
                ha="center", va="top", fontsize=6.6, color=RED,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.9,
                                shrinkA=1.0, shrinkB=4.0), zorder=9)

    for sgn, lab, val in ((-1.0, "patient right", share_r),
                          (+1.0, "patient left", share_l)):
        ax.text(sgn * 30.0, 37.0, f"{lab}\n{val:.1f} % of bite force",
                ha="center", va="center", fontsize=7.0, color=INK,
                fontweight="bold", linespacing=1.5)

    # Size key: three reference forces, drawn with the same scaling as above.
    ref = np.array([0.25, 0.55, 1.0]) * pf.max()
    for i, f in enumerate(ref):
        ax.scatter([-8.0 + 8.0 * i], [-14.0],
                   s=10.0 + 118.0 * f / pf.max(), c="#cccccc",
                   edgecolors=MUTED, linewidths=0.5)
        ax.text(-8.0 + 8.0 * i, -18.4, f"{f:.0f}", ha="center", va="center",
                fontsize=6.0, color=SUB)
    ax.text(0.0, -21.4, "contact force (N)", ha="center", va="center",
            fontsize=6.3, color=SUB)

    fit_limits(ax, (-36.0, 36.0), (-23.5, 43.0), MAIN)
    ax.set_title("Maxillary occlusal contact pressure map", fontsize=8.8,
                 fontweight="bold", color=INK, pad=4)
    safe_colorbar(fig, sc, ax, label="contact pressure (MPa)", width=0.11,
                  pad=0.09, labelsize=6.6)

    # ---- (b) force shares ----------------------------------------------------
    ax_b = axes_inches(fig, AUX_T)
    bars = [("left / right", [(share_r, "right", BLUE),
                              (share_l, "left", GREEN)]),
            ("anterior / posterior", [(share_a, "anterior", GOLD),
                                      (share_p, "posterior", BLUE)])]
    for k, (name, parts) in enumerate(bars):
        left = 0.0
        for val, lab, col in parts:
            ax_b.barh(k, val, left=left, height=0.46, color=col,
                      edgecolor="white", linewidth=0.8)
            ax_b.text(left + val / 2.0, k, f"{lab}\n{val:.1f} %",
                      ha="center", va="center", fontsize=6.4, color="white",
                      linespacing=1.4)
            left += val
    ax_b.axvline(50.0, color=RULE, lw=0.9, ls=(0, (2.6, 2.0)))
    ax_b.text(50.0, 1.62, "even split", ha="center", va="bottom",
              fontsize=6.2, color=RULE)
    ax_b.set_yticks([0, 1])
    ax_b.set_yticklabels(["L / R", "ant / post"], fontsize=6.6)
    ax_b.set_ylim(-0.55, 1.95)
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("share of total bite force (%)", fontsize=8.0)
    ax_b.tick_params(labelsize=6.6)
    ax_b.text(1.5, -0.48, f"total {pf.sum():.0f} N over n={len(rows)} "
              f"contacts; imbalance {abs(share_l - share_r):.1f} points",
              ha="left", va="bottom", fontsize=6.2, color=SUB)
    ax_b.set_title("Where the load sits", fontsize=8.5, fontweight="bold",
                   color=INK, pad=3)

    # ---- (c) contact sequence ------------------------------------------------
    ax_c = axes_inches(fig, AUX_B)
    names = ["1", "2", "3", "4", "5", "6", "7"]
    mean_p = {}
    for t in teeth:
        m = [r["press"] for r in rows if r["tooth"] == t["name"]]
        mean_p[t["name"]] = float(np.mean(m))
    lo, hi = min(mean_p.values()), max(mean_p.values())
    onset = {k: (1.0 - (v - lo) / (hi - lo)) * 0.72 * CLOSE_MS
             for k, v in mean_p.items()}
    first = min(onset, key=onset.get)

    for i, nm in enumerate(names):
        for sgn, side, col in ((-0.19, "R", BLUE), (+0.19, "L", GREEN)):
            key = f"{side}{nm}"
            t0 = onset[key]
            ax_c.barh(i + sgn, CLOSE_MS - t0, left=t0, height=0.32,
                      color=col, edgecolor="white", linewidth=0.4)
    ax_c.axvline(CLOSE_MS, color=RULE, lw=0.8)
    ax_c.text(CLOSE_MS - 2.0, 6.98, "full intercuspation", ha="right",
              va="bottom", fontsize=6.2, color=RULE)
    ax_c.plot([onset[first]], [names.index(first[1:]) +
                               (0.19 if first[0] == "L" else -0.19)],
              marker="o", ms=4.2, mfc="none", mec=RED, mew=1.1, zorder=5)
    ax_c.text(2.0, 6.98, f"first contact: tooth {first}", ha="left",
              va="bottom", fontsize=6.2, color=RED)
    ax_c.set_yticks(np.arange(len(names)))
    ax_c.set_yticklabels([f"tooth {n}" for n in names], fontsize=6.4)
    ax_c.set_ylim(-1.05, len(names) + 0.30)
    ax_c.set_xlim(0, CLOSE_MS * 1.06)
    ax_c.set_xlabel("time from first contact (ms)", fontsize=8.0)
    ax_c.tick_params(labelsize=6.6)
    ax_c.grid(axis="x", color=HAIR, lw=0.6)
    ax_c.set_axisbelow(True)
    ax_c.text(2.0, -1.00, "blue = right side,  green = left side", ha="left",
              va="bottom", fontsize=6.2, color=SUB)
    ax_c.set_title("Contact sequence through closure", fontsize=8.5,
                   fontweight="bold", color=INK, pad=3)

    heavy = "left" if share_l > share_r else "right"
    panel_labels_inches(fig, [ax, ax_b, ax_c], dx=-0.52, dy=0.10)
    savefig(fig, "820_dnt_occlusal_force_map.png")


if __name__ == "__main__":
    main()
