"""Axial dose colour wash for a synthetic five-field photon plan.

One dose grid is built once from five coplanar beams and everything else in
the figure reads from it: the isodose lines, the lateral profile, the penumbra
width and the conformity indices are all measured on the same array, so the
numbers cannot drift away from the picture.

The main panel is an ``set_aspect("equal")`` image, so its colour bar is
pinned with ``figkit.safe_colorbar`` (an ``inset_axes`` on the *applied* axes
box) rather than ``fig.colorbar``, which would park the bar an inch away from
the shrunken image.  The two smaller panels on the right carry the evidence:
how sharp the lateral fall-off is, and how tightly each isodose surface wraps
the target.

Synthetic illustration for a visual reference library - not patient data.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"

FIGW, FIGH = 10.4, 6.4
RX = 60.0                       # prescription dose (Gy)
FOV = 15.0                      # half field of view (cm)
NPIX = 240
GANTRY = (0.0, 72.0, 144.0, 216.0, 288.0)     # gantry angles (deg)
ISO = (3.2, 1.4)                # isocentre / target centre (cm)
PTV_R = 2.7                     # target radius (cm)
LEVELS = (95.0, 80.0, 50.0, 30.0)             # isodose levels (% of Rx)

PTV_C = "#b23a48"
CORD_C = "#3b5b92"
LUNG_C = "#2e7d5b"


def ellipse(X, Y, cx, cy, a, b):
    return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0


def anatomy(X, Y, rng):
    """Grey-scale transverse slice plus the delineated structure masks."""
    body = ellipse(X, Y, 0.0, 0.0, 13.0, 9.4)
    lung_r = ellipse(X, Y, 6.1, 1.2, 4.3, 5.4)
    lung_l = ellipse(X, Y, -6.1, 1.2, 4.3, 5.4)
    vertebra = ellipse(X, Y, 0.0, -6.4, 2.3, 1.9)
    canal = ellipse(X, Y, 0.0, -6.1, 0.95, 0.95)
    ptv = (X - ISO[0]) ** 2 + (Y - ISO[1]) ** 2 <= PTV_R ** 2

    img = np.full(X.shape, 0.03)
    img[body] = 0.44
    img[lung_r | lung_l] = 0.11
    img[vertebra] = 0.82
    img[canal] = 0.38
    img[ptv] = 0.52
    img += rng.normal(0.0, 0.018, X.shape)
    img = gaussian_filter(img, 1.5)
    return np.clip(img, 0.0, 1.0), body, ptv, canal, (lung_r | lung_l)


def beam_dose(X, Y, angle_deg, half_width, offset=0.0, mu=0.042, sigma=0.40):
    """One coplanar beam: flat lateral profile with an exponential fall-off."""
    a = np.deg2rad(angle_deg)
    ux, uy = np.sin(a), -np.cos(a)                 # source -> isocentre
    depth = (X - ISO[0]) * ux + (Y - ISO[1]) * uy  # + = downstream
    lateral = -(X - ISO[0]) * uy + (Y - ISO[1]) * ux - offset
    lat = 0.5 * (erf((half_width - lateral) / sigma) +
                 erf((half_width + lateral) / sigma))
    return lat * np.exp(-mu * (depth + 12.0))


def beam_tail(angle_deg, radius):
    """Point on a circle of ``radius`` upstream of the isocentre."""
    a = np.deg2rad(angle_deg)
    u = np.array([np.sin(a), -np.cos(a)])
    c = np.array(ISO)
    b = float(c @ u)
    t = b + np.sqrt(max(b * b - float(c @ c) + radius ** 2, 0.0))
    return c - t * u, u


def edge_80_20(pos_cm, prof):
    """80 %-20 % width (mm) of the right-hand shoulder, and its two abscissae."""
    top = float(prof.max())
    i_top = int(np.argmax(prof))
    p, x = prof[i_top:], pos_cm[i_top:]
    x80 = float(np.interp(-0.80 * top, -p, x))
    x20 = float(np.interp(-0.20 * top, -p, x))
    return abs(x20 - x80) * 10.0, x80, x20


def main():
    set_pub_style()
    rng = np.random.default_rng(871)

    g = np.linspace(-FOV, FOV, NPIX)
    X, Y = np.meshgrid(g, g)
    img, body, ptv, canal, lung = anatomy(X, Y, rng)

    # Small per-beam weight and alignment errors keep the plan from being
    # unrealistically perfect - the conformity panel would otherwise be flat.
    offsets = rng.normal(0.0, 0.16, len(GANTRY))
    weights = rng.normal(1.0, 0.09, len(GANTRY))
    dose = np.zeros_like(X)
    for ang, off, w in zip(GANTRY, offsets, weights):
        dose += w * beam_dose(X, Y, ang, PTV_R + 0.12, offset=off)
    single = beam_dose(X, Y, GANTRY[0], PTV_R + 0.12, offset=offsets[0])
    dose = gaussian_filter(dose, 1.1)
    scale = RX / np.median(dose[ptv])          # normalise to the prescription
    dose *= scale
    single *= scale
    dose = np.where(body, dose, 0.0)

    fig = plt.figure(figsize=(FIGW, FIGH))
    ax = axes_inches(fig, (0.88, 1.00, 4.44, 4.44))
    ax_prof = axes_inches(fig, (7.10, 3.70, 2.90, 1.74))
    ax_ci = axes_inches(fig, (7.10, 0.95, 2.90, 1.74))
    despine(ax_prof)
    despine(ax_ci)

    # ---- main panel: colour wash over the grey-scale slice ----------------
    ext = [-FOV, FOV, -FOV, FOV]
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, extent=ext,
              origin="lower", interpolation="bilinear")
    # A ramped alpha instead of a hard mask: the low end of ``magma`` is very
    # dark, so a flat alpha turns the 30-40 % region into a black polygon that
    # hides the anatomy it is supposed to sit on.
    rel = dose / RX
    a_map = np.clip(0.26 + 0.60 * (rel - 0.30) / 0.78, 0.0, 0.86)
    a_map[rel < 0.30] = 0.0
    im = ax.imshow(dose, cmap="magma", vmin=0.30 * RX, vmax=1.08 * RX,
                   extent=ext, origin="lower", alpha=a_map,
                   interpolation="bilinear")
    ax.set_aspect("equal")

    cs = ax.contour(X, Y, rel * 100.0, levels=sorted(LEVELS),
                    colors="white", linewidths=0.75, alpha=0.95)
    labels = ax.clabel(cs, fmt=lambda v: f"{v:.0f}%", fontsize=6.0,
                       inline=True, inline_spacing=2, colors="white")
    for t in labels:
        t.set_path_effects([pe.withStroke(linewidth=1.6, foreground="#1a1a1a")])

    for mask, col, lw in ((ptv, PTV_C, 1.5), (canal, CORD_C, 1.2),
                          (lung, LUNG_C, 1.0)):
        ax.contour(X, Y, mask.astype(float), levels=[0.5], colors=[col],
                   linewidths=lw)

    for ang in GANTRY:
        p0, u = beam_tail(ang, 13.9)
        p1 = p0 + 3.2 * u
        ax.annotate("", xy=tuple(p1), xytext=tuple(p0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#c9a227",
                                    mutation_scale=8, shrinkA=0, shrinkB=0))
        pt = p0 - 0.62 * u
        halo_text(ax, pt[0], pt[1], f"{ang:.0f}$^\\circ$", ha="center",
                  va="center", fontsize=6.0, color="#c9a227", halo="#1a1a1a",
                  halo_lw=1.5)

    y_line = ISO[1]
    ax.plot([-FOV + 0.4, FOV - 0.4], [y_line, y_line], color="#cccccc",
            lw=0.8, ls=(0, (3.0, 2.0)))
    halo_text(ax, -FOV + 0.7, y_line + 0.45, "profile (b)", ha="left",
              va="bottom", fontsize=6.2, color="white", halo="#1a1a1a")

    ax.set_xlim(-FOV, FOV)
    ax.set_ylim(-FOV, FOV)
    ax.set_xlabel("lateral position (cm)", fontsize=8.0)
    ax.set_ylabel("anterior-posterior position (cm)", fontsize=8.0)
    ax.tick_params(labelsize=7.0, color="#8a8a8a")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("#8a8a8a")
    ax.legend(handles=[Patch(facecolor="none", edgecolor=PTV_C, label="PTV"),
                       Patch(facecolor="none", edgecolor=CORD_C,
                             label="spinal canal"),
                       Patch(facecolor="none", edgecolor=LUNG_C,
                             label="lung")],
              loc="lower left", ncol=3, frameon=False, fontsize=6.8,
              handlelength=1.3, handletextpad=0.45, columnspacing=1.1,
              labelcolor="white", bbox_to_anchor=(0.005, 0.005),
              borderpad=0.25)
    safe_colorbar(fig, im, ax, label="absorbed dose (Gy)", width=0.12,
                  pad=0.09, labelsize=6.8)

    # ---- aux 1: lateral dose profile through the isocentre ----------------
    row = int(np.argmin(np.abs(g - y_line)))
    prof = dose[row, :] / RX * 100.0
    prof_1 = np.where(body[row, :], single[row, :], 0.0) / RX * 100.0
    pen, x80, x20 = edge_80_20(g, prof_1)     # measured on one field's edge
    top = float(prof.max())
    top1 = float(prof_1.max())

    ax_prof.plot(g, prof, color="#b23a48", lw=1.4, label="all 5 fields")
    ax_prof.fill_between(g, 0.0, prof, color="#b23a48", alpha=0.10)
    ax_prof.plot(g, prof_1, color=GREY, lw=1.0, ls=(0, (2.6, 1.5)),
                 label="single field, 0$^\\circ$")
    for xv in (x80, x20):
        ax_prof.plot([xv, xv], [0.0, 0.86 * top1], color="#3a3a3a", lw=0.6)
    ax_prof.annotate("", xy=(x20, 0.62 * top1), xytext=(x80, 0.62 * top1),
                     arrowprops=dict(arrowstyle="<|-|>", lw=0.8,
                                     color="#3a3a3a", mutation_scale=6))
    ax_prof.text(x20 + 0.6, 0.66 * top1,
                 f"80-20 penumbra\n{pen:.1f} mm", fontsize=6.3, color=INK,
                 ha="left", va="bottom", linespacing=1.5)
    ax_prof.set_xlim(-FOV, FOV)
    ax_prof.set_ylim(0.0, 1.26 * top)
    ax_prof.set_xlabel("lateral position (cm)", fontsize=7.2)
    ax_prof.set_ylabel("dose (% of Rx)", fontsize=7.2)
    ax_prof.tick_params(labelsize=6.5)
    ax_prof.legend(loc="upper left", frameon=False, fontsize=6.2,
                   handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
                   borderpad=0.1)
    ax_prof.set_title("Dose profile through the isocentre", fontsize=8.5,
                      fontweight="bold", color=INK, pad=4.0)

    # ---- aux 2: conformity of every isodose surface -----------------------
    v_ptv = float(ptv.sum())
    lv = np.array(sorted(LEVELS, reverse=True))
    paddick, spill, cover = [], [], []
    for level in lv:
        iso_mask = (dose >= level / 100.0 * RX) & body
        v_iso = float(iso_mask.sum())
        v_in = float((iso_mask & ptv).sum())
        paddick.append(v_in ** 2 / max(v_ptv * v_iso, 1.0))
        spill.append(1.0 - v_in / max(v_iso, 1.0))
        cover.append(v_in / v_ptv)
    pos = np.arange(len(lv), dtype=float)
    ax_ci.bar(pos - 0.19, paddick, width=0.36, color="#c9a227",
              edgecolor="white", linewidth=0.5, label="Paddick CI")
    ax_ci.bar(pos + 0.19, spill, width=0.36, color="#3b5b92",
              edgecolor="white", linewidth=0.5, label="spill outside PTV")
    for i, (p, s) in enumerate(zip(paddick, spill)):
        ax_ci.text(i - 0.19, p + 0.025, f"{p:.2f}", ha="center", va="bottom",
                   fontsize=6.0, color=SUB)
        ax_ci.text(i + 0.19, s + 0.025, f"{s:.2f}", ha="center", va="bottom",
                   fontsize=6.0, color=SUB)
    ax_ci.set_xticks(pos)
    ax_ci.set_xticklabels([f"{v:.0f} %" for v in lv], fontsize=6.5)
    ax_ci.set_ylim(0.0, 1.30)
    ax_ci.set_yticks(np.arange(0.0, 1.01, 0.25))
    ax_ci.tick_params(axis="y", labelsize=6.5)
    ax_ci.set_xlabel("isodose surface (% of Rx)", fontsize=7.2)
    ax_ci.set_ylabel("index (-)", fontsize=7.2)
    ax_ci.text(0.985, 0.985, f"target coverage by the\n95 % surface = "
               f"{cover[0]:.2f}", transform=ax_ci.transAxes, ha="right",
               va="top", fontsize=6.2, color=SUB, linespacing=1.6)
    ax_ci.legend(loc="upper left", frameon=False, fontsize=6.2,
                 handlelength=1.2, handletextpad=0.45, labelspacing=0.3,
                 borderpad=0.1, bbox_to_anchor=(-0.012, 1.005))
    ax_ci.set_title("Conformity of each isodose surface", fontsize=8.5,
                    fontweight="bold", color=INK, pad=4.0)

    d_ptv = np.sort(dose[ptv])
    d2 = float(np.percentile(d_ptv, 98.0))
    d98 = float(np.percentile(d_ptv, 2.0))
    d50 = float(np.percentile(d_ptv, 50.0))
    hi = (d2 - d98) / d50
    cord_max = float(dose[canal].max())



    panel_labels_inches(fig, [ax], labels=["(a)"], dx=-0.62, dy=0.20)
    panel_labels_inches(fig, [ax_prof, ax_ci], labels=["(b)", "(c)"],
                        dx=-0.62, dy=0.20)
    savefig(fig, "871_rtx_dose_distribution.png")


if __name__ == "__main__":
    main()
