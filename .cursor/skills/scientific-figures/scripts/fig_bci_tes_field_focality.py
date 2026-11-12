"""Transcranial electric stimulation: where the field lands and how tight it is.

Main panel: an axial head slice at true scale (``set_aspect("equal")``) with
the electric-field magnitude of a 4x1 high-definition ring painted inside the
brain compartment, grey equipotential lines, the half-maximum contour that
defines focality, and the scalp electrodes drawn where the current enters and
leaves.  Auxiliary panels (both 58 per cent of the main panel width): the
field along a radial line from the cortical surface to the midline for three
montages, and the focality-versus-target-intensity trade-off swept over the
ring radius with the three montages marked on it.

The field is solved, not drawn: each electrode is a regularised point current
source on the scalp of a homogeneous 0.33 S/m conductor, the potential is the
sum of their Green's functions, and E = -grad V is taken with ``np.gradient``
on the same millimetre grid that is displayed.  Every number quoted - target
intensity in V/m, half-maximum area in cm2, depth of the half-maximum - is
measured on that grid.  A homogeneous head is a coarse approximation to a
layered one; this is a reference illustration, not a dosing calculation.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse, Rectangle
from pubstyle import set_pub_style, savefig, get_palette
from figkit import axes_inches, panel_labels_inches, safe_colorbar, halo_text

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREY = "#b3b3b3"
BOX = dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="none",
           alpha=0.86)

SIGMA = 0.33                 # head conductivity, S/m
R_ELEC = 12.0                # electrode radius used to regularise the source, mm
SHUNT = 0.18                 # lumped skull-shunting factor, dimensionless
A_SCALP, B_SCALP = 75.0, 90.0
A_SKULL, B_SKULL = 69.0, 83.0
A_CSF, B_CSF = 65.0, 79.0
A_BRAIN, B_BRAIN = 62.0, 76.0
XLIM, YLIM = (-120.0, 120.0), (-95.0, 95.0)
NX, NY = 361, 286
TARGET = (0.0, 68.0)         # cortical target, mm
I_TOTAL = 2.0                # injected current, mA

MONTAGES = [
    ("bipolar pads, 35 cm$^2$", [(78.0, +I_TOTAL), (252.0, -I_TOTAL)], BLUE),
    ("4x1 HD ring, 42$^\\circ$", [(90.0, +I_TOTAL), (48.0, -I_TOTAL / 2.0),
                                  (132.0, -I_TOTAL / 2.0)], RED),
    ("close bipolar, 30$^\\circ$", [(105.0, +I_TOTAL), (75.0, -I_TOTAL)],
     TEAL),
]
RING_SWEEP = np.linspace(18.0, 95.0, 26)

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def scalp_point(deg):
    """Electrode centre on the scalp ellipse, mm."""
    t = np.deg2rad(deg)
    return A_SCALP * np.cos(t), B_SCALP * np.sin(t)


def grid():
    x = np.linspace(XLIM[0], XLIM[1], NX)
    y = np.linspace(YLIM[0], YLIM[1], NY)
    return x, y, np.meshgrid(x, y)


def inside(xx, yy, a, b):
    return (xx / a) ** 2 + (yy / b) ** 2 <= 1.0


def solve(xx, yy, sources):
    """Potential (V) and field magnitude (V/m) for a list of (deg, mA)."""
    v = np.zeros_like(xx)
    for deg, ma in sources:
        px, py = scalp_point(deg)
        d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2 + R_ELEC ** 2) * 1e-3
        v += (ma * 1e-3) / (2.0 * np.pi * SIGMA * d)
    step = (xx[0, 1] - xx[0, 0]) * 1e-3
    gy, gx = np.gradient(v, step, step)
    # A homogeneous head has no skull to shunt current tangentially, so the
    # raw solution overstates the intracranial field by roughly a factor of
    # five; one lumped factor keeps the V/m axis in the range a layered model
    # would give without pretending to be one.
    return v, SHUNT * np.hypot(gx, gy)


def metrics(xx, yy, emag, brain, cell_cm2):
    """Target intensity, half-max area and peak field inside the brain."""
    peak = float(emag[brain].max())
    hot = brain & (emag >= 0.5 * peak)
    i = int(np.argmin(np.abs(yy[:, 0] - TARGET[1])))
    j = int(np.argmin(np.abs(xx[0] - TARGET[0])))
    return dict(target=float(emag[i, j]), peak=peak,
                area=float(hot.sum()) * cell_cm2)


def head_shells(ax):
    for a, b, fc, ec in ((A_SCALP, B_SCALP, "#ececec", "#9a9a9a"),
                         (A_SKULL, B_SKULL, "#f6f6f6", "#b0b0b0"),
                         (A_CSF, B_CSF, "#e6ecf3", "#b0b0b0")):
        ax.add_patch(Ellipse((0.0, 0.0), 2 * a, 2 * b, facecolor=fc,
                             edgecolor=ec, lw=0.8, zorder=1))


def electrodes(ax, sources):
    for deg, ma in sources:
        px, py = scalp_point(deg)
        col = RED if ma > 0 else BLUE
        t = np.deg2rad(deg)
        ang = np.rad2deg(np.arctan2(B_SCALP * np.cos(t), -A_SCALP * np.sin(t)))
        pad = Rectangle((-9.0, -3.0), 18.0, 6.0, facecolor=col,
                        edgecolor="white", lw=0.8, zorder=8)
        tr = (plt.matplotlib.transforms.Affine2D()
              .rotate_deg(ang + 90.0).translate(px, py) + ax.transData)
        pad.set_transform(tr)
        ax.add_patch(pad)
        # Slide the label along the scalp before pushing it outward, so the
        # anode's label clears the top of the axes instead of being cropped.
        lab_deg = deg - 16.0 if np.cos(t) >= 0.0 else deg + 16.0
        lx, ly = np.array(scalp_point(lab_deg)) * 1.11
        halo_text(ax, lx, ly, "{:+.1f} mA".format(ma), color=col,
                  halo="white", halo_lw=2.2, fontsize=6.2, ha="center",
                  va="center", zorder=9)


def head_panel(fig, ax, x, y, xx, yy, v, emag, brain, met, name):
    head_shells(ax)
    warm = mcolors.LinearSegmentedColormap.from_list(
        "warm_cut", get_palette("cb_sequential_warm", 16)[:13])
    field = np.ma.masked_where(~brain, emag)
    im = ax.imshow(field, origin="lower", cmap=warm, vmin=0.0,
                   vmax=float(met["peak"]),
                   extent=(x[0], x[-1], y[0], y[-1]), aspect="equal",
                   interpolation="bilinear", zorder=2)
    ax.add_patch(Ellipse((0.0, 0.0), 2 * A_BRAIN, 2 * B_BRAIN,
                         facecolor="none", edgecolor="#7a7a7a", lw=0.9,
                         zorder=5))
    ax.contour(x, y, np.ma.masked_where(~brain, v),
               levels=np.linspace(float(v[brain].min()),
                                  float(v[brain].max()), 11),
               colors="#8a8a8a", linewidths=0.45, alpha=0.75, zorder=3)
    ax.contour(x, y, np.ma.masked_where(~brain, emag),
               levels=[0.5 * met["peak"]], colors=[INK], linewidths=1.3,
               zorder=6)
    ax.plot([TARGET[0]], [TARGET[1]], marker="o", ms=6.0, mfc="none",
            mec=INK, mew=1.3, zorder=9)
    electrodes(ax, dict(MONTAGES_BY_NAME)[name])

    ax.annotate("cortical target\n{:.2f} V/m".format(met["target"]),
                xy=TARGET, xytext=(-104.0, 22.0), fontsize=6.4, color=INK,
                ha="left", va="center", linespacing=1.45, bbox=BOX, zorder=10,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8,
                                shrinkA=2, shrinkB=5))
    ax.text(-116.0, -84.0,
            "black outline: half of the peak field, {:.2f} V/m\n"
            "it encloses {:.1f} cm$^2$ of this slice"
            .format(0.5 * met["peak"], met["area"]),
            fontsize=6.4, color=INK, ha="left", va="bottom",
            linespacing=1.5, bbox=BOX, zorder=10)
    ax.text(116.0, -84.0, "grey lines: equipotentials\nof the solved "
            "potential", fontsize=6.2, color=SUB, ha="right", va="bottom",
            linespacing=1.5, bbox=BOX, zorder=10)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("left - right (mm)")
    ax.set_ylabel("posterior - anterior (mm)")
    ax.tick_params(width=0.8, length=3.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cb = safe_colorbar(fig, im, ax, width=0.11, pad=0.09, labelsize=6.2)
    cb.ax.set_ylabel("field magnitude in brain tissue (V/m)", fontsize=6.2)
    ax.set_title("{:s}: a {:.0f} mA ring puts {:.2f} V/m on target"
                 .format(name, I_TOTAL, met["target"]), fontsize=9.0,
                 fontweight="bold", color=INK, pad=6.0)


def depth_panel(ax, x, y, fields, met_all):
    depth = np.linspace(0.0, 70.0, 141)
    for (name, _, col), emag in zip(MONTAGES, fields):
        prof = []
        for d in depth:
            j = int(np.argmin(np.abs(x - 0.0)))
            i = int(np.argmin(np.abs(y - (B_BRAIN - d))))
            prof.append(emag[i, j])
        prof = np.array(prof)
        ax.plot(depth, prof, color=col, lw=1.5, zorder=5, label=name)
        half = 0.5 * prof.max()
        k = int(np.argmax(prof < half)) if (prof < half).any() else len(prof) - 1
        ax.plot([depth[k]], [prof[k]], marker="o", ms=3.6, mfc="white",
                mec=col, mew=1.0, zorder=7)
    ax.axvline(B_BRAIN - TARGET[1], color=GREY, lw=0.9, ls=(0, (2.5, 2.0)),
               zorder=3)
    ax.text(B_BRAIN - TARGET[1] + 1.6, 0.02, "target depth {:.0f} mm"
            .format(B_BRAIN - TARGET[1]), fontsize=6, color=SUB,
            ha="left", va="bottom", rotation=90, bbox=BOX, zorder=8)
    ax.set_xlim(0.0, depth[-1])
    ax.set_ylim(0.0, None)
    ax.set_xlabel("depth below the cortical surface (mm)", fontsize=6.8)
    ax.set_ylabel("field magnitude (V/m)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Radial decay under the anterior scalp", fontsize=8.0,
                 fontweight="bold", color=INK)


def tradeoff_panel(ax, sweep, met_all):
    inten = np.array([s[1] for s in sweep])
    area = np.array([s[2] for s in sweep])
    ax.plot(inten, area, color=MUTED, lw=1.2, marker="o", ms=2.6,
            mfc="white", mec=MUTED, mew=0.5, zorder=4,
            label="HD ring, half-angle swept")
    for k, off in ((0, (6.0, -5.0)), (len(sweep) - 1, (6.0, 3.0))):
        ax.annotate("{:.0f}$^\\circ$".format(sweep[k][0]),
                    xy=(inten[k], area[k]), xytext=off,
                    textcoords="offset points", fontsize=6, color=MUTED,
                    ha="left", va="center", bbox=BOX, zorder=7)
    offs = [(7.0, 6.0, "left", "bottom"), (-7.0, 9.0, "right", "bottom"),
            (-7.0, -9.0, "right", "top")]
    for (name, _, col), met, (ox, oy, ha, va) in zip(MONTAGES, met_all, offs):
        ax.plot([met["target"]], [met["area"]], marker="o", ms=5.4,
                mfc=col, mec="white", mew=0.9, zorder=8)
        ax.annotate(name.split(",")[0], xy=(met["target"], met["area"]),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=6, color=col, ha=ha, va=va,
                    bbox=BOX, zorder=9, annotation_clip=True,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.6,
                                    shrinkA=1, shrinkB=3))
    lo = min(area.min(), min(m["area"] for m in met_all))
    hi = max(area.max(), max(m["area"] for m in met_all))
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_xlabel("field on target (V/m)", fontsize=6.8)
    ax.set_ylabel("half-maximum area (cm$^2$)", fontsize=6.8)
    ax.tick_params(labelsize=6.0, width=0.7, length=2.5)
    ax.grid(lw=0.5, alpha=0.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=6, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, borderaxespad=0.3, labelspacing=0.22)
    ax.set_title("Tighter costs intensity, {:.0f} to {:.0f}$^\\circ$ ring"
                 .format(RING_SWEEP[0], RING_SWEEP[-1]), fontsize=8.0,
                 fontweight="bold", color=INK)


MONTAGES_BY_NAME = [(m[0], m[1]) for m in MONTAGES]


def audit(fig, foot, axes):
    fig.canvas.draw()
    fb = foot.get_window_extent()
    for ax in axes:
        ab = ax.get_window_extent()
        for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
            tb = t.get_window_extent()
            if tb.width <= 0.0 or not t.get_text():
                continue
            if tb.x1 < ab.x0 - 2.0 or tb.x0 > ab.x1 + 2.0:
                continue
            if fb.overlaps(tb):
                print("OVERLAP with", repr(t.get_text()))


def main():
    set_pub_style()
    np.random.default_rng(2291)
    x, y, (xx, yy) = grid()
    brain = inside(xx, yy, A_BRAIN, B_BRAIN)
    cell_cm2 = ((x[1] - x[0]) * (y[1] - y[0])) / 100.0

    fields, pots, met_all = [], [], []
    for _, src, _ in MONTAGES:
        v, e = solve(xx, yy, src)
        fields.append(e)
        pots.append(v)
        met_all.append(metrics(xx, yy, e, brain, cell_cm2))

    sweep = []
    for half_ang in RING_SWEEP:
        src = [(90.0, +I_TOTAL), (90.0 - half_ang, -I_TOTAL / 2.0),
               (90.0 + half_ang, -I_TOTAL / 2.0)]
        _, e = solve(xx, yy, src)
        m = metrics(xx, yy, e, brain, cell_cm2)
        sweep.append((half_ang, m["target"], m["area"]))

    for (name, _, _), m in zip(MONTAGES, met_all):
        print("{:24s} target {:.3f} V/m  peak {:.3f} V/m  half-max area "
              "{:5.1f} cm2".format(name, m["target"], m["peak"], m["area"]))
    print("ring sweep: {:.1f} cm2 at {:.0f} deg to {:.1f} cm2 at {:.0f} deg"
          .format(sweep[0][2], RING_SWEEP[0], sweep[-1][2], RING_SWEEP[-1]))
    assert met_all[1]["area"] < met_all[0]["area"], "the ring is the tighter"
    assert sweep[0][2] < sweep[-1][2], "a wider ring spreads the field"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_h = axes_inches(fig, MAIN)
    ax_d = axes_inches(fig, AUX_T)
    ax_t = axes_inches(fig, AUX_B)

    head_panel(fig, ax_h, x, y, xx, yy, pots[1], fields[1], brain, met_all[1],
               MONTAGES[1][0])
    depth_panel(ax_d, x, y, fields, met_all)
    tradeoff_panel(ax_t, sweep, met_all)



    panel_labels_inches(fig, [ax_h, ax_d, ax_t], dx=-0.62, dy=0.14)
    savefig(fig, "1057_bci_tes_field_focality.png")


if __name__ == "__main__":
    main()
