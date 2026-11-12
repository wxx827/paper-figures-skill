"""Potential-pH diagram for iron in water, with the three regions named.

Every boundary on the sheet is one Nernst expression evaluated over the pH
axis, not a traced line: the standard potential, the electron count and the
proton count of each half reaction are the only inputs, and the dissolved
iron activity is fixed at 1e-6 mol/L so the diagram means "corroding" in the
engineering sense.  The region colouring is produced by classifying a grid of
(pH, E) points with the *same* line functions, so a coloured field can never
disagree with the boundary drawn across it, and the triple points quoted in
the annotations are solved numerically from those functions rather than read
off the picture.  The two dashed lines a and b are the water stability window;
their slope is exactly -59.16 mV per pH unit, which the script asserts.  The
side panels re-read the same classified grid: how much of the water window is
corroding at each pH, and the slope every boundary carries.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
EMBER = "#c0603f"
SAND = "#d98c5f"
BLUE = "#3b5b92"
TEAL = "#2a7f7a"
GREY = "#8a8a8a"
STEEL = "#b8bcc2"

SEED = 1068
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

K = 0.05916                 # 2.303 R T / F at 25 C, V per decade
LOG_C = -6.0                # log10 of the dissolved iron activity, mol/L
PH_LO, PH_HI = -1.0, 16.0
E_LO, E_HI = -1.45, 1.85

# Region codes shared by the field, the labels and the side panels.
FE, FE2, FE3, MAG, HEM, HFE = 0, 1, 2, 3, 4, 5
REGION_COLOR = [STEEL, "#e8a99a", "#f0c9a8", "#7d9dc4", "#a8cfc9", "#e3b7ae"]
REGION_NAME = ["Fe (immunity)", "Fe$^{2+}$ (corrosion)", "Fe$^{3+}$ (corrosion)",
               "Fe$_3$O$_4$ (passivation)", "Fe$_2$O$_3$ (passivation)",
               "HFeO$_2^-$ (corrosion)"]
CLASS_OF = {FE: "immunity", FE2: "corrosion", FE3: "corrosion",
            MAG: "passivation", HEM: "passivation", HFE: "corrosion"}


# ---------------------------------------------------------------------------
# Half reactions.  E = E0 - (K / n) * (m * pH + s * log10 c)
# ---------------------------------------------------------------------------
def nernst(e0, n, m, s=0.0):
    """Return E(pH) for one half reaction at the fixed ion activity."""
    def line(ph):
        return e0 - (K / n) * (m * np.asarray(ph, dtype=float) + s * LOG_C)
    return line


LINES = {
    "a": nernst(0.000, 2, 2),                    # 2 H+ + 2 e- = H2
    "b": nernst(1.229, 4, 4),                    # O2 + 4 H+ + 4 e- = 2 H2O
    "Fe2+/Fe": nernst(-0.440, 2, 0, -1.0),       # Fe2+ + 2 e- = Fe
    "Fe3O4/Fe": nernst(-0.085, 8, 8),            # Fe3O4 + 8 H+ + 8 e- = 3 Fe + 4 H2O
    "Fe3O4/Fe2+": nernst(0.980, 2, 8, 3.0),      # Fe3O4 + 8 H+ + 2 e- = 3 Fe2+ + 4 H2O
    "Fe2O3/Fe3O4": nernst(0.221, 2, 2),          # 3 Fe2O3 + 2 H+ + 2 e- = 2 Fe3O4 + H2O
    "Fe2O3/Fe2+": nernst(0.728, 2, 6, 2.0),      # Fe2O3 + 6 H+ + 2 e- = 2 Fe2+ + 3 H2O
    "Fe3+/Fe2+": nernst(0.771, 1, 0, 0.0),       # Fe3+ + e- = Fe2+
    "Fe/HFeO2-": nernst(0.493, 2, 3, -1.0),      # HFeO2- + 3 H+ + 2 e- = Fe + 2 H2O
}


def slope_of(name):
    """dE/dpH in mV per pH unit, taken from the function that is drawn."""
    f = LINES[name]
    return 1000.0 * float(f(9.0) - f(8.0))


def cross(f, g, lo, hi):
    return float(brentq(lambda p: f(p) - g(p), lo, hi))


def triple_points():
    ph_a = cross(LINES["Fe2+/Fe"], LINES["Fe3O4/Fe2+"], 4.0, 12.0)
    ph_b = cross(LINES["Fe3O4/Fe2+"], LINES["Fe2O3/Fe2+"], 2.0, 12.0)
    ph_c = cross(LINES["Fe2O3/Fe2+"], LINES["Fe3+/Fe2+"], -1.0, 6.0)
    ph_alk = cross(LINES["Fe3O4/Fe"], LINES["Fe/HFeO2-"], 9.0, 16.0)
    return ph_a, ph_b, ph_c, ph_alk


def classify(ph, ee, ph_c, ph_alk):
    """Region code for a (pH, E) mesh, built from the same line functions."""
    e_metal = np.minimum(np.minimum(LINES["Fe2+/Fe"](ph), LINES["Fe3O4/Fe"](ph)),
                         LINES["Fe/HFeO2-"](ph))
    e_aq_top = np.minimum(LINES["Fe3O4/Fe2+"](ph), LINES["Fe2O3/Fe2+"](ph))
    e_ox = LINES["Fe2O3/Fe3O4"](ph)
    e_fe3 = LINES["Fe3+/Fe2+"](ph)

    reg = np.full(ee.shape, HEM, dtype=int)
    alk = ph >= ph_alk
    reg = np.where(alk & (ee < e_ox), HFE, reg)
    acid = ~alk
    reg = np.where(acid & (ee < e_aq_top) & (ee > e_fe3) & (ph < ph_c), FE3, reg)
    reg = np.where(acid & (ee < e_aq_top) &
                   ~((ee > e_fe3) & (ph < ph_c)), FE2, reg)
    reg = np.where(acid & (ee >= e_aq_top) & (ee < e_ox), MAG, reg)
    reg = np.where(ee < e_metal, FE, reg)
    return reg


def label_anchor(reg, code, ph, ee):
    """Deepest interior point of a region, so a label cannot touch a line."""
    mask = (reg == code)
    if not mask.any():
        return None
    pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    dist = distance_transform_edt(pad)[1:-1, 1:-1]
    j, i = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(ph[0, i]), float(ee[j, 0])


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def main_panel(ax, ph, ee, reg, tps):
    ph_a, ph_b, ph_c, ph_alk = tps
    cmap = ListedColormap(REGION_COLOR)
    norm = BoundaryNorm(np.arange(-0.5, len(REGION_COLOR)), cmap.N)
    ax.pcolormesh(ph, ee, reg, cmap=cmap, norm=norm, shading="auto",
                  zorder=1, rasterized=True)

    grid = np.linspace(PH_LO, PH_HI, 400)
    segs = (("Fe2+/Fe", PH_LO, ph_a, RED),
            ("Fe3O4/Fe", ph_a, ph_alk, BLUE),
            ("Fe/HFeO2-", ph_alk, PH_HI, RED),
            ("Fe3O4/Fe2+", ph_b, ph_a, BLUE),
            ("Fe2O3/Fe2+", ph_c, ph_b, BLUE),
            ("Fe2O3/Fe3O4", ph_b, PH_HI, TEAL),
            ("Fe3+/Fe2+", PH_LO, ph_c, EMBER))
    for name, lo, hi, col in segs:
        x = grid[(grid >= lo) & (grid <= hi)]
        x = np.concatenate(([lo], x, [hi]))
        ax.plot(x, LINES[name](x), color=col, lw=1.5, zorder=6)
    ax.plot([ph_alk, ph_alk], [LINES["Fe/HFeO2-"](ph_alk),
                               LINES["Fe2O3/Fe3O4"](ph_alk)],
            color=TEAL, lw=1.5, zorder=6)
    ax.plot([ph_c, ph_c], [LINES["Fe3+/Fe2+"](ph_c), E_HI], color=EMBER,
            lw=1.5, zorder=6)

    for name, txt in (("a", "a: 2H$^+$ + 2e$^-$ = H$_2$"),
                      ("b", "b: O$_2$ + 4H$^+$ + 4e$^-$ = 2H$_2$O")):
        ax.plot(grid, LINES[name](grid), color=INK, lw=1.15,
                ls=(0, (5.0, 2.4)), zorder=8)
        y = float(LINES[name](12.6))
        ax.text(12.6, y + 0.05, txt, fontsize=6.4, color=INK, ha="center",
                va="bottom", rotation=-17.5, rotation_mode="anchor", zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor="none", alpha=0.86))

    for code in (FE, FE2, FE3, MAG, HEM, HFE):
        anchor = label_anchor(reg, code, ph, ee)
        if anchor is None:
            continue
        col = RED if CLASS_OF[code] == "corrosion" else (
            BLUE if CLASS_OF[code] == "passivation" else "#3f4650")
        ax.text(anchor[0], anchor[1], REGION_NAME[code], fontsize=7.4,
                color=col, ha="center", va="center", fontweight="bold",
                zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.plot([ph_a, ph_b, ph_c, ph_alk],
            [LINES["Fe2+/Fe"](ph_a), LINES["Fe3O4/Fe2+"](ph_b),
             LINES["Fe3+/Fe2+"](ph_c), LINES["Fe3O4/Fe"](ph_alk)],
            marker="o", ms=3.6, mfc="white", mec=INK, mew=1.0, ls="none",
            zorder=11)
    ax.annotate("triple point Fe / Fe$^{2+}$ / Fe$_3$O$_4$\nat pH %.2f, "
                "%.3f V" % (ph_a, LINES["Fe2+/Fe"](ph_a)),
                xy=(ph_a, LINES["Fe2+/Fe"](ph_a)), xytext=(2.1, -1.16),
                fontsize=6.4, color=INK, ha="left", va="bottom",
                linespacing=1.32, zorder=12,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7, shrinkA=3,
                                shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor="none", alpha=0.86))
    ax.annotate("Fe$_2$O$_3$ + 6H$^+$ + 2e$^-$ = 2Fe$^{2+}$ + 3H$_2$O\n"
                "%.0f mV per pH unit at a$_{Fe^{2+}}$ = 10$^{%.0f}$"
                % (slope_of("Fe2O3/Fe2+"), LOG_C),
                xy=(4.2, float(LINES["Fe2O3/Fe2+"](4.2))), xytext=(5.6, 1.42),
                fontsize=6.4, color=BLUE, ha="left", va="top",
                linespacing=1.32, zorder=12,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkA=3,
                                shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.26", facecolor="white",
                          edgecolor="none", alpha=0.86))

    handles = [Line2D([], [], color=INK, lw=1.15, ls=(0, (5.0, 2.4)),
                      label="water stability lines a and b"),
               Line2D([], [], color=RED, lw=1.5, label="metal / ion boundary"),
               Line2D([], [], color=BLUE, lw=1.5, label="oxide / ion boundary"),
               Line2D([], [], color=TEAL, lw=1.5, label="oxide / oxide boundary")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#cccccc",
                    borderpad=0.42, handlelength=1.9)
    leg.get_frame().set_alpha(0.9)
    leg.set_zorder(12)

    ax.set_xlim(PH_LO, PH_HI)
    ax.set_ylim(E_LO, E_HI)
    ax.set_xlabel("pH")
    ax.set_ylabel("potential E (V vs SHE)")
    ax.set_xticks(np.arange(0, 17, 2))
    ax.tick_params(length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Fe / H$_2$O at 25 " + u"\u00b0" + "C, dissolved iron fixed "
                 "at 10$^{-6}$ mol/L", fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def window_panel(ax, ph, ee, reg):
    """How the water window splits between the three regions, pH by pH."""
    ph1 = ph[0]
    inside = (ee >= LINES["a"](ph)) & (ee <= LINES["b"](ph))
    total = inside.sum(axis=0).astype(float)
    total[total == 0] = np.nan
    fracs = {}
    for tag in ("immunity", "passivation", "corrosion"):
        codes = [c for c, t in CLASS_OF.items() if t == tag]
        m = np.isin(reg, codes) & inside
        fracs[tag] = 100.0 * m.sum(axis=0) / total

    ax.stackplot(ph1, fracs["corrosion"], fracs["passivation"],
                 fracs["immunity"],
                 colors=["#e8a99a", "#8fb3d4", STEEL], edgecolor="white",
                 lw=0.35, zorder=3)
    i_best = int(np.nanargmax(fracs["passivation"]))
    ph_best = float(ph1[i_best])
    ax.axvline(ph_best, color=BLUE, lw=0.9, ls=(0, (2.6, 1.8)), zorder=5)
    ax.text(ph_best + 0.5, 52.0, "widest passive\nwindow at pH %.1f\n(%.0f %% "
            "of the a-b band)" % (ph_best, fracs["passivation"][i_best]),
            fontsize=6.0, color=BLUE, ha="left", va="center", linespacing=1.30,
            zorder=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                edgecolor="none", alpha=0.85))
    ax.text(0.6, 12.0, "corrosion", fontsize=6.2, color="#8c2f2f", ha="left",
            va="center", zorder=8)
    ax.set_xlim(PH_LO, PH_HI)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("pH", fontsize=6.8)
    ax.set_ylabel("share of the water window (%)", fontsize=6.3)
    ax.set_xticks(np.arange(0, 17, 4))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("What a vertical cut through the diagram contains",
                 fontsize=8.0, fontweight="bold", color=INK)
    return ph_best, float(fracs["passivation"][i_best])


def slope_panel(ax):
    names = ["a", "b", "Fe3O4/Fe", "Fe2O3/Fe3O4", "Fe/HFeO2-", "Fe3O4/Fe2+",
             "Fe2O3/Fe2+", "Fe2+/Fe"]
    show = ["line a", "line b", "Fe$_3$O$_4$/Fe", "Fe$_2$O$_3$/Fe$_3$O$_4$",
            "Fe/HFeO$_2^-$", "Fe$_3$O$_4$/Fe$^{2+}$", "Fe$_2$O$_3$/Fe$^{2+}$",
            "Fe$^{2+}$/Fe"]
    vals = np.array([slope_of(n) for n in names])
    cols = [INK, INK, BLUE, TEAL, RED, BLUE, BLUE, RED]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", lw=0.5,
            zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v - 4.0, yy, "%.1f" % v, fontsize=6.0, color=SUB, ha="right",
                va="center", zorder=6)
    ax.axvline(-K * 1000.0, color=GREY, lw=0.9, ls=(0, (2.6, 1.8)), zorder=4)
    ax.text(-K * 1000.0 - 3.0, -0.85, "-%.2f mV/pH:\none proton per electron"
            % (K * 1000.0), fontsize=6.0, color=SUB, ha="right", va="bottom",
            linespacing=1.30, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="none", alpha=0.85))
    ax.set_yticks(y)
    ax.set_yticklabels(show, fontsize=6.2)
    ax.set_xlim(-260.0, 30.0)
    ax.set_ylim(-1.4, len(names) - 0.4)
    ax.set_xlabel("boundary slope dE/dpH (mV per pH unit)", fontsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Every slope is protons per electron", fontsize=8.0,
                 fontweight="bold", color=INK)
    return vals


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
    np.random.default_rng(SEED)

    ph1 = np.linspace(PH_LO, PH_HI, 760)
    e1 = np.linspace(E_LO, E_HI, 700)
    ph, ee = np.meshgrid(ph1, e1)
    tps = triple_points()
    reg = classify(ph, ee, tps[2], tps[3])

    for name in ("a", "b"):
        assert abs(slope_of(name) + K * 1000.0) < 1e-9, \
            "the water stability lines must fall at -59.16 mV per pH unit"
    assert tps[2] < tps[1] < tps[0] < tps[3], \
        "the triple points have to appear in ascending pH order"
    for code in (FE, FE2, FE3, MAG, HEM, HFE):
        assert (reg == code).any(), "every region must survive the grid"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_win = axes_inches(fig, AUX_T)
    ax_slope = axes_inches(fig, AUX_B)

    main_panel(ax_main, ph, ee, reg, tps)
    ph_best, frac_best = window_panel(ax_win, ph, ee, reg)
    slope_panel(ax_slope)

    inside = (ee >= LINES["a"](ph)) & (ee <= LINES["b"](ph))
    pass_share = 100.0 * np.isin(reg, [MAG, HEM])[inside].mean()
    corr_share = 100.0 * np.isin(reg, [FE2, FE3, HFE])[inside].mean()



    panel_labels_inches(fig, [ax_main, ax_win, ax_slope], dx=-0.62, dy=0.14)
    savefig(fig, "1068_cor_pourbaix_diagram.png")


if __name__ == "__main__":
    main()
