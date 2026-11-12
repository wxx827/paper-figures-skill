"""Shear and bending moment of a three-span continuous beam.

The left column is the classic set: the load case on top with its supports, the
shear diagram beneath it, and the bending moment diagram at the bottom drawn on
the tension side - sagging positive, plotted downward - so the curve always
falls on the face of the beam that is in tension.  The subordinate panels show
what changes when the live load is patterned rather than applied everywhere,
which is what actually sizes the reinforcement, and the reactions the
foundations have to carry.

The beam is solved, not sketched.  A two-node Euler-Bernoulli stiffness model
gives the support reactions; shear is their running sum minus the load applied
to the left of the section; the moment is the integral of that shear, with the
built-in end moment recovered from the condition M = 0 at the far pin rather
than read off the stiffness solution, which gives an independent check on the
sign convention.  Synthetic beam, arbitrary but self-consistent numbers.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Ellipse

from pubstyle import set_pub_style, savefig, despine
from figkit import axes_inches, panel_labels_inches, shared_limits

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
BASE = "#808080"

SPANS = [7.0, 9.0, 6.0]                       # m
NODE_X = [0.0, 7.0, 11.5, 16.0, 22.0]         # m; 11.5 carries the point load
SUPPORTS = [(0.0, "fixed"), (7.0, "roller"), (16.0, "pin"), (22.0, "roller")]
UDL_FULL = [22.0, 18.0, 18.0, 26.0]           # kN/m, one entry per element
POINT = [(11.5, 90.0)]                        # (x in m, downward kN)
DEAD_FRAC = 0.38                              # share of the UDL that is dead
EI = 1.0e5                                    # kN m2 (scale only; M is static)
L_TOT = NODE_X[-1]

# Live-load patterns: which *spans* (not elements) carry the full UDL.
PATTERNS = [
    ("all spans loaded", (0, 1, 2), True),
    ("spans 1 and 3", (0, 2), False),
    ("span 2 only", (1,), True),
    ("spans 1 and 2", (0, 1), True),
    ("spans 2 and 3", (1, 2), True),
]
SPAN_OF_ELEM = [0, 1, 1, 2]


def element_loads(active):
    return [UDL_FULL[e] if SPAN_OF_ELEM[e] in active else
            DEAD_FRAC * UDL_FULL[e] for e in range(len(UDL_FULL))]


def solve_reactions(w_el, with_point):
    """Two-node Euler-Bernoulli solve; returns upward reactions in kN."""
    n = len(NODE_X)
    k_g = np.zeros((2 * n, 2 * n))
    f_g = np.zeros(2 * n)
    for e in range(n - 1):
        ln = NODE_X[e + 1] - NODE_X[e]
        k_e = EI / ln ** 3 * np.array([
            [12.0, 6 * ln, -12.0, 6 * ln],
            [6 * ln, 4 * ln * ln, -6 * ln, 2 * ln * ln],
            [-12.0, -6 * ln, 12.0, -6 * ln],
            [6 * ln, 2 * ln * ln, -6 * ln, 4 * ln * ln]])
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        k_g[np.ix_(idx, idx)] += k_e
        w = w_el[e]
        f_g[idx] += np.array([-w * ln / 2.0, -w * ln ** 2 / 12.0,
                              -w * ln / 2.0, w * ln ** 2 / 12.0])
    if with_point:
        for xp, p in POINT:
            f_g[2 * NODE_X.index(xp)] -= p

    fixed = []
    for xs, kind in SUPPORTS:
        i = NODE_X.index(xs)
        fixed.append(2 * i)
        if kind == "fixed":
            fixed.append(2 * i + 1)
    free = [d for d in range(2 * n) if d not in fixed]
    u = np.zeros(2 * n)
    u[free] = np.linalg.solve(k_g[np.ix_(free, free)], f_g[free])
    r = k_g @ u - f_g
    return {xs: float(r[2 * NODE_X.index(xs)]) for xs, _ in SUPPORTS}


def internal_forces(x, w_el, with_point, reactions):
    """Shear (kN) and sagging-positive moment (kN m) along the beam."""
    v = np.zeros_like(x)
    for xs, r in reactions.items():
        v += np.where(x >= xs, r, 0.0)
    for e in range(len(w_el)):
        a, b = NODE_X[e], NODE_X[e + 1]
        v -= w_el[e] * np.clip(x - a, 0.0, b - a)
    if with_point:
        for xp, p in POINT:
            v -= np.where(x > xp, p, 0.0)
    m = np.concatenate([[0.0], np.cumsum(np.diff(x) * 0.5 * (v[1:] + v[:-1]))])
    # The far support is a roller with no overhang, so M must vanish there;
    # that fixes the built-in moment at A without trusting a stiffness sign.
    m = m - m[-1]
    return v, m


def load_case(active, with_point, x):
    w_el = element_loads(active)
    reactions = solve_reactions(w_el, with_point)
    v, m = internal_forces(x, w_el, with_point, reactions)
    return dict(w_el=w_el, reactions=reactions, v=v, m=m)


def symbol_scales(ax):
    """Data units per inch on each axis, so symbols can be drawn isotropic."""
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x1 - x0) / (bb.width * fw), (y1 - y0) / (bb.height * fh)


def support_symbol(ax, xs, kind, sx, sy, h=0.13):
    """Pin / roller / fixed symbol, sized in inches so it never skews."""
    dy, dx = h * sy, 0.60 * h * sx
    if kind == "fixed":
        ax.add_patch(Rectangle((xs - 1.5 * dx, -1.55 * dy), 1.5 * dx,
                               3.1 * dy, facecolor="none", edgecolor=RULE,
                               lw=0.9, hatch="////", zorder=5))
        ax.plot([xs, xs], [-1.55 * dy, 1.55 * dy], color=RULE, lw=1.6,
                zorder=6)
        return
    ax.add_patch(Polygon([[xs, 0.0], [xs - dx, -dy], [xs + dx, -dy]],
                         facecolor="white", edgecolor=RULE, lw=0.9, zorder=6))
    if kind == "roller":
        for s in (-0.5, 0.5):
            ax.add_patch(Ellipse((xs + s * dx, -dy - 0.28 * dy),
                                 0.56 * dx, 0.56 * dy, facecolor="white",
                                 edgecolor=RULE, lw=0.8, zorder=6))
        y_g = -dy - 0.56 * dy
    else:
        y_g = -dy
    ax.plot([xs - 1.5 * dx, xs + 1.5 * dx], [y_g, y_g], color=RULE, lw=1.0,
            zorder=5)
    for k in range(4):
        gx = xs - 1.35 * dx + k * 0.9 * dx
        ax.plot([gx, gx - 0.35 * dx], [y_g, y_g - 0.42 * dy], color=RULE,
                lw=0.7, zorder=5)


def load_panel(ax, case):
    ax.set_xlim(-1.1, L_TOT + 1.1)
    ax.set_ylim(-1.25, 1.95)
    ax.plot([0.0, L_TOT], [0.0, 0.0], color=RULE, lw=2.4,
            solid_capstyle="butt", zorder=4)
    w_max = max(UDL_FULL)
    for e, w in enumerate(case["w_el"]):
        a, b = NODE_X[e], NODE_X[e + 1]
        hgt = 0.30 + 0.62 * w / w_max
        ax.add_patch(Rectangle((a, 0.0), b - a, hgt, facecolor=BLUE,
                               alpha=0.16, edgecolor=BLUE, lw=0.7, zorder=2))
        for xa in np.linspace(a + 0.32, b - 0.32, max(int((b - a) / 1.1), 2)):
            ax.annotate("", xy=(xa, 0.045), xytext=(xa, hgt),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.7,
                                        mutation_scale=5), zorder=3)
        ax.text(0.5 * (a + b), hgt + 0.06, f"{w:.0f} kN/m", ha="center",
                va="bottom", fontsize=6.3, color=BLUE)
    for xp, p in POINT:
        ax.annotate("", xy=(xp, 0.06), xytext=(xp, 1.72),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                    mutation_scale=8), zorder=6)
        ax.text(xp + 0.25, 1.70, f"{p:.0f} kN", ha="left", va="top",
                fontsize=6.6, color=RED)

    sx, sy = symbol_scales(ax)
    for xs, kind in SUPPORTS:
        support_symbol(ax, xs, kind, sx, sy)
    for xs, kind in SUPPORTS:
        ax.text(xs, -1.20, kind, ha="center", va="bottom", fontsize=6.2,
                color=MUTED)
    for i, ln in enumerate(SPANS):
        a = sum(SPANS[:i])
        ax.annotate("", xy=(a, -0.62), xytext=(a + ln, -0.62),
                    arrowprops=dict(arrowstyle="<|-|>", color=SUB, lw=0.7,
                                    mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.text(a + ln / 2.0, -0.58, f"L{i + 1} = {ln:.0f} m", ha="center",
                va="bottom", fontsize=6.3, color=SUB)
    ax.axis("off")
    ax.set_title("Three-span continuous beam, all spans loaded",
                 fontsize=9.0, fontweight="bold", color=INK)


def shear_panel(ax, x, case):
    v = case["v"]
    ax.axhline(0.0, color=RULE, lw=0.9, zorder=4)
    ax.fill_between(x, 0.0, v, where=v >= 0, color=BLUE, alpha=0.22, lw=0,
                    zorder=2)
    ax.fill_between(x, 0.0, v, where=v <= 0, color=BLUE, alpha=0.10, lw=0,
                    zorder=2)
    ax.plot(x, v, color=BLUE, lw=1.5, zorder=5)
    i_hi, i_lo = int(np.argmax(v)), int(np.argmin(v))
    for i, va in ((i_hi, "bottom"), (i_lo, "top")):
        ax.plot([x[i]], [v[i]], marker="o", ms=3.6, mfc="white", mec=BLUE,
                mew=1.1, zorder=7)
        ax.annotate(f"{v[i]:+.0f} kN", xy=(x[i], v[i]),
                    xytext=(0, 4.0 if va == "bottom" else -4.0),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=6.4, color=BLUE)
    for xs, _ in SUPPORTS:
        ax.axvline(xs, color="#e0e0e0", lw=0.7, zorder=1)
    ax.set_xlim(-1.1, L_TOT + 1.1)
    ax.set_ylim(*shared_limits([v], pad=0.17))
    ax.set_ylabel("shear $V$ (kN)", fontsize=7.8)
    ax.tick_params(labelsize=6.9, labelbottom=False)
    despine(ax)
    ax.set_axisbelow(True)


def moment_panel(ax, x, case):
    m = case["m"]
    ax.axhline(0.0, color=RULE, lw=0.9, zorder=4)
    ax.fill_between(x, 0.0, m, where=m >= 0, color=RED, alpha=0.22, lw=0,
                    zorder=2)
    ax.fill_between(x, 0.0, m, where=m <= 0, color=GREEN, alpha=0.20, lw=0,
                    zorder=2)
    ax.plot(x, m, color=RED, lw=1.7, zorder=5)
    for xs, _ in SUPPORTS:
        ax.axvline(xs, color="#e0e0e0", lw=0.7, zorder=1)

    i_sag = int(np.argmax(m))
    ax.plot([x[i_sag]], [m[i_sag]], marker="o", ms=4.2, mfc="white", mec=RED,
            mew=1.2, zorder=7)
    # Below the zero line and left of the sagging bulge is the one pocket of
    # this panel that no part of the diagram reaches into.
    ax.annotate(f"max sagging {m[i_sag]:.0f} " + r"$kN{\cdot}m$"
                + f"\nat x = {x[i_sag]:.2f} m", xy=(x[i_sag], m[i_sag]),
                xytext=(x[i_sag] - 5.4, m[i_sag] * 0.70), ha="center",
                va="center", fontsize=6.4, color=RED, linespacing=1.35,
                zorder=8,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                shrinkA=3.0, shrinkB=3.0))
    hog = []
    for xs, _ in SUPPORTS:
        i = int(np.argmin(np.abs(x - xs)))
        if m[i] < -1.0:
            hog.append((xs, m[i]))
            ax.plot([xs], [m[i]], marker="o", ms=3.8, mfc="white", mec=GREEN,
                    mew=1.1, zorder=7)
            # The axis is inverted, so a positive offset moves the label clear
            # of the hogging peak rather than into it.
            ax.annotate(f"{m[i]:.0f}", xy=(xs, m[i]), xytext=(0, 5.0),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.4, color=GREEN)
    sign = np.sign(m)
    cross = x[:-1][(sign[:-1] * sign[1:] < 0)]
    for xc in cross:
        ax.plot([xc], [0.0], marker="D", ms=3.0, mfc=GOLD, mec="white",
                mew=0.6, zorder=8)

    ax.set_xlim(-1.1, L_TOT + 1.1)
    ax.invert_yaxis()          # sagging positive, drawn on the tension side
    ax.set_xlabel("distance along the beam (m)", fontsize=7.8)
    ax.set_ylabel("moment $M$ (" + r"$kN{\cdot}m$" + ")", fontsize=7.8)
    ax.tick_params(labelsize=6.9)
    despine(ax)
    ax.set_axisbelow(True)
    ax.text(0.012, 0.955, "sagging (+) drawn downward", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.3, color=MUTED)
    return m[i_sag], x[i_sag], hog, cross


def envelope_panel(ax, x, cases, prim):
    stack = np.array([c["m"] for c in cases])
    hi, lo = stack.max(axis=0), stack.min(axis=0)
    ax.fill_between(x, lo, hi, color=BASE, alpha=0.22, lw=0, zorder=2)
    ax.plot(x, hi, color=RED, lw=1.2, zorder=5)
    ax.plot(x, lo, color=GREEN, lw=1.2, zorder=5)
    ax.plot(x, prim["m"], color=INK, lw=1.2, ls=(0, (4.0, 2.0)), zorder=6)
    ax.axhline(0.0, color=RULE, lw=0.8, zorder=4)
    for xs, _ in SUPPORTS:
        ax.axvline(xs, color="#e0e0e0", lw=0.7, zorder=1)
    ax.set_xlim(-1.1, L_TOT + 1.1)
    ax.invert_yaxis()
    ax.set_xlabel("distance along the beam (m)", fontsize=7.6)
    ax.set_ylabel("moment envelope (" + r"$kN{\cdot}m$" + ")", fontsize=7.6)
    ax.tick_params(labelsize=6.8)
    despine(ax)
    ax.legend([plt.Line2D([], [], color=RED, lw=1.2),
               plt.Line2D([], [], color=GREEN, lw=1.2),
               plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (4.0, 2.0)))],
              ["max sagging", "max hogging", "all spans loaded"],
              fontsize=6.3, frameon=False, loc="lower left", ncol=1,
              handlelength=1.7, labelspacing=0.28)
    ax.set_title(f"Patterned live load over {len(cases)} cases",
                 fontsize=8.4, fontweight="bold", color=INK)
    return hi, lo


def reaction_panel(ax, cases, prim):
    xs = [s[0] for s in SUPPORTS]
    names = [f"{chr(65 + i)}\n{s[1]}" for i, s in enumerate(SUPPORTS)]
    base = np.array([prim["reactions"][k] for k in xs])
    allr = np.array([[c["reactions"][k] for k in xs] for c in cases])
    lo, hi = allr.min(axis=0), allr.max(axis=0)
    pos = np.arange(len(xs))
    ax.bar(pos, base, width=0.56, color=BLUE, edgecolor="white", lw=0.7,
           zorder=3)
    ax.errorbar(pos, base, yerr=[base - lo, hi - base], fmt="none",
                ecolor=RULE, elinewidth=0.8, capsize=2.6, zorder=5)
    for p, b, h in zip(pos, base, hi):
        ax.text(p, h + 0.03 * hi.max(), f"{b:.0f}", ha="center", va="bottom",
                fontsize=6.4, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels(names, fontsize=6.4)
    ax.set_ylim(0.0, hi.max() * 1.24)
    ax.set_ylabel("support reaction (kN)", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=6.8)
    despine(ax)
    ax.grid(axis="y", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.965, "bars: all spans loaded; whiskers: pattern range",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.3,
            color=SUB)
    ax.set_title("Reactions and their pattern range", fontsize=8.4,
                 fontweight="bold", color=INK)
    return base, lo, hi


def main():
    set_pub_style()
    rng = np.random.default_rng(85706)
    x = np.linspace(0.0, L_TOT, 4401)

    # A small perturbation on the point load, so no printed value is fixed.
    POINT[0] = (POINT[0][0], POINT[0][1] * (1.0 + 0.02 * rng.normal()))

    cases = [load_case(a, p, x) for _, a, p in PATTERNS]
    prim = cases[0]

    total_load = sum(w * (NODE_X[e + 1] - NODE_X[e])
                     for e, w in enumerate(prim["w_el"])) + POINT[0][1]
    assert abs(sum(prim["reactions"].values()) - total_load) < 1e-6
    assert abs(prim["m"][-1]) < 1e-9

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_load = axes_inches(fig, (0.94, 4.74, 4.94, 0.86))
    ax_shear = axes_inches(fig, (0.94, 2.98, 4.94, 1.50))
    ax_mom = axes_inches(fig, (0.94, 0.94, 4.94, 1.72))
    ax_env = axes_inches(fig, (7.32, 3.42, 2.68, 1.94))
    ax_rxn = axes_inches(fig, (7.32, 0.94, 2.68, 1.94))

    load_panel(ax_load, prim)
    shear_panel(ax_shear, x, prim)
    m_sag, x_sag, hog, cross = moment_panel(ax_mom, x, prim)
    hi, lo = envelope_panel(ax_env, x, cases, prim)
    base, r_lo, r_hi = reaction_panel(ax_rxn, cases, prim)
    # Both moment panels show the same quantity, so they share one scale.
    span = shared_limits([prim["m"], hi, lo], pad=0.10)
    for a in (ax_mom, ax_env):
        a.set_ylim(span[1], span[0])           # inverted: tension side down

    worst = min(hog, key=lambda h: h[1])
    panel_labels_inches(fig, [ax_load, ax_shear, ax_mom, ax_env, ax_rxn],
                        dx=-0.60, dy=0.14)
    savefig(fig, "857_str_moment_diagram.png")


if __name__ == "__main__":
    main()
