"""The Lawson diagram: where six decades of tokamaks sit against ignition.

Both boundaries are derived rather than traced.  The D-T reactivity is the
Maxwellian average <sigma v> = sqrt(8 / pi m_r) T^-3/2 * integral sigma(E) E
exp(-E/T) dE, integrated numerically over the Bosch-Hale S-factor form of the
cross section, so the curve carries the 64 keV resonance with it.  Steady
power balance for a 50:50 D-T plasma with a bremsstrahlung sink then gives the
triple product required for a given gain Q,

    n T tau_E  =  3 T^2 / ( (1/4) <sigma v> E_alpha (1 + 5/Q) / 3 - C_b sqrt(T) T / 3 )

whose Q -> infinity limit is the ignition contour and whose Q = 1 branch is
scientific breakeven; the bremsstrahlung term is what bends both curves into a
vertical asymptote near 4 keV.  Every experimental point is placed by solving
the same relation for its published gain, so a device plotted above the Q = 1
contour has Q > 1 by construction and cannot disagree with its own label.  The
right column re-reads the same table: triple product against calendar year on
a log axis with a fitted doubling time, and the gains as a bar chart.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, lighten, get_cmap
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"
PURPLE = "#8e6bbf"
GREEN = "#2e7d5b"
GOLD = "#c9a227"
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"

KEV = 1.602176634e-16          # J per keV
MB = 1.0e-31                   # m^2 per millibarn
M_R_DT = 1.2074 * 1.66053907e-27      # D-T reduced mass, kg
E_ALPHA = 3520.0               # keV carried by the alpha
E_FUS = 17600.0                # keV released per D-T reaction
C_BREM = 5.34e-37              # W m^3 keV^-1/2, n_e n_i sqrt(T) bremsstrahlung

# Bosch-Hale S-factor for T(d,n)alpha, E in keV, sigma in millibarn.
BG_DT = 34.3827
A_DT = (6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0)
B_DT = (6.38e1, -9.95e-1, 6.981e-5, 1.728e-4)

#        label          year   T_i(keV)   Q        marker
DEVICES = [
    ("T-3", 1968, 0.40, 4.0e-9, "o"),
    ("Alcator A", 1975, 0.85, 1.0e-6, "o"),
    ("PLT", 1978, 6.0, 3.0e-3, "o"),
    ("Alcator C", 1983, 1.5, 2.0e-4, "o"),
    ("TFTR", 1986, 20.0, 8.0e-3, "o"),
    ("JET", 1991, 18.0, 1.2e-1, "o"),
    ("JT-60U", 1993, 30.0, 4.0e-1, "o"),
    ("TFTR (D-T)", 1994, 32.0, 2.7e-1, "o"),
    ("JET (DTE1)", 1997, 28.0, 6.2e-1, "o"),
    ("JT-60U (eq.)", 1998, 45.0, 1.25, "o"),
    ("JET (DTE2)", 2021, 13.0, 3.3e-1, "o"),
    ("EAST", 2023, 5.0, 8.0e-3, "o"),
]
TARGETS = [("ITER", 16.0, 10.0, (10.0, 6.0), "left", "bottom"),
           ("SPARC", 9.0, 11.0, (-10.0, -3.0), "right", "top")]

FW, FH = 10.4, 6.4
MAIN = (0.86, 1.06, 4.86, 4.14)
AUX1 = (6.62, 3.30, 2.80, 1.90)
AUX2 = (6.62, 1.06, 2.80, 1.52)


# ------------------------------------------------------------------ physics
def sigma_dt(e_kev):
    """D-T fusion cross section in millibarn, E is the centre-of-mass energy."""
    e = np.asarray(e_kev, dtype=float)
    num = A_DT[0] + e * (A_DT[1] + e * (A_DT[2] + e * (A_DT[3] + e * A_DT[4])))
    den = 1.0 + e * (B_DT[0] + e * (B_DT[1] + e * (B_DT[2] + e * B_DT[3])))
    s = num / den
    return s / (e * np.exp(BG_DT / np.sqrt(e)))


def reactivity(temp_kev, n_node=3000, e_max=900.0):
    """<sigma v> in m^3/s by Maxwellian integration of the cross section."""
    e = np.logspace(np.log10(0.3), np.log10(e_max), n_node)
    sig = sigma_dt(e) * MB                      # m^2
    t = np.atleast_1d(np.asarray(temp_kev, dtype=float))
    integ = np.trapezoid(
        sig[None, :] * (e[None, :] * KEV) *
        np.exp(-e[None, :] / t[:, None]), e * KEV, axis=1)
    pre = np.sqrt(8.0 / (np.pi * M_R_DT)) * (t * KEV) ** -1.5
    out = pre * integ
    return out if np.ndim(temp_kev) else float(out[0])


def triple_for_q(temp_kev, q):
    """Required n T tau_E in m^-3 keV s for gain q, with bremsstrahlung."""
    t = np.asarray(temp_kev, dtype=float)
    sv = reactivity(t)
    heat = 0.25 * sv * E_ALPHA * KEV * (1.0 + 5.0 / q)   # W m^3 per n^2
    loss = C_BREM * np.sqrt(t)
    net = heat - loss
    with np.errstate(divide="ignore", invalid="ignore"):
        ntau = np.where(net > 0.0, 3.0 * t * KEV / net, np.nan)
    return ntau * t


def q_of(temp_kev, ntt):
    """Invert the same relation: the gain implied by a (T, nTtau) point."""
    t = float(temp_kev)
    sv = reactivity(t)
    need = 3.0 * t * KEV * t / ntt + C_BREM * np.sqrt(t)
    return 5.0 / (need / (0.25 * sv * E_ALPHA * KEV) - 1.0)


# ------------------------------------------------------------------ panels
def main_panel(ax, tab, tgt):
    t = np.logspace(np.log10(0.30), np.log10(200.0), 700)
    ign = triple_for_q(t, 1.0e9)
    brk = triple_for_q(t, 1.0)

    ax.fill_between(t, ign, 1e25, color=lighten(RED, 0.90), lw=0.0, zorder=1)
    ax.plot(t, ign, color=RED, lw=2.0, zorder=6)
    ax.plot(t, brk, color=ORANGE, lw=1.5, ls=(0, (5.0, 2.4)), zorder=6)
    ax.text(160.0, float(np.interp(160.0, t, brk)) * 1.7, "breakeven, Q = 1",
            ha="right", va="bottom", fontsize=6.4, color=ORANGE, zorder=9)
    for qv, col in ((0.1, PURPLE), (0.01, BLUE)):
        ntt = triple_for_q(t, qv)
        ax.plot(t, ntt, color=col, lw=0.8, ls=(0, (1.6, 1.8)), zorder=5,
                alpha=0.85)
        ax.text(160.0, float(np.interp(160.0, t, ntt)) * 1.7,
                "Q = {:g}".format(qv), ha="right", va="bottom", fontsize=6.2,
                color=col, zorder=9)

    k = int(np.nanargmin(ign))
    ax.plot([t[k]], [ign[k]], marker="v", ms=4.4, color=RED, zorder=8)
    ax.annotate("cheapest ignition at\nT$_i$ = {:.0f} keV,  "
                "nT$\\tau_E$ = {:.1f}$\\times$10$^{{21}}$".format(
                    t[k], ign[k] / 1e21),
                xy=(t[k], ign[k]), xytext=(0.40, 8.0e19),
                ha="left", va="bottom", fontsize=6.3, color=RED,
                linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                alpha=0.75))
    ax.text(7.0, 4.2e22, "IGNITION\nQ $\\rightarrow \\infty$", ha="left",
            va="top", fontsize=7.4, color=RED, fontweight="bold",
            linespacing=1.25, zorder=9)

    cmap = get_cmap("cb_sequential_blue")
    yrs = np.array([r["year"] for r in tab], dtype=float)
    y0, y1 = yrs.min(), yrs.max()
    for r in tab:
        c = cmap(0.28 + 0.68 * (r["year"] - y0) / (y1 - y0))
        ax.plot([r["T"]], [r["ntt"]], marker="o", ms=6.2, mfc=c,
                mec="white", mew=0.8, zorder=10)
        ax.annotate(r["name"], xy=(r["T"], r["ntt"]),
                    xytext=(r["dx"], r["dy"]), textcoords="offset points",
                    ha=r["ha"], va=r["va"], fontsize=6.0, color=INK,
                    zorder=11)

    for name, tt, qq, ntt, off, ha, va in tgt:
        ax.plot([tt], [ntt], marker="*", ms=13.0, mfc="none", mec=GREEN,
                mew=1.4, zorder=11)
        ax.annotate("{:s}\ntarget Q = {:.0f}".format(name, qq), xy=(tt, ntt),
                    xytext=off, textcoords="offset points", ha=ha, va=va,
                    fontsize=6.3, color=GREEN, fontweight="bold",
                    linespacing=1.25, zorder=12)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.30, 200.0)
    ax.set_ylim(1e15, 6e22)
    ax.set_xlabel("ion temperature T$_i$ (keV)")
    ax.set_ylabel("triple product n$_i$T$_i\\tau_E$  "
                  "($m^{-3}\\,keV\\,s$)")
    ax.tick_params(labelsize=6.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(which="major", lw=0.5, alpha=0.45)
    ax.set_title("Fusion triple product against ion temperature",
                 fontsize=9.0, fontweight="bold", color=INK, pad=5.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(y0, y1))
    cax = ax.inset_axes([0.035, 0.918, 0.24, 0.024])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([1970, 1995, 2020])
    cb.ax.tick_params(labelsize=6, length=2.0, width=0.6)
    cb.outline.set_linewidth(0.5)
    cb.set_label("year of the shot", fontsize=6.1, color=SUB, labelpad=1.5)
    return t[k], ign[k]


def year_panel(ax, tab, tgt):
    yrs = np.array([r["year"] for r in tab], dtype=float)
    ntt = np.array([r["ntt"] for r in tab], dtype=float)
    order = np.argsort(yrs)
    yrs, ntt = yrs[order], ntt[order]
    slope, icpt = np.polyfit(yrs, np.log10(ntt), 1)
    dbl = np.log10(2.0) / slope

    xs = np.linspace(1965, 2040, 60)
    ax.plot(xs, 10.0 ** (icpt + slope * xs), color=SUB, lw=1.0,
            ls=(0, (4.5, 2.4)), zorder=3)
    cmap = get_cmap("cb_sequential_blue")
    ax.plot(yrs, ntt, color=lighten(BLUE, 0.55), lw=1.0, zorder=4)
    for y, v in zip(yrs, ntt):
        c = cmap(0.28 + 0.68 * (y - yrs.min()) / (yrs.max() - yrs.min()))
        ax.plot([y], [v], marker="o", ms=4.8, mfc=c, mec="white", mew=0.7,
                zorder=5)
    for name, _tt, _qq, v, _o, _h, _v in tgt:
        yr = 2036.0 if name == "ITER" else 2027.0
        ax.plot([yr], [v], marker="*", ms=10.0, mfc="none", mec=GREEN,
                mew=1.2, zorder=6)
        ax.text(yr, v * 1.9, name, ha="center", va="bottom", fontsize=6.0,
                color=GREEN)
    ax.text(0.035, 0.955,
            "fitted doubling time {:.1f} years".format(dbl),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=SUB)
    ax.set_yscale("log")
    ax.set_xlim(1963, 2042)
    ax.set_ylim(1e14, 3e22)
    ax.set_xlabel("year", fontsize=7.4)
    ax.set_ylabel("nT$\\tau_E$ ($m^{-3}\\,keV\\,s$)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=0.5, alpha=0.45)
    ax.set_title("A Moore's law for the triple product", fontsize=8.2,
                 fontweight="bold", color=INK)
    return dbl


def gain_panel(ax, tab, tgt):
    rows = sorted(tab, key=lambda r: r["Q"])[-8:]
    names = [r["name"] for r in rows] + [t[0] for t in tgt]
    vals = [r["Q"] for r in rows] + [t[2] for t in tgt]
    cols = [BLUE] * len(rows) + [GREEN] * len(tgt)
    pos = np.arange(len(names), dtype=float)
    ax.barh(pos, vals, height=0.68, color=cols, edgecolor="white", lw=0.6,
            zorder=4)
    for p, v in zip(pos, vals):
        ax.text(v * 1.22, p, "{:.2g}".format(v), ha="left", va="center",
                fontsize=6.0, color=SUB, zorder=6)
    ax.axvline(1.0, color=ORANGE, lw=1.0, ls=(0, (4.0, 2.2)), zorder=5)
    ax.text(1.18, -0.62, "scientific\nbreakeven Q = 1", ha="left",
            va="bottom", fontsize=6.1, color=ORANGE, linespacing=1.25)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 6e1)
    ax.set_ylim(-0.75, len(names) - 0.15)
    ax.set_yticks(pos)
    ax.set_yticklabels(names, fontsize=6.1)
    ax.set_xlabel("energy gain Q (-)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=6.6)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.5, alpha=0.45)
    ax.set_title("Gain achieved, and gain promised", fontsize=8.2,
                 fontweight="bold", color=INK)


def build():
    off = {"T-3": (7, -2, "left", "top"),
           "Alcator A": (7, -1, "left", "top"),
           "PLT": (-7, 1, "right", "bottom"),
           "Alcator C": (7, 2, "left", "bottom"),
           "TFTR": (-7, 0, "right", "center"),
           "JET": (-7, 2, "right", "bottom"),
           "JT-60U": (7, -6, "left", "top"),
           "TFTR (D-T)": (7, -7, "left", "top"),
           "JET (DTE1)": (-7, 3, "right", "bottom"),
           "JT-60U (eq.)": (6, 3, "left", "bottom"),
           "JET (DTE2)": (-7, 1, "right", "bottom"),
           "EAST": (7, -2, "left", "top")}
    tab = []
    for name, yr, tt, qq, _m in DEVICES:
        ntt = float(triple_for_q(np.array([tt]), qq)[0])
        dx, dy, ha, va = off[name]
        tab.append(dict(name=name, year=yr, T=tt, Q=qq, ntt=ntt, dx=dx,
                        dy=dy, ha=ha, va=va))
    tgt = [(n, tt, qq, float(triple_for_q(np.array([tt]), qq)[0]), off, ha,
            va) for n, tt, qq, off, ha, va in TARGETS]
    return tab, tgt


def main():
    set_pub_style()
    np.random.default_rng(919)
    tab, tgt = build()

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAIN)
    ax_y = axes_inches(fig, AUX1)
    ax_q = axes_inches(fig, AUX2)
    t_opt, ntt_opt = main_panel(ax_m, tab, tgt)
    dbl = year_panel(ax_y, tab, tgt)
    gain_panel(ax_q, tab, tgt)

    jet = [r for r in tab if r["name"] == "JET (DTE1)"][0]
    check = q_of(jet["T"], jet["ntt"])
    sv10 = reactivity(10.0)
    panel_labels_inches(fig, [ax_m, ax_y, ax_q], dx=-0.62, dy=0.13)
    savefig(fig, "919_fus_lawson_criterion.png")


if __name__ == "__main__":
    main()
