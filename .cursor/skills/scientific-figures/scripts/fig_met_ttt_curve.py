"""Isothermal transformation diagram with three continuous cooling paths.

The main panel is the classic double-C diagram of a low-alloy steel on a
logarithmic time axis: a pearlite nose, a bay, a bainite nose, and the
martensite start and finish isotherms.  Three Newtonian cooling paths are
laid over it, and what each one produces is *not* asserted - it is worked
out by integrating Scheil's additive rule along the path against the same
start and finish curves that are drawn, so the product name and the hardness
printed at the end of every path follow from the geometry.  The two smaller
panels sweep the same simulation over cooling rate (a Jominy-style hardness
curve) and break the three paths into their phase inventories.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig, lighten
from figkit import axes_inches, panel_labels_inches

RED = "#b23a48"        # fastest quench / high-temperature liquidus family
GOLD = "#c9a227"       # bainite
GREEN = "#2e7d5b"      # pearlite
PURPLE = "#8e6bbf"     # martensite
BLUE = "#3b5b92"       # transformation start / finish curves
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#808080"

AE1 = 727.0            # eutectoid temperature, deg C
MS, MF = 220.0, 60.0   # martensite start / finish, deg C
T_AUST = 880.0         # austenitising temperature, deg C
T_ROOM = 25.0
T_BAY = 500.0          # bay temperature separating the two product buckets

TN_P, TT_P = 620.0, 8.0    # pearlite nose: temperature (deg C), time (s)
TN_B, TT_B = 380.0, 3.0    # bainite nose
M_UP, B_LO = 2.6, 1.80     # pearlite branch exponents
C_UP, D_LO = 2.80, 1.90    # bainite branch exponents
FIN_FACT = 14.0            # finish curve sits this many times to the right

HV_MART, HV_AUST = 700.0, 250.0
K_KM = 0.022               # Koistinen-Marburger coefficient, 1/deg C
T_MIN, T_MAX = 1e-1, 1e5   # time axis span, s


def t_start(temp):
    """Transformation start time (s); the two lobes add their rates.

    Adding ``1/t`` rather than taking ``min`` is what rounds the bay: a plain
    minimum leaves a kink at the crossover that no measured diagram has.
    """
    t = np.asarray(temp, dtype=float)
    gap = np.maximum(AE1 - t, 1e-6)
    pear = np.where(t >= TN_P,
                    TT_P * ((AE1 - TN_P) / gap) ** M_UP,
                    TT_P * np.exp(B_LO * (TN_P - t) / 100.0))
    bain = np.where(t >= TN_B,
                    TT_B * np.exp(C_UP * (t - TN_B) / 100.0),
                    TT_B * np.exp(D_LO * (TN_B - t) / 100.0))
    both = pear * bain / (pear + bain)
    return np.where(t >= AE1, np.inf, both)


def hv_of(temp):
    """Hardness of the diffusional product formed at ``temp``."""
    return float(np.clip(HV_MART - 1.05 * (temp - MS), 150.0, HV_MART))


def cooling_path(tau, n=12000):
    t = np.geomspace(1e-3, 1e6, n)
    return t, T_ROOM + (T_AUST - T_ROOM) * np.exp(-t / tau)


def simulate(tau, n=12000):
    """Scheil additive rule down one cooling path; returns the inventory."""
    t, temp = cooling_path(tau, n)
    d_t = np.diff(t, prepend=0.0)
    ts = t_start(temp)
    tf = ts * FIN_FACT

    s_nuc = 0.0
    f_dif = 0.0
    f_p = 0.0
    f_b = 0.0
    hv_sum = 0.0
    t_nuc = np.nan
    for i in range(n):
        ti = temp[i]
        if ti <= MS:
            break
        if ti >= AE1 or not np.isfinite(ts[i]):
            continue
        if s_nuc < 1.0:
            s_nuc += d_t[i] / ts[i]
            if s_nuc >= 1.0:
                t_nuc = ti
            continue
        if f_dif < 1.0:
            d_f = min(d_t[i] / max(tf[i] - ts[i], 1e-9), 1.0 - f_dif)
            f_dif += d_f
            if ti >= T_BAY:
                f_p += d_f
            else:
                f_b += d_f
            hv_sum += d_f * hv_of(ti)

    f_m = (1.0 - f_dif) * (1.0 - np.exp(-K_KM * (MS - T_ROOM)))
    f_ret = max(1.0 - f_dif - f_m, 0.0)
    hv = hv_sum + f_m * HV_MART + f_ret * HV_AUST
    return dict(pearlite=f_p, bainite=f_b, martensite=f_m, retained=f_ret,
                hv=hv, t_nuc=t_nuc, rate=(700.0 - T_ROOM) / tau)


def product_name(res):
    parts = [("pearlite", res["pearlite"]), ("bainite", res["bainite"]),
             ("martensite", res["martensite"])]
    return max(parts, key=lambda p: p[1])[0]


def main_panel(ax, paths):
    grid = np.linspace(MS, AE1 - 1.0, 900)
    ts = t_start(grid)
    tf = ts * FIN_FACT
    ok = np.isfinite(ts) & (ts < T_MAX)

    ax.axhspan(AE1, 900.0, color=lighten(RED, 0.86), zorder=0)
    ax.axhspan(T_MIN * 0.0, MS, color=lighten(PURPLE, 0.86), zorder=0)
    ax.fill_betweenx(grid[ok], T_MIN, ts[ok], color=lighten(RED, 0.93),
                     zorder=0)
    ax.fill_betweenx(grid[ok], ts[ok], np.minimum(tf[ok], T_MAX),
                     color=lighten(BLUE, 0.84), zorder=1)
    hot = ok & (grid >= T_BAY)
    cold = ok & (grid < T_BAY)
    ax.fill_betweenx(grid[hot], np.minimum(tf[hot], T_MAX), T_MAX,
                     color=lighten(GREEN, 0.86), zorder=0)
    ax.fill_betweenx(grid[cold], np.minimum(tf[cold], T_MAX), T_MAX,
                     color=lighten(GOLD, 0.86), zorder=0)

    ax.plot(ts[ok], grid[ok], color=BLUE, lw=1.7, zorder=5)
    ax.plot(tf[ok], grid[ok], color=BLUE, lw=1.3, ls=(0, (5, 2.2)), zorder=5)
    ax.axhline(AE1, color=GREY, lw=1.1, ls=(0, (6, 2, 1, 2)), zorder=4)
    ax.axhline(MS, color=PURPLE, lw=1.3, ls=(0, (4, 2.2)), zorder=4)
    ax.axhline(MF, color=PURPLE, lw=1.0, ls=(0, (2, 2.0)), zorder=4)

    ax.text(T_MAX * 0.55, AE1 + 12.0, r"$A_{e1}$ = " + f"{AE1:.0f}" +
            r" $^\circ$C", ha="right", va="bottom", fontsize=6.6, color=GREY)
    ax.text(T_MAX * 0.55, MS + 10.0, r"$M_s$ = " + f"{MS:.0f}" +
            r" $^\circ$C", ha="right", va="bottom", fontsize=6.6,
            color=PURPLE)
    ax.text(T_MAX * 0.55, MF + 10.0, r"$M_f$ = " + f"{MF:.0f}" +
            r" $^\circ$C", ha="right", va="bottom", fontsize=6.6,
            color=PURPLE)

    nose_p = float(t_start(TN_P))
    nose_b = float(t_start(TN_B))
    ax.plot([nose_p, nose_b], [TN_P, TN_B], marker="o", ms=4.2, ls="none",
            color=BLUE, mfc="white", mew=1.0, zorder=7)
    ax.text(nose_p * 1.7, TN_P + 8.0, f"pearlite nose {nose_p:.0f} s",
            ha="left", va="bottom", fontsize=6.4, color=BLUE, zorder=7)
    ax.text(nose_b * 1.8, TN_B + 7.0, f"bainite nose {nose_b:.0f} s",
            ha="left", va="bottom", fontsize=6.4, color=BLUE, zorder=7)
    t_bay = float(t_start(T_BAY))
    ax.text(t_bay * 1.5, T_BAY - 6.0, f"bay {t_bay:.0f} s", ha="left",
            va="top", fontsize=6.4, color=BLUE, zorder=7)

    t_ref = float(t_start(706.0))
    ax.text(t_ref * 0.55, 708.0, "start (1 %)", ha="right", va="bottom",
            fontsize=6.6, color=BLUE, zorder=7)
    ax.text(t_ref * FIN_FACT * 2.6, 700.0, "finish (99 %)", ha="left",
            va="top", fontsize=6.6, color=BLUE, zorder=7)

    ax.text(3.5e4, 655.0, "pearlite", fontsize=8.6, color="#1f5a44",
            ha="center", va="center", zorder=6)
    ax.text(3.5e4, 330.0, "bainite", fontsize=8.6, color="#8a6a12",
            ha="center", va="center", zorder=6)
    ax.text(2.4e4, 125.0, "martensite", fontsize=8.6, color="#5f4a86",
            ha="center", va="center", zorder=6)
    ax.text(0.14, 520.0, "austenite", fontsize=7.6, color="#8c2f2f",
            ha="left", va="center", zorder=6)

    for tau, color, tag, res in paths:
        t, temp = cooling_path(tau)
        keep = (t >= T_MIN) & (temp >= 40.0)
        ax.plot(t[keep], temp[keep], color=color, lw=1.9, zorder=8)
        if np.isfinite(res["t_nuc"]):
            ax.plot([t[int(np.argmin(np.abs(temp - res["t_nuc"])))]],
                    [res["t_nuc"]], marker="o", ms=4.8, color=color,
                    mfc="white", mew=1.3, zorder=9)

    for row, (tau, color, tag, res) in enumerate(paths):
        ax.text(8.4e4, 878.0 - 33.0 * row,
                f"{tag}:  {res['rate']:.3g}" + r" $^\circ$C/s"
                f"  $\\rightarrow$  {product_name(res)},  "
                f"{res['hv']:.0f} HV",
                ha="right", va="center", fontsize=6.6, color=color, zorder=9)

    ax.set_xscale("log")
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(0.0, 900.0)
    ax.set_xlabel("time (s, log scale)")
    ax.set_ylabel(r"temperature ($^\circ$C)")
    ax.set_yticks(np.arange(0, 901, 100))
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.tick_params(labelsize=7.2)
    ax.set_title("Isothermal diagram with three continuous cooling paths",
                 fontsize=9.0, fontweight="bold", color=INK)


def jominy_panel(ax, sweep, paths):
    rate = np.array([r["rate"] for r in sweep])
    hv = np.array([r["hv"] for r in sweep])
    ax.plot(rate, hv, color=INK, lw=1.6, zorder=4)
    ax.fill_between(rate, 150.0, hv, color=lighten(INK, 0.90), zorder=1)
    for tau, color, tag, res in paths:
        ax.plot([res["rate"]], [res["hv"]], marker="o", ms=5.0, color=color,
                mfc="white", mew=1.4, zorder=6)
        ax.text(res["rate"], res["hv"] + 26.0, tag, ha="center", va="bottom",
                fontsize=6.3, color=color)
    half = 0.5 * (hv.min() + hv.max())
    r_crit = float(np.interp(half, hv[np.argsort(hv)],
                             rate[np.argsort(hv)]))
    ax.axvline(r_crit, color=GREY, lw=0.9, ls=(0, (3, 2.2)), zorder=3)
    ax.text(r_crit * 1.25, hv.min() + 22.0,
            f"half hardening\n{r_crit:.0f}" + r" $^\circ$C/s", ha="left",
            va="bottom", fontsize=6.3, color=SUB)

    ax.set_xscale("log")
    ax.set_xlim(rate.min(), rate.max())
    ax.set_ylim(150.0, 760.0)
    ax.set_xlabel(r"cooling rate at 700 $^\circ$C ($^\circ$C/s, log scale)",
                  fontsize=7.4)
    ax.set_ylabel("hardness (HV)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Hardenability sweep over the same model", fontsize=8.2,
                 fontweight="bold", color=INK)


def inventory_panel(ax, paths):
    tags = [p[2] for p in paths]
    keys = ["pearlite", "bainite", "martensite", "retained"]
    cols = [lighten(GREEN, 0.32), lighten(GOLD, 0.28),
            lighten(PURPLE, 0.34), lighten(GREY, 0.55)]
    names = ["pearlite", "bainite", "martensite", "retained $\\gamma$"]

    y = np.arange(len(tags))[::-1]
    left = np.zeros(len(tags))
    for key, col, name in zip(keys, cols, names):
        val = np.array([p[3][key] for p in paths]) * 100.0
        ax.barh(y, val, left=left, height=0.56, color=col,
                edgecolor="white", linewidth=0.7, label=name, zorder=3)
        for yi, (v, l0) in enumerate(zip(val, left)):
            if v >= 11.0:
                ax.text(l0 + v / 2.0, y[yi], f"{v:.0f}", ha="center",
                        va="center", fontsize=6.2, color=INK, zorder=5)
        left = left + val

    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}\n{p[3]['hv']:.0f} HV"
                        for t, p in zip(tags, paths)], fontsize=6.5)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("volume fraction (%)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="x", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.2, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=4, handlelength=1.0,
              columnspacing=0.9, handletextpad=0.4)
    ax.set_title("Product inventory of the three paths", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    paths = [(2.0, RED, "fast", None),
             (130.0, GOLD, "medium", None),
             (3000.0, GREEN, "slow", None)]
    paths = [(tau, col, tag, simulate(tau)) for tau, col, tag, _ in paths]
    taus = np.geomspace(0.5, 2.0e4, 34)
    sweep = [simulate(t, n=9000) for t in taus]

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.80, 0.92, 5.55, 4.52))
    ax_jom = axes_inches(fig, (7.20, 3.55, 2.95, 1.89))
    ax_inv = axes_inches(fig, (7.20, 1.10, 2.95, 1.71))

    main_panel(ax_main, paths)
    jominy_panel(ax_jom, sweep, paths)
    inventory_panel(ax_inv, paths)

    hv_fast = paths[0][3]["hv"]
    hv_slow = paths[2][3]["hv"]
    panel_labels_inches(fig, [ax_main, ax_jom, ax_inv], dx=-0.46, dy=0.10)
    savefig(fig, "829_met_ttt_curve.png")


if __name__ == "__main__":
    main()
