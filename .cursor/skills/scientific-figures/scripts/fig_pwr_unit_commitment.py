"""Unit commitment solved hour by hour, then read back three different ways.

For each of the 24 hours the residual after must-take renewables and hydro is
covered by committing the cheapest prefix of the thermal fleet that can carry
both that residual and the spinning-reserve requirement.  Every committed unit
is first loaded to its technical minimum and the remainder is handed down the
merit order, so the stack adds up to the served demand exactly - the assertion
in ``main`` is what makes the picture trustworthy rather than decorative.  The
commitment matrix drives the status bars, and the reserve panel is the gap
between the committed capacity and what the fleet is actually producing.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
ORANGE = "#c0603f"
PURPLE = "#7b5ea7"

COL = {
    "solar": "#6bbf8a",
    "wind": "#2a7f7a",
    "hydro": "#3b5b92",
    "nuclear": "#4a4f5c",
    "coal": "#7a6a5d",
    "CCGT": "#a89078",
    "OCGT": "#c4a98d",
    "battery": PURPLE,
}

# name, fuel, Pmax (MW), Pmin as a fraction of Pmax, marginal cost USD/MWh
THERMAL = [
    ("NUC-1", "nuclear", 800.0, 0.75, 10.0),
    ("NUC-2", "nuclear", 800.0, 0.75, 10.6),
    ("COAL-A", "coal", 450.0, 0.45, 30.0),
    ("COAL-B", "coal", 450.0, 0.45, 33.0),
    ("COAL-C", "coal", 450.0, 0.45, 36.0),
    ("CCGT-1", "CCGT", 300.0, 0.35, 46.0),
    ("CCGT-2", "CCGT", 300.0, 0.35, 50.0),
    ("CCGT-3", "CCGT", 300.0, 0.35, 56.0),
    ("OCGT-1", "OCGT", 150.0, 0.20, 95.0),
    ("OCGT-2", "OCGT", 150.0, 0.20, 105.0),
]

RES_FRAC = 0.03          # reserve carried against forecast error
ETA = 0.88               # battery round-trip efficiency
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def profiles(rng):
    """Hourly load, must-take renewables and the battery schedule, in MW."""
    h = np.arange(24) + 0.5
    load = (4200.0
            + 700.0 * np.exp(-0.5 * ((h - 9.0) / 2.3) ** 2)
            + 1150.0 * np.exp(-0.5 * ((h - 19.3) / 2.0) ** 2)
            - 750.0 * np.exp(-0.5 * ((h - 4.0) / 3.0) ** 2)
            + rng.normal(0.0, 24.0, 24))
    solar = 1500.0 * np.clip(np.sin(np.pi * (h - 6.2) / 12.4), 0.0, None) ** 1.4
    wind = 900.0 + 260.0 * np.sin(2 * np.pi * (h - 2.0) / 19.0) \
        + rng.normal(0.0, 55.0, 24)
    wind = np.clip(wind, 380.0, None)
    hydro = np.full(24, 500.0)

    charge = np.zeros(24)
    charge[[1, 2, 3, 11, 12, 13]] = [90.0, 110.0, 100.0, 150.0, 165.0, 150.0]
    shape = np.zeros(24)
    shape[[17, 18, 19, 20]] = [0.7, 1.0, 1.0, 0.6]
    discharge = shape * (ETA * charge.sum() / shape.sum())
    return load, solar, wind, hydro, charge, discharge


def commit(residual, load):
    """Cheapest feasible prefix per hour, then merit-order loading."""
    pmax = np.array([u[2] for u in THERMAL])
    pmin = np.array([u[2] * u[3] for u in THERMAL])
    cmax, cmin = np.cumsum(pmax), np.cumsum(pmin)
    biggest = pmax.max()

    gen = np.zeros((len(THERMAL), 24))
    on = np.zeros((len(THERMAL), 24), dtype=bool)
    req = RES_FRAC * load + biggest
    curtail = np.zeros(24)
    head = np.zeros(24)

    for t in range(24):
        need = residual[t]
        want = need + req[t]
        k = int(np.searchsorted(cmax, want) + 1)
        k = min(max(k, int(np.searchsorted(cmax, need) + 1)), len(THERMAL))
        if cmin[k - 1] > need:                 # minimum generation binds
            curtail[t] = cmin[k - 1] - need
            need = cmin[k - 1]
        on[:k, t] = True
        gen[:k, t] = pmin[:k]
        spare = need - pmin[:k].sum()
        for i in range(k):
            step = min(spare, pmax[i] - pmin[i])
            gen[i, t] += step
            spare -= step
        head[t] = cmax[k - 1] - gen[:k, t].sum()
    return gen, on, req, head, curtail


def stepify(y):
    """Duplicate each sample so a 24-value series draws as 24 flat hours."""
    return np.repeat(np.asarray(y, dtype=float), 2)


def main_panel(ax, bands, load, served, charge, curtail):
    x = np.repeat(np.arange(25), 2)[1:-1]
    base = np.zeros(48)
    for name, col, series in bands:
        top = base + stepify(series)
        ax.fill_between(x, base, top, facecolor=col, edgecolor="white",
                        linewidth=0.45, zorder=3, label=name)
        base = top

    ax.fill_between(x, 0.0, -stepify(charge), facecolor=PURPLE, alpha=0.42,
                    edgecolor="white", linewidth=0.45, zorder=3)
    ax.plot(x, stepify(served), color=INK, lw=1.0, ls=(0, (3.5, 2.0)),
            zorder=7)
    ax.plot(x, stepify(load), color=ORANGE, lw=2.0, zorder=8)
    ax.axhline(0.0, color="#999999", lw=0.7, zorder=6)

    pk = int(np.argmax(load))
    ax.annotate("system peak {:.0f} MW at {:02d}:00".format(load[pk], pk),
                xy=(pk + 0.5, load[pk]), xytext=(23.7, 6180.0),
                fontsize=6.8, color=ORANGE, fontweight="bold", ha="right",
                va="top", zorder=10,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8,
                                shrinkA=2, shrinkB=3))
    ax.text(12.5, -430.0, "battery charging {:.0f} MWh, added to the demand "
            "the stack must serve".format(charge.sum()), ha="center",
            va="center", fontsize=6.4, color=PURPLE, zorder=9)
    handles = [Patch(facecolor=c, edgecolor="white", lw=0.5, label=n)
               for n, c, _s in bands]
    handles += [
        Patch(facecolor=PURPLE, alpha=0.42, edgecolor="white", lw=0.5,
              label="battery charging"),
        Line2D([], [], color=ORANGE, lw=2.0, label="system load"),
        Line2D([], [], color=INK, lw=1.0, ls=(0, (3.5, 2.0)),
               label="load plus charging (served)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.2, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.4, ncol=3, columnspacing=0.9,
              borderaxespad=0.3, labelspacing=0.26)

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(-700.0, 6300.0)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("hour of day (hours)")
    ax.set_ylabel("power (MW)")
    ax.grid(axis="y", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Hourly dispatch stack, balanced to the served demand",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def status_panel(ax, on, starts):
    n = len(THERMAL)
    for i, (name, fuel, _pm, _pf, _mc) in enumerate(THERMAL):
        y = n - 1 - i
        ax.add_patch(plt.Rectangle((0.0, y - 0.34), 24.0, 0.68,
                                   facecolor="#f2f2f2", edgecolor="none",
                                   zorder=1))
        t = 0
        while t < 24:
            if on[i, t]:
                t1 = t
                while t1 < 24 and on[i, t1]:
                    t1 += 1
                ax.add_patch(plt.Rectangle((t, y - 0.34), t1 - t, 0.68,
                                           facecolor=COL[fuel],
                                           edgecolor="white", linewidth=0.5,
                                           zorder=3))
                if t > 0:
                    ax.plot([t], [y + 0.55], marker="v", ms=3.2, mfc=RED,
                            mec="white", mew=0.5, clip_on=False, zorder=5)
                t = t1
            else:
                t += 1
    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([u[0] for u in THERMAL], fontsize=6.0)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xlabel("hour of day (hours)", fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Commitment status, {:d} starts".format(starts),
                 fontsize=8.0, fontweight="bold", color=INK)
    ax.plot([], [], marker="v", ms=3.2, mfc=RED, mec="white", mew=0.5, lw=0,
            label="start-up")
    ax.legend(loc="lower left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.9,
              handlelength=1.0, borderaxespad=0.25)


def reserve_panel(ax, req, head):
    x = np.repeat(np.arange(25), 2)[1:-1]
    hs, rs = stepify(head), stepify(req)
    ax.fill_between(x, 0.0, hs, facecolor="#cfe0d8", edgecolor="#2e7d5b",
                    linewidth=0.9, zorder=3, label="headroom on committed "
                                                   "units")
    ax.fill_between(x, hs, rs, where=rs > hs, facecolor="none", hatch="////",
                    edgecolor=RED, linewidth=0.0, zorder=4)
    ax.plot(x, rs, color=RED, lw=1.5, zorder=6, label="reserve requirement")

    short = req - head
    if short.max() > 0.0:
        t = int(np.argmax(short))
        ax.annotate("{:.0f} MW short at {:02d}:00".format(short[t], t),
                    xy=(t + 0.5, 0.5 * (head[t] + req[t])),
                    xytext=(0.8, req.max() * 1.42), fontsize=6.2,
                    color=RED, ha="left", va="top", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.7,
                                    shrinkA=2, shrinkB=3))
    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(0.0, max(head.max(), req.max()) * 1.30)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.set_xlabel("hour of day (hours)", fontsize=6.8)
    ax.set_ylabel("spinning reserve (MW)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.92, handlelength=1.2,
              borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Reserve carried against the largest unit", fontsize=8.0,
                 fontweight="bold", color=INK)
    return short


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
    rng = np.random.default_rng(20240617)
    load, solar, wind, hydro, charge, discharge = profiles(rng)

    served = load + charge
    residual = served - solar - wind - hydro - discharge
    gen, on, req, head, curtail = commit(residual, load)
    solar_used = solar - curtail                     # min-generation spillage

    by_fuel = {}
    for (name, fuel, _pm, _pf, _mc), g in zip(THERMAL, gen):
        by_fuel[fuel] = by_fuel.get(fuel, np.zeros(24)) + g

    bands = [("solar", COL["solar"], solar_used),
             ("wind", COL["wind"], wind),
             ("hydro", COL["hydro"], hydro),
             ("nuclear", COL["nuclear"], by_fuel["nuclear"]),
             ("coal", COL["coal"], by_fuel["coal"]),
             ("CCGT", COL["CCGT"], by_fuel["CCGT"]),
             ("OCGT", COL["OCGT"], by_fuel.get("OCGT", np.zeros(24))),
             ("battery discharge", COL["battery"], discharge)]

    total = sum(b[2] for b in bands)
    assert np.allclose(total, served, atol=1e-6), "stack must equal demand"
    assert np.all(gen >= -1e-9) and np.all(gen <= np.array(
        [u[2] for u in THERMAL])[:, None] + 1e-9), "unit limits violated"
    assert abs(discharge.sum() - ETA * charge.sum()) < 1e-6, "battery energy"

    starts = int(np.sum(on[:, 1:] & ~on[:, :-1]))
    thermal_energy = gen.sum()
    vre_share = 100.0 * (solar_used.sum() + wind.sum()) / served.sum()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_stat = axes_inches(fig, AUX_T)
    ax_res = axes_inches(fig, AUX_B)

    main_panel(ax_main, bands, load, served, charge, curtail)
    status_panel(ax_stat, on, starts)
    short = reserve_panel(ax_res, req, head)



    panel_labels_inches(fig, [ax_main, ax_stat, ax_res], dx=-0.62, dy=0.14)
    savefig(fig, "967_pwr_unit_commitment.png")


if __name__ == "__main__":
    main()
