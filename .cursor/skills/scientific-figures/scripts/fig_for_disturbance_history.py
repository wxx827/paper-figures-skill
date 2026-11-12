"""Disturbance history written into the age mosaic of a managed landscape.

One event list drives the whole figure.  Thirty-four patches - irregular fire
blobs, angular clearcuts, wind-aligned blowdowns - are stamped onto an
old-growth age matrix in chronological order, so a later event really does
erase the age signature of an earlier one and the map is a genuine "time
since last disturbance" surface rather than a painting of one.

Both side panels are read back off that same raster.  The area chart counts
the cells each patch actually covers, so its bars sum to the disturbed
fraction quoted on the map, and the recovery panel places every patch at its
own age with sample plots drawn around it, then refits the exponential
recovery rate per disturbance type from those points.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from pubstyle import set_pub_style, savefig, get_palette
from figkit import add_scale_bar, axes_inches, panel_labels_inches, safe_colorbar

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"

TYPES = ["fire", "harvest", "windthrow"]
TYPE_COLOR = {"fire": "#b23a48", "harvest": "#3b5b92", "windthrow": "#c9a227"}
TYPE_LS = {"fire": "-", "harvest": (0, (4.0, 2.0)), "windthrow": (0, (1.3, 1.6))}
# exponential recovery: NDVI(t) = inf - (inf - n0) exp(-k t)
TYPE_RECOVERY = {"fire": (0.14, 0.082), "harvest": (0.29, 0.150),
                 "windthrow": (0.36, 0.112)}
NDVI_INF = 0.88

FW, FH = 10.4, 6.4
MAP_RECT = (0.66, 1.04, 5.24, 4.14)
KM_Y = 4.6
KM_X = KM_Y * MAP_RECT[2] / MAP_RECT[3]
NX, NY = 296, 260
CELL_HA = (KM_X * 1000.0 / NX) * (KM_Y * 1000.0 / NY) / 10000.0
YEAR_NOW = 2024
YEAR_LO, YEAR_HI = 1956, 2022
BIN = 5
TYPE_COUNT = {"fire": 11, "harvest": 14, "windthrow": 9}
PLOTS_PER_PATCH = 4
# forestry age classes: a mosaic is read as classes, not as a smooth ramp
AGE_BREAKS = [0, 10, 20, 30, 45, 60, 80, 110, 140, 175]


def blob(rng, cx, cy, r, elong=1.0, angle=0.0, rough=0.24, n=110):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ph = rng.uniform(0.0, 2.0 * np.pi, 3)
    rad = r * (1.0 + rough * (0.60 * np.sin(3 * th + ph[0]) +
                              0.40 * np.sin(5 * th + ph[1]) +
                              0.28 * np.sin(7 * th + ph[2])))
    x, y = rad * np.cos(th) * elong, rad * np.sin(th) / elong
    ca, sa = np.cos(angle), np.sin(angle)
    return np.column_stack([cx + x * ca - y * sa, cy + x * sa + y * ca])


def clearcut(rng, cx, cy, r, angle):
    """Angular block with jittered corners - a cut boundary follows a plan."""
    w, h = r * rng.uniform(1.1, 1.9), r * rng.uniform(0.7, 1.2)
    base = np.array([[-w, -h], [w, -h], [w * rng.uniform(0.7, 1.0), h],
                     [-w * rng.uniform(0.7, 1.0), h]], dtype=float)
    base += rng.normal(0.0, r * 0.07, base.shape)
    ca, sa = np.cos(angle), np.sin(angle)
    rot = np.column_stack([base[:, 0] * ca - base[:, 1] * sa,
                           base[:, 0] * sa + base[:, 1] * ca])
    return rot + np.array([cx, cy])


def build_landscape(rng):
    rough = gaussian_filter(rng.normal(0.0, 1.0, (NY, NX)), 17.0)
    age = np.clip(122.0 + 46.0 * rough / np.abs(rough).max() * 3.2,
                  84.0, 172.0)

    gx = (np.arange(NX) + 0.5) * KM_X / NX
    gy = (np.arange(NY) + 0.5) * KM_Y / NY
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    # each type is spread over the whole record, otherwise the recovery fit
    # for a type with no recent events is pure extrapolation
    plan = []
    for kind, n in TYPE_COUNT.items():
        yrs = np.linspace(YEAR_LO, YEAR_HI, n) + rng.normal(0.0, 2.6, n)
        plan += [(int(np.clip(y, YEAR_LO, YEAR_HI)), kind) for y in yrs]
    plan.sort()

    events = []
    for year, kind in plan:
        cx = rng.uniform(0.35, KM_X - 0.35)
        cy = rng.uniform(0.30, KM_Y - 0.30)
        if kind == "harvest":
            r = rng.uniform(0.16, 0.30)
            poly = clearcut(rng, cx, cy, r, rng.uniform(-0.45, 0.45))
        elif kind == "fire":
            r = rng.uniform(0.22, 0.46)
            poly = blob(rng, cx, cy, r, elong=rng.uniform(0.85, 1.25),
                        angle=rng.uniform(0.0, np.pi), rough=0.30)
        else:                       # windthrow: elongated along the storm axis
            r = rng.uniform(0.14, 0.26)
            poly = blob(rng, cx, cy, r, elong=rng.uniform(2.1, 3.2),
                        angle=np.deg2rad(rng.normal(28.0, 8.0)), rough=0.20)
        inside = Path(poly).contains_points(pts).reshape(NY, NX)
        if inside.sum() < 12:
            continue
        age[inside] = float(YEAR_NOW - year)
        events.append(dict(year=int(year), kind=kind, poly=poly,
                           cells=int(inside.sum()),
                           area=int(inside.sum()) * CELL_HA))
    return age, events


def draw_map(ax, fig, age, events):
    cmap = mpl.colors.ListedColormap(
        get_palette("cb_sequential_green", len(AGE_BREAKS) - 1))
    norm = mpl.colors.BoundaryNorm(AGE_BREAKS, cmap.N)
    im = ax.imshow(age, origin="lower", extent=[0.0, KM_X, 0.0, KM_Y],
                   cmap=cmap, norm=norm, interpolation="nearest", zorder=1)
    ax.set_aspect("equal")
    for ev in events:
        poly = np.vstack([ev["poly"], ev["poly"][:1]])
        ax.plot(poly[:, 0], poly[:, 1], color=TYPE_COLOR[ev["kind"]],
                lw=0.95, ls=TYPE_LS[ev["kind"]], zorder=4)
    ax.set_xlim(0.0, KM_X)
    ax.set_ylim(0.0, KM_Y)
    add_scale_bar(ax, 1.0, "1 km", loc=(0.035, 0.045), height=0.016,
                  fontsize=6.2, color="#2b2b2b", text_color="white",
                  halo="#1a1a1a")
    ax.set_xlabel("easting (km)", fontsize=8.0)
    ax.set_ylabel("northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
        s.set_color(RULE)
    ax.set_title("Time since last disturbance, with event boundaries",
                 fontsize=8.5, fontweight="bold")
    safe_colorbar(fig, im, ax, label="stand age (years)", width=0.12,
                  pad=0.09, ticks=AGE_BREAKS)
    return im


def draw_area(ax, events):
    edges = np.arange(YEAR_LO - (YEAR_LO % BIN), YEAR_HI + BIN + 1, BIN)
    mid = edges[:-1] + BIN / 2.0
    stack = {k: np.zeros(mid.size) for k in TYPES}
    for ev in events:
        j = int(np.clip(np.searchsorted(edges, ev["year"], "right") - 1,
                        0, mid.size - 1))
        stack[ev["kind"]][j] += ev["area"]
    bottom = np.zeros(mid.size)
    for k in TYPES:
        ax.bar(mid, stack[k], width=BIN * 0.86, bottom=bottom,
               color=TYPE_COLOR[k], edgecolor="white", linewidth=0.5,
               zorder=3, label=k)
        bottom += stack[k]
    ax.set_xlabel("disturbance year (-)", fontsize=7.8)
    ax.set_ylabel("area affected (ha)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_xlim(edges[0] - 1, edges[-1] + 1)
    ax.set_title("Disturbed area by five-year period", fontsize=8.5,
                 fontweight="bold")
    return stack, mid, bottom


def recovery(t, k, n0):
    return NDVI_INF - (NDVI_INF - n0) * np.exp(-k * t)


def draw_recovery(ax, rng, events):
    fits = {}
    for k in TYPES:
        tt, vv = [], []
        for ev in (e for e in events if e["kind"] == k):
            age = YEAR_NOW - ev["year"]
            n0, rate = TYPE_RECOVERY[k]
            mu = recovery(age, rate, n0)
            tt += [age] * PLOTS_PER_PATCH
            vv += list(mu + rng.normal(0.0, 0.042, PLOTS_PER_PATCH))
        tt, vv = np.array(tt, float), np.clip(np.array(vv), 0.03, 0.96)
        par, _ = curve_fit(recovery, tt, vv, p0=[0.10, 0.25],
                           bounds=([0.008, 0.02], [0.60, 0.55]),
                           maxfev=30000)
        fits[k] = par
        ax.scatter(tt, vv, s=6.0, color=TYPE_COLOR[k], alpha=0.65,
                   edgecolors="white", linewidths=0.25, zorder=4)
        gx = np.linspace(0.0, YEAR_NOW - YEAR_LO, 200)
        gy = recovery(gx, *par)
        sd = float(np.std(vv - recovery(tt, *par)))
        ax.fill_between(gx, gy - sd, gy + sd, color=TYPE_COLOR[k], alpha=0.16,
                        linewidth=0.0, zorder=3)
        ax.plot(gx, gy, color=TYPE_COLOR[k], lw=1.3, zorder=5)
    ax.axhline(NDVI_INF, color="#808080", lw=0.8, ls=(0, (4, 2.4)), zorder=2)
    ax.text(1.0, NDVI_INF + 0.015, "mature-stand NDVI", ha="left",
            va="bottom", fontsize=6.2, color=MUTED)
    ax.set_xlim(0.0, YEAR_NOW - YEAR_LO)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("years since disturbance (years)", fontsize=7.8)
    ax.set_ylabel("NDVI (-)", fontsize=7.8)
    ax.tick_params(labelsize=6.8)
    ax.set_title("Post-disturbance recovery trajectories", fontsize=8.5,
                 fontweight="bold")
    # time to close 80 % of the gap to the mature-stand value
    txt = "\n".join(f"{k}  $t_{{80}}$ = {np.log(5.0) / fits[k][0]:.0f} yr"
                    for k in TYPES)
    ax.text(0.975, 0.055, txt, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.3, color=SUB, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.92))
    return fits


def main():
    set_pub_style()
    rng = np.random.default_rng(687)
    age, events = build_landscape(rng)

    fig = plt.figure(figsize=(FW, FH))
    ax_m = axes_inches(fig, MAP_RECT)
    ax_a = axes_inches(fig, (7.24, 3.28, 2.80, 1.90))
    ax_r = axes_inches(fig, (7.24, 1.04, 2.80, 1.58))

    draw_map(ax_m, fig, age, events)
    stack, mid, total = draw_area(ax_a, events)
    fits = draw_recovery(ax_r, rng, events)

    handles = [Line2D([0], [0], color=TYPE_COLOR[k], lw=1.4, ls=TYPE_LS[k],
                      label=k) for k in TYPES]

    panel_labels_inches(fig, [ax_m, ax_a, ax_r], dx=-0.36, dy=0.20)

    land_ha = KM_X * KM_Y * 100.0
    dist_ha = sum(e["area"] for e in events)
    young = float(np.mean(age <= 30.0))
    busiest = int(mid[int(np.argmax(total))])

    print(f"events={len(events)} dist_ha={dist_ha:.1f} land={land_ha:.0f} "
          f"young={young:.3f} busiest={busiest} "
          f"k={{'{TYPES[0]}': {fits['fire'][0]:.4f}, "
          f"'harvest': {fits['harvest'][0]:.4f}, "
          f"'windthrow': {fits['windthrow'][0]:.4f}}}")
    savefig(fig, "687_for_disturbance_history.png")


if __name__ == "__main__":
    main()
