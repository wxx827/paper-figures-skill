"""Figure 616 - Focal mechanisms across a three-fault system.

The beach balls are not drawn from a library of glyph shapes.  Each one is the
P-wave radiation pattern of a moment tensor built from that event's strike,
dip and rake with the Aki and Richards formulation, evaluated on a lower
hemisphere and projected with the equal-area (Schmidt) rule; the black lobes
are simply where the radiation is compressional.  Strike-slip, thrust and
normal events therefore differ in shape because the physics differs, not
because three glyphs were pasted in.

The strike of every mechanism is measured from the fault trace it belongs to,
so the map and the mechanisms cannot drift apart.  Panel (b) plots the
eigenvectors of the same tensors and panel (c) reduces them to one axial mean,
the maximum horizontal compression direction quoted on both panels.

Black and white beach balls are a hard domain convention and are kept.
"""
import textwrap

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.stats import gaussian_kde

from pubstyle import set_pub_style, savefig, get_palette, COLORS
from figkit import add_scale_bar, halo_text, panel_label_inches

X1, Y1 = 110.0, 100.0                 # map extent, km
N_EVENTS = 42                         # events per fault system
N_BALLS = 9                           # mechanisms drawn on the map
GRID = 141                            # beach-ball raster, samples per side

P_COL = "#b23a48"                     # compression axis, house red
T_COL = "#3b5b92"                     # tension axis, house blue
FAULT = "#3a3a3a"

# (name, trace vertices in km, dip, rake, colour of the label)
FAULTS = [
    dict(name="F1 dextral strike-slip",
         trace=[(20.0, 4.0), (26.0, 34.0), (32.0, 62.0), (41.0, 96.0)],
         dip=86.0, rake=176.0, kind="strike-slip"),
    dict(name="F2 thrust front",
         trace=[(5.0, 66.0), (35.0, 71.0), (68.0, 74.0), (105.0, 82.0)],
         dip=34.0, rake=92.0, kind="thrust"),
    dict(name="F3 normal fault",
         trace=[(72.0, 8.0), (82.0, 26.0), (93.0, 52.0)],
         dip=54.0, rake=-88.0, kind="normal"),
]


def ball_radius(mw):
    """Beach-ball radius in map kilometres."""
    return 2.6 + 1.15 * (float(mw) - 4.0)


def trace_strike(trace):
    """Mean strike of a polyline, in degrees clockwise from north."""
    p = np.asarray(trace, dtype=float)
    d = p[-1] - p[0]
    return float(np.degrees(np.arctan2(d[0], d[1])) % 360.0)


def moment_tensor(strike, dip, rake):
    """Double-couple moment tensor in (north, east, down), Aki and Richards."""
    s, d, l = np.deg2rad([strike, dip, rake])
    m = np.empty((3, 3))
    m[0, 0] = -(np.sin(d) * np.cos(l) * np.sin(2 * s)
                + np.sin(2 * d) * np.sin(l) * np.sin(s) ** 2)
    m[0, 1] = (np.sin(d) * np.cos(l) * np.cos(2 * s)
               + 0.5 * np.sin(2 * d) * np.sin(l) * np.sin(2 * s))
    m[0, 2] = -(np.cos(d) * np.cos(l) * np.cos(s)
                + np.cos(2 * d) * np.sin(l) * np.sin(s))
    m[1, 1] = (np.sin(d) * np.cos(l) * np.sin(2 * s)
               - np.sin(2 * d) * np.sin(l) * np.cos(s) ** 2)
    m[1, 2] = -(np.cos(d) * np.cos(l) * np.sin(s)
                - np.cos(2 * d) * np.sin(l) * np.cos(s))
    m[2, 2] = np.sin(2 * d) * np.sin(l)
    m[1, 0], m[2, 0], m[2, 1] = m[0, 1], m[0, 2], m[1, 2]
    return m


def pt_axes(m):
    """Trend and plunge of the P and T axes of a moment tensor, in degrees."""
    w, v = np.linalg.eigh(m)
    out = []
    for vec in (v[:, 0], v[:, -1]):            # most negative, most positive
        if vec[2] < 0.0:
            vec = -vec
        out.append((float(np.degrees(np.arctan2(vec[1], vec[0])) % 360.0),
                    float(np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0))))))
    return out[0], out[1]


def eq_area(trend, plunge):
    """Lower-hemisphere equal-area projection; radius 1 at zero plunge."""
    i = np.deg2rad(90.0 - np.asarray(plunge, dtype=float))
    r = np.sqrt(2.0) * np.sin(i / 2.0)
    t = np.deg2rad(np.asarray(trend, dtype=float))
    return r * np.sin(t), r * np.cos(t)


def unit_disk():
    """Ray directions for every sample of the projected lower hemisphere."""
    g = np.linspace(-1.0, 1.0, GRID)
    gx, gy = np.meshgrid(g, g)
    r = np.hypot(gx, gy)
    inside = r <= 1.0
    plunge = np.pi / 2.0 - 2.0 * np.arcsin(np.clip(r / np.sqrt(2.0), 0, 1))
    trend = np.arctan2(gx, gy)
    rays = np.stack([np.cos(plunge) * np.cos(trend),
                     np.cos(plunge) * np.sin(trend),
                     np.sin(plunge)])
    return gx, gy, inside, rays


def beach_ball(ax, m, x0, y0, radius, gx, gy, inside, rays, lw=0.7,
               zorder=8):
    """Draw one mechanism; black lobes are the compressional quadrants."""
    amp = np.einsum("ab,axy,bxy->xy", m, rays, rays)
    amp = np.where(inside, amp, np.nan)
    ax.add_patch(Circle((x0, y0), radius, facecolor="white",
                        edgecolor="none", zorder=zorder))
    if np.nanmax(amp) > 0.0:
        ax.contourf(x0 + radius * gx, y0 + radius * gy, amp,
                    levels=[0.0, float(np.nanmax(amp)) + 1e-9],
                    colors=["#1a1a1a"], zorder=zorder + 1)
    ax.add_patch(Circle((x0, y0), radius, facecolor="none",
                        edgecolor="#1a1a1a", linewidth=lw,
                        zorder=zorder + 2))


def sample_events(rng, fault, strike):
    """Epicentres, magnitudes and mechanism angles along one fault."""
    p = np.asarray(fault["trace"], dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = rng.uniform(0.0, s[-1], N_EVENTS)
    x = np.interp(u, s, p[:, 0]) + rng.normal(0.0, 3.4, N_EVENTS)
    y = np.interp(u, s, p[:, 1]) + rng.normal(0.0, 3.4, N_EVENTS)
    mw = 3.6 + rng.exponential(0.62, N_EVENTS)          # Gutenberg-Richter
    return dict(x=x, y=y, mw=np.clip(mw, 3.6, 7.0),
                strike=strike + rng.normal(0.0, 7.0, N_EVENTS),
                dip=fault["dip"] + rng.normal(0.0, 5.0, N_EVENTS),
                rake=fault["rake"] + rng.normal(0.0, 9.0, N_EVENTS))


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(6160)

    cat = []
    for f in FAULTS:
        st = trace_strike(f["trace"])
        f["strike"] = st
        ev = sample_events(rng, f, st)
        for k in range(N_EVENTS):
            m = moment_tensor(ev["strike"][k], ev["dip"][k], ev["rake"][k])
            (pt, pp), (tt, tp) = pt_axes(m)
            cat.append(dict(x=ev["x"][k], y=ev["y"][k], mw=ev["mw"][k], m=m,
                            kind=f["kind"], p_tr=pt, p_pl=pp, t_tr=tt,
                            t_pl=tp))
    mw = np.array([e["mw"] for e in cat])
    p_tr = np.array([e["p_tr"] for e in cat])
    p_pl = np.array([e["p_pl"] for e in cat])
    t_tr = np.array([e["t_tr"] for e in cat])
    t_pl = np.array([e["t_pl"] for e in cat])

    # Axial mean of the P-axis trends: doubling the angles first is what makes
    # a 350 deg and a 10 deg axis average to north rather than to south.
    fold = p_tr % 180.0
    th = np.deg2rad(2.0 * fold)
    shmax = 0.5 * np.degrees(np.arctan2(np.sin(th).mean(),
                                        np.cos(th).mean())) % 180.0
    msum = sum(e["m"] / np.linalg.norm(e["m"]) for e in cat)
    ev_sum = np.linalg.eigvalsh(msum)
    ratio = float((ev_sum[1] - ev_sum[0]) / (ev_sum[2] - ev_sum[0]))

    # The nine largest events that do not overlap once drawn.
    order = np.argsort(-mw)
    chosen = []
    for j in order:
        r = ball_radius(cat[j]["mw"])
        if all(np.hypot(cat[j]["x"] - cat[k]["x"], cat[j]["y"] - cat[k]["y"])
               > (r + ball_radius(cat[k]["mw"])) * 1.35
               for k in chosen):
            chosen.append(j)
        if len(chosen) == N_BALLS:
            break

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 0.58],
                          left=0.055, right=0.975, top=0.882, bottom=0.120,
                          wspace=0.24, hspace=0.40)
    ax = fig.add_subplot(gs[:, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1], projection="polar")

    # ---- (a) the map --------------------------------------------------------
    ax.set_facecolor("#f5f5f5")
    ax.scatter([e["x"] for e in cat], [e["y"] for e in cat],
               s=1.6 * 2.2 ** (mw - 3.6), facecolor="#9a9a9a",
               edgecolor="none", alpha=0.75, zorder=3)
    for f in FAULTS:
        p = np.asarray(f["trace"], dtype=float)
        ax.plot(p[:, 0], p[:, 1], color=FAULT, lw=1.6, zorder=5,
                solid_capstyle="round")
        if f["kind"] == "thrust":                     # teeth on the hanging wall
            for a, b in zip(p[:-1], p[1:]):
                for s in np.linspace(0.06, 0.94, 9):
                    c = a + s * (b - a)
                    d = (b - a) / np.linalg.norm(b - a)
                    nvec = np.array([-d[1], d[0]]) * 2.4
                    ax.plot([c[0], c[0] + nvec[0]], [c[1], c[1] + nvec[1]],
                            color=FAULT, lw=1.0, zorder=5)
        halo_text(ax, p[-1, 0], p[-1, 1] + 2.6, f["name"], fontsize=6.4,
                  ha="right", va="bottom", color="#1a1a1a", halo="white",
                  halo_lw=2.2, zorder=7)

    gx, gy, inside, rays = unit_disk()
    for j in chosen:
        e = cat[j]
        r = ball_radius(e["mw"])
        beach_ball(ax, e["m"], e["x"], e["y"], r, gx, gy, inside, rays)
        halo_text(ax, e["x"], e["y"] - r - 1.2, f"M$_w$ {e['mw']:.1f}",
                  fontsize=6.0, ha="center", va="top", color="#1a1a1a",
                  halo="white", halo_lw=2.0, zorder=12)

    ax.set_xlim(0.0, X1)
    ax.set_ylim(0.0, Y1)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (km)", fontsize=8.0)
    ax.set_ylabel("northing (km)", fontsize=8.0)
    ax.tick_params(labelsize=7.0)
    ax.set_title(f"{len(cat)} events, {len(chosen)} largest shown as "
                 "lower-hemisphere mechanisms", fontsize=8.8,
                 fontweight="bold", pad=4.0)
    add_scale_bar(ax, 20.0, "20 km", loc=(0.435, 0.045), height=0.020,
                  fontsize=6.2, n_seg=2)
    ax.legend(handles=[
        Line2D([0], [0], color=FAULT, lw=1.6, label="mapped fault trace"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="#9a9a9a", markeredgecolor="none",
               label="epicentre, area $\\propto$ M$_w$"),
    ], loc="lower right", frameon=True, framealpha=0.90, facecolor="white",
        edgecolor="#cccccc", borderpad=0.4, fontsize=6.4, handlelength=1.8)

    # ---- (b) P and T axes on an equal-area net ------------------------------
    px, py = eq_area(p_tr, p_pl)
    tx, ty = eq_area(t_tr, t_pl)
    dens = gaussian_kde(np.vstack([px, py]), bw_method=0.30)
    zz = np.where(inside, dens(np.vstack([gx.ravel(), gy.ravel()]))
                  .reshape(gx.shape), np.nan)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "dens", get_palette("neutral", 9)[:5])
    ax_s.contourf(gx, gy, zz, levels=7, cmap=cmap, zorder=2)
    ax_s.scatter(tx, ty, s=7.0, facecolor="none", edgecolor=T_COL,
                 linewidth=0.55, zorder=4)
    ax_s.scatter(px, py, s=6.0, facecolor=P_COL, edgecolor="none", alpha=0.85,
                 zorder=5)

    th_c = np.linspace(0, 2 * np.pi, 721)
    ax_s.plot(np.sin(th_c), np.cos(th_c), color=COLORS["rule"], lw=0.9,
              zorder=6)
    for a in np.deg2rad([shmax, shmax + 180.0]):
        ax_s.annotate("", xy=(0.55 * np.sin(a), 0.55 * np.cos(a)),
                      xytext=(1.13 * np.sin(a), 1.13 * np.cos(a)),
                      annotation_clip=False, zorder=8,
                      arrowprops=dict(arrowstyle="-|>", lw=1.3, color=P_COL,
                                      shrinkA=0, shrinkB=0, mutation_scale=8))
    for lab, dx, dy in (("N", 0.0, 1.10), ("E", 1.10, 0.0),
                        ("S", 0.0, -1.10), ("W", -1.10, 0.0)):
        ax_s.text(dx, dy, lab, fontsize=6.4, color=COLORS["sub"],
                  ha="center", va="center")
    ax_s.set_xlim(-1.32, 1.32)
    ax_s.set_ylim(-1.32, 1.32)
    ax_s.set_aspect("equal")
    ax_s.axis("off")
    ax_s.set_title("P and T axes, equal-area net", fontsize=8.5,
                   fontweight="bold", pad=3.5)
    ax_s.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="none", markersize=2.9,
               markerfacecolor=P_COL, markeredgecolor="none",
               label="P axis (shading: density)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=2.9,
               markerfacecolor="none", markeredgecolor=T_COL,
               markeredgewidth=0.55, label="T axis"),
    ], loc="lower left", bbox_to_anchor=(-0.06, -0.055), frameon=False,
        fontsize=6.3, handletextpad=0.5)

    # ---- (c) P-axis rose and the inversion result ---------------------------
    width = np.deg2rad(10.0)
    edges = np.arange(0.0, 360.1, 10.0)
    both = np.concatenate([fold, fold + 180.0])
    counts, _ = np.histogram(both, bins=edges)
    centres = np.deg2rad(edges[:-1] + 5.0)
    ax_r.bar(centres, counts, width=width, bottom=0.0, color=P_COL,
             edgecolor="white", linewidth=0.35, alpha=0.85, zorder=3)
    rmax = counts.max() * 1.14
    for a in np.deg2rad([shmax, shmax + 180.0]):
        ax_r.plot([a, a], [0.0, rmax], color="#1a1a1a", lw=1.5, zorder=5)
    ax_r.set_theta_zero_location("N")
    ax_r.set_theta_direction(-1)
    ax_r.set_ylim(0.0, rmax)
    ax_r.set_yticks([])
    ax_r.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax_r.set_xticklabels(["N", "", "E", "", "S", "", "W", ""], fontsize=6.4)
    ax_r.tick_params(pad=0.5)
    ax_r.grid(lw=0.4, alpha=0.55)
    ax_r.set_title("P-axis azimuth rose", fontsize=8.5, fontweight="bold",
                   pad=6.0)
    ax_r.text(0.5, -0.14, f"S$_{{Hmax}}$ {shmax:.0f}$^\\circ$, stress ratio "
              f"R = {ratio:.2f}", transform=ax_r.transAxes, fontsize=6.8,
              color="#1a1a1a", ha="center", va="top")

    ax.apply_aspect()
    ax_s.apply_aspect()
    panel_label_inches(fig, ax, "(a)", dx=-0.44, dy=0.09)
    panel_label_inches(fig, ax_s, "(b)", dx=-0.16, dy=0.09)
    panel_label_inches(fig, ax_r, "(c)", dx=-0.16, dy=0.14)

    horiz = float((np.abs(p_pl) < 30.0).mean())

    savefig(fig, "616_eqk_focal_mechanism.png")


if __name__ == "__main__":
    main()
