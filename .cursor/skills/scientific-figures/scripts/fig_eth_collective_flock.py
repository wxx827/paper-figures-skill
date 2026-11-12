"""Snapshot of a flock at the moment collective order has taken hold.

Everything on the page comes out of one zonal interaction simulation
(repulsion, alignment, attraction) in which only the width of the alignment
zone changes half way through the run.  The main panel is the claim: the group
is a single coherent body moving in one direction, and the local alignment
colouring shows the disorder that survives is at the edges, not in the core.
The order parameter panel is the control that makes the snapshot meaningful -
without it a reader cannot tell whether the arrows line up because the animals
interact or because they were initialised that way.  The nearest-neighbour
distribution fixes the spatial scale the interaction zones are set in.
"""
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import ConvexHull

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import (axes_inches, panel_labels_inches, safe_colorbar,
                    add_scale_bar, halo_text)

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
RED = "#b23a48"
BLUE = "#3b5b92"

N_AGENT = 180
N_STEP = 900
DT = 0.15                 # s
SPEED = 1.0               # m/s
ZOR = 0.55                # repulsion radius, m
ZOA = 9.0                 # attraction radius, m
ZOO_LATE = 6.0            # alignment radius after the switch, m
SWITCH = 420              # step at which the alignment zone opens
RAMP = 90                 # steps over which it opens
TURN_MAX = 0.34           # rad per step
NOISE = 0.065             # rad
W_ORI = 2.2               # weight of alignment against attraction


def zoo_at(step):
    return ZOO_LATE * float(np.clip((step - SWITCH) / RAMP, 0.0, 1.0))


def simulate(seed=2026):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.0, 3.2, (N_AGENT, 2))
    ang = rng.uniform(-np.pi, np.pi, N_AGENT)
    vel = np.column_stack([np.cos(ang), np.sin(ang)])
    phi = np.empty(N_STEP)

    for s in range(N_STEP):
        zoo = zoo_at(s)
        diff = pos[None, :, :] - pos[:, None, :]
        dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
        np.fill_diagonal(dist, np.inf)
        unit = diff / dist[:, :, None]

        rep_m = dist < ZOR
        rep = -(unit * rep_m[:, :, None]).sum(axis=1)
        ori_m = (dist >= ZOR) & (dist < zoo)
        ori = (vel[None, :, :] * ori_m[:, :, None]).sum(axis=1)
        att_m = (dist >= max(zoo, ZOR)) & (dist < ZOA)
        att = (unit * att_m[:, :, None]).sum(axis=1)

        def unitise(v):
            n = np.hypot(v[:, 0], v[:, 1])[:, None]
            return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)

        want = W_ORI * unitise(ori) + unitise(att)
        want = np.where(rep_m.any(axis=1)[:, None], rep, want)
        want = unitise(want)
        blank = np.hypot(want[:, 0], want[:, 1]) < 1e-9
        want[blank] = vel[blank]

        target = np.arctan2(want[:, 1], want[:, 0]) \
            + rng.normal(0.0, NOISE, N_AGENT)
        cur = np.arctan2(vel[:, 1], vel[:, 0])
        d = (target - cur + np.pi) % (2.0 * np.pi) - np.pi
        cur = cur + np.clip(d, -TURN_MAX, TURN_MAX)
        vel = np.column_stack([np.cos(cur), np.sin(cur)])
        pos = pos + SPEED * DT * vel
        phi[s] = float(np.hypot(*vel.mean(axis=0)))
    return pos - pos.mean(axis=0), vel, phi


def local_alignment(pos, vel, radius=ZOO_LATE):
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    m = dist < radius
    s = (vel[None, :, :] * m[:, :, None]).sum(axis=1)
    return np.hypot(s[:, 0], s[:, 1]) / m.sum(axis=1)


def nnd(pos):
    diff = pos[None, :, :] - pos[:, None, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def frame_to_box(xs, ys, box_w, box_h, pad=0.12):
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    mx, my = pad * (x1 - x0), pad * (y1 - y0)
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    want, have = box_w / box_h, (x1 - x0) / (y1 - y0)
    if have < want:
        e = ((y1 - y0) * want - (x1 - x0)) / 2.0
        x0, x1 = x0 - e, x1 + e
    else:
        e = ((x1 - x0) / want - (y1 - y0)) / 2.0
        y0, y1 = y0 - e, y1 + e
    return x0, x1, y0, y1


def flock_panel(fig, ax, pos, vel, align, extent):
    hull = ConvexHull(pos)
    hv = np.vstack([pos[hull.vertices], pos[hull.vertices][:1]])
    ax.plot(hv[:, 0], hv[:, 1], color="#b3b3b3", lw=0.9, ls=(0, (4, 2.4)),
            zorder=2)

    norm = mpl.colors.Normalize(0.0, 1.0)
    q = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], align,
                  cmap="viridis", norm=norm, angles="xy", scale_units="xy",
                  scale=1.0 / 0.95, width=0.0055, headwidth=3.4,
                  headlength=4.2, headaxislength=3.6, zorder=4)

    c = pos.mean(axis=0)
    heading = vel.mean(axis=0)
    heading = heading / np.hypot(*heading)
    ax.plot([c[0]], [c[1]], marker="X", ms=8.0, mfc=RED, mec="white",
            mew=1.1, zorder=7)
    span = 0.30 * (extent[1] - extent[0])
    ax.add_patch(FancyArrowPatch(
        tuple(c - heading * span * 0.18), tuple(c + heading * span * 0.82),
        arrowstyle="-|>", mutation_scale=16, lw=2.0, color=RED, alpha=0.9,
        shrinkA=0, shrinkB=0, zorder=6))
    ang = np.degrees(np.arctan2(heading[1], heading[0])) % 360.0
    tip = c + heading * span * 0.86
    halo_text(ax, tip[0], tip[1] + 0.6, f"group heading {ang:.0f} deg",
              ha="center", va="bottom", fontsize=6.8, color=RED,
              halo="white", halo_lw=2.2, zorder=8)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x relative to group centroid (m)")
    ax.set_ylabel("y relative to group centroid (m)")
    ax.grid(lw=0.45, alpha=0.5)
    ax.set_title("A single polarised body with disorder only at the edge",
                 fontsize=9.0, fontweight="bold", color=INK)
    annotate_box(ax, f"n = {pos.shape[0]} individuals\n"
                     f"group extent {hull.volume:.0f} m$^2$\n"
                     f"mean local alignment {align.mean():.2f}",
                 loc="upper left", fontsize=6.6, color=SUB)
    add_scale_bar(ax, 5.0, "5 m", loc=(0.035, 0.050), height=0.016,
                  text_color=INK, halo="white", fontsize=6.4)
    safe_colorbar(fig, q, ax, label="local alignment (-)", width=0.10,
                  pad=0.07, labelsize=6.6, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    return float(hull.volume)


def order_panel(ax, phi):
    t = np.arange(phi.size) * DT
    ax.axvspan(0.0, SWITCH * DT, facecolor="#f5f5f5", edgecolor="none",
               zorder=0)
    ax.plot(t, phi, color=BLUE, lw=1.2, zorder=4)
    lo = float(phi[SWITCH // 2:SWITCH].mean())
    hi = float(phi[-120:].mean())
    ax.axhline(lo, color=MUTED, lw=0.9, ls=(0, (3, 2.2)), zorder=3)
    ax.axhline(hi, color=RED, lw=0.9, ls=(0, (3, 2.2)), zorder=3)
    ax.axvline(SWITCH * DT, color=RULE, lw=0.9, ls=(0, (2.4, 2.0)), zorder=3)
    ax.text(0.025, 0.955, f"disordered mean $\\varphi$ = {lo:.2f}\n"
                          f"polarised mean $\\varphi$ = {hi:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
            color=SUB, zorder=6,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6, alpha=0.93))
    ax.text(SWITCH * DT - 3.0, 0.48, "alignment zone opens", fontsize=6.3,
            color=RULE, ha="right", va="center",
            transform=ax.get_xaxis_transform())
    ax.text(0.982, 0.045, "snapshot in (a) is the final step",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3,
            color=SUB)
    ax.set_xlabel("time (s)", fontsize=7.4)
    ax.set_ylabel(r"polarisation $\varphi$ (-)", fontsize=7.4)
    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Order appears only once alignment is switched on",
                 fontsize=8.2, fontweight="bold", color=INK)
    return lo, hi


def nnd_panel(ax, d):
    ax.hist(d, bins=np.linspace(0.0, float(np.percentile(d, 99.5)) * 1.1, 22),
            color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    med = float(np.median(d))
    ax.axvline(med, color=RED, lw=1.2, ls=(0, (3.2, 2.0)), zorder=4)
    ax.text(med * 1.08, 0.96, f"median {med:.2f} m", fontsize=6.3, color=RED,
            ha="left", va="top", transform=ax.get_xaxis_transform())
    ax.axvline(ZOR, color=MUTED, lw=0.9, ls=(0, (2.2, 1.8)), zorder=4)
    ax.annotate(f"repulsion radius\n{ZOR:.2f} m", xy=(ZOR, 0.62),
                xytext=(ZOR * 0.42, 0.72), textcoords=ax.get_xaxis_transform(),
                xycoords=ax.get_xaxis_transform(), fontsize=6.3, color=MUTED,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xlabel("nearest-neighbour distance (m)", fontsize=7.4)
    ax.set_ylabel(f"individuals of n={d.size} (-)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.45, alpha=0.6)
    ax.set_title("Spacing is set by the repulsion zone", fontsize=8.2,
                 fontweight="bold", color=INK)
    return med


def main():
    set_pub_style()
    pos, vel, phi = simulate()
    align = local_alignment(pos, vel)
    d = nnd(pos)
    extent = frame_to_box(pos[:, 0], pos[:, 1], 5.05, 4.62)

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (0.86, 0.80, 5.00, 4.62))
    ax_phi = axes_inches(fig, (7.02, 3.40, 2.84, 2.02))
    ax_nnd = axes_inches(fig, (7.02, 0.80, 2.84, 2.02))

    area = flock_panel(fig, ax_main, pos, vel, align, extent)
    lo, hi = order_panel(ax_phi, phi)
    med = nnd_panel(ax_nnd, d)

    panel_labels_inches(fig, [ax_main, ax_phi, ax_nnd], dx=-0.52, dy=0.14)
    savefig(fig, "779_eth_collective_flock.png")


if __name__ == "__main__":
    main()
