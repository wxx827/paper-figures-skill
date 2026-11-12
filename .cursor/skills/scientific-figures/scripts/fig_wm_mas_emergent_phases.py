"""Emergent behaviour phases: three metrics, three named phases, three maps.

One 6-agent occupancy simulation produces everything here.  At each of 61
checkpoints the agents take 30 moves on an 18 x 12 arena; a cell is stamped
with the role of its last visitor, and a cross-role contact is logged whenever
two agents of different roles end up in neighbouring cells.  The three curves
are read straight off those grids -- coverage is the stamped fraction,
specialisation is the neighbour purity of the stamps, conflict rate is the
contact count per 100 moves -- and the three snapshots are the very grids that
produced the plotted points, so the caption arithmetic under each map closes
against the curve above it.

Phase edges are found in the data (conflict rate halving, specialisation
crossing 50%), never hand-placed.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import COLORS, savefig, set_pub_style

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
ACCENT = "#b5544a"
BAND = "#dfe6ee"            # clearly darker than 0.92, so it reads as ink
ROLE_COL = [BLUE[1], BLUE[3], BLUE[5]]
ROLE_NAME = ["haulers", "scouts", "builders"]
EMPTY = "#f4f8fb"

GW, GH = 18, 12
ROLE = [0, 0, 1, 1, 2, 2]
# One 6 x 6 patrol block per agent; the two blocks of a role form its band, and
# the band is what "specialisation" is measured against.
BLOCK = [(0, 5, 0, 5), (0, 5, 6, 11),        # haulers  - left band
         (12, 17, 0, 5), (12, 17, 6, 11),    # scouts   - right band
         (6, 11, 0, 5), (6, 11, 6, 11)]      # builders - middle band
SPAWN = (8, 5)
T, MOVES, STEP_K, EPISODES = 61, 46, 5, 9

# territory[y, x] = the role that owns the cell, from BLOCK
TERRITORY = np.zeros((GH, GW), int)
for _a, (_x0, _x1, _y0, _y1) in enumerate(BLOCK):
    TERRITORY[_y0:_y1 + 1, _x0:_x1 + 1] = ROLE[_a]


def _path(x0, x1, y0, y1):
    """Boustrophedon sweep of one block, the policy an agent converges to."""
    out = []
    for k, y in enumerate(range(y0, y1 + 1)):
        xs = range(x0, x1 + 1) if k % 2 == 0 else range(x1, x0 - 1, -1)
        out += [(x, y) for x in xs]
    return out


PATHS = [_path(*b) for b in BLOCK]


def episode(rng, skill):
    """One episode: returns the stamped grid and the three measurements."""
    grid = np.full((GH, GW), -1)
    pos = np.array([SPAWN] * 6)
    ptr = [0] * 6
    contacts = []
    for _ in range(MOVES):
        for a in range(6):
            if rng.random() < skill:
                p = PATHS[a]
                tx, ty = p[min(ptr[a], len(p) - 1)]
                if tx == pos[a, 0] and ty == pos[a, 1]:
                    ptr[a] += 1                     # arrive and advance in one
                    tx, ty = p[min(ptr[a], len(p) - 1)]
                dx, dy = tx - pos[a, 0], ty - pos[a, 1]
                if abs(dx) >= abs(dy):
                    pos[a, 0] += np.sign(dx)
                else:
                    pos[a, 1] += np.sign(dy)
            else:
                k = rng.integers(0, 4)
                pos[a] += [(1, 0), (-1, 0), (0, 1), (0, -1)][k]
            pos[a, 0] = np.clip(pos[a, 0], 0, GW - 1)
            pos[a, 1] = np.clip(pos[a, 1], 0, GH - 1)
            grid[pos[a, 1], pos[a, 0]] = ROLE[a]
        for i in range(6):
            for j in range(i + 1, 6):
                if ROLE[i] != ROLE[j] and \
                        np.max(np.abs(pos[i] - pos[j])) <= 1:
                    contacts.append(tuple((pos[i] + pos[j]) / 2.0))
    vis = grid >= 0
    onown = int(((grid == TERRITORY) & vis).sum())
    return dict(grid=grid, cells=int(vis.sum()), own=onown,
                cov=100.0 * vis.mean(),
                spec=100.0 * onown / max(int(vis.sum()), 1),
                nconf=len(contacts),
                rate=100.0 * len(contacts) / MOVES,
                conf=sorted(set(contacts)))


def simulate(seed=1590):
    """61 checkpoints x 5 episodes; every number is measured on a grid.

    The curves are the checkpoint means; the snapshot panels draw episode 1 of
    three checkpoints and the figure marks that episode's own value on the
    curve, so the caption arithmetic is a point the reader can see.
    """
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        # how reliably an agent follows its own sweep instead of milling about
        skill = 0.02 + 0.96 / (1.0 + np.exp(-(t - 0.42 * T) / (0.16 * T)))
        eps = [episode(rng, skill) for _ in range(EPISODES)]
        out.append(dict(eps=eps, skill=skill,
                        cov=float(np.mean([e["cov"] for e in eps])),
                        spec=float(np.mean([e["spec"] for e in eps])),
                        rate=float(np.mean([e["rate"] for e in eps]))))
    return out


def main():
    set_pub_style(base_font=8)
    fr = simulate()
    steps = np.arange(T) * STEP_K
    cov = np.array([f["cov"] for f in fr])
    spec = np.array([f["spec"] for f in fr])
    rate = np.array([f["rate"] for f in fr])

    # ---- phase edges, found in the data ------------------------------------
    # Phase I ends once the conflict rate has lost 30% of its peak; phase II
    # ends once specialisation is inside 85% of its final level and stays
    # there.  Both rules are printed in the subtitle.
    above = np.flatnonzero(rate > 0.80 * rate.max())
    i1 = int(above[-1] + 1)
    below = np.flatnonzero(spec < 0.95 * spec[-1])
    i2 = int(below[-1] + 1)
    assert 0 < i1 < i2 < T - 1, "phase edges must stay ordered"
    edges = [0, i1, i2, T - 1]
    mids = [(edges[k] + edges[k + 1]) // 2 for k in range(3)]
    names = ["Uncoordinated exploration", "Territory contention",
             "Stable division of labour"]

    fig = plt.figure(figsize=(10.4, 6.0))
    ax = fig.add_axes([0.058, 0.470, 0.905, 0.395])
    ylo, yhi = 0.0, max(rate.max(), 100.0) * 1.28

    ax.fill_between([steps[i1], steps[i2]], ylo, yhi, color=BAND,
                    edgecolor="none", zorder=0)
    for e in (i1, i2):                     # explicit endpoints, not axvline
        ax.plot([steps[e], steps[e]], [ylo, yhi], color="#93a5b4", lw=0.9,
                ls=(0, (3.5, 2.2)), zorder=2)

    lines = [
        ax.plot(steps, cov, color=BLUE[1], lw=2.0, zorder=5,
                label=f"arena coverage (%), {cov[0]:.1f} "
                      f"$\\rightarrow$ {cov[-1]:.1f}")[0],
        ax.plot(steps, spec, color=BLUE[3], lw=2.0, ls=(0, (5, 2)), zorder=5,
                label=f"role specialisation (%), {spec[0]:.1f} "
                      f"$\\rightarrow$ {spec[-1]:.1f}")[0],
        ax.plot(steps, rate, color=ACCENT, lw=2.0, ls=(0, (1.4, 1.4)),
                zorder=5, label=f"cross-role conflicts per 100 moves, "
                f"{rate[0]:.0f} $\\rightarrow$ {rate[-1]:.0f}")[0]]
    leg0 = ax.legend(handles=lines, loc="lower left",
                     bbox_to_anchor=(0.012, 0.030), fontsize=7.2,
                     frameon=True, facecolor="white", edgecolor="#c3ccd4",
                     framealpha=0.96, handlelength=2.4, labelspacing=0.45)
    leg0.set_zorder(8)
    for txt in leg0.get_texts():
        txt.set_color(INK)

    for k in range(3):
        xc = 0.5 * (steps[edges[k]] + steps[edges[k + 1]])
        ax.text(xc, yhi - 0.02 * yhi,
                f"Phase {'I II III'.split()[k]}\n{names[k]}",
                ha="center", va="top", fontsize=7.4, fontweight="bold",
                color=BLUE[0], zorder=7, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.30", fc="white", ec=BLUE[5],
                          lw=0.6, alpha=0.96))

    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("environment steps (thousands)", fontsize=8.0,
                  color=COLORS["sub"], labelpad=2.0)
    ax.set_ylabel("percent, or contacts per 100 moves", fontsize=8.0,
                  color=COLORS["sub"], labelpad=2.5)
    ax.tick_params(labelsize=7.2, length=2.8)
    ax.grid(axis="y", color="#e6eaee", lw=0.6)
    ax.set_axisbelow(True)

    # ---- one snapshot per phase, the grid that made those points ----------
    tw = 0.205
    th = tw * 10.4 / 6.0 * (GH / GW)
    y0 = 0.178
    lo, hi = 0.058 + tw / 2, 0.963 - tw / 2
    for k, m in enumerate(mids):
        f = fr[m]["eps"][0]                  # the episode this panel draws
        for arr, col in ((f["cov"], BLUE[1]), (f["spec"], BLUE[3]),
                         (f["rate"], ACCENT)):
            ax.plot([steps[m]], [arr], marker="o", ms=5.2, mfc="white",
                    mec=col, mew=1.4, zorder=6)
        xa = ax.transData.transform((steps[m], 0))[0]
        xc = float(np.clip(fig.transFigure.inverted().transform((xa, 0))[0],
                           lo, hi))
        axs = fig.add_axes([xc - tw / 2, y0, tw, th])
        img = np.zeros((GH, GW, 3))
        img[:] = np.array(plt.matplotlib.colors.to_rgb(EMPTY))
        for r in range(3):
            img[f["grid"] == r] = plt.matplotlib.colors.to_rgb(ROLE_COL[r])
        axs.imshow(img, interpolation="nearest", zorder=1,
                   extent=[0, GW, GH, 0])
        for gx in range(GW + 1):
            axs.plot([gx, gx], [0, GH], color="white", lw=0.35, zorder=2)
        for gy in range(GH + 1):
            axs.plot([0, GW], [gy, gy], color="white", lw=0.35, zorder=2)
        for (cx, cy) in f["conf"]:
            axs.plot([cx + 0.5], [cy + 0.5], marker="x", ms=3.6, mew=1.1,
                     color=ACCENT, zorder=4)
        axs.set_xlim(0, GW)
        axs.set_ylim(GH, 0)
        axs.set_xticks([])
        axs.set_yticks([])
        for s in axs.spines.values():
            s.set_color(BLUE[2])
            s.set_linewidth(0.8)

        # the connector: phase band -> its snapshot
        fig.add_artist(Line2D([xc, xc], [y0 + th + 0.004, 0.462],
                              color="#93a5b4", lw=0.8, zorder=1))
        cells = f["cells"]
        assert abs(100.0 * cells / (GW * GH) - f["cov"]) < 1e-9
        assert abs(100.0 * f["own"] / cells - f["spec"]) < 1e-9
        axs.text(0.5, -0.10, f"{steps[m]:d}k steps", transform=axs.transAxes,
                 ha="center", va="top", fontsize=7.6, fontweight="bold",
                 color=INK)
        axs.text(0.5, -0.27,
                 f"coverage {cells:d} / {GW * GH:d} = {f['cov']:.1f}%\n"
                 f"on own band {f['own']:d} / {cells:d} = {f['spec']:.1f}%\n"
                 f"{f['nconf']:d} contacts / {MOVES:d} moves x 100 = "
                 f"{f['rate']:.0f}",
                 transform=axs.transAxes, ha="center",
                 va="top", fontsize=6.6, color=COLORS["sub"], linespacing=1.5)

    key = [Line2D([], [], marker="s", ls="none", ms=6.4, mfc=ROLE_COL[r],
                  mec="white", mew=0.6, label=f"cells last visited by "
                  f"{ROLE_NAME[r]}") for r in range(3)]
    key += [Line2D([], [], marker="s", ls="none", ms=6.4, mfc=EMPTY,
                   mec="#c3ccd4", mew=0.6, label="never visited"),
            Line2D([], [], marker="x", ls="none", ms=4.6, mew=1.1,
                   color=ACCENT, label="cross-role contact"),
            Line2D([], [], marker="o", ls="none", ms=5.2, mfc="white",
                   mec=BLUE[0], mew=1.4,
                   label="the plotted episode drawn below")]
    leg = fig.legend(handles=key, loc="lower center",
                     bbox_to_anchor=(0.510, 0.012), ncol=6, fontsize=7.0,
                     frameon=False, handletextpad=0.5, columnspacing=1.6)
    for txt in leg.get_texts():
        txt.set_color(COLORS["sub"])

    # wrapped: as one line this subtitle is 12.0 in wide and drags the tight
    # bbox out to 12.5 in, which drops the delivered h/w to 0.466

    savefig(fig, "1185_wm_mas_emergent_phases.png")


if __name__ == "__main__":
    main()
