"""One year of reservoir operation: inflow, stage and generation.

Three panels share a daily time axis so the causal chain is read vertically:
a storm arrives (a), the pool rises against the seasonal flood-limit level
(b), and the plant runs at whatever head the pool gives it (c).  The figure is
produced by an actual mass-balance simulation - storage is stepped day by day
with a stage-storage curve, releases are capped by turbine capacity and by the
dead pool, and any volume that cannot be stored below the current flood limit
is spilled.  Spill days are therefore detected, not drawn by hand, and they
are highlighted in red across all three panels.

The operating chart on the right is the same year plotted as stage against
time over the rule zones that governed it.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig, annotate_box
from figkit import halo_text, panel_labels_inches

INK, RULE, SUB, MUTED = "#1a1a1a", "#333333", "#555555", "#777777"
C_IN, C_OUT, C_LEVEL = "#3b5b92", "#2a7f7a", "#3b5b92"
C_SPILL, C_LIMIT = "#b23a48", "#c9a227"

Z_DEAD, Z_NORMAL, Z_CHECK, TAIL = 205.0, 245.0, 249.0, 190.0
S_DEAD, S_NORMAL, S_MAX = 2.0, 12.0, 13.4     # 10^8 m3
Q_TURB_MAX, SPILL_MAX, CAP = 320.0, 2600.0, 140.0   # m3/s, m3/s, MW
ETA = 0.86
DT = 86400.0 / 1.0e8                          # m3/s -> 10^8 m3 per day
FLOOD = (152, 273)                            # day of year, inclusive

MONTH_START = np.concatenate([[0], np.cumsum(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])])
MONTH_LAB = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
             "Oct", "Nov", "Dec"]


def z_of_s(s):
    s = np.clip(s, S_DEAD, S_MAX)
    return Z_DEAD + (Z_NORMAL - Z_DEAD) * ((s - S_DEAD) /
                                           (S_NORMAL - S_DEAD)) ** 0.62


def s_of_z(z):
    z = np.clip(z, Z_DEAD, Z_CHECK)
    return S_DEAD + (S_NORMAL - S_DEAD) * ((z - Z_DEAD) /
                                           (Z_NORMAL - Z_DEAD)) ** (1 / 0.62)


def flood_limit():
    """Phased flood-limit level: drawn down for the main flood, refilled after."""
    tl = np.full(365, Z_NORMAL)
    tl[151:181] = 238.0
    tl[181:243] = 235.0
    tl[243:273] = 240.0
    tl[273:305] = np.linspace(240.0, Z_NORMAL, 305 - 273)
    return tl


def make_inflow(rng):
    d = np.arange(365)
    base = 42.0 + 130.0 * np.exp(-0.5 * ((d - 206.0) / 44.0) ** 2)
    noise = rng.normal(0.0, 1.0, 365)
    for _ in range(3):
        noise = np.convolve(noise, np.ones(9) / 9.0, mode="same")
    q = base * (1.0 + 1.2 * noise)
    for _ in range(6):                       # storm hydrographs
        t0 = int(rng.integers(150, 285))
        amp = float(rng.uniform(150.0, 720.0))
        dur = float(rng.uniform(1.8, 3.2))
        q += amp * np.exp(-0.5 * ((d - t0) / dur) ** 2)
    return np.clip(q, 22.0, None)


def firm_demand():
    q = np.full(365, 85.0)
    q[90:181] = 120.0                        # irrigation draw
    q[181:304] = 125.0
    return q


def simulate(inflow, target, demand):
    s = s_of_z(241.0)
    n = inflow.size
    out = dict(level=np.zeros(n), turbine=np.zeros(n), spill=np.zeros(n),
               power=np.zeros(n), storage=np.zeros(n))
    for i in range(n):
        z = z_of_s(s)
        head = max(z - TAIL, 1.0)
        q_cap = CAP * 1000.0 / (9.81 * ETA * head)
        q_t = min(Q_TURB_MAX, q_cap, demand[i])
        q_t = min(q_t, max((s - S_DEAD) / DT + inflow[i], 0.0))
        s_tar = s_of_z(target[i])
        # above the flood limit the plant is pushed to its maximum before a
        # drop is spilled; skipping this step turns every wet day into a
        # spill day, which is neither realistic nor readable
        if s + (inflow[i] - q_t) * DT > s_tar:
            q_t = min(Q_TURB_MAX, q_cap,
                      max(q_t, (s - s_tar) / DT + inflow[i]))
        s_new = s + (inflow[i] - q_t) * DT
        spill = 0.0
        if s_new > s_tar:
            spill = min((s_new - s_tar) / DT, SPILL_MAX)
            s_new -= spill * DT
        if s_new > S_MAX:                    # forced spill above the crest
            spill += (s_new - S_MAX) / DT
            s_new = S_MAX
        s = max(s_new, S_DEAD)
        out["storage"][i] = s
        out["level"][i] = z_of_s(s)
        out["turbine"][i] = q_t
        out["spill"][i] = spill
        out["power"][i] = 9.81 * ETA * q_t * max(z_of_s(s) - TAIL, 0.0) / 1000.0
    return out


def month_axis(ax):
    """Month ticks on the shared axis.

    Only the bottom panel may set tick *labels*: with ``sharex`` the formatter
    is shared too, so ``set_xticklabels([])`` on an upper panel would blank
    the bottom one as well.  Upper panels hide theirs with ``labelbottom``.
    """
    ax.set_xlim(0, 365)
    ax.set_xticks(MONTH_START)
    ax.set_xticklabels(MONTH_LAB, fontsize=6.5)
    ax.tick_params(axis="x", length=2.2)


def main():
    set_pub_style()
    rng = np.random.default_rng(604)

    day = np.arange(365)
    inflow = make_inflow(rng)
    target = flood_limit()
    res = simulate(inflow, target, firm_demand())
    outflow = res["turbine"] + res["spill"]
    spilling = res["spill"] > 1.0

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 0.58],
                          height_ratios=[1.0, 1.0, 0.80],
                          left=0.058, right=0.980, top=0.855, bottom=0.168,
                          wspace=0.20, hspace=0.22)
    ax_q = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[1, 0], sharex=ax_q)
    ax_p = fig.add_subplot(gs[2, 0], sharex=ax_q)
    ax_r = fig.add_subplot(gs[:, 1])

    # ---- season and spill bands, repeated on every stacked panel -----------
    edges = np.flatnonzero(np.diff(np.r_[0, spilling.astype(int), 0]))
    for ax in (ax_q, ax_z, ax_p):
        ax.axvspan(FLOOD[0], FLOOD[1], color="#3d6b8e", alpha=0.065, lw=0,
                   zorder=0)
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(a - 0.5, b - 0.5, color=C_SPILL, alpha=0.20, lw=0,
                       zorder=1)

    # ---- (a) inflow and outflow -------------------------------------------
    ax_q.plot(day, inflow, color=C_IN, lw=1.1, zorder=4, label="inflow")
    ax_q.plot(day, outflow, color=C_OUT, lw=1.0, zorder=5, label="outflow")
    ax_q.fill_between(day, 0, inflow, color=C_IN, alpha=0.10, lw=0, zorder=2)
    ax_q.set_ylabel("discharge (m$^{3}$ s$^{-1}$)", fontsize=7.6)
    ax_q.set_ylim(0, max(inflow.max(), outflow.max()) * 1.12)
    ax_q.tick_params(labelsize=6.6, labelbottom=False)
    ax_q.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_q.set_title("Inflow against total release", fontsize=8.5,
                   fontweight="bold")

    # ---- (b) pool level ----------------------------------------------------
    ax_z.plot(day, res["level"], color=C_LEVEL, lw=1.4, zorder=5)
    ax_z.fill_between(day, Z_DEAD, res["level"], color=C_LEVEL, alpha=0.10,
                      lw=0, zorder=2)
    ax_z.axhline(Z_NORMAL, color="#2e7d5b", lw=0.9, ls=(0, (5, 2)), zorder=4)
    ax_z.axhline(Z_DEAD, color="#808080", lw=0.9, ls=(0, (5, 2)), zorder=4)
    ax_z.plot(day, target, color=C_LIMIT, lw=1.2, ls=(0, (4, 2)), zorder=6)
    for y, txt, col in ((Z_NORMAL, "normal pool 245 m", "#2e7d5b"),
                        (Z_DEAD, "dead water level 205 m", "#808080")):
        ax_z.text(362, y + 1.2, txt, ha="right", va="bottom", fontsize=6.3,
                  color=col, zorder=7)
    ax_z.text(212, 227.5, "phased flood-limit level", ha="center", va="top",
              fontsize=6.3, color=C_LIMIT, zorder=7)
    ax_z.set_ylabel("pool level (m)", fontsize=7.6)
    ax_z.set_ylim(Z_DEAD - 3.0, Z_CHECK + 1.5)
    ax_z.tick_params(labelsize=6.6, labelbottom=False)
    ax_z.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_z.set_title("Pool level against the operating rules", fontsize=8.5,
                   fontweight="bold")

    # ---- (c) generation ----------------------------------------------------
    ax_p.bar(day, res["power"], width=1.0, color="#3d6b8e", lw=0, zorder=3)
    ax_p.axhline(CAP, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=4)
    ax_p.text(362, CAP + 4.0, f"installed capacity {CAP:.0f} MW", ha="right",
              va="bottom", fontsize=6.3, color=MUTED)
    ax_p.set_ylabel("output (MW)", fontsize=7.6)
    ax_p.set_ylim(0, CAP * 1.30)
    ax_p.tick_params(labelsize=6.6)
    ax_p.grid(axis="y", lw=0.5, color="#ececec", zorder=0)
    ax_p.set_title("Daily generation", fontsize=8.5, fontweight="bold")
    month_axis(ax_p)

    # ---- (d) operating chart ----------------------------------------------
    anchors = np.array([230, 226, 222, 219, 220, 224, 228, 230, 231, 233,
                        233, 232], dtype=float)
    mid = MONTH_START + np.diff(np.r_[MONTH_START, 365]) / 2.0
    lower = np.interp(day, np.r_[mid - 365, mid, mid + 365],
                      np.r_[anchors, anchors, anchors])
    reduced = lower - 9.0
    zones = [(target, np.full(365, Z_CHECK), "#e8ccd0",
              "increased output / flood control"),
             (lower, target, "#cfdce9", "guaranteed output"),
             (reduced, lower, "#f0e3bd", "reduced output"),
             (np.full(365, Z_DEAD), reduced, "#e6e6e6", "dead storage")]
    for lo, hi, col, name in zones:
        ax_r.fill_between(day, lo, hi, color=col, lw=0, zorder=2)
        halo_text(ax_r, 182, 0.5 * (float(np.asarray(lo)[182]) +
                                    float(np.asarray(hi)[182])), name,
                  ha="center", va="center", fontsize=6.2, color="#4a4a4a",
                  halo="white", halo_lw=1.8, zorder=9)
    ax_r.plot(day, target, color=C_LIMIT, lw=1.1, ls=(0, (4, 2)), zorder=5)
    ax_r.plot(day, lower, color="#7d8b96", lw=0.9, zorder=5)
    ax_r.plot(day, res["level"], color=C_LEVEL, lw=1.5, zorder=7)
    ax_r.scatter(day[spilling], res["level"][spilling], s=7.0,
                 color=C_SPILL, zorder=8, linewidths=0)
    ax_r.set_ylim(Z_DEAD - 1.5, Z_CHECK + 1.0)
    ax_r.set_ylabel("pool level (m)", fontsize=7.4)
    ax_r.set_xlabel("month", fontsize=7.4)
    ax_r.tick_params(labelsize=6.4)
    ax_r.set_title("Operating chart for the same year", fontsize=8.5,
                   fontweight="bold")
    ax_r.set_xlim(0, 365)
    ax_r.set_xticks(MONTH_START[::2])
    ax_r.set_xticklabels(MONTH_LAB[::2], fontsize=6.4)

    # ---- derived headline numbers -----------------------------------------
    spill_days = int(spilling.sum())
    spill_vol = float(res["spill"].sum() * DT)
    energy = float(res["power"].sum() * 24.0 / 1000.0)
    inflow_vol = float(inflow.sum() * DT)
    annotate_box(ax_q,
                 f"inflow {inflow_vol:.1f} \u00d7 10$^{8}$ m$^{3}$   "
                 f"spill {spill_vol:.2f} \u00d7 10$^{8}$ m$^{3}$ on "
                 f"{spill_days} days", loc="upper right", fontsize=6.5,
                 margin=0.020)

    fig.legend(handles=[
        Line2D([0], [0], color=C_IN, lw=1.2, label="inflow"),
        Line2D([0], [0], color=C_OUT, lw=1.2, label="total release"),
        Line2D([0], [0], color=C_LIMIT, lw=1.2, ls=(0, (4, 2)),
               label="flood-limit level"),
        Line2D([0], [0], color="#3d6b8e", lw=5, alpha=0.28,
               label="flood season"),
        Line2D([0], [0], color=C_SPILL, lw=5, alpha=0.35,
               label="spill in progress")],
        loc="lower center", bbox_to_anchor=(0.5, 0.090), ncol=5,
        frameon=False, fontsize=7.0, handlelength=1.9, columnspacing=1.9)

    panel_labels_inches(fig, [ax_q, ax_z, ax_p, ax_r], dx=-0.42, dy=0.09)


    print(f"inflow={inflow_vol:.2f}  spill={spill_vol:.2f} on {spill_days} d"
          f"  energy={energy:.0f} GWh")
    print(f"level min={res['level'].min():.2f}  max={res['level'].max():.2f}"
          f"  storage min={res['storage'].min():.2f}")
    savefig(fig, "604_wat_reservoir_operation.png")


if __name__ == "__main__":
    main()
