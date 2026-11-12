"""Psychoacoustic metrics of a cabin recording during a two-gear drive-away.

Four time histories - loudness, sharpness, roughness and fluctuation strength
- all driven by one shared engine-speed profile, each with its 5 % exceedance
percentile marked.  The radar on the right condenses those percentiles into
Zwicker's psychoacoustic annoyance and compares the cabin with its target.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import (set_pub_style, savefig, style_axes, add_panel_label,
                      CATEGORICAL, COLORS)
from figkit import halo_text

T_END = 12.0
KNOTS_T = [0.0, 1.2, 5.4, 5.75, 6.35, 8.0, 12.0]
KNOTS_R = [900.0, 1500.0, 4300.0, 4250.0, 2500.0, 3400.0, 4900.0]
# metric -> (unit, colour, target value for the reference cabin)
TARGETS = {"Loudness $N$": ("sone", 26.0), "Sharpness $S$": ("acum", 1.55),
           "Roughness $R$": ("asper", 0.42),
           "Fluctuation $F$": ("vacil", 0.55)}


def smooth(v, width=9.0):
    k = np.exp(-0.5 * (np.arange(-40, 41) / width) ** 2)
    n = k.size // 2
    padded = np.pad(np.asarray(v, dtype=float), n, mode="edge")
    return np.convolve(padded, k / k.sum(), mode="same")[n:-n]


def drive_cycle(t):
    rpm = np.interp(t, KNOTS_T, KNOTS_R)
    return smooth(rpm, 12.0)


def metrics(t, rpm, rng):
    """The four metrics, all functions of the same speed profile."""
    n = 6.0 + 0.0052 * rpm + 1.4 * np.sin(2 * np.pi * t / 3.1)
    s = 1.02 + 1.7e-4 * rpm
    r = 0.14 + 0.42 * np.exp(-0.5 * ((rpm - 2450.0) / 720.0) ** 2)
    f = 0.06 + 0.62 * np.exp(-0.5 * ((rpm - 1550.0) / 520.0) ** 2)
    out = []
    for base, jit in ((n, 0.9), (s, 0.035), (r, 0.030), (f, 0.035)):
        out.append(np.maximum(base + smooth(rng.normal(0, jit, t.size), 4.0),
                              1e-3))
    return out


def annoyance(n5, s5, r5, f5):
    """Zwicker psychoacoustic annoyance from the 5 % exceedance values."""
    ws = (s5 - 1.75) * 0.25 * np.log10(n5 + 10.0) if s5 > 1.75 else 0.0
    wfr = (2.18 / n5 ** 0.4) * (0.4 * f5 + 0.6 * r5)
    return n5 * (1.0 + np.sqrt(ws ** 2 + wfr ** 2)), ws, wfr


def main():
    set_pub_style()
    rng = np.random.default_rng(420)

    t = np.linspace(0.0, T_END, 1400)
    rpm = drive_cycle(t)
    series = metrics(t, rpm, rng)
    names = list(TARGETS)
    p95 = [float(np.percentile(v, 95.0)) for v in series]

    pa, w_s, w_fr = annoyance(*p95)
    pa_target, _, _ = annoyance(*[TARGETS[k][1] for k in names])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.88],
                          wspace=0.28, hspace=0.42, left=0.058, right=0.985,
                          top=0.880, bottom=0.105)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    ax_r = fig.add_subplot(gs[:, 2], projection="polar")

    for ax, name, val, col, lab, sym in zip(axes, names, series, CATEGORICAL,
                                            "abcd", "NSRF"):
        unit, target = TARGETS[name]
        ax.fill_between(t, 0.0, val, color=col, alpha=0.16, lw=0)
        ax.plot(t, val, color=col, lw=1.4, zorder=4)
        pv = float(np.percentile(val, 95.0))
        ax.axhline(pv, color=COLORS["rule"], lw=0.9, ls=(0, (4.0, 2.2)),
                   zorder=5)
        ax.axhline(target, color=COLORS["baseline"], lw=0.9,
                   ls=(0, (1.6, 1.8)), zorder=5)
        ax.text(0.985, pv, f"${sym}_5$ = {pv:.2f} {unit}",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=6.8, color=COLORS["rule"])
        ax.set_ylim(0.0, max(val.max(), target) * 1.30)
        ax.set_xlim(0.0, T_END)
        ax.set_ylabel(f"{name} ({unit})", fontsize=8.0)
        ax.tick_params(labelsize=7.0)
        if lab in "cd":
            ax.set_xlabel("Time (s)", fontsize=8.0)
        else:
            ax.tick_params(labelbottom=False)
        ax.grid(axis="y", lw=0.5, alpha=0.55)
        style_axes(ax)
        add_panel_label(ax, f"({lab})", x=-0.14, y=1.04)

    ax_sp = axes[0].twinx()
    ax_sp.plot(t, rpm, color=COLORS["faint"], lw=0.9, zorder=2)
    ax_sp.set_ylim(0, 6400)
    ax_sp.set_yticks([])
    for side in ("top", "right"):
        ax_sp.spines[side].set_visible(False)
    ax_sp.text(T_END * 0.52, rpm.max() * 0.60,
               f"engine speed {rpm.min() / 1000:.1f}-{rpm.max() / 1000:.1f} krpm",
               fontsize=6.4, color=COLORS["muted"], ha="left", va="top")

    # ---- (e) annoyance radar ----------------------------------------------
    labels = [n.split(" ")[0] for n in names] + ["Annoyance"]
    meas = [p95[i] / TARGETS[names[i]][1] for i in range(4)] + \
           [pa / pa_target]
    units = [f"{p95[i]:.2f} {TARGETS[names[i]][0]}" for i in range(4)] + \
            [f"{pa:.1f} PA"]
    ang = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    ring = np.append(ang, ang[0])

    ax_r.plot(ring, [1.0] * len(ring), color=COLORS["baseline"], lw=1.1,
              ls=(0, (4.0, 2.2)), zorder=4, label="target cabin")
    ax_r.plot(ring, np.append(meas, meas[0]), color=COLORS["accent"], lw=1.7,
              zorder=6, label="measured")
    ax_r.fill(ring, np.append(meas, meas[0]), color=COLORS["accent"],
              alpha=0.16, zorder=5)
    ax_r.scatter(ang, meas, s=18, color=COLORS["accent"], zorder=7,
                 edgecolor="white", linewidth=0.7)
    ax_r.set_theta_offset(np.pi / 2.0)
    ax_r.set_theta_direction(-1)
    ax_r.set_rlabel_position(180.0)
    ax_r.set_xticks(ang)
    ax_r.set_xticklabels(labels, fontsize=7.4)
    ax_r.set_ylim(0.0, max(max(meas), 1.0) * 1.36)
    ax_r.set_yticks([0.5, 1.0, 1.5])
    ax_r.set_yticklabels(["0.5", "1.0", "1.5"], fontsize=6.5)
    ax_r.tick_params(axis="x", pad=7.0)
    ax_r.grid(lw=0.5, alpha=0.6)
    ax_r.set_title("Ratio to target (-)", fontsize=8.5, fontweight="bold",
                   pad=14)
    ax_r.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=2,
                fontsize=7.0, frameon=False)
    for a, m, u in zip(ang, meas, units):
        halo_text(ax_r, a, m * 0.78, u, ha="center", va="center",
                  fontsize=6.2, color=COLORS["sub"], halo="white",
                  halo_lw=2.2)
    ax_r.text(0.5, -0.32, "$PA = N_5\\,(1 + \\sqrt{w_S^2 + w_{FR}^2})$\n"
              f"$w_S$ = {w_s:.2f},   $w_{{FR}}$ = {w_fr:.2f}",
              transform=ax_r.transAxes, ha="center", va="top", fontsize=6.8,
              color=COLORS["sub"], linespacing=1.6)
    add_panel_label(ax_r, "(e)", x=-0.14, y=1.06)

    over = [labels[i] for i, m in enumerate(meas) if m > 1.0]
    savefig(fig, "420_acu_psychoacoustics.png")


if __name__ == "__main__":
    main()
