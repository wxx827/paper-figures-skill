"""Four circadian phase markers on a 24-hour clock face, with light and lag.

The main panel is a circular plot because circadian phase is a circular
quantity: 23:40 and 00:20 are forty minutes apart, and a linear axis would put
them at opposite ends of the page.  Each marker gets its own ring so the four
distributions never overlap, and each carries its individual observations, its
mean direction and a 95 % confidence arc; the Rayleigh statistics quoted in the
legend are computed from those same angles.  The two smaller panels supply what
the clock face cannot: how a light pulse moves the clock depending on when it
lands, and how many days the clock needs to catch up after a long flight.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"

PURPLE = "#8e6bbf"
BLUE = "#3b5b92"
GOLD = "#c9a227"
RED = "#b23a48"

N_SUBJ = 18
# (name, mean clock hour, concentration, ring radius, colour)
MARKERS = [("DLMO", 21.4, 14.0, 0.52, PURPLE),
           ("core temperature minimum", 4.7, 11.0, 0.68, BLUE),
           ("cortisol peak", 8.2, 9.0, 0.84, GOLD),
           ("activity midpoint (M10)", 14.3, 5.5, 1.00, RED)]
NIGHT = (22.0, 6.0)          # shaded biological night on the clock face


def h2t(h):
    return np.asarray(h, dtype=float) / 24.0 * 2.0 * np.pi


def rayleigh(a):
    """Mean direction, resultant length, p value and the 95 % half-width."""
    n = a.size
    c, s = float(np.cos(a).sum()), float(np.sin(a).sum())
    rr = float(np.hypot(c, s))
    r = rr / n
    z = rr * rr / n
    p = np.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * n) -
                      (24.0 * z - 132.0 * z ** 2 + 76.0 * z ** 3 -
                       9.0 * z ** 4) / (288.0 * n * n))
    mu = float(np.arctan2(s, c)) % (2.0 * np.pi)
    arg = np.sqrt(max(n * n - (n * n - rr * rr) * np.exp(3.841 / n), 0.0)) / rr
    ci = float(np.arccos(np.clip(arg, -1.0, 1.0)))
    return mu, r, float(min(max(p, 0.0), 1.0)), ci


def phase_samples(seed=21):
    rng = np.random.default_rng(seed)
    out = []
    for name, hour, kappa, ring, col in MARKERS:
        a = rng.vonmises(h2t(hour), kappa, N_SUBJ) % (2.0 * np.pi)
        out.append((name, a, ring, col) + rayleigh(a))
    return out


def prc_curve(seed=31):
    """Type-1 light PRC: dead zone, delay lobe, advance lobe."""
    rng = np.random.default_rng(seed)
    ct = np.linspace(0.0, 24.0, 481)
    shift = (-2.25 * np.exp(-0.5 * ((ct - 15.6) / 2.7) ** 2)
             + 1.55 * np.exp(-0.5 * ((ct - 21.8) / 2.3) ** 2)
             + 1.55 * np.exp(-0.5 * ((ct - 21.8 - 24.0) / 2.3) ** 2))
    ct_pts = np.arange(1.0, 24.0, 2.0)
    obs = np.interp(ct_pts, ct, shift) + rng.normal(0.0, 0.16, ct_pts.size)
    err = np.full(ct_pts.size, 0.0) + rng.uniform(0.14, 0.30, ct_pts.size)
    return ct, shift, ct_pts, obs, err


def zero_crossings(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    idx = np.flatnonzero(np.sign(ys[:-1]) != np.sign(ys[1:]))
    return [float(xs[i] - ys[i] * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))
            for i in idx]


def jetlag(seed=41):
    rng = np.random.default_rng(seed)
    d = np.arange(0.0, 15.0, 1.0)
    out = {}
    for name, tau, col in [("eastward, 8 h advance", 6.4, RED),
                           ("westward, 8 h delay", 4.1, BLUE)]:
        y = 8.0 * np.exp(-d / tau) + rng.normal(0.0, 0.16, d.size)
        y[0] = 8.0
        out[name] = (np.clip(y, 0.0, None), col)
    return d, out


def clock_panel(ax, groups):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    th = np.linspace(h2t(NIGHT[0]), h2t(NIGHT[1] + 24.0), 200)
    ax.fill_between(th, 0.0, 1.16, color="#eeeeee", linewidth=0.0, zorder=0)
    ax.text(h2t(2.0), 1.11, "biological night", ha="center", va="center",
            fontsize=6.4, color=MUTED, zorder=1)

    ring_th = np.linspace(0.0, 2.0 * np.pi, 361)
    for _, _, ring, _, _, _, _, _ in groups:
        ax.plot(ring_th, np.full_like(ring_th, ring), color="#d5d5d5",
                lw=0.6, zorder=1)

    handles = []
    for name, a, ring, col, mu, r, p, ci in groups:
        jitter = np.linspace(-0.030, 0.030, a.size)
        ax.scatter(a, ring + jitter, s=6.5, color=col, alpha=0.75,
                   edgecolors="none", zorder=4)
        ax.plot([mu, mu], [0.0, ring], color=col, lw=0.9, alpha=0.40,
                zorder=3)
        arc = np.linspace(mu - ci, mu + ci, 60)
        ax.plot(arc, np.full_like(arc, ring), color=col, lw=3.4, alpha=0.45,
                solid_capstyle="butt", zorder=5)
        ax.plot([mu, mu], [ring - 0.055, ring + 0.055], color=col, lw=2.0,
                zorder=6)
        hh = (mu / (2.0 * np.pi) * 24.0) % 24.0
        ptxt = "p < 0.001" if p < 0.001 else "p = %.3f" % p
        handles.append(Line2D([], [], color=col, lw=2.0, marker="o", ms=3.0,
                              label="%s  %02d:%02d   r = %.2f, %s"
                              % (name, int(hh), int(round((hh % 1) * 60)) % 60,
                                 r, ptxt)))

    ax.set_xticks(h2t(np.arange(0.0, 24.0, 2.0)))
    ax.set_xticklabels(["%02d:00" % h for h in range(0, 24, 2)], fontsize=7.0)
    ax.set_ylim(0.0, 1.16)
    ax.set_yticks([])
    ax.set_rlabel_position(0.0)
    ax.grid(color="#e2e2e2", lw=0.5)
    ax.spines["polar"].set_color("#cccccc")
    ax.spines["polar"].set_linewidth(0.8)
    ax.set_title("Four phase markers, one clock face", fontsize=9.0,
                 fontweight="bold", color=INK, pad=11)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=6.6, frameon=False, handlelength=1.5,
              handletextpad=0.5, columnspacing=1.4, labelspacing=0.35)


def prc_panel(ax, ct, shift, ct_pts, obs, err):
    ax.axhline(0.0, color="#808080", lw=0.8, zorder=1)
    ax.fill_between(ct, shift, 0.0, where=shift < 0.0, color=BLUE, alpha=0.18,
                    linewidth=0.0, zorder=2, interpolate=True)
    ax.fill_between(ct, shift, 0.0, where=shift > 0.0, color=RED, alpha=0.18,
                    linewidth=0.0, zorder=2, interpolate=True)
    ax.plot(ct, shift, color=RULE, lw=1.5, zorder=4)
    ax.errorbar(ct_pts, obs, yerr=err, fmt="o", ms=2.8, lw=0.0,
                elinewidth=0.7, capsize=1.6, color=INK, mfc="white",
                mew=0.8, zorder=5)

    cross = zero_crossings(ct, shift, 16.0, 24.0)
    if cross:
        xc = cross[0]
        ax.plot([xc], [0.0], marker="o", ms=3.6, color=INK, zorder=6)
        ax.annotate("delay-to-advance crossover\nCT %.1f h" % xc,
                    xy=(xc, 0.0), xytext=(xc - 0.6, 1.30), ha="right",
                    va="bottom", fontsize=6.4, color=INK,
                    arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7))
    ax.text(5.5, -1.65, "dead zone", ha="center", va="center", fontsize=6.6,
            color=MUTED)
    ax.text(15.6, -2.55, "delays", ha="center", va="bottom", fontsize=6.6,
            color=BLUE)
    ax.text(23.7, 0.12, "advances", ha="right", va="bottom", fontsize=6.6,
            color=RED)

    ax.set_xlim(0.0, 24.0)
    ax.set_ylim(-3.0, 2.4)
    ax.set_xticks(np.arange(0.0, 24.1, 4.0))
    ax.set_xlabel("circadian time of the light pulse (h)", fontsize=7.4)
    ax.set_ylabel("phase shift (h)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.set_title("Light phase-response curve", fontsize=8.2,
                 fontweight="bold", color=INK)


def jetlag_panel(ax, d, series):
    for name, (y, col) in series.items():
        rate = (y[0] - y[5]) / 5.0
        ax.plot(d, y, color=col, lw=1.4, marker="o", ms=2.8, mfc="white",
                mew=0.8, label="%s, %.2f h/day" % (name, rate))
        i = int(np.argmax(y <= 1.0)) if np.any(y <= 1.0) else len(d) - 1
        ax.plot([d[i]], [y[i]], marker="v", ms=4.0, color=col, zorder=5)
        ax.text(d[i], y[i] + 0.42, "%d d" % int(d[i]), ha="center",
                va="bottom", fontsize=6.3, color=col)
    ax.axhline(1.0, color="#808080", lw=0.8, ls=(0, (3, 2.2)))
    ax.text(0.2, 1.18, "re-entrained ( < 1 h )", ha="left", va="bottom",
            fontsize=6.2, color=MUTED)
    ax.set_xlim(0.0, 14.6)
    ax.set_ylim(0.0, 9.6)
    ax.set_xlabel("days after the flight (d)", fontsize=7.4)
    ax.set_ylabel("phase misalignment (h)", fontsize=7.4)
    ax.tick_params(labelsize=6.5)
    ax.grid(axis="y", lw=0.5, alpha=0.7)
    ax.legend(fontsize=6.4, frameon=False, loc="upper right",
              handlelength=1.3, borderaxespad=0.2)
    ax.set_title("Phase recovery after an eight-hour shift", fontsize=8.2,
                 fontweight="bold", color=INK)


def main():
    set_pub_style()
    groups = phase_samples()
    ct, shift, ct_pts, obs, err = prc_curve()
    d, lag = jetlag()

    fig = plt.figure(figsize=(10.4, 6.4))
    ax_main = axes_inches(fig, (1.00, 1.18, 4.30, 4.30), projection="polar")
    ax_top = axes_inches(fig, (6.40, 3.58, 3.10, 1.90))
    ax_bot = axes_inches(fig, (6.40, 0.98, 3.10, 1.90))

    clock_panel(ax_main, groups)
    prc_panel(ax_top, ct, shift, ct_pts, obs, err)
    jetlag_panel(ax_bot, d, lag)

    dlmo = groups[0]
    cbt = groups[1]
    gap = ((cbt[4] - dlmo[4]) % (2.0 * np.pi)) / (2.0 * np.pi) * 24.0
    note = ("Synthetic cohort of %d participants per marker, angles drawn from "
            "von Mises distributions; ring radius only separates the four "
            "markers and carries no magnitude. Thick arcs are 95 %% confidence "
            "intervals on the mean direction and the Rayleigh resultant r and "
            "p value are computed from the plotted angles. Circadian time 12 h "
            "is subjective dusk." % N_SUBJ)
    panel_labels_inches(fig, [ax_main, ax_top, ax_bot], dx=-0.50, dy=0.13)
    savefig(fig, "788_slp_circadian_phase.png")


if __name__ == "__main__":
    main()
