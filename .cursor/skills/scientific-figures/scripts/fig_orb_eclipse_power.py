"""Sunlight, eclipse and the battery over three orbits at a 25 deg beta angle.

The shadow is not a switch: the illuminated fraction is the overlap of the
solar disk with the Earth's disk as seen from the spacecraft, so umbra,
penumbra and full sun all fall out of one geometric function of the beta
angle and the argument of latitude.  Array output is that fraction times the
cosine of the incidence on a zenith-mounted panel, the battery is the running
integral of generation minus load, and both side panels re-run the same
integration across beta so the eclipse length, the average power and the
depth of discharge belong to the same model as the timeline.
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
BLUE = "#3b5b92"
DEEP = "#22406e"
TEAL = "#2a7f7a"
GOLD = "#c9a227"
SLATE = "#4a4f5c"
WARM = "#f3e2b3"
SHADE = "#d7dbe0"

MU = 398600.4418        # km3/s2
R_E = 6378.14           # km
H_SAT = 550.0           # km
BETA_DESIGN = 25.0      # deg between the sun line and the orbit plane
SUN_PHASE = 40.0        # deg, in-plane direction of the sun at the epoch
RHO_SUN = 0.2666        # deg, apparent solar radius at 1 AU

P_ARRAY = 420.0         # W at normal incidence
P_PAYLOAD = 190.0       # W while the imager runs
P_DOWNLINK = 70.0       # W while the transmitter is on
PAYLOAD_WIN = (20.0, 32.0)     # min after each orbit start
DOWNLINK_WIN = (68.0, 77.0)    # min after each orbit start
CAP_AH = 30.0           # A h of battery
BUS_V = 28.0            # V
ETA_CHG = 0.92
SOC0 = 92.0             # per cent at the start
DOD_LIMIT = 30.0        # per cent
N_ORBIT = 3
DT = 2.0                # s

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.92, 1.02, 4.58, 4.48)
AUX_T = (7.52, 3.63, 2.66, 1.87)
AUX_B = (7.52, 1.02, 2.66, 1.80)


def radius():
    return R_E + H_SAT


def period():
    return 2.0 * np.pi * np.sqrt(radius() ** 3 / MU)


def earth_angular_radius():
    return np.degrees(np.arcsin(R_E / radius()))


def beta_full_sun():
    """Beta angle (deg) above which the orbit never enters shadow."""
    return earth_angular_radius()


def lit_fraction(u_deg, beta_deg):
    """Fraction of the solar disk still visible, 0 in umbra and 1 in sun."""
    rho_e = earth_angular_radius()
    cos_th = -np.cos(np.radians(beta_deg)) * np.cos(
        np.radians(u_deg - SUN_PHASE))
    theta = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))
    out = np.ones_like(theta)
    out[theta <= rho_e - RHO_SUN] = 0.0
    edge = (theta > rho_e - RHO_SUN) & (theta < rho_e + RHO_SUN)
    if np.any(edge):
        d = np.radians(theta[edge])
        r1, r2 = np.radians(RHO_SUN), np.radians(rho_e)
        a1 = np.arccos(np.clip((d ** 2 + r1 ** 2 - r2 ** 2)
                               / (2.0 * d * r1), -1.0, 1.0))
        a2 = np.arccos(np.clip((d ** 2 + r2 ** 2 - r1 ** 2)
                               / (2.0 * d * r2), -1.0, 1.0))
        tri = 0.5 * np.sqrt(np.maximum(
            (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2),
            0.0))
        overlap = r1 ** 2 * a1 + r2 ** 2 * a2 - tri
        out[edge] = np.clip(1.0 - overlap / (np.pi * r1 ** 2), 0.0, 1.0)
    return out


def load_profile(t, p_bus):
    """Bus load (W) against time (s)."""
    per = period()
    phase = np.mod(t, per) / 60.0
    load = np.full_like(t, p_bus)
    load += np.where((phase >= PAYLOAD_WIN[0]) & (phase < PAYLOAD_WIN[1]),
                     P_PAYLOAD, 0.0)
    load += np.where((phase >= DOWNLINK_WIN[0]) & (phase < DOWNLINK_WIN[1]),
                     P_DOWNLINK, 0.0)
    return load


def orbit_balance(p_bus, beta_deg):
    """Net energy (W h) into the battery over one orbit at a given bus load."""
    per = period()
    t = np.arange(0.0, per, DT)
    nu = lit_fraction(np.degrees(2.0 * np.pi * t / per), beta_deg)
    gen = P_ARRAY * np.cos(np.radians(beta_deg)) * nu
    net = gen - load_profile(t, p_bus)
    return float(np.sum(np.where(net > 0.0, net * ETA_CHG, net))
                 * DT / 3600.0)


def solve_bus_power(beta_deg, lo=20.0, hi=400.0):
    """Housekeeping load that makes one orbit close on zero net energy."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if orbit_balance(mid, beta_deg) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(beta_deg, p_bus, n_orbit=N_ORBIT, soc0=SOC0):
    """Timeline of illumination, array power, load and battery state."""
    per = period()
    t = np.arange(0.0, n_orbit * per + DT, DT)
    u = np.degrees(2.0 * np.pi * t / per)
    nu = lit_fraction(u, beta_deg)
    gen = P_ARRAY * np.cos(np.radians(beta_deg)) * nu
    load = load_profile(t, p_bus)
    net = gen - load
    energy = np.where(net > 0.0, net * ETA_CHG, net)     # W into the battery
    cap_wh = CAP_AH * BUS_V
    soc = soc0 + np.cumsum(energy) * DT / 3600.0 / cap_wh * 100.0
    soc = np.clip(soc, 0.0, 100.0)
    return dict(t=t, u=u, nu=nu, gen=gen, load=load, soc=soc, per=per,
                cap_wh=cap_wh)


def eclipse_spans(t, nu, thresh=0.5):
    """Start and end times (s) of every shadow interval in the run."""
    dark = nu < thresh
    edges = np.diff(dark.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0] + 1)
    if dark[0]:
        starts = [0] + starts
    if dark[-1]:
        stops = stops + [dark.size - 1]
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, stops)]


def penumbra_seconds(beta_deg, p_bus):
    """Time (s) spent with the solar disk partly covered, per shadow entry."""
    sim = simulate(beta_deg, p_bus, n_orbit=1)
    partial = (sim["nu"] > 1e-6) & (sim["nu"] < 1.0 - 1e-6)
    return float(np.sum(partial) * DT / 2.0)


def beta_sweep(betas, p_bus):
    ecl, avg_p, dod = [], [], []
    for b in betas:
        sim = simulate(b, p_bus)
        spans = eclipse_spans(sim["t"], sim["nu"])
        one = np.mean([s[1] - s[0] for s in spans]) / 60.0 if spans else 0.0
        ecl.append(one)
        avg_p.append(float(np.mean(sim["gen"])))
        dod.append(100.0 - float(np.min(sim["soc"])))
    return np.array(ecl), np.array(avg_p), np.array(dod)


def timeline_panel(ax, sim, spans):
    t = sim["t"] / 60.0
    for k, (a, b) in enumerate(spans):
        ax.axvspan(a / 60.0, b / 60.0, color=SHADE, zorder=1,
                   label="eclipse" if k == 0 else None)
    ax.fill_between(t, 0.0, sim["gen"], color=WARM, edgecolor="none",
                    zorder=2)
    ax.plot(t, sim["gen"], color=GOLD, lw=1.5, zorder=6,
            label="array output")
    ax.plot(t, sim["load"], color=ORANGE, lw=1.3, zorder=6,
            label="bus load")
    ax.axhline(0.0, color="#9aa3ab", lw=0.7, zorder=3)

    top = float(np.max(sim["gen"]))
    ax.annotate("array flat at {:.0f} W: the single-axis yoke holds the "
                "incidence\nat the beta angle whenever the sun is up"
                .format(top), xy=(t[int(0.62 * t.size)], top),
                xytext=(t[-1] * 0.26, 128.0), fontsize=6.3, color="#8a6d10",
                ha="left", va="center", linespacing=1.32, zorder=9,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8,
                                shrinkA=2, shrinkB=3))
    ax.text(t[-1] * 0.30, np.max(sim["load"]) + 10.0,
            "imaging block {:.0f} W".format(np.max(sim["load"])),
            fontsize=6.2, color=ORANGE, ha="left", va="bottom")

    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(-12.0, 520.0)
    ax.set_xlabel("time from the start of the run (min)")
    ax.set_ylabel("power (W)")
    ax.tick_params(labelsize=6.8)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top",):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(t, sim["soc"], color=DEEP, lw=1.8, zorder=7,
             label="battery state of charge")
    lo = float(np.min(sim["soc"]))
    j = int(np.argmin(sim["soc"]))
    ax2.plot([t[j]], [lo], marker="o", ms=5.0, mfc=DEEP, mec="white",
             mew=0.9, zorder=8)
    ax2.annotate("lowest state of charge {:.1f} per cent,\n"
                 "{:.1f} A h drawn from {:.0f} A h"
                 .format(lo, (SOC0 - lo) / 100.0 * CAP_AH, CAP_AH),
                 xy=(t[j], lo), xytext=(t[j] + 11.0, 57.0), fontsize=6.3,
                 color=DEEP, ha="left", va="center", linespacing=1.32,
                 zorder=9, arrowprops=dict(arrowstyle="-", color=DEEP,
                                           lw=0.8, shrinkA=2, shrinkB=3))
    ax2.set_ylim(50.0, 101.0)
    ax2.set_ylabel("state of charge (%)", color=DEEP)
    ax2.tick_params(labelsize=6.8, colors=DEEP)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(DEEP)

    handles = [Line2D([], [], color=GOLD, lw=1.5, label="array output (W)"),
               Line2D([], [], color=ORANGE, lw=1.3, label="bus load (W)"),
               Line2D([], [], color=DEEP, lw=1.8,
                      label="state of charge (%)"),
               Patch(facecolor=SHADE, edgecolor="none",
                     label="umbra")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=6.4,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.94, handlelength=1.7, ncol=2,
                    columnspacing=1.0, borderaxespad=0.35, labelspacing=0.26)
    leg.set_zorder(11)
    ax.set_title("Three orbits at a beta angle of {:.0f} deg"
                 .format(BETA_DESIGN), fontsize=9.0, fontweight="bold",
                 color=INK, pad=6.0)


def sweep_panel(ax, betas, ecl, avg_p):
    ax.plot(betas, ecl, color=SLATE, lw=1.6, zorder=6)
    ax.fill_between(betas, 0.0, ecl, color="#e2e5e9", zorder=2)
    b_star = beta_full_sun()
    for s in (-1.0, 1.0):
        ax.axvline(s * b_star, color=GOLD, lw=1.0, ls=(0, (4.0, 2.2)),
                   zorder=5)
    ax.text(0.0, ecl.max() * 1.06, "no eclipse beyond {:.1f} deg"
            .format(b_star), fontsize=6.1, color="#8a6d10", ha="center",
            va="bottom")
    ax.plot([BETA_DESIGN], [np.interp(BETA_DESIGN, betas, ecl)], marker="o",
            ms=4.4, mfc=RED, mec="white", mew=0.8, zorder=8)
    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(0.0, ecl.max() * 1.30)
    ax.set_xticks(range(-90, 91, 30))
    ax.set_xlabel("beta angle (deg)", fontsize=6.8)
    ax.set_ylabel("eclipse per orbit (min)", fontsize=6.8, color=SLATE)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top",):
        ax.spines[side].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(betas, avg_p, color=GOLD, lw=1.4, ls=(0, (4.0, 2.0)), zorder=6)
    ax2.set_ylabel("orbit-average array power (W)", fontsize=6.8,
                   color="#8a6d10")
    ax2.tick_params(labelsize=6.4, colors="#8a6d10")
    ax2.set_ylim(0.0, avg_p.max() * 1.30)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GOLD)
    ax.set_title("Eclipse and harvest against beta", fontsize=8.0,
                 fontweight="bold", color=INK)


def dod_panel(ax, betas, dod):
    inside = dod <= DOD_LIMIT
    ax.fill_between(betas, 0.0, dod, where=inside, color="#dce7e2",
                    interpolate=True, zorder=2)
    ax.fill_between(betas, 0.0, dod, where=~inside, color="#f2dcdf",
                    interpolate=True, zorder=2)
    ax.plot(betas, dod, color=TEAL, lw=1.6, zorder=6)
    ax.axhline(DOD_LIMIT, color=RED, lw=1.1, ls=(0, (4.0, 2.2)), zorder=5)
    ax.text(-88.0, DOD_LIMIT + 0.6, "{:.0f} per cent cycling limit"
            .format(DOD_LIMIT), fontsize=6.1, color=RED, ha="left",
            va="bottom")
    here = float(np.interp(BETA_DESIGN, betas, dod))
    ax.plot([BETA_DESIGN], [here], marker="o", ms=4.6, mfc=RED, mec="white",
            mew=0.8, zorder=8)
    ax.annotate("design beta {:.0f} deg\ndraws {:.1f} per cent"
                .format(BETA_DESIGN, here), xy=(BETA_DESIGN, here),
                xytext=(BETA_DESIGN + 12.0, dod.max() * 0.82), fontsize=6.1,
                color=INK, ha="left", va="center", linespacing=1.3, zorder=9,
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xlim(-90.0, 90.0)
    ax.set_ylim(0.0, max(dod.max(), DOD_LIMIT) * 1.28)
    ax.set_xticks(range(-90, 91, 30))
    ax.set_xlabel("beta angle (deg)", fontsize=6.8)
    ax.set_ylabel("depth of discharge (%)", fontsize=6.8)
    ax.tick_params(labelsize=6.4)
    ax.grid(lw=0.5, alpha=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("How hard the battery is worked", fontsize=8.0,
                 fontweight="bold", color=INK)


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
    p_bus = solve_bus_power(BETA_DESIGN)
    sim = simulate(BETA_DESIGN, p_bus)
    spans = eclipse_spans(sim["t"], sim["nu"])
    ecl_min = np.mean([s[1] - s[0] for s in spans]) / 60.0
    pen_s = penumbra_seconds(BETA_DESIGN, p_bus)
    betas = np.arange(-88.0, 88.1, 4.0)
    ecl, avg_p, dod = beta_sweep(betas, p_bus)
    dod_here = 100.0 - float(np.min(sim["soc"]))

    assert len(spans) == N_ORBIT, "one shadow per orbit expected"
    assert 0.0 < ecl_min < sim["per"] / 60.0, "eclipse longer than the orbit"
    assert np.all(ecl[np.abs(betas) > beta_full_sun()] == 0.0), \
        "no eclipse is allowed above the threshold beta"
    assert 0.0 < dod_here < 100.0, "battery must stay inside its range"
    assert abs(np.max(sim["gen"])
               - P_ARRAY * np.cos(np.radians(BETA_DESIGN))) < 1.0, \
        "peak array power must be the beta cosine"
    assert abs(orbit_balance(p_bus, BETA_DESIGN)) < 1e-3, \
        "the design orbit must close on zero net energy"
    assert abs(sim["soc"][-1] - SOC0) < 0.5, "the cycle must repeat"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_time = axes_inches(fig, MAIN)
    ax_sweep = axes_inches(fig, AUX_T)
    ax_dod = axes_inches(fig, AUX_B)

    timeline_panel(ax_time, sim, spans)
    sweep_panel(ax_sweep, betas, ecl, avg_p)
    dod_panel(ax_dod, betas, dod)



    panel_labels_inches(fig, [ax_time, ax_sweep, ax_dod], dx=-0.66, dy=0.14)
    savefig(fig, "1024_orb_eclipse_power.png")


if __name__ == "__main__":
    main()
