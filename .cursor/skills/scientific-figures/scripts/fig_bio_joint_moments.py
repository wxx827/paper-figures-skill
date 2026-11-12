"""Knee kinetics through the gait cycle: moment, mechanical power and the
vertical ground reaction force on one shared cycle axis.

Power is not drawn independently - it is the product of the internal knee
moment and the joint angular velocity, so the sign of every shaded region
follows from the two panels above it.  Positive and negative work are the
time integrals of those same regions, printed in J/kg.  The dashed rules mark
the gait events and cross all three panels, which is the only way to read
"the absorption burst starts at opposite heel strike" off a figure.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, style_axes, COLORS
from figkit import shared_limits, panel_labels_inches

GEN = "#2e7d5b"          # house green  - energy generation (positive work)
ABS_ = "#8e6bbf"         # house purple - energy absorption (negative work)
EVENT = "#b23a48"        # house red    - the single emphasised event, toe-off

N_POINT = 401
STRIDE_S = 1.07          # stride time (s), sets the physical time axis
TOE_OFF = 62.0
EVENTS = [(10.0, "OTO"), (50.0, "OHS"), (TOE_OFF, "TO")]


def _bump(p, centre, width, amp):
    d = (p - centre + 50.0) % 100.0 - 50.0
    return amp * np.exp(-0.5 * (d / width) ** 2)


def _smooth_noise(rng, n, span=41):
    x = rng.normal(0.0, 1.0, n)
    k = np.exp(-0.5 * np.linspace(-3.0, 3.0, span) ** 2)
    k /= k.sum()
    pad = span // 2
    s = np.convolve(np.r_[x[-pad:], x, x[:pad]], k, mode="same")[pad:-pad]
    return (s - s.min()) / (np.ptp(s) + 1e-9)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build(rng):
    p = np.linspace(0.0, 100.0, N_POINT)
    t = p / 100.0 * STRIDE_S

    angle = 2.0 + _bump(p, 16.0, 8.0, 16.0) + _bump(p, 72.0, 12.0, 58.0)
    moment = (_bump(p, 15.0, 7.0, 0.62) - _bump(p, 42.0, 10.0, 0.30)
              + _bump(p, 57.0, 7.0, 0.14) - _bump(p, 88.0, 9.0, 0.16))

    # Extension is the positive direction for both moment and velocity, so
    # power = M_ext * omega_ext with omega_ext = -d(theta_flex)/dt.
    omega = -np.gradient(np.deg2rad(angle), t)
    power = moment * omega

    core = (96.0 + 16.0 * np.cos(2 * np.pi * (p - 25.0) / 26.0)
            - 5.0 * np.exp(-0.5 * ((p - 52.0) / 13.0) ** 2))
    window = _sigmoid((p - 3.0) / 0.8) * _sigmoid((59.0 - p) / 1.0)
    grf = np.clip(core * window, 0.0, None)

    sd = {"moment": 0.05 + 0.07 * _smooth_noise(rng, p.size),
          "power": 0.10 + 0.16 * _smooth_noise(rng, p.size),
          "grf": (2.5 + 4.5 * _smooth_noise(rng, p.size)) * (grf > 1.0)}
    return p, t, {"moment": moment, "power": power, "grf": grf}, sd


def main():
    set_pub_style()
    rng = np.random.default_rng(511)
    p, t, sig, sd = build(rng)

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(3, 1, hspace=0.26, left=0.105, right=0.975,
                          top=0.885, bottom=0.148)
    ax_m = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1], sharex=ax_m)
    ax_g = fig.add_subplot(gs[2], sharex=ax_m)

    for ax in (ax_m, ax_p, ax_g):
        ax.axvspan(0.0, TOE_OFF, facecolor="#f5f5f5", edgecolor="none",
                   zorder=0)
        ax.axhline(0.0, color=COLORS["baseline"], lw=0.7, ls=(0, (3.0, 2.4)),
                   zorder=1)
        for x, _ in EVENTS:
            ax.axvline(x, color=EVENT if x == TOE_OFF else COLORS["baseline"],
                       lw=0.9, ls=(0, (4.0, 2.4)),
                       alpha=1.0 if x == TOE_OFF else 0.9, zorder=2)
        ax.set_xlim(0.0, 100.0)
        ax.tick_params(labelsize=7.0)
        style_axes(ax)

    # ---- (a) internal knee moment ------------------------------------------
    m, s = sig["moment"], sd["moment"]
    ax_m.fill_between(p, m - s, m + s, color=COLORS["rule"], alpha=0.16, lw=0,
                      zorder=3)
    ax_m.plot(p, m, color=COLORS["rule"], lw=1.5, zorder=4)
    ax_m.set_ylim(*shared_limits([m - s, m + s], pad=0.16))
    ax_m.set_ylabel("Knee moment\n(N\u00b7m/kg)", fontsize=8.0,
                    linespacing=1.25)
    ax_m.set_title("Internal knee moment, extensor positive", fontsize=8.5,
                   fontweight="bold")
    ax_m.tick_params(labelbottom=False)
    i_peak = int(np.argmax(m))
    ax_m.annotate(f"weight-acceptance peak {m[i_peak]:.2f} N\u00b7m/kg",
                  xy=(p[i_peak], m[i_peak]), xytext=(p[i_peak] + 6.0,
                                                     m.max() * 1.02),
                  ha="left", va="bottom", fontsize=6.6, color=COLORS["ink"],
                  arrowprops=dict(arrowstyle="-", lw=0.7,
                                  color=COLORS["muted"]))

    # ---- (b) mechanical power, split into generation and absorption --------
    pw, s = sig["power"], sd["power"]
    ax_p.fill_between(p, 0.0, pw, where=pw >= 0.0, interpolate=True,
                      color=GEN, alpha=0.55, lw=0, zorder=3,
                      label="generation")
    ax_p.fill_between(p, 0.0, pw, where=pw <= 0.0, interpolate=True,
                      color=ABS_, alpha=0.45, lw=0, zorder=3,
                      label="absorption")
    ax_p.plot(p, pw, color=COLORS["rule"], lw=1.3, zorder=4)
    ax_p.set_ylim(*shared_limits([pw - s, pw + s], pad=0.22))
    ax_p.set_ylabel("Knee power\n(W/kg)", fontsize=8.0, linespacing=1.25)
    ax_p.set_title("Joint power = moment \u00d7 angular velocity",
                   fontsize=8.5, fontweight="bold")
    ax_p.tick_params(labelbottom=False)
    w_pos = float(np.trapezoid(np.clip(pw, 0.0, None), t))
    w_neg = float(np.trapezoid(np.clip(pw, None, 0.0), t))
    ax_p.text(0.012, 0.95, f"generation {w_pos:+.2f} J/kg",
              transform=ax_p.transAxes, ha="left", va="top", fontsize=6.6,
              color=GEN)
    ax_p.text(0.988, 0.06, f"absorption {w_neg:+.2f} J/kg",
              transform=ax_p.transAxes, ha="right", va="bottom", fontsize=6.6,
              color=ABS_)
    ax_p.text(TOE_OFF / 2.0, 0.95, "stance", transform=
              ax_p.get_xaxis_transform(), ha="center", va="top", fontsize=6.6,
              color=COLORS["muted"])
    ax_p.text((TOE_OFF + 100.0) / 2.0, 0.95, "swing", transform=
              ax_p.get_xaxis_transform(), ha="center", va="top", fontsize=6.6,
              color=COLORS["muted"])

    # ---- (c) vertical ground reaction force ---------------------------------
    g, s = sig["grf"], sd["grf"]
    ax_g.fill_between(p, g - s, g + s, color=COLORS["rule"], alpha=0.16, lw=0,
                      zorder=3)
    ax_g.plot(p, g, color=COLORS["rule"], lw=1.5, zorder=4)
    ax_g.axhline(100.0, color=COLORS["muted"], lw=0.7, ls=(0, (1.6, 1.8)),
                 zorder=2)
    ax_g.text(99.0, 100.0, "body weight", ha="right", va="bottom",
              fontsize=6.2, color=COLORS["muted"])
    ax_g.set_ylim(-20.0, shared_limits([g + s], pad=0.10)[1])
    ax_g.set_ylabel("Vertical GRF\n(%BW)", fontsize=8.0, linespacing=1.25)
    ax_g.set_xlabel("Gait cycle (%)", fontsize=8.0)
    ax_g.set_title("Vertical ground reaction force", fontsize=8.5,
                   fontweight="bold")
    stance = p <= TOE_OFF
    i1 = int(np.argmax(np.where(p < 35.0, g, -1.0)))
    i2 = int(np.argmax(np.where((p > 42.0) & stance, g, -1.0)))
    iv = int(np.argmin(np.where((p > 25.0) & (p < 48.0), g, 1e3)))
    for i, lab, dx in ((i1, "F1", 0.0), (iv, "valley", 0.0), (i2, "F2", 5.0)):
        ax_g.annotate(f"{lab} {g[i]:.0f}", xy=(p[i], g[i]),
                      xytext=(p[i] + dx,
                              g[i] + (9.0 if lab != "valley" else -14.0)),
                      ha="center", va="bottom" if lab != "valley" else "top",
                      fontsize=6.4, color=COLORS["ink"],
                      arrowprops=dict(arrowstyle="-", lw=0.6,
                                      color=COLORS["muted"]))

    # ---- gait events, labelled once along the bottom panel -----------------
    for x, lab in [(0.0, "HS")] + EVENTS:
        ax_g.text(x + 1.0, 0.015, lab, transform=ax_g.get_xaxis_transform(),
                  ha="left", va="bottom", fontsize=6.4,
                  color=EVENT if x == TOE_OFF else COLORS["sub"],
                  fontweight="bold" if x == TOE_OFF else "normal")

    panel_labels_inches(fig, [ax_m, ax_p, ax_g], dx=-0.60, dy=0.08)
    # Wrapped: a single-line footer wider than the canvas makes the
    # bbox_inches="tight" export expand instead of clip (DESIGN_SYSTEM 12.11).
    savefig(fig, "511_bio_joint_moments.png")


if __name__ == "__main__":
    main()
