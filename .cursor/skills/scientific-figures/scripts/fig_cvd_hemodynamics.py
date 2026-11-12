"""Left-ventricular haemodynamics: one cycle model seen four ways.

All four panels are rendered from a *single* cardiac-cycle generator.  The
pressure-volume loop, the pressure-time tracing, the aortic flow waveform and
the cardiac-output curve are therefore not four illustrations that happen to
agree - they are four projections of the same arrays, so the ejection phase
shaded in (b) is exactly the leftward limb of the loop in (a) and the flow in
(c) is its volume derivative.

Preload is the only thing varied between the three coloured conditions:
end-systolic volume is pinned by the contractility line (ESPVR) and the
afterload, so a larger end-diastolic volume buys stroke volume directly -
the Frank-Starling mechanism, drawn rather than asserted.  Every quoted
number (EDV, ESV, SV, EF, peak velocity, VTI, peak cardiac output) is measured
from the plotted arrays.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches, shared_limits

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GRIDC = "#e2e2e2"
ACCENT = "#b23a48"          # the single emphasis object: operating point
BAND_IVC = "#e8e2ef"
BAND_EJ = "#dfe8dd"

# Baseline first (house blue), then the two comparison loading states.
PRELOAD = [(112.0, "#3b5b92", "preload \u2212"),
           (128.0, "#2e7d5b", "baseline"),
           (146.0, "#c9a227", "preload +")]

EES, V0_ES = 2.70, 18.0     # ESPVR: end-systolic elastance (mmHg/mL)
A_ED, B_ED, V0_ED = 0.62, 0.0245, 14.0   # EDPVR stiffness
P_END = 96.0                # end-systolic (aortic) pressure, mmHg
P_OPEN = 80.0               # aortic valve opening pressure, mmHg
AO_AREA = 3.6               # cm^2, for velocity conversion
HR0 = 72.0

FRAC = {"ivc": 0.07, "eject": 0.30, "ivr": 0.08, "fill": 0.55}
N = {"ivc": 60, "eject": 260, "ivr": 60, "fill": 320}


def edpvr(v):
    return A_ED * (np.exp(B_ED * (v - V0_ED)) - 1.0)


def cycle(edv, period):
    """One cardiac cycle starting at end-diastole; returns t, V, P and phases.

    Phases run isovolumic contraction -> ejection -> isovolumic relaxation ->
    filling, which is also the order the loop is traversed anticlockwise.
    """
    esv = P_END / EES + V0_ES
    sv = edv - esv
    p_ed = edpvr(edv)
    p_peak = 104.0 + 0.30 * sv

    t, v, p, marks = [], [], [], {}
    t0 = 0.0
    for name in ("ivc", "eject", "ivr", "fill"):
        dur = FRAC[name] * period
        s = np.linspace(0.0, 1.0, N[name])
        marks[name] = (t0, t0 + dur)
        if name == "ivc":
            vv = np.full_like(s, edv)
            pp = p_ed + (P_OPEN - p_ed) * s ** 1.8
        elif name == "eject":
            # Skewed flow pulse first, volume by integration: this is what
            # gives panel (c) a physiological acceleration time instead of an
            # instantaneous peak at valve opening.
            q = s ** 0.85 * (1.0 - s) ** 1.9
            cum = np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1]))))
            vv = edv - sv * cum / cum[-1]
            pp = ((1.0 - s) * P_OPEN + s * P_END +
                  (p_peak - P_OPEN) * np.sin(np.pi * s ** 0.85))
        elif name == "ivr":
            vv = np.full_like(s, esv)
            pp = P_END * (1.0 - s) ** 2.1 + edpvr(esv) * s ** 2.1
        else:
            vv = esv + sv * (1.0 - (1.0 - s) ** 0.55)
            pp = edpvr(vv) * (0.80 + 0.20 * s)
        t.append(t0 + dur * s)
        v.append(vv)
        p.append(pp)
        t0 += dur
    t, v, p = (np.concatenate(a) for a in (t, v, p))
    # Neighbouring phases share their boundary sample; a repeated timestamp
    # makes np.gradient divide by zero when panel (c) differentiates volume.
    keep = np.concatenate(([True], np.diff(t) > 1e-9))
    return t[keep], v[keep], p[keep], marks, esv, sv, p_ed


def aortic_pressure(t, p_lv, marks, period):
    """Aortic tracing: tracks the LV during ejection, then runs off."""
    t_open, t_close = marks["eject"]
    p_ao = np.empty_like(t)
    ej = (t >= t_open) & (t <= t_close)
    p_ao[ej] = p_lv[ej] - 3.0
    tail = ~ej
    dt = np.where(t[tail] >= t_close, t[tail] - t_close,
                  t[tail] + period - t_close)
    p_notch = p_lv[ej][-1] - 3.0
    p_ao[tail] = P_OPEN + (p_notch - P_OPEN) * np.exp(-dt / 0.26)
    p_ao += 5.5 * np.exp(-0.5 * ((t - t_close - 0.020) / 0.013) ** 2)
    return p_ao


def main():
    set_pub_style()
    period = 60.0 / HR0

    runs = []
    for edv, colour, label in PRELOAD:
        t, v, p, marks, esv, sv, p_ed = cycle(edv, period)
        runs.append(dict(edv=edv, colour=colour, label=label, t=t, v=v, p=p,
                         marks=marks, esv=esv, sv=sv, p_ed=p_ed,
                         ef=sv / edv * 100.0))
    base = runs[1]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4))
    (ax_pv, ax_pt), (ax_q, ax_co) = axes

    # Panels (a) and (b) show the same physical quantity, so one ruler - with
    # head-room above the loops reserved for the legend and phase captions.
    p_lim = (0.0, shared_limits(
        [r["p"] for r in runs] +
        [aortic_pressure(base["t"], base["p"], base["marks"], period)],
        pad=0.12)[1] * 1.14)

    # ---- (a) pressure-volume loops ----------------------------------------
    vgrid = np.linspace(30.0, 165.0, 200)
    ax_pv.plot(vgrid, edpvr(vgrid), color="#9a9a9a", lw=1.0, ls=(0, (4, 2.4)),
               zorder=2)
    ves = np.linspace(V0_ES, 66.0, 60)
    ax_pv.plot(ves, EES * (ves - V0_ES), color="#9a9a9a", lw=1.0,
               ls=(0, (4, 2.4)), zorder=2)
    ax_pv.text(160.0, edpvr(160.0) + 3.0, "EDPVR", fontsize=6.4, color=MUTED,
               ha="right", va="bottom")
    ax_pv.text(41.0, EES * (41.0 - V0_ES) - 2.0, "ESPVR", fontsize=6.4,
               color=MUTED, ha="left", va="top", rotation=54)

    for r in runs:
        ax_pv.plot(np.append(r["v"], r["v"][0]), np.append(r["p"], r["p"][0]),
                   color=r["colour"], lw=1.5, zorder=4,
                   label=(f"{r['label']}:  EDV {r['edv']:.0f} mL,  "
                          f"SV {r['sv']:.0f} mL,  EF {r['ef']:.0f}%"))
    ax_pv.plot([base["esv"]], [P_END], marker="o", ms=4.0, mfc=ACCENT,
               mec="white", mew=0.7, zorder=6)
    ax_pv.plot([base["edv"]], [base["p_ed"]], marker="o", ms=4.0, mfc=ACCENT,
               mec="white", mew=0.7, zorder=6)
    ax_pv.annotate(r"$V_{ES}$" + f" = {base['esv']:.0f} mL",
                   xy=(base["esv"], P_END), xytext=(33.0, 74.0), ha="left",
                   fontsize=6.8, color=INK,
                   arrowprops=dict(arrowstyle="-", lw=0.7, color="#9a9a9a"))
    ax_pv.annotate(r"$V_{ED}$" + f" = {base['edv']:.0f} mL",
                   xy=(base["edv"], base["p_ed"]),
                   xytext=(base["edv"] - 8, 34), fontsize=6.8, color=INK,
                   arrowprops=dict(arrowstyle="-", lw=0.7, color="#9a9a9a"))
    y_sv = 16.0
    ax_pv.add_patch(FancyArrowPatch((base["esv"], y_sv), (base["edv"], y_sv),
                                    arrowstyle="<|-|>", mutation_scale=7,
                                    lw=0.9, color=INK, shrinkA=0, shrinkB=0,
                                    zorder=6))
    ax_pv.text((base["esv"] + base["edv"]) / 2.0, y_sv + 3.0,
               f"stroke volume {base['sv']:.0f} mL", ha="center", va="bottom",
               fontsize=6.8, color=INK)
    ax_pv.set_xlabel("left-ventricular volume (mL)", fontsize=8.0)
    ax_pv.set_ylabel("left-ventricular pressure (mmHg)", fontsize=8.0)
    ax_pv.set_xlim(30.0, 168.0)
    ax_pv.set_ylim(*p_lim)
    ax_pv.set_title("Pressure\u2013volume loops across three preloads",
                    fontsize=8.8, fontweight="bold", loc="left")
    ax_pv.legend(loc="upper left", frameon=False, fontsize=6.6,
                 handlelength=1.4, borderaxespad=0.2, labelspacing=0.30)
    ax_pv.grid(color=GRIDC, lw=0.5)
    ax_pv.tick_params(labelsize=7.0)

    # ---- (b) pressure tracings --------------------------------------------
    t2 = np.concatenate([base["t"], base["t"] + period])
    lv2 = np.concatenate([base["p"], base["p"]])
    ao = aortic_pressure(base["t"], base["p"], base["marks"], period)
    ao2 = np.concatenate([ao, ao])
    for k in range(2):
        for name, colour in (("ivc", BAND_IVC), ("eject", BAND_EJ)):
            a, b = base["marks"][name]
            ax_pt.axvspan(a + k * period, b + k * period, color=colour, lw=0,
                          zorder=0)
    ax_pt.plot(t2, lv2, color="#3b5b92", lw=1.5, zorder=4,
               label="left ventricle")
    ax_pt.plot(t2, ao2, color="#b23a48", lw=1.5, zorder=3, label="aorta")
    a, b = base["marks"]["ivc"]
    ax_pt.annotate("isovolumic contraction", xy=((a + b) / 2.0 + period,
                                                 p_lim[1] * 0.80),
                   xytext=((a + b) / 2.0 + period - 0.10, p_lim[1] * 0.955),
                   ha="right", va="center", fontsize=6.4, color=SUB,
                   arrowprops=dict(arrowstyle="->", lw=0.7, color="#9a9a9a"))
    for k in range(2):
        a, b = base["marks"]["eject"]
        ax_pt.text((a + b) / 2.0 + k * period, p_lim[1] * 0.955, "ejection",
                   ha="center", va="center", fontsize=6.6, color=SUB)
    t_close = base["marks"]["eject"][1]
    ax_pt.annotate("dicrotic notch", xy=(t_close + 0.02, ao[
        np.argmin(np.abs(base["t"] - (t_close + 0.02)))]),
        xytext=(t_close + 0.22, p_lim[1] * 0.70), fontsize=6.4, color=SUB,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#9a9a9a"))
    ax_pt.set_xlabel("time (s)", fontsize=8.0)
    ax_pt.set_ylabel("pressure (mmHg)", fontsize=8.0)
    ax_pt.set_xlim(0.0, 2 * period)
    ax_pt.set_ylim(*p_lim)
    ax_pt.set_title("Ventricular and aortic pressure over two cycles",
                    fontsize=8.8, fontweight="bold", loc="left")
    ax_pt.legend(loc="upper right", frameon=False, fontsize=7.0,
                 handlelength=1.4, borderaxespad=0.2)
    ax_pt.grid(axis="y", color=GRIDC, lw=0.5)
    ax_pt.tick_params(labelsize=7.0)

    # ---- (c) aortic flow velocity ------------------------------------------
    dv = -np.gradient(base["v"], base["t"])          # mL/s leaving the LV
    dv[dv < 0.0] = 0.0
    vel = dv / AO_AREA                                # cm/s
    vel2 = np.concatenate([vel, vel])
    vti = np.trapezoid(vel, base["t"])                # cm per beat
    ax_q.fill_between(t2, 0.0, vel2, color="#8fc0d9", alpha=0.55, lw=0,
                      zorder=2)
    ax_q.plot(t2, vel2, color="#2e74a8", lw=1.4, zorder=3)
    i_pk = int(np.argmax(vel))
    ax_q.plot([base["t"][i_pk]], [vel[i_pk]], marker="o", ms=4.0, mfc=ACCENT,
              mec="white", mew=0.7, zorder=5)
    ax_q.annotate(f"peak {vel[i_pk]:.0f} cm/s",
                  xy=(base["t"][i_pk], vel[i_pk]),
                  xytext=(base["t"][i_pk] + 0.16, vel[i_pk] * 0.97),
                  fontsize=6.8, color=INK, va="center",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="#9a9a9a"))
    ax_q.text(0.985, 0.94,
              f"VTI = {vti:.1f} cm per beat\nacceleration time = "
              f"{(base['t'][i_pk] - base['marks']['eject'][0]) * 1000:.0f} ms",
              transform=ax_q.transAxes, ha="right", va="top", fontsize=6.6,
              color=SUB, linespacing=1.4)
    ax_q.set_xlabel("time (s)", fontsize=8.0)
    ax_q.set_ylabel("aortic flow velocity (cm/s)", fontsize=8.0)
    ax_q.set_xlim(0.0, 2 * period)
    ax_q.set_ylim(0.0, vel.max() * 1.28)
    ax_q.set_title("Aortic velocity is the volume derivative of the loop",
                   fontsize=8.8, fontweight="bold", loc="left")
    ax_q.grid(axis="y", color=GRIDC, lw=0.5)
    ax_q.tick_params(labelsize=7.0)

    # ---- (d) cardiac output vs heart rate ----------------------------------
    hr = np.linspace(40.0, 190.0, 260)
    for r in runs:
        # Systole shortens with rate, but far more slowly than the cycle does,
        # so diastole - and with it stroke volume - is what gets squeezed out.
        t_sys = 0.10 + 0.20 * np.sqrt(60.0 / hr)
        fill_time = 60.0 / hr - t_sys
        sv_hr = r["sv"] * (1.0 - np.exp(-fill_time / 0.16))
        co = hr * sv_hr / 1000.0
        ax_co.plot(hr, co, color=r["colour"], lw=1.5, zorder=3,
                   label=r["label"])
        if r is base:
            i_max = int(np.argmax(co))
            ax_co.plot([hr[i_max]], [co[i_max]], marker="o", ms=4.4,
                       mfc=ACCENT, mec="white", mew=0.7, zorder=6)
            ax_co.annotate(f"peak {co[i_max]:.1f} L/min at "
                           f"{hr[i_max]:.0f} bpm",
                           xy=(hr[i_max], co[i_max]),
                           xytext=(hr[i_max] + 6, co[i_max] - 1.9),
                           fontsize=6.8, color=INK,
                           arrowprops=dict(arrowstyle="->", lw=0.7,
                                           color="#9a9a9a"))
            co_rest = np.interp(HR0, hr, co)
            ax_co.plot([HR0], [co_rest], marker="s", ms=4.0, mfc="white",
                       mec=INK, mew=0.9, zorder=6)
            ax_co.annotate(f"resting {HR0:.0f} bpm, "
                           f"{co_rest:.1f} L/min", xy=(HR0, co_rest),
                           xytext=(44, co_rest - 2.0), fontsize=6.8,
                           color=SUB, va="top",
                           arrowprops=dict(arrowstyle="->", lw=0.7,
                                           color="#9a9a9a"))
    ax_co.set_xlabel("heart rate (bpm)", fontsize=8.0)
    ax_co.set_ylabel(r"cardiac output $\dot{Q}$ (L/min)", fontsize=8.0)
    ax_co.set_xlim(40.0, 190.0)
    ax_co.set_ylim(0.0, None)
    ax_co.set_title("Output peaks, then filling time runs out",
                    fontsize=8.8, fontweight="bold", loc="left")
    ax_co.text(0.985, 0.05, "loading states and colours as in panel (a)",
               transform=ax_co.transAxes, ha="right", va="bottom",
               fontsize=6.4, color=MUTED)
    ax_co.grid(color=GRIDC, lw=0.5)
    ax_co.tick_params(labelsize=7.0)

    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.6, w_pad=2.6)
    panel_labels_inches(fig, [ax_pv, ax_pt, ax_q, ax_co], dx=-0.44, dy=0.06)
    savefig(fig, "548_cvd_hemodynamics.png")


if __name__ == "__main__":
    main()
