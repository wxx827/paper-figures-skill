"""A generator trip integrated from the swing equation, then taken apart.

One 1100 MW infeed is lost from a 10 GVA system and the frequency is stepped
forward with 2H df/dt = P_governor + P_ffr - P_loss - D df, so the nadir, the
rate of change and the settling point in the main panel are all read out of
the same trajectory rather than typed in.  The upper side panel is that
balance rearranged: at every instant the inertial, governor, battery and load
damping contributions add up to the lost infeed, which the script asserts.
The lower panel re-runs the whole integration across a range of system
inertia, with and without the fast frequency response.
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
TEAL = "#2a7f7a"
SLATE = "#4a4f5c"
TAN = "#a89078"
PURPLE = "#7b5ea7"

SBASE = 10000.0          # MVA
F0 = 50.0                # Hz
P_LOSS = 1000.0          # MW of infeed lost
H_SYS = 4.5              # s, so 45 GWs of stored kinetic energy
DROOP = 0.05             # governor droop, p.u.
T_GOV = 6.0              # governor plus turbine time constant, s
DAMP = 2.5               # load damping, p.u. MW per p.u. Hz
FFR_CAP = 400.0          # MW of battery fast frequency response
FFR_DELAY, FFR_RISE = 0.35, 0.45      # s
UFLS = 49.2              # Hz, first under-frequency load shedding stage
T_EVENT = 2.0
T_END = 40.0
DT = 0.002

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.02, 4.62, 4.48)
AUX_T = (7.44, 3.61, 2.68, 1.89)
AUX_B = (7.44, 1.02, 2.68, 1.82)


def simulate(h=H_SYS, ffr=True, p_loss=P_LOSS):
    """Explicit Euler on the swing equation; everything returned in MW."""
    n = int(round(T_END / DT)) + 1
    t = np.arange(n) * DT
    df = np.zeros(n)              # frequency deviation, p.u.
    gov = np.zeros(n)             # governor output, p.u.
    acc = np.zeros(n)             # d(df)/dt, p.u. per second
    fr = np.zeros(n)
    loss = np.where(t >= T_EVENT, p_loss / SBASE, 0.0)

    for k in range(n):
        if ffr:
            since = t[k] - T_EVENT - FFR_DELAY
            ramp = float(np.clip(since / FFR_RISE, 0.0, 1.0))
            fr[k] = ramp * min(FFR_CAP / SBASE, max(-df[k], 0.0) * 60.0)
        acc[k] = (gov[k] + fr[k] - loss[k] - DAMP * df[k]) / (2.0 * h)
        if k + 1 < n:
            df[k + 1] = df[k] + DT * acc[k]
            gov[k + 1] = gov[k] + DT * (-df[k] / DROOP - gov[k]) / T_GOV

    parts = dict(inertia=-2.0 * h * acc * SBASE,
                 governor=gov * SBASE,
                 ffr=fr * SBASE,
                 damping=-DAMP * df * SBASE)
    return t, F0 * (1.0 + df), parts, loss * SBASE


def metrics(t, f):
    i = int(np.argmin(f))
    w = int(round(0.5 / DT))
    j = int(round(T_EVENT / DT))
    rocof = (f[j + w] - f[j]) / 0.5
    settle = float(np.mean(f[-int(round(5.0 / DT)):]))
    return dict(nadir=float(f[i]), t_nadir=float(t[i] - T_EVENT),
                rocof=float(rocof), settle=settle)


def main_panel(ax, t, f_on, f_off, m_on, m_off):
    ax.axhspan(UFLS - 0.6, UFLS, color="#f4e3e5", zorder=0)
    ax.axhline(UFLS, color=RED, lw=1.1, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(0.4, UFLS - 0.012, "49.2 Hz, first load-shedding stage",
            ha="left", va="top", fontsize=6.4, color=RED, zorder=7)
    ax.axhline(F0, color="#999999", lw=0.8, zorder=3)
    ax.axvline(T_EVENT, color=SUB, lw=0.9, ls=(0, (3.0, 2.0)), zorder=3)
    ax.text(T_EVENT - 0.25, 50.055, "trip of\n{:.0f} MW".format(P_LOSS),
            ha="right", va="top", fontsize=6.8, color=INK, fontweight="bold",
            linespacing=1.30, zorder=8)

    ax.plot(t, f_off, color=ORANGE, lw=1.7, zorder=6)
    ax.plot(t, f_on, color=BLUE, lw=2.0, zorder=7)

    for f, m, col, tag, dy in ((f_off, m_off, ORANGE, "no battery", -0.012),
                               (f_on, m_on, BLUE,
                                "with {:.0f} MW FFR".format(FFR_CAP),
                                -0.075)):
        ti = T_EVENT + m["t_nadir"]
        ax.plot([ti], [m["nadir"]], marker="o", ms=5.6, mfc=col, mec="white",
                mew=1.0, zorder=9)
        ax.annotate("{:s}\nnadir {:.2f} Hz at {:.1f} s"
                    .format(tag, m["nadir"], m["t_nadir"]), xy=(ti, m["nadir"]),
                    xytext=(ti + 2.6, m["nadir"] + dy), fontsize=6.6,
                    color=col, ha="left", va="top", linespacing=1.32,
                    fontweight="bold", zorder=10,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                    shrinkA=2, shrinkB=3))

    r = m_off["rocof"]
    x0, x1 = T_EVENT, T_EVENT + 1.6
    ax.plot([x0, x1], [F0, F0 + r * (x1 - x0)], color=SLATE, lw=1.1,
            ls=(0, (2.5, 1.6)), zorder=8)
    ax.annotate("initial RoCoF {:.2f} Hz/s, measured\nover the first 500 ms"
                .format(r), xy=(x0 + 0.6, F0 + r * 0.6), xytext=(5.8, 50.062),
                ha="left", va="top", fontsize=6.5, color=SLATE,
                linespacing=1.32, zorder=9,
                arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.7,
                                shrinkA=2, shrinkB=3))

    ax.annotate("", xy=(T_END - 3.0, m_on["settle"]),
                xytext=(T_END - 3.0, F0),
                arrowprops=dict(arrowstyle="<|-|>", color=TEAL, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    ax.text(T_END - 3.4, 0.5 * (F0 + m_on["settle"]),
            "droop settling\n{:.0f} mHz".format(
                1000.0 * (F0 - m_on["settle"])), ha="right", va="center",
            fontsize=6.4, color=TEAL, linespacing=1.32, zorder=9)

    handles = [Line2D([], [], color=BLUE, lw=2.0,
                      label="inertia 45 GWs, with {:.0f} MW of fast response"
                            .format(FFR_CAP)),
               Line2D([], [], color=ORANGE, lw=1.7,
                      label="inertia 45 GWs, governors only"),
               Line2D([], [], color=SLATE, lw=1.1, ls=(0, (2.5, 1.6)),
                      label="initial rate of change of frequency"),
               Patch(facecolor="#f4e3e5", edgecolor="none",
                     label="load-shedding band")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.4, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.94,
              handlelength=1.8, borderaxespad=0.35, labelspacing=0.28)

    ax.set_xlim(0.0, T_END)
    ax.set_ylim(49.10, 50.07)
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("system frequency (Hz)")
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Frequency after the loss of a {:.0f} MW infeed"
                 .format(P_LOSS), fontsize=9.0, fontweight="bold", color=INK,
                 pad=6.0)


def balance_panel(ax, t, parts, loss):
    keys = [("inertia", SLATE, "inertial release"),
            ("ffr", PURPLE, "battery FFR"),
            ("governor", TAN, "governor response"),
            ("damping", TEAL, "load damping")]
    base = np.zeros_like(t)
    for key, col, lab in keys:
        top = base + parts[key]
        ax.fill_between(t, base, top, facecolor=col, edgecolor="none",
                        alpha=0.92, zorder=3, label=lab)
        base = top
    ax.plot(t, loss, color=RED, lw=1.3, zorder=6, label="lost infeed")

    ax.set_xlim(T_EVENT - 0.5, T_EVENT + 18.0)
    ax.set_ylim(-90.0, P_LOSS * 1.42)
    ax.set_xlabel("time (seconds)", fontsize=6.8)
    ax.set_ylabel("contribution (MW)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.2, ncol=2, columnspacing=0.8,
              borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Where the lost megawatts come from", fontsize=8.0,
                 fontweight="bold", color=INK)


def sweep_panel(ax):
    energies = np.linspace(20.0, 70.0, 21)
    out = {}
    for ffr in (True, False):
        vals = []
        for e in energies:
            t, f, _p, _l = simulate(h=e / (SBASE / 1000.0), ffr=ffr)
            vals.append(metrics(t, f)["nadir"])
        out[ffr] = np.array(vals)

    lo = min(out[False].min(), out[True].min())
    hi = max(out[False].max(), out[True].max())
    ax.axhline(UFLS, color=RED, lw=1.1, ls=(0, (4.5, 2.4)), zorder=4)
    ax.text(21.0, UFLS + 0.012, "load-shedding threshold", ha="left",
            va="bottom", fontsize=6.1, color=RED, zorder=6)
    ax.plot(energies, out[False], color=ORANGE, lw=1.6, marker="o", ms=2.8,
            zorder=6, label="governors only")
    ax.plot(energies, out[True], color=BLUE, lw=1.8, marker="o", ms=2.8,
            zorder=7, label="with {:.0f} MW FFR".format(FFR_CAP))
    ax.fill_between(energies, out[False], out[True], color="#dbe4ef",
                    alpha=0.75, zorder=3)

    span = hi - lo
    e_now = H_SYS * SBASE / 1000.0
    ax.axvline(e_now, color=SUB, lw=0.9, ls=(0, (3.0, 2.0)), zorder=5)
    ax.text(e_now + 1.0, lo - 0.06 * span, "today {:.0f} GWs".format(e_now),
            ha="left", va="bottom", fontsize=6.1, color=SUB, zorder=6)

    cross = np.interp(UFLS, out[False], energies)
    ax.plot([cross], [UFLS], marker="o", ms=4.6, mfc=ORANGE, mec="white",
            mew=0.9, zorder=8)
    ax.annotate("shedding below\n{:.0f} GWs".format(cross), xy=(cross, UFLS),
                xytext=(cross + 2.0, lo - 0.09 * span), fontsize=6.1,
                color=ORANGE, ha="left", va="bottom", zorder=9,
                linespacing=1.30,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7,
                                shrinkA=2, shrinkB=3))

    ax.set_xlim(energies[0] - 2.0, energies[-1] + 2.0)
    ax.set_ylim(lo - 0.16 * span, hi + 0.10 * span)
    ax.set_xlabel("system inertia (GW s)", fontsize=6.8)
    ax.set_ylabel("frequency nadir (Hz)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.4, borderaxespad=0.25, labelspacing=0.24)
    ax.set_title("Nadir against system inertia", fontsize=8.0,
                 fontweight="bold", color=INK)
    return cross


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
    t, f_on, parts, loss = simulate(ffr=True)
    _t2, f_off, _p2, _l2 = simulate(ffr=False)
    m_on, m_off = metrics(t, f_on), metrics(t, f_off)

    total = sum(parts[k] for k in ("inertia", "governor", "ffr", "damping"))
    assert np.allclose(total, loss, atol=1e-6), "the balance must close"
    num = np.gradient(f_on / F0 - 1.0, DT)
    smooth = np.abs(t - T_EVENT) > 0.01        # skip the step discontinuity
    smooth[[0, -1]] = False
    assert np.max(np.abs(-2.0 * H_SYS * num * SBASE
                         - parts["inertia"])[smooth]) < 12.0, "swing mismatch"
    assert m_on["nadir"] > m_off["nadir"], "FFR must raise the nadir"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_bal = axes_inches(fig, AUX_T)
    ax_swp = axes_inches(fig, AUX_B)

    main_panel(ax_main, t, f_on, f_off, m_on, m_off)
    balance_panel(ax_bal, t, parts, loss)
    cross = sweep_panel(ax_swp)



    panel_labels_inches(fig, [ax_main, ax_bal, ax_swp], dx=-0.62, dy=0.14)
    savefig(fig, "969_pwr_frequency_event.png")


if __name__ == "__main__":
    main()
