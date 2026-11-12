"""Electroglottogram, inverse-filtered glottal flow and its derivative.

Three synchronous views of one larynx, drawn for a modal and a breathy voice on
a single clock.  The flow is a Rosenberg pulse whose rise and fall are set
independently, the electroglottogram is the complement of the glottal area put
through a contact lag, and the derivative is a finite difference of the flow
array that is actually plotted - so the maximum flow declination rate marked on
the bottom trace is the minimum of the drawn curve and nothing else.  Every
quotient is a measurement on those arrays: the open quotient is the fraction of
the period during which the flow exceeds one percent of its peak, the speed
quotient is the ratio of the rising to the falling part of that interval, and
the contact quotient is the fraction of the period the electroglottogram spends
above a criterion level, which is why the second panel sweeps that criterion
instead of quoting a single number.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#8a8a8a"
HAIR = "#cccccc"

MODAL = "#2a7f7a"
MODAL_F = "#dcedeb"
BREATHY = "#b23a48"
BREATHY_F = "#f4dfe2"
AMBER = "#c0603f"
DEEP = "#134a76"

SEED = 1046
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

FS = 40000.0
F0 = 118.0
T0 = 1.0 / F0
N_CYC = 4


def rosenberg(t, t0, oq, sq):
    """One glottal-flow pulse on the phase axis ``t`` in [0, t0)."""
    top = oq * t0
    tp = top * sq / (1.0 + sq)
    tn = top - tp
    g = np.zeros_like(t)
    a = t < tp
    g[a] = 0.5 * (1.0 - np.cos(np.pi * t[a] / tp))
    b = (t >= tp) & (t < top)
    g[b] = np.cos(0.5 * np.pi * (t[b] - tp) / tn)
    return g


def smooth(x, n):
    k = np.hanning(n)
    k /= k.sum()
    return np.convolve(np.r_[x[-n:], x, x[:n]], k, mode="same")[n:-n]


def source(rng, oq, sq, leak, n_cyc=N_CYC, jitter=0.0, oq_sd=0.0, sq_sd=0.0):
    """Flow, its derivative and the electroglottogram over ``n_cyc`` cycles."""
    flow, egg, marks = [], [], [0]
    for _ in range(n_cyc):
        t0 = T0 * (1.0 + rng.normal(0.0, jitter))
        oq_c = float(np.clip(oq * (1.0 + rng.normal(0.0, oq_sd)), 0.15, 0.96))
        sq_c = max(sq * (1.0 + rng.normal(0.0, sq_sd)), 0.4)
        n = int(round(t0 * FS))
        t = np.arange(n) / FS
        g = rosenberg(t, t0, oq_c, sq_c)
        flow.append(g + leak)
        # Vocal-fold contact is the complement of the opening, smoothed: the
        # electroglottogram never has the corners the area function does, and
        # an incompletely closing glottis loses contact depth as well.
        c = 1.0 - np.clip(g / (g.max() + 1e-12), 0.0, 1.0)
        egg.append(smooth(c, max(int(0.055 * n), 3)) * (1.0 - 1.90 * leak))
        marks.append(marks[-1] + n)
    flow = np.concatenate(flow)
    egg = np.concatenate(egg)
    # A trace of smoothing before differentiating: an ideal Rosenberg closure
    # is a step in slope, and its raw difference is a spike no inverse filter
    # would ever deliver.
    flow = smooth(flow, int(0.030 * T0 * FS) | 1)
    t = np.arange(len(flow)) / FS
    dflow = np.gradient(flow, 1.0 / FS)
    return dict(t=t, flow=flow, dflow=dflow, egg=egg,
                marks=np.array(marks[:-1]), n=np.diff(marks))


def quotients(src):
    """Open and speed quotient per cycle, straight off the flow array."""
    oq, sq = [], []
    for m, n in zip(src["marks"], src["n"]):
        g = src["flow"][m:m + n]
        g = g - g.min()
        on = g > 0.01 * g.max()
        if not on.any():
            continue
        i0, i1 = int(np.argmax(on)), int(n - 1 - np.argmax(on[::-1]))
        ip = i0 + int(np.argmax(g[i0:i1 + 1]))
        oq.append((i1 - i0 + 1) / n)
        sq.append(max(ip - i0, 1) / max(i1 - ip, 1))
    return np.array(oq), np.array(sq)


def contact_quotient(src, crit):
    """Fraction of each period the electroglottogram spends above ``crit``."""
    out = []
    for m, n in zip(src["marks"], src["n"]):
        e = src["egg"][m:m + n]
        lvl = e.min() + crit * (e.max() - e.min())
        out.append((e > lvl).sum() / n)
    return float(np.mean(out))


def main_panel(ax, mo, br, qm, qb):
    rows = (("electroglottogram\n(relative contact)", 2.15, "egg", 1.0),
            ("glottal flow\n(inverse filtered)", 0.0, "flow", 1.0),
            ("flow derivative\n(dU/dt)", -2.25, "dflow", 1.0))
    t_ms = mo["t"] * 1e3

    m0, m1 = mo["marks"][1], mo["marks"][1] + mo["n"][1]
    g = mo["flow"][m0:m1] - mo["flow"][m0:m1].min()
    on = g > 0.01 * g.max()
    o0 = m0 + int(np.argmax(on))
    o1 = m0 + int(len(on) - 1 - np.argmax(on[::-1]))
    ax.axvspan(t_ms[o0], t_ms[o1], color="#eef4f3", zorder=1)
    ax.text(0.5 * (t_ms[o0] + t_ms[o1]), 3.62, "open phase", fontsize=6.3,
            color=MODAL, ha="center", va="bottom", fontweight="bold",
            zorder=9)
    for xm in (t_ms[m0], t_ms[m1 - 1]):
        ax.axvline(xm, color=GREY, lw=0.7, ls=(0, (2.0, 2.0)), zorder=2)
    ax.annotate("", xy=(t_ms[m1 - 1], 3.42), xytext=(t_ms[m0], 3.42),
                arrowprops=dict(arrowstyle="<->", color=SUB, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=9)
    ax.text(t_ms[m0] - 0.22, 3.42, "T0 = %.2f ms" % (1e3 / F0), fontsize=6.3,
            color=SUB, ha="right", va="center", zorder=9)

    # One scale per row, taken from the modal trace and reused for the
    # breathy one: autoscaling each voice would hide the very loss of
    # amplitude the figure exists to show.
    scale = {key: float(np.abs(mo[key]).max()) for _, _, key, _ in rows}
    for src, col, lw in ((mo, MODAL, 1.6), (br, BREATHY, 1.5)):
        for name, off, key, gain in rows:
            y = src[key].astype(float) / (scale[key] + 1e-12)
            ax.plot(src["t"] * 1e3, y * 0.86 * gain + off, color=col, lw=lw,
                    zorder=6)

    for name, off, key, _ in rows:
        ax.text(-0.30, off, name, fontsize=6.8, color=INK, ha="right",
                va="center", linespacing=1.32, zorder=9)
        ax.axhline(off, color=HAIR, lw=0.6, zorder=2)

    sc_d = float(np.abs(mo["dflow"]).max())
    i = m0 + int(np.argmin(mo["dflow"][m0:m1]))
    yv = mo["dflow"] / sc_d * 0.86 - 2.25
    ax.plot([t_ms[i]], [yv[i]], marker="o", ms=4.4, mfc="white", mec=MODAL,
            mew=1.3, zorder=10)
    ax.annotate("maximum flow declination rate %.0f (a.u.) / s, modal"
                % mo["dflow"][m0:m1].min(), xy=(t_ms[i], yv[i]),
                xytext=(t_ms[i] + 0.55, -4.18), fontsize=6.3, color=MODAL,
                ha="left", va="bottom", zorder=10,
                arrowprops=dict(arrowstyle="-", color=MODAL, lw=0.8,
                                shrinkA=3, shrinkB=3))
    j = m0 + int(np.argmin(br["dflow"][m0:m1]))
    yb = br["dflow"][j] / sc_d * 0.86 - 2.25
    ax.plot([t_ms[j]], [yb], marker="o", ms=4.0, mfc="white", mec=BREATHY,
            mew=1.2, zorder=10)
    ax.text(t_ms[j] + 0.22, yb - 0.10, "breathy only %.0f (a.u.) / s"
            % br["dflow"][m0:m1].min(), fontsize=6.3, color=BREATHY,
            ha="left", va="top", zorder=10)

    ax.text(t_ms[-1] - 0.12, 1.32, "the breathy source never closes: its flow "
            "floor stays\nabove zero, its contact trace loses depth and its "
            "derivative\nnever reaches the modal minimum", fontsize=6.3,
            color=SUB, ha="right", va="bottom", linespacing=1.36, zorder=9)

    handles = [
        Line2D([], [], color=MODAL, lw=1.6, label="modal voice: OQ %.2f, "
               "SQ %.2f" % (qm[0].mean(), qm[1].mean())),
        Line2D([], [], color=BREATHY, lw=1.5, label="breathy voice: OQ %.2f, "
               "SQ %.2f" % (qb[0].mean(), qb[1].mean())),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=6.4,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.95, handlelength=2.0, borderaxespad=0.4,
                    labelspacing=0.30)
    leg.set_zorder(12)

    ax.set_xlim(-0.05, t_ms[-1] + 0.05)
    ax.set_ylim(-4.30, 3.95)
    ax.set_yticks([])
    ax.set_xlabel("time (ms), %d cycles at F0 = %.0f Hz" % (N_CYC, F0))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", lw=0.5, alpha=0.30)
    ax.set_title("One larynx, three synchronous views, two phonation types",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)


def criterion_panel(ax, mo, br):
    crit = np.linspace(0.15, 0.65, 60)
    for src, col, tag in ((mo, MODAL, "modal"), (br, BREATHY, "breathy")):
        cq = np.array([contact_quotient(src, c) for c in crit])
        ax.plot(crit * 100.0, cq * 100.0, color=col, lw=1.6, zorder=6,
                label=tag)
        k = int(np.argmin(np.abs(crit - 0.25)))
        ax.plot([25.0], [cq[k] * 100.0], marker="o", ms=4.0, mfc="white",
                mec=col, mew=1.2, zorder=8)
        up = col is MODAL
        ax.text(26.5, cq[k] * 100.0 + (3.4 if up else -12.0),
                "CQ25 = %.0f %%" % (cq[k] * 100.0), fontsize=6.1, color=col,
                ha="left", va="bottom" if up else "top", zorder=9)
    ax.axvline(25.0, color=GREY, lw=0.8, ls=(0, (3.0, 2.2)), zorder=4)
    ax.text(62.0, 12.0, "the gap between the two voices\nsurvives any "
            "criterion, so the\nchoice of level is not what\ncarries the "
            "result", fontsize=6.0, color=SUB, ha="right", va="bottom",
            linespacing=1.34, zorder=9)
    leg = ax.legend(loc="upper right", fontsize=6.1, frameon=True,
                    facecolor="white", edgecolor="#dddddd", framealpha=0.95,
                    handlelength=1.7, borderaxespad=0.35, labelspacing=0.26)
    leg.set_zorder(10)
    ax.set_xlim(14.0, 66.0)
    ax.set_ylim(0.0, 92.0)
    ax.set_xlabel("contact criterion, percentage of EGG range (%)",
                  fontsize=6.6)
    ax.set_ylabel("contact quotient (%)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Contact quotient against the criterion", fontsize=8.0,
                 fontweight="bold", color=INK)


def cycle_panel(ax, series):
    for (tag, col, oq) in series:
        k = np.arange(1, len(oq) + 1)
        ax.plot(k, oq * 100.0, color=col, lw=1.1, marker="o", ms=2.4,
                mfc="white", mec=col, mew=0.7, zorder=6)
        m, s = oq.mean() * 100.0, oq.std(ddof=1) * 100.0
        ax.plot([0, len(oq)], [m, m], color=col, lw=0.8, ls=(0, (4.0, 2.2)),
                zorder=4)
        ax.fill_between([0, len(oq)], m - s, m + s, color=col, alpha=0.12,
                        zorder=2)
        ax.text(len(oq) + 1.0, m, "%.0f\n+/- %.1f %%" % (m, s), fontsize=6.0,
                color=col, ha="left", va="center", linespacing=1.3, zorder=9)
        up = col is MODAL
        ax.text(1.0, m + (s + 2.2) * (1 if up else -1), tag, fontsize=6.2,
                color=col, ha="left", va="bottom" if up else "top",
                fontweight="bold", zorder=9)
    ax.set_xlim(0.0, len(series[0][2]) + 7.5)
    ax.set_ylim(30.0, 100.0)
    ax.set_xlabel("cycle index (n = %d consecutive periods)"
                  % len(series[0][2]), fontsize=6.6)
    ax.set_ylabel("open quotient (%)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Open quotient, cycle by cycle", fontsize=8.0,
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
    rng = np.random.default_rng(SEED)

    mo = source(rng, oq=0.55, sq=2.10, leak=0.00)
    br = source(rng, oq=0.86, sq=1.15, leak=0.20)
    qm, qb = quotients(mo), quotients(br)

    long_m = source(rng, 0.55, 2.10, 0.00, n_cyc=36, jitter=0.010,
                    oq_sd=0.045, sq_sd=0.07)
    long_b = source(rng, 0.86, 1.15, 0.20, n_cyc=36, jitter=0.022,
                    oq_sd=0.065, sq_sd=0.12)
    oq_m = quotients(long_m)[0]
    oq_b = quotients(long_b)[0]

    cq_m = contact_quotient(mo, 0.25)
    cq_b = contact_quotient(br, 0.25)
    mfdr_m = float(mo["dflow"].min())
    mfdr_b = float(br["dflow"].min())

    assert qb[0].mean() > qm[0].mean() + 0.20, \
        "the breathy source must stay open for longer"
    assert cq_m > cq_b + 0.10, "modal contact must exceed breathy contact"
    assert mfdr_m < mfdr_b, \
        "the modal derivative must reach the deeper negative peak"
    assert br["flow"].min() > 0.10, "the breathy source must not close"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_cq = axes_inches(fig, AUX_T)
    ax_cy = axes_inches(fig, AUX_B)

    main_panel(ax_main, mo, br, qm, qb)
    criterion_panel(ax_cq, mo, br)
    cycle_panel(ax_cy, [("modal", MODAL, oq_m), ("breathy", BREATHY, oq_b)])



    panel_labels_inches(fig, [ax_main, ax_cq, ax_cy], dx=-0.62, dy=0.14)
    savefig(fig, "1046_vox_glottal_inverse_filter.png")


if __name__ == "__main__":
    main()
