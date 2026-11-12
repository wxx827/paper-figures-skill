"""Voice range profile: the region of the frequency-intensity plane a voice can reach.

The phonetogram is an *envelope* figure, so everything quoted about it is
geometry.  Each contour is built by closing an upper and a lower sound-pressure
limit against each other with a taper that forces them to meet at the ends of
the usable frequency range, which is what makes the outline a simple closed
polygon; its area is then the shoelace sum over the plotted vertices, taken in
semitone-decibel units because those are the ones that do not change when the
frequency axis is drawn logarithmically.  The dysphonic contour is a strict
subset of the normal one, and that containment is asserted vertex by vertex
before the figure is written, so the loss of area, of pitch range and of
dynamic range that the panels report cannot disagree with the drawing.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path

from pubstyle import set_pub_style, savefig

from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
GREY = "#8a8a8a"
HAIR = "#cccccc"

NORM = "#2a7f7a"
NORM_F = "#d7ebe9"
PATH = "#b23a48"
PATH_F = "#f2dade"
AMBER = "#c0603f"
DEEP = "#134a76"

SEED = 1045
FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.86, 1.02, 4.66, 4.46)
AUX_T = (7.42, 3.61, 2.70, 1.87)
AUX_B = (7.42, 1.02, 2.70, 1.82)

F_REF = 65.406            # C2, the semitone origin used throughout


def st2hz(st):
    return F_REF * 2.0 ** (np.asarray(st, dtype=float) / 12.0)


def hz2st(hz):
    return 12.0 * np.log2(np.asarray(hz, dtype=float) / F_REF)


def wiggle(rng, st, amp, n_comp=4):
    """Smooth low-order ripple so a contour is not a textbook parabola."""
    out = np.zeros_like(st)
    for k in range(1, n_comp + 1):
        out += (amp / k) * np.sin(2.0 * np.pi * k * (st - st[0])
                                  / (st[-1] - st[0]) + rng.uniform(0, 2 * np.pi))
    return out


def envelope(rng, st_lo, st_hi, soft, loud, taper=0.16, n=240):
    """Closed phonetogram contour as (semitone, dB SPL) vertices.

    ``soft`` and ``loud`` are callables giving the quiet and loud limits.  The
    two are pulled together by a raised-cosine taper at both ends, which is
    what closes the outline instead of leaving two open curves.
    """
    st = np.linspace(st_lo, st_hi, n)
    u = (st - st_lo) / (st_hi - st_lo)
    lo = soft(st) + wiggle(rng, st, 0.55)
    hi = loud(st) + wiggle(rng, st, 0.75)
    mid = 0.5 * (hi + lo)
    dyn = hi - lo
    w = np.ones_like(u)
    m = u < taper
    w[m] = 0.5 - 0.5 * np.cos(np.pi * u[m] / taper)
    m = u > 1.0 - taper
    w[m] = 0.5 - 0.5 * np.cos(np.pi * (1.0 - u[m]) / taper)
    hi_c, lo_c = mid + 0.5 * dyn * w, mid - 0.5 * dyn * w
    poly = np.column_stack([np.r_[st, st[::-1]], np.r_[hi_c, lo_c[::-1]]])
    return dict(st=st, hi=hi_c, lo=lo_c, poly=poly)


def shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def metrics(env):
    st, hi, lo = env["st"], env["hi"], env["lo"]
    live = (hi - lo) > 0.6
    return dict(area=shoelace(env["poly"]),
                st_range=float(st[live].max() - st[live].min()),
                f_lo=float(st2hz(st[live].min())),
                f_hi=float(st2hz(st[live].max())),
                spl_max=float(hi.max()), spl_min=float(lo.min()),
                dyn_max=float((hi - lo).max()),
                dyn_st=float(st[np.argmax(hi - lo)]))


def main_panel(ax, nrm, dys, mn, md):
    ax.add_patch(Rectangle((110.0, 57.0), 190.0 - 110.0, 78.0 - 57.0,
                           facecolor="#eeeeee", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((110.0, 57.0), 190.0 - 110.0, 78.0 - 57.0,
                           facecolor="none", edgecolor=GREY, lw=0.9,
                           ls=(0, (3.6, 2.2)), zorder=8))
    ax.text(110.0, 55.4, "habitual speaking range", fontsize=6.1, color=GREY,
            ha="left", va="top", zorder=9)

    for env, fc, ec, lw in ((nrm, NORM_F, NORM, 1.7), (dys, PATH_F, PATH, 1.7)):
        f = st2hz(env["poly"][:, 0])
        ax.fill(f, env["poly"][:, 1], facecolor=fc, edgecolor="none",
                alpha=0.85, zorder=3)
        ax.plot(np.r_[f, f[0]], np.r_[env["poly"][:, 1], env["poly"][0, 1]],
                color=ec, lw=lw, zorder=6)

    for env, col, tag, xt, yt in ((nrm, NORM, "normal", None, 2.6),
                                  (dys, PATH, "dysphonic", 540.0, 0.0)):
        i = int(np.argmax(env["hi"]))
        fx, fy = st2hz(env["st"][i]), env["hi"][i]
        ax.plot([fx], [fy], marker="o", ms=4.2, mfc="white", mec=col,
                mew=1.3, zorder=9)
        lab = "%s peak %.0f dB SPL" % (tag, fy)
        if xt is None:
            ax.text(fx * 1.05, fy + yt, lab, fontsize=6.3, color=col,
                    ha="left", va="bottom", fontweight="bold", zorder=10)
        else:
            ax.annotate(lab, xy=(fx, fy), xytext=(xt, fy + yt), fontsize=6.3,
                        color=col, ha="left", va="center", fontweight="bold",
                        zorder=10,
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                        shrinkA=2, shrinkB=4))

    for env, col, y in ((nrm, NORM, 38.5), (dys, PATH, 34.0)):
        m = mn if env is nrm else md
        ax.annotate("", xy=(m["f_hi"], y), xytext=(m["f_lo"], y),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.0,
                                    shrinkA=0, shrinkB=0), zorder=7)
        ax.text(np.sqrt(m["f_lo"] * m["f_hi"]), y + 0.9,
                "%.0f Hz to %.0f Hz, %.1f semitones"
                % (m["f_lo"], m["f_hi"], m["st_range"]), fontsize=6.2,
                color=col, ha="center", va="bottom", zorder=9)

    ax.text(560.0, 47.0, "the dysphonic profile is a strict\nsubset of the "
            "normal one: it loses\n%.0f %% of the area, %.1f semitones\nof "
            "pitch and %.0f dB of dynamic range"
            % (100.0 * (1.0 - md["area"] / mn["area"]),
               mn["st_range"] - md["st_range"],
               mn["dyn_max"] - md["dyn_max"]),
            fontsize=6.4, color=SUB, ha="left", va="bottom", linespacing=1.38,
            zorder=10)

    handles = [
        Line2D([], [], color=NORM, lw=1.7, label="normal voice, area %.0f "
               "semitone dB" % mn["area"]),
        Line2D([], [], color=PATH, lw=1.7, label="dysphonic voice, area %.0f "
               "semitone dB" % md["area"]),
        Line2D([], [], color=GREY, lw=0.9, ls=(0, (3.6, 2.2)),
               label="habitual speech reference band"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=6.3,
                    frameon=True, facecolor="white", edgecolor="#dddddd",
                    framealpha=0.95, handlelength=2.0, borderaxespad=0.4,
                    labelspacing=0.30)
    leg.set_zorder(12)

    ax.set_xscale("log")
    ax.set_xlim(72.0, 1180.0)
    ax.set_ylim(32.0, 122.0)
    ax.set_xticks([82.4, 110.0, 146.8, 196.0, 261.6, 349.2, 466.2, 622.3,
                   830.6, 1108.7])
    ax.set_xticklabels(["82", "110", "147", "196", "262", "349", "466", "622",
                        "831", "1109"])
    ax.minorticks_off()
    ax.set_xlabel("fundamental frequency (Hz), logarithmic")
    ax.set_ylabel("sound pressure level at 30 cm (dB SPL)")
    ax.grid(lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # A secondary_xaxis would inherit the parent's logarithmic *scale* and
    # relabel the semitone ticks as powers of ten, so the second ruler is a
    # twin axes carrying the same log mapping with hand-placed ticks.
    sec = ax.twiny()
    sec.set_xscale("log")
    sec.set_xlim(ax.get_xlim())
    st_ticks = [6, 12, 18, 24, 30, 36, 42, 48]
    sec.set_xticks(st2hz(st_ticks))
    sec.set_xticklabels([str(s) for s in st_ticks])
    sec.minorticks_off()
    sec.tick_params(labelsize=6.4)
    sec.set_xlabel("semitones above C2 (65.4 Hz)", fontsize=7.0)
    for side in ("left", "right", "bottom"):
        sec.spines[side].set_visible(False)

    ax.set_title("One closed contour per voice; the area inside it is the "
                 "phonatory capacity", fontsize=9.0, fontweight="bold",
                 color=INK, pad=22.0)


def dynamic_panel(ax, nrm, dys, mn, md):
    for env, col, m, tag in ((nrm, NORM, mn, "normal"),
                             (dys, PATH, md, "dysphonic")):
        f = st2hz(env["st"])
        ax.plot(f, env["hi"] - env["lo"], color=col, lw=1.6, zorder=6,
                label=tag)
        ax.plot([st2hz(m["dyn_st"])], [m["dyn_max"]], marker="o", ms=3.8,
                mfc="white", mec=col, mew=1.2, zorder=8)
    ax.axhline(15.0, color=GREY, lw=0.8, ls=(0, (3.2, 2.2)), zorder=4)
    ax.text(1130.0, 16.2, "15 dB working minimum", fontsize=6.0, color=GREY,
            ha="right", va="bottom", zorder=8)
    ax.text(500.0, 2.0, "widest at %.0f Hz: %.0f dB against %.0f dB"
            % (st2hz(mn["dyn_st"]), mn["dyn_max"], md["dyn_max"]),
            fontsize=6.1, color=SUB, ha="left", va="bottom", zorder=9)
    leg = ax.legend(loc="upper right", fontsize=6.1, frameon=True,
                    facecolor="white", edgecolor="#dddddd", framealpha=0.95,
                    handlelength=1.7, borderaxespad=0.35, labelspacing=0.26)
    leg.set_zorder(10)
    ax.set_xscale("log")
    ax.set_xlim(76.0, 1180.0)
    ax.set_ylim(0.0, 52.0)
    ax.set_xticks([100.0, 200.0, 400.0, 800.0])
    ax.set_xticklabels(["100", "200", "400", "800"])
    ax.minorticks_off()
    ax.set_xlabel("fundamental frequency (Hz)", fontsize=6.8)
    ax.set_ylabel("dynamic range (dB)", fontsize=6.6)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Height of the contour, frequency by frequency",
                 fontsize=8.0, fontweight="bold", color=INK)


def summary_panel(ax, mn, md):
    rows = [
        ("profile area", "%.0f", "semitone dB", mn["area"], md["area"]),
        ("pitch range", "%.1f", "semitones", mn["st_range"], md["st_range"]),
        ("dynamic range", "%.0f", "dB", mn["dyn_max"], md["dyn_max"]),
        ("highest F0", "%.0f", "Hz", mn["f_hi"], md["f_hi"]),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (name, fmt, unit, a, b) in zip(y, rows):
        ax.barh(yi + 0.19, 100.0, height=0.34, color=NORM_F, edgecolor=NORM,
                linewidth=0.9, zorder=5)
        ax.barh(yi - 0.19, 100.0 * b / a, height=0.34, color=PATH_F,
                edgecolor=PATH, linewidth=0.9, zorder=5)
        ax.text(101.5, yi + 0.19, (fmt + " %s") % (a, unit), fontsize=6.0,
                color=NORM, ha="left", va="center", zorder=8)
        ax.text(100.0 * b / a + 1.5, yi - 0.19, (fmt + " %s") % (b, unit),
                fontsize=6.0, color=PATH, ha="left", va="center", zorder=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_xlim(0.0, 152.0)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("dysphonic value as a percentage of the normal value (%)",
                  fontsize=6.6)
    ax.grid(axis="x", lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Every summary shrinks together", fontsize=8.0,
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

    nrm = envelope(rng, hz2st(82.0), hz2st(1050.0),
                   soft=lambda s: 47.0 + 0.42 * (s - 4.0),
                   loud=lambda s: 92.0 + 0.62 * s - 0.0125 * s ** 2)
    dys = envelope(rng, hz2st(126.0), hz2st(470.0),
                   soft=lambda s: 61.0 + 0.34 * (s - 12.0),
                   loud=lambda s: 84.0 + 0.30 * s - 0.0105 * s ** 2)

    mn, md = metrics(nrm), metrics(dys)

    inside = Path(nrm["poly"]).contains_points(dys["poly"])
    assert inside.all(), \
        "the dysphonic contour has to lie wholly inside the normal one"
    assert md["area"] < 0.5 * mn["area"], "the profile must lose real area"
    assert md["st_range"] < mn["st_range"] - 8.0, "pitch range must contract"
    assert md["dyn_max"] < mn["dyn_max"] - 8.0, "dynamic range must contract"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_dyn = axes_inches(fig, AUX_T)
    ax_sum = axes_inches(fig, AUX_B)

    main_panel(ax_main, nrm, dys, mn, md)
    dynamic_panel(ax_dyn, nrm, dys, mn, md)
    summary_panel(ax_sum, mn, md)



    panel_labels_inches(fig, [ax_main, ax_dyn, ax_sum], dx=-0.62, dy=0.30)
    savefig(fig, "1045_vox_phonetogram_vrp.png")


if __name__ == "__main__":
    main()
