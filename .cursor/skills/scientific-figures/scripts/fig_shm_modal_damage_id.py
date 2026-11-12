"""Vibration-based damage identification on a 30 m simply supported girder.

Main panel: the first three bending modes of the healthy and the damaged
girder, drawn one above the other on their own offset baselines over the
undeformed axis, with the two stiffness-reduced regions ruled across every
mode.  Auxiliary (both at 58 % of the main panel width): the drop in natural
frequency mode by mode, and the mode-shape curvature difference along the
span, which is what actually localises the damage.

Nothing is typed in.  A two-node Euler-Bernoulli beam with consistent mass is
assembled over forty elements, the healthy and the damaged stiffness matrices
differ only in the flexural rigidity of five elements, and both eigenproblems
are solved with ``scipy.linalg.eigh``.  Frequencies are sqrt(lambda) / 2 pi,
curvature is the second difference of the plotted nodal displacements, and the
severity quoted at each peak is 1 - kappa_healthy / kappa_damaged, the local
moment-curvature reading of the same arrays.
"""
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.linalg import eigh

from pubstyle import set_pub_style, savefig
from figkit import axes_inches, panel_labels_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HEALTHY = "#2a7f7a"
DAMAGED = "#b23a48"
ORANGE = "#c0603f"
BLUE = "#3b5b92"

SPAN = 30.0
NEL = 40
EI0 = 3.25e10 * 0.35          # N m^2, concrete box girder
MASS = 3000.0                 # kg per metre
NMODES = 6
# element index slice -> fractional loss of flexural rigidity
DAMAGE = [((11, 14), 0.35), ((26, 28), 0.20)]

FIG_W, FIG_H = 11.0, 6.5
MAIN = (0.88, 1.14, 5.46, 4.34)
AUX_T = (7.64, 3.62, 3.16, 1.86)
AUX_B = (7.64, 1.14, 3.16, 1.86)


def stiffness_factors():
    """Per-element EI multiplier for the healthy and the damaged girder."""
    healthy = np.ones(NEL)
    damaged = np.ones(NEL)
    for (a, b), loss in DAMAGE:
        damaged[a:b] = 1.0 - loss
    return healthy, damaged


def assemble(factors):
    """Global stiffness and consistent mass of the beam, in SI units."""
    le = SPAN / NEL
    ndof = 2 * (NEL + 1)
    kk = np.zeros((ndof, ndof))
    mm = np.zeros((ndof, ndof))
    ke = np.array([
        [12.0, 6.0 * le, -12.0, 6.0 * le],
        [6.0 * le, 4.0 * le ** 2, -6.0 * le, 2.0 * le ** 2],
        [-12.0, -6.0 * le, 12.0, -6.0 * le],
        [6.0 * le, 2.0 * le ** 2, -6.0 * le, 4.0 * le ** 2]]) / le ** 3
    me = np.array([
        [156.0, 22.0 * le, 54.0, -13.0 * le],
        [22.0 * le, 4.0 * le ** 2, 13.0 * le, -3.0 * le ** 2],
        [54.0, 13.0 * le, 156.0, -22.0 * le],
        [-13.0 * le, -3.0 * le ** 2, -22.0 * le, 4.0 * le ** 2]]) \
        * MASS * le / 420.0
    for e in range(NEL):
        d = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        kk[np.ix_(d, d)] += EI0 * factors[e] * ke
        mm[np.ix_(d, d)] += me
    return kk, mm


def solve_modes(factors):
    """Natural frequencies in Hz and unit-normalised nodal mode shapes."""
    kk, mm = assemble(factors)
    fixed = [0, 2 * NEL]
    free = np.array([i for i in range(2 * (NEL + 1)) if i not in fixed])
    lam, vec = eigh(kk[np.ix_(free, free)], mm[np.ix_(free, free)])
    freqs = np.sqrt(np.maximum(lam[:NMODES], 0.0)) / (2.0 * np.pi)
    shapes = np.zeros((NMODES, NEL + 1))
    rots = np.zeros((NMODES, NEL + 1))
    for n in range(NMODES):
        full = np.zeros(2 * (NEL + 1))
        full[free] = vec[:, n]
        v, th = full[0::2], full[1::2]
        lead = int(np.argmax(np.abs(v) > 0.45 * np.max(np.abs(v))))
        if v[lead] < 0.0:
            v, th = -v, -th
        # Displacements are rescaled for drawing; rotations are left in the
        # mass-normalised scale eigh returns, which is what makes healthy and
        # damaged curvatures comparable.
        shapes[n], rots[n] = v / np.max(np.abs(v)), th
    return freqs, shapes, rots


def curvature(rot):
    """Element-midpoint curvature, in 1/m.

    For a Hermite beam element the second derivative of the interpolation at
    mid-length reduces to the rotation difference over the element length, so
    curvature is read element by element and never straddles a stiffness
    discontinuity the way a nodal second difference does.
    """
    return np.diff(rot) / (SPAN / NEL)


def damage_index(rt_h, rt_d, nuse=4):
    """Per-element stiffness loss from the curvature ratio, as a fraction.

    Local flexural rigidity times curvature is the same modal moment before
    and after damage, so the least-squares curvature ratio over the first
    ``nuse`` modes estimates EI_damaged / EI_healthy element by element.
    """
    kh = np.array([curvature(rt_h[n]) for n in range(nuse)])
    kd = np.array([curvature(rt_d[n]) for n in range(nuse)])
    num = (kh ** 2).sum(axis=0)
    den = (np.abs(kh) * np.abs(kd)).sum(axis=0)
    return 1.0 - num / np.maximum(den, 1e-18)


def find_peaks(di, xm, thresh=0.05, guard=4):
    """Well-separated local maxima of the loss estimate."""
    out = []
    for i in np.argsort(di)[::-1]:
        if di[i] < thresh:
            break
        if any(abs(i - j) < guard for j, _p, _s in out):
            continue
        out.append((int(i), float(xm[i]), float(np.clip(di[i], 0.0, 0.95))))
    return sorted(out, key=lambda t: t[1])


def true_zones():
    """Damaged element slices as (x_start, x_end, loss) in metres."""
    le = SPAN / NEL
    return [(a * le, b * le, loss) for (a, b), loss in DAMAGE]


def main_panel(ax, x, freqs_h, freqs_d, sh_h, sh_d):
    step = 2.7
    for z0, z1, _loss in true_zones():
        ax.axvspan(z0, z1, color=DAMAGED, alpha=0.10, lw=0.0, zorder=1)
    amp = 5.0 * np.floor(0.75 / max(np.abs(sh_d[:3] - sh_h[:3]).max(), 1e-9)
                         / 5.0)
    ticks, labels = [], []
    for n in range(3):
        off = step * n
        ax.axhline(off, color="#bbbbbb", lw=0.8, ls=(0, (5, 3)), zorder=2)
        ax.plot([0.0, SPAN], [off, off], color="#999999", lw=1.4, zorder=2)
        for xs in (0.0, SPAN):
            ax.plot([xs], [off], marker="^", ms=5.0, mfc="#dddddd",
                    mec="#777777", mew=0.6, zorder=3)
        ax.plot(x, off + sh_h[n], color=HEALTHY, lw=1.7, zorder=5)
        ax.plot(x, off + sh_d[n], color=DAMAGED, lw=1.4,
                ls=(0, (4.0, 2.0)), zorder=6)
        ax.fill_between(x, off, off + amp * (sh_d[n] - sh_h[n]),
                        facecolor=ORANGE, alpha=0.35, lw=0.0, zorder=4)
        ax.plot(x, off + amp * (sh_d[n] - sh_h[n]), color=ORANGE, lw=0.9,
                zorder=5)
        drop = 100.0 * (freqs_h[n] - freqs_d[n]) / freqs_h[n]
        ax.text(0.45, off + 1.34, "Mode {:d}   {:.2f} Hz to {:.2f} Hz "
                "({:.1f} % drop)".format(n + 1, freqs_h[n], freqs_d[n], drop),
                fontsize=6.8, color=INK, ha="left", va="bottom", zorder=8)
        ticks += [off - 1.0, off, off + 1.0]
        labels += ["-1", "0", "1"]

    ax.set_xlim(-0.9, SPAN + 0.9)
    ax.set_ylim(-1.6, step * 2 + 2.15)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_xlabel("position along span (m)")
    ax.set_ylabel("normalised modal displacement (offset per mode)")
    ax.grid(axis="x", lw=0.5, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("First three bending modes, healthy against damaged girder",
                 fontsize=9.0, fontweight="bold", color=INK, pad=6.0)
    handles = [Line2D([], [], color=HEALTHY, lw=1.7, label="healthy"),
               Line2D([], [], color=DAMAGED, lw=1.4, ls=(0, (4.0, 2.0)),
                      label="damaged"),
               Patch(facecolor=ORANGE, alpha=0.35,
                     label="shape difference x {:.0f}".format(amp)),
               Patch(facecolor=DAMAGED, alpha=0.10,
                     label="stiffness-reduced zone")]
    ax.legend(handles=handles, loc="lower center", fontsize=6.4, ncol=4,
              frameon=True, facecolor="white", edgecolor="#dddddd",
              framealpha=0.94, handlelength=1.5, columnspacing=1.0,
              borderaxespad=0.25)


def freq_panel(ax, freqs_h, freqs_d):
    idx = np.arange(1, NMODES + 1)
    drop = 100.0 * (freqs_h - freqs_d) / freqs_h
    ax.bar(idx, drop, width=0.62, facecolor=DAMAGED, edgecolor="white",
           linewidth=0.6, zorder=3)
    for i, d in zip(idx, drop):
        ax.text(i, d + 0.05, "{:.2f}".format(d), ha="center", va="bottom",
                fontsize=6, color=SUB, zorder=5)
    ax.axhline(1.0, color=BLUE, lw=1.0, ls=(0, (3.5, 2.0)), zorder=4,
               label="1 % detectability floor")
    ax.set_xticks(idx)
    ax.set_xticklabels(["{:d}\n{:.2f} Hz".format(i, f)
                        for i, f in zip(idx, freqs_h)], fontsize=6)
    ax.set_xlabel("mode order and healthy frequency (Hz)", fontsize=6.8)
    ax.set_ylabel("frequency drop (%)", fontsize=6.8)
    ax.set_ylim(0.0, drop.max() * 1.32)
    ax.grid(axis="y", lw=0.5, alpha=0.40)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.0, frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.92, handlelength=1.3,
              borderaxespad=0.25)
    ax.set_title("Frequency shift is global, not local", fontsize=8.0,
                 fontweight="bold", color=INK)
    return drop


def locate_panel(ax, xm, di, peaks):
    le = SPAN / NEL
    truth = np.zeros(NEL)
    for (a, b), loss in DAMAGE:
        truth[a:b] = 100.0 * loss
    for z0, z1, _loss in true_zones():
        ax.axvspan(z0, z1, color=DAMAGED, alpha=0.12, lw=0.0, zorder=1)
    est = 100.0 * di
    ax.axhline(0.0, color="#999999", lw=0.7, zorder=3)
    ax.fill_between(xm, 0.0, est, step="mid", facecolor=ORANGE, alpha=0.32,
                    lw=0.0, zorder=2)
    ax.step(xm, est, where="mid", color=ORANGE, lw=1.2, zorder=4,
            label="estimated")
    ax.step(np.r_[0.0, xm + 0.5 * le], np.r_[truth[0], truth], where="pre",
            color=DAMAGED, lw=1.1, ls=(0, (3.5, 2.0)), zorder=5,
            label="imposed")
    for _i, xp, sev in peaks:
        left = xp > 0.5 * SPAN
        ax.annotate("{:.1f} m\n{:.0f} % loss".format(xp, 100.0 * sev),
                    xy=(xp, 100.0 * sev),
                    xytext=(xp - 1.4 if left else xp + 1.4,
                            100.0 * sev + 5.0),
                    fontsize=6, color=DAMAGED,
                    ha="right" if left else "left", va="bottom",
                    zorder=8, arrowprops=dict(arrowstyle="-", color=DAMAGED,
                                              lw=0.6, shrinkA=1, shrinkB=3))
    ax.set_xlim(-0.9, SPAN + 0.9)
    ax.set_ylim(-9.0, 52.0)
    ax.set_xlabel("position along span (m)", fontsize=6.8)
    ax.set_ylabel("stiffness loss (%)", fontsize=6.8)
    ax.grid(lw=0.5, alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=6.0, frameon=True,
              facecolor="white", edgecolor="#dddddd", framealpha=0.92,
              handlelength=1.3, borderaxespad=0.25, labelspacing=0.22)
    ax.set_title("Curvature ratio locates and sizes it", fontsize=8.0,
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
    np.random.default_rng(31415)          # fixed for reproducibility

    fac_h, fac_d = stiffness_factors()
    freqs_h, sh_h, rt_h = solve_modes(fac_h)
    freqs_d, sh_d, rt_d = solve_modes(fac_d)
    x = np.linspace(0.0, SPAN, NEL + 1)
    xm = (np.arange(NEL) + 0.5) * (SPAN / NEL)

    di = damage_index(rt_h, rt_d)
    peaks = find_peaks(di, xm)
    assert np.all(freqs_d < freqs_h), "damage must lower every frequency"

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_main = axes_inches(fig, MAIN)
    ax_freq = axes_inches(fig, AUX_T)
    ax_loc = axes_inches(fig, AUX_B)

    main_panel(ax_main, x, freqs_h, freqs_d, sh_h, sh_d)
    drop = freq_panel(ax_freq, freqs_h, freqs_d)
    locate_panel(ax_loc, xm, di, peaks)


    zones = true_zones()

    panel_labels_inches(fig, [ax_main, ax_freq, ax_loc], dx=-0.62, dy=0.14)
    savefig(fig, "984_shm_modal_damage_id.png")


if __name__ == "__main__":
    main()
