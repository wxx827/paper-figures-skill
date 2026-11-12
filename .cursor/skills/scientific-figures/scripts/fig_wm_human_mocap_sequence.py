"""Figure 1163 - One walk, three readings: overlay, joint angles, speed.

There is exactly one piece of data here: ``MOTION``, a (240, 18, 3) array of
joint positions in metres.  Everything else is a projection of it.

* (a) eight of those frames, drawn light to dark, plus the ground-plane
  projection of each toe and of the pelvis path;
* (b) six flexion angles, measured from the same joint triples;
* (c) the walking speed, differentiated from the same pelvis track.

The gait itself is closed-loop: the speed profile is chosen first, distance
is its integral, and the stride phase is distance divided by a fixed stride
length.  So when the subject speeds up the legs cycle faster in (a) - the
overlay and the curve cannot disagree about when the acceleration happened.
Both right-hand panels carry the eight sample ticks, which is the explicit
cross-reference BRIEF_WM section 1.6 asks for.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from pubstyle import set_pub_style, savefig, style_axes, get_palette
from figkit import image_axes, background_card
from _wm_scenes import build, human_pose, project, BONES

H, W = 177, 344
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
DEEP, MID = "#20455f", "#2e74a8"

DZ = 3.50
PROJ = dict(f=1.30, cam_y=1.02, cam_z=-DZ, yaw=np.pi / 2, cx=0.090,
            cy=0.527, aspect=W / H)
STRIDE = 1.42                                  # metres per gait cycle

# ---- the one array everything else is derived from -------------------------
T = np.linspace(0.0, 3.20, 240)
SPEED = (0.80 + 0.85 / (1.0 + np.exp(-(T - 1.05) / 0.30))
         - 0.45 / (1.0 + np.exp(-(T - 2.55) / 0.22)))
DIST = np.concatenate([[0.0], np.cumsum(np.diff(T) * 0.5 *
                                        (SPEED[1:] + SPEED[:-1]))])
MOTION = np.stack([human_pose("walk", d / STRIDE, forward=d) for d in DIST])

KEY = np.linspace(6, len(T) - 7, 8).astype(int)


def flexion(a, b, c):
    """Flexion angle at joint ``b``, in degrees, for the whole sequence."""
    v1, v2 = MOTION[:, a] - MOTION[:, b], MOTION[:, c] - MOTION[:, b]
    cosv = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) *
                                      np.linalg.norm(v2, axis=1) + 1e-9)
    return 180.0 - np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))


# Only the leg joints are plotted.  The elbow angle of this skeleton is a
# fixed 31.5 deg for the whole take - the arms swing from the shoulder - so
# the two elbow series were flat, identical, and the dashed one sat exactly
# on top of the solid one: six entries in the legend, five lines on the page.
ANGLES = [("L knee", flexion(10, 11, 12), DEEP, "-"),
          ("R knee", flexion(14, 15, 16), DEEP, (0, (3.2, 1.8))),
          ("L hip", flexion(1, 10, 11), MID, "-"),
          ("R hip", flexion(1, 14, 15), MID, (0, (3.2, 1.8)))]
PELVIS_SPEED = np.gradient(MOTION[:, 0, 2], T)

# stance = the ankle is resting on the floor plane, read off the same array
ANKLE_Y = MOTION[:, [12, 16], 1]
STANCE = ANKLE_Y < (ANKLE_Y.min() + 0.020)


def runs(mask):
    """Contiguous True intervals of a boolean series, as (t0, t1) pairs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((T[i], T[j]))
            i = j + 1
        else:
            i += 1
    return out


STEPS = [runs(STANCE[:, 0]), runs(STANCE[:, 1])]
N_STEPS = sum(len(r) for r in STEPS)


def px(P3):
    p, _ = project(np.atleast_2d(P3), **PROJ)
    return p * np.array([W, H])


def main():
    set_pub_style(base_font=8)

    sc = build("studio", seed=6801, shape=(H, W), horizon=0.250,
               cam=dict(exposure=1.04, contrast=1.12, dof=1.3,
                        vignette=0.20))
    ramp = get_palette("cb_sequential_blue", 11)[3:]      # skip near-white

    fig = plt.figure(figsize=(10.4, 6.0))
    # The margins are set by what hangs outside the axes, not by the axes:
    # the (d) lane labels sit left of `left` and the (c) tick labels right of
    # `right`.  At the old 0.988 the latter landed outside the canvas,
    # `bbox_inches="tight"` widened the export to 10.75 in, and the delivered
    # h/w came out 0.546 instead of the 0.577 declared here.
    background_card(fig, (0.660, 0.052, 0.312, 0.845), facecolor="#f2f4f6",
                    radius=0.010, zorder=-5)
    gl = fig.add_gridspec(1, 1, left=0.026, right=0.640, top=0.845,
                          bottom=0.312)
    gd = fig.add_gridspec(1, 1, left=0.026, right=0.640, top=0.222,
                          bottom=0.115)
    gr = fig.add_gridspec(2, 1, left=0.712, right=0.958, top=0.845,
                          bottom=0.078, hspace=0.395)

    # ---- (a) the overlay --------------------------------------------------
    ax = fig.add_subplot(gl[0, 0])
    image_axes(ax, color="#9a9a9a", lw=0.6)
    ax.set_aspect("auto")
    ax.imshow(sc["rgb"], interpolation="bilinear", aspect="auto")

    path = px(np.stack([MOTION[:, 0, 0] * 0.0, np.zeros(len(T)),
                        MOTION[:, 0, 2]], axis=1))
    ax.plot(path[:, 0], path[:, 1], color="#e8eef3", lw=1.0, alpha=0.75,
            zorder=3)

    for k, i in enumerate(KEY):
        col = ramp[k]
        P = MOTION[i]
        for j in (13, 17):                       # toe -> ground projection
            g = px(np.array([P[j, 0], 0.0, P[j, 2]]))[0]
            ax.add_patch(Ellipse(g, 0.052 * W * 0.42, 0.021 * H,
                                 facecolor=col, edgecolor="#14202c",
                                 lw=0.4, alpha=0.75, zorder=4))
        pts = px(P)
        for (a, b, r) in BONES:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color="#14202c", lw=3.2 * (0.5 + 6 * r),
                    solid_capstyle="round", zorder=5 + k, alpha=0.55)
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color=col, lw=2.2 * (0.5 + 6 * r),
                    solid_capstyle="round", zorder=5 + k)
        ax.scatter(pts[:, 0], pts[:, 1], s=5.0, facecolors="#ffffff",
                   edgecolors="#14202c", linewidths=0.35, zorder=14 + k)
        ax.add_patch(Ellipse(pts[3], 0.030 * W, 0.055 * H, facecolor=col,
                             edgecolor="#14202c", lw=0.6, zorder=14 + k))
        ax.text(pts[3, 0], pts[3, 1] - 0.050 * H, f"{T[i]:.1f}", fontsize=6.2,
                color="#14202c", ha="center", va="bottom", zorder=20,
                bbox=dict(boxstyle="square,pad=0.16", facecolor="#ffffff",
                          edgecolor="none", alpha=0.78))

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.text(0.012, 0.965, "(a)   Eight frames of one 3.2 s walk, light to "
            "dark;  ovals are the toes projected onto the floor",
            transform=ax.transAxes, fontsize=7.4, color=INK, ha="left",
            va="top", zorder=22,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#ffffff",
                      edgecolor="#b9c6d3", lw=0.55, alpha=0.93))
    rx0, rw = 0.762 * W, 0.0225 * W
    for k, col in enumerate(ramp):
        ax.add_patch(plt.Rectangle((rx0 + k * rw, 0.048 * H), rw, 0.040 * H,
                                   facecolor=col, edgecolor="none",
                                   zorder=21))
    ax.add_patch(plt.Rectangle((rx0, 0.048 * H), len(ramp) * rw, 0.040 * H,
                               facecolor="none", edgecolor="#14202c",
                               lw=0.5, zorder=22))
    ax.text(rx0 + 0.5 * len(ramp) * rw, 0.038 * H,
            f"t = {T[KEY[0]]:.1f} s  \u2192  {T[KEY[-1]]:.1f} s",
            fontsize=6.4, color="#14202c", ha="center", va="bottom",
            zorder=22,
            bbox=dict(boxstyle="square,pad=0.16", facecolor="#ffffff",
                      edgecolor="none", alpha=0.82))

    # ---- (d) stance phases, read off the ankle heights --------------------
    axd = fig.add_subplot(gd[0, 0])
    # "left" / "right", not "left foot" / "right foot": the longer pair hung
    # 0.28 in off the left edge of the canvas, and the panel title above
    # already says these lanes are foot contacts
    for row, (name, col) in enumerate((("left", DEEP), ("right", MID))):
        for (t0, t1) in STEPS[row]:
            axd.add_patch(plt.Rectangle((t0, row - 0.30), max(t1 - t0, 0.012),
                                        0.60, facecolor=col,
                                        edgecolor="#14202c", lw=0.5,
                                        zorder=3))
        axd.text(-0.012, row, name, transform=axd.get_yaxis_transform(),
                 ha="right", va="center", fontsize=6.8, color=SUB)
    for i in KEY:
        axd.axvline(T[i], color="#c3ccd4", lw=0.6, zorder=1)
    axd.set_xlim(T[0], T[-1])
    axd.set_ylim(1.62, -0.62)
    axd.set_yticks([])
    axd.set_xlabel("time (s)", fontsize=7.4, labelpad=2.2)
    axd.tick_params(labelsize=6.8)
    axd.spines["left"].set_visible(False)
    style_axes(axd)
    axd.text(0.0, 1.34, f"(d)   stance phases:  {N_STEPS} foot contacts in "
             f"{T[-1] - T[0]:.1f} s, closing up as the speed rises",
             transform=axd.transAxes, fontsize=7.4, color=INK, ha="left",
             va="top")

    # ---- (b) joint angles -------------------------------------------------
    axb = fig.add_subplot(gr[0, 0])
    axb.patch.set_visible(False)
    for name, series, col, ls in ANGLES:
        axb.plot(T, series, color=col, lw=1.35, ls=ls, label=name, zorder=4)
    for i in KEY:
        axb.axvline(T[i], color="#c3ccd4", lw=0.6, zorder=1)
    axb.set_ylabel("flexion (deg)", fontsize=7.4, labelpad=2.0)
    axb.set_xlim(T[0], T[-1])
    axb.set_ylim(-22.0, 62.0)
    axb.tick_params(labelsize=6.8)
    axb.set_yticks([0, 20, 40, 60])
    axb.set_xticklabels([])
    axb.legend(ncol=2, fontsize=6.2, loc="lower center", frameon=True,
               facecolor="#ffffff", edgecolor="#cccccc", framealpha=0.92,
               columnspacing=1.0, handlelength=1.9, handletextpad=0.42,
               borderpad=0.30, labelspacing=0.24)
    axb.text(0.0, 1.13, "(b)  knee and hip flexion, solid left / dashed "
             "right", transform=axb.transAxes, fontsize=7.4, color=INK,
             ha="left", va="top")
    style_axes(axb)

    # ---- (c) speed --------------------------------------------------------
    axc = fig.add_subplot(gr[1, 0])
    axc.patch.set_visible(False)
    axc.fill_between(T, 0, PELVIS_SPEED, color="#dfeaf3", zorder=1)
    axc.plot(T, PELVIS_SPEED, color=DEEP, lw=1.7, zorder=4)
    for i in KEY:
        axc.axvline(T[i], color="#c3ccd4", lw=0.6, zorder=1)
        axc.scatter([T[i]], [PELVIS_SPEED[i]], s=17,
                    facecolors=ramp[list(KEY).index(i)],
                    edgecolors="#14202c", linewidths=0.6, zorder=6)
    imax = int(np.argmax(PELVIS_SPEED))
    # the eight sample ticks run the full height of this panel, so every
    # label inside it needs a plate under it
    axc.annotate(f"peak {PELVIS_SPEED[imax]:.2f} m/s at t = {T[imax]:.2f} s",
                 xy=(T[imax], PELVIS_SPEED[imax]),
                 xytext=(T[imax] + 0.10, PELVIS_SPEED[imax] + 0.30),
                 fontsize=6.5, color=DEEP, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED),
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                           alpha=0.92))
    axc.set_ylim(0, PELVIS_SPEED.max() * 1.55)
    axc.set_xlim(T[0], T[-1])
    axc.set_ylabel("speed (m s$^{-1}$)", fontsize=7.4, labelpad=2.0)
    axc.set_xlabel("time (s)", fontsize=7.4, labelpad=2.4)
    axc.tick_params(labelsize=6.8)
    axc.text(0.0, 1.16, "(c)  pelvis speed, differentiated from the same "
             "track", transform=axc.transAxes, fontsize=7.4, color=INK,
             ha="left", va="top")
    axc.text(0.985, 0.055, "grey ticks mark the eight skeletons in (a)",
             transform=axc.transAxes, fontsize=6.3, color=MUTED, ha="right",
             va="bottom",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                       alpha=0.92))
    style_axes(axc)

    savefig(fig, "1163_wm_human_mocap_sequence.png")


if __name__ == "__main__":
    main()
