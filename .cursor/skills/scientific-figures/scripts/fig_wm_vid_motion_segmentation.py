"""Figure 1169 - Motion segmentation of a conveyor frame.

The scene is the ``conveyor`` kind added to the shared library for this batch:
a fixed oblique camera over a production line, which conveniently contains
three independent motion sources - the belt (and the parts riding it), one
part being ejected off the end, and a pusher arm on its own sweep cycle -
against static machine structure.

The pipeline is a single chain and every printed number is a reading of it:

    designed flow  ->  k-means (k = 4) on the per-pixel (u, v)
                   ->  labels        ->  masks drawn in (a)
                                     ->  areas printed in (b)
                                     ->  cluster means = the arrows in (a)
                                     ->  the very same points plotted in (c)

Area closure.  Every pixel receives exactly one label, so the four shares are
a partition of the frame and must print as 100.0 %.  Rounding each share
independently would land on 99.9 or 100.1, so the printed values come from a
largest-remainder allocation: each is within 0.05 pp of its true share and the
four of them sum to 100.0 exactly.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, add_panel_label
from figkit import image_axes
from _wm_scenes import (build, conveyor_parts, scapsule, sellipse,
                        _pusher_pose, CONVEYOR_TOP, CONVEYOR_BOT)

INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
# (a) is a full-bleed photograph, so this figure is already dense; its two
# side panels only need grouping, and their card is therefore kept clearly
# *above* the 0.94 ink threshold rather than on it.
HAIR, CARD = "#d5d5d5", "#f5f5f5"
DEEP = "#134a76"
SEED = 1574  # original id; figure number is 1169
H, W = 180, 228
TFRAME = 0.30
CAM = dict(contrast=1.18, sharpen=1.30, exposure=1.06, sat=0.66)

# (name, u, v) - the motion sources the frame is built from.  k-means gets the
# pixels only; each recovered centroid is named by its nearest design vector.
DESIGN = [("static structure", 0.00, 0.00, "#7b8794"),
          ("belt and parts", 6.60, 0.00, "#2e74a8"),
          ("ejected part", 2.20, 5.80, "#2a7f7a"),
          ("sweep arm", -3.80, 2.60, "#6b5fa8")]
# where each region's caption parks inside the tile, as a fraction of the
# frame; a leader line runs from there to the mask's own centroid, so the
# caption can be kept clear of the arrows without floating free
LABEL_AT = [(0.020, 0.988, "left", "bottom"),
            (0.020, 0.462, "left", "top"),
            (0.988, 0.988, "right", "bottom"),
            (0.020, 0.020, "left", "top")]


def flow_field(anchors):
    """Region-wise constant velocity plus a little smooth measurement noise."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    v = yy / H
    u_f = np.zeros((H, W))
    v_f = np.zeros((H, W))

    belt = ((v >= CONVEYOR_TOP) & (v <= CONVEYOR_BOT)).astype(float)
    for (px, py, s, _fam, _col) in conveyor_parts(TFRAME, H, W):
        belt = np.maximum(belt, sellipse(H, W, px, py - 0.9 * s, 1.7 * s,
                                         1.6 * s, 0.8))
    u_f += DESIGN[1][1] * belt
    v_f += DESIGN[1][2] * belt

    ej = [a for a in anchors if a["name"] == "part"][-1]
    m_ej = sellipse(H, W, ej["cx"], ej["cy"], 1.5 * ej["rx"], 2.0 * ej["ry"],
                    0.8)
    u_f = u_f * (1 - m_ej) + DESIGN[2][1] * m_ej
    v_f = v_f * (1 - m_ej) + DESIGN[2][2] * m_ej

    pv, ph = _pusher_pose(TFRAME, H, W)
    m_ar = np.clip(scapsule(H, W, pv, ph, 0.026 * W, 1.0)
                   + sellipse(H, W, ph[0], ph[1], 0.048 * W, 0.042 * H, 0.9),
                   0, 1)
    u_f = u_f * (1 - m_ar) + DESIGN[3][1] * m_ar
    v_f = v_f * (1 - m_ar) + DESIGN[3][2] * m_ar

    u_f += 0.55 * gaussian_filter(rng.standard_normal((H, W)), 5.0) * 6.0
    v_f += 0.55 * gaussian_filter(rng.standard_normal((H, W)), 5.0) * 6.0
    return np.dstack([u_f, v_f])


def kmeans(X, k, seed, iters=60):
    """Lloyd with a k-means++ start; deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(0, X.shape[0])]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.asarray(C)[None]) ** 2).sum(axis=2),
                   axis=1)
        C.append(X[int(np.argmax(d * rng.random(d.shape)))])
    C = np.asarray(C, dtype=float)
    lab = np.zeros(X.shape[0], dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            if np.any(lab == j):
                C[j] = X[lab == j].mean(axis=0)
    return lab, C


def largest_remainder(counts, total=100.0, dp=1):
    """Round shares so that the printed values sum to ``total`` exactly."""
    q = 10 ** dp
    exact = counts / counts.sum() * total
    floors = np.floor(exact * q) / q
    left = int(round((total - floors.sum()) * q))
    order = np.argsort(-(exact - floors))
    out = floors.copy()
    for i in range(left):
        out[order[i % out.size]] += 1.0 / q
    return out


def main():
    set_pub_style(base_font=8.5)
    sc = build("conveyor", seed=SEED, shape=(H, W), t=TFRAME, cam=CAM)
    frame, anchors = sc["rgb"], sc["anchors"]
    flow = flow_field(anchors)

    X = flow.reshape(-1, 2)
    lab, C = kmeans(X, len(DESIGN), SEED)
    # name each recovered cluster by the design vector it landed nearest to
    dv = np.array([[d[1], d[2]] for d in DESIGN])
    pick = np.argmin(((C[:, None, :] - dv[None]) ** 2).sum(axis=2), axis=1)
    if len(set(pick.tolist())) != len(DESIGN):        # degenerate start
        raise RuntimeError("clusters did not separate; change SEED")
    lab = np.asarray(pick)[lab].reshape(H, W)
    means = np.array([flow[lab == j].mean(axis=0) for j in range(len(DESIGN))])
    counts = np.array([(lab == j).sum() for j in range(len(DESIGN))],
                      dtype=float)
    share = largest_remainder(counts)
    speed = np.hypot(means[:, 0], means[:, 1])

    fig = plt.figure(figsize=(10.4, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.44, 1.0],
                          height_ratios=[0.86, 1.0], left=0.050, right=0.972,
                          top=0.838, bottom=0.098, wspace=0.185, hspace=0.360)
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])

    # ---- (a) the frame with the four masks --------------------------------
    image_axes(axa, color="#3a3a3a", lw=0.8)
    axa.set_aspect("auto")
    lum = (frame @ np.array([0.30, 0.59, 0.11]))[..., None]
    base = np.clip(0.16 + 0.82 * lum, 0, 1).repeat(3, axis=2)
    over = base.copy()
    for j, (name, _u, _v, col) in enumerate(DESIGN):
        m = (lab == j)[..., None].astype(float)
        rgbc = np.array(plt.matplotlib.colors.to_rgb(col))
        over = over * (1 - 0.40 * m) + rgbc[None, None, :] * 0.40 * m
    axa.imshow(over, interpolation="nearest", aspect="auto")
    # thin boundary between labels, so four flat washes read as four regions
    edge = np.zeros((H, W), dtype=bool)
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    axa.imshow(np.dstack([np.zeros((H, W, 3)), 0.55 * edge]),
               interpolation="nearest", aspect="auto", zorder=3)

    rowi = np.mgrid[0:H, 0:W][0]
    for j, (name, _u, _v, col) in enumerate(DESIGN):
        m = lab == j
        if j == 0:
            # the static label covers housing *and* floor, so its overall
            # centroid would land on the belt; use the floor part of it
            m = m & (rowi > CONVEYOR_BOT * H + 4)
        ys, xs = np.nonzero(m)
        cx, cy = float(xs.mean()), float(ys.mean())
        dx, dy = means[j] * 5.2
        if speed[j] > 0.25:
            for lw, cc, ms_ in ((2.8, "white", 13), (1.5, col, 10)):
                axa.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                             zorder=7 if cc == "white" else 8,
                             arrowprops=dict(arrowstyle="-|>",
                                             mutation_scale=ms_, lw=lw,
                                             color=cc, shrinkA=0, shrinkB=0))
        else:
            axa.plot([cx], [cy], marker="o", ms=8.0, mfc="none", mec="white",
                     mew=2.0, zorder=7)
            axa.plot([cx], [cy], marker="o", ms=8.0, mfc="none", mec=col,
                     mew=1.0, zorder=8)
        axa.plot([cx], [cy], marker="o", ms=3.2, mfc="white", mec=col,
                 mew=1.0, zorder=9)
        lx, ly, ha, va = LABEL_AT[j]
        axa.annotate(f"{name}\n({round(means[j][0], 1) + 0.0:+.1f}, "
                     f"{round(means[j][1], 1) + 0.0:+.1f}) px/frame",
                     xy=(cx, cy), xytext=(lx * W, ly * H), fontsize=6.6,
                     color=col, ha=ha, va=va, linespacing=1.45, zorder=10,
                     bbox=dict(boxstyle="round,pad=0.24", fc="white",
                               ec="none", alpha=0.92),
                     arrowprops=dict(arrowstyle="-", lw=0.7, color="white",
                                     alpha=0.85, shrinkA=3, shrinkB=6))
    axa.set_title("Four motion-consistent regions, recovered by clustering "
                  "the flow", fontsize=9.4, fontweight="bold", pad=4.0,
                  color=INK, loc="left")
    add_panel_label(axa, "(a)", x=-0.060, y=1.005, size=9.5)

    # ---- (b) area shares ---------------------------------------------------
    axb.set_facecolor(CARD)
    ypos = np.arange(len(DESIGN))
    axb.barh(ypos, share, height=0.60,
             color=[d[3] for d in DESIGN], edgecolor="white", linewidth=0.6,
             zorder=3)
    for j in range(len(DESIGN)):
        inside = share[j] > 0.34 * share.max()
        axb.text(share[j] - 1.4 if inside else share[j] + 1.4, ypos[j],
                 f"{share[j]:.1f} %", ha="right" if inside else "left",
                 va="center", fontsize=7.2, zorder=6,
                 color="white" if inside else INK,
                 fontweight="bold" if inside else "normal",
                 bbox=None if inside else dict(facecolor="white",
                                               edgecolor="none", pad=1.0,
                                               alpha=0.92))
    axb.set_yticks(ypos)
    axb.set_yticklabels([d[0] for d in DESIGN], fontsize=7.2)
    axb.set_xlim(0, max(share) * 1.20)
    axb.set_ylim(len(DESIGN) - 0.45, -0.55)
    axb.set_xlabel("share of the frame (%)", fontsize=7.8)
    axb.tick_params(labelsize=7.0, length=2.2)
    axb.grid(True, axis="x", lw=0.45, color="#e2e2e2")
    axb.set_axisbelow(True)
    axb.text(0.975, 0.045, f"{' + '.join(f'{s:.1f}' for s in share)}"
             f" = {share.sum():.1f} %", transform=axb.transAxes, ha="right",
             va="bottom", fontsize=6.8, color=DEEP, zorder=7,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9d8",
                       lw=0.55, alpha=0.94))
    axb.set_title("One label per pixel, so the shares partition the frame",
                  fontsize=8.4, fontweight="bold", pad=4.0, color=INK,
                  loc="left")
    add_panel_label(axb, "(b)", x=-0.235, y=1.005, size=9.5)

    # ---- (c) the clusters in velocity space --------------------------------
    rng = np.random.default_rng(SEED + 1)
    sub = rng.choice(X.shape[0], 2600, replace=False)
    axc.set_facecolor(CARD)
    lf = lab.reshape(-1)
    for j, (name, _u, _v, col) in enumerate(DESIGN):
        s = sub[lf[sub] == j]
        # no ax.legend() here on purpose: the four named bars in (b) are the
        # legend, and they are the same four clusters, so there is nothing
        # that can promise a fifth series
        axc.scatter(X[s, 0], X[s, 1], s=2.6, color=col, alpha=0.42,
                    linewidths=0, zorder=3)
    for j, (name, _u, _v, col) in enumerate(DESIGN):
        axc.plot([means[j][0]], [means[j][1]], marker="X", ms=7.2,
                 mfc=col, mec="white", mew=1.0, zorder=6)
    axc.plot([0, 0], [-6.0, 10.5], color="#b3b3b3", lw=0.7, zorder=2)
    axc.plot([-9.5, 12.0], [0, 0], color="#b3b3b3", lw=0.7, zorder=2)
    axc.set_xlim(-9.5, 12.0)
    # v is an image-row displacement, so +v points *down* in (a).  The axis is
    # inverted to match: a cluster that sits below the origin here is drawn
    # with a downward arrow there, and the two panels stop being mirrors.
    axc.set_ylim(10.5, -6.0)
    axc.set_xlabel("horizontal flow $u$ (px / frame)", fontsize=7.8)
    axc.set_ylabel("vertical flow $v$ (px / frame, $+$ is down)",
                   fontsize=7.8)
    axc.tick_params(labelsize=7.0, length=2.4)
    axc.grid(True, lw=0.45, color="#e6e6e6")
    axc.set_axisbelow(True)
    axc.text(0.975, 0.955, "X = cluster mean,\nthe arrows drawn in (a)",
             transform=axc.transAxes, ha="right", va="top", fontsize=6.8,
             color=SUB, linespacing=1.5, zorder=7,
             bbox=dict(boxstyle="round,pad=0.26", fc="white", ec=HAIR,
                       lw=0.55, alpha=0.94))
    axc.set_title(f"The same {len(sub):,} pixels in velocity space",
                  fontsize=8.4, fontweight="bold", pad=4.0, color=INK,
                  loc="left")
    add_panel_label(axc, "(c)", x=-0.145, y=1.005, size=9.5)

    # ---- headline ----------------------------------------------------------
    fast = int(np.argmax(speed))
    savefig(fig, "1169_wm_vid_motion_segmentation.png")


if __name__ == "__main__":
    main()
