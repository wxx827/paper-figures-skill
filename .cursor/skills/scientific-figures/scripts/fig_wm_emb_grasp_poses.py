"""Twelve candidate grasps scored on a single point-cloud observation.

Synthetic data: the cloud comes from ``mug_cloud`` and the candidate list and
their scores from ``CANDIDATES`` - replace those two and the rest of the
figure is unchanged.

The mug is a synthetic point cloud (cylinder shell, base disc, torus handle)
with per-point depth shading, and every candidate is drawn the way a grasp
planner reports one: two finger segments, the palm bar that joins them and a
short approach shaft, all placed from an (approach, opening) frame rather
than sketched by hand.  Line colour is the predicted grasp score on one
shared cool ramp, so the 3D view and the ranked bars on the right are read
against the same scale.

The colour range is taken over every candidate before anything is drawn, and
the bars are sorted by the same numbers that colour the poses.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubstyle import set_pub_style, savefig, style_axes
from figkit import panel_label_inches

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
CLOUD = "#8e99a4"
ACCENT = "#b23a48"

SEED = 1530  # original id; figure number is 1125
# house sequential blue, truncated so even the weakest candidate stays
# legible as a line on white paper
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "grasp_blue", plt.get_cmap("PuBu")(np.linspace(0.42, 1.0, 256)))

R_BODY, H_BODY = 0.30, 0.56
HANDLE_C = np.array([0.30 + 0.155, 0.0, 0.02])
HANDLE_R, HANDLE_T = 0.155, 0.034


# ---- object point cloud ----------------------------------------------------
def mug_cloud(rng, n_side=5200, n_base=900, n_handle=1700):
    th = rng.uniform(0, 2 * np.pi, n_side)
    z = rng.uniform(-H_BODY / 2, H_BODY / 2, n_side)
    r = R_BODY + rng.normal(0, 0.004, n_side)
    side = np.column_stack([r * np.cos(th), r * np.sin(th), z])

    tb = rng.uniform(0, 2 * np.pi, n_base)
    rb = R_BODY * np.sqrt(rng.uniform(0, 1, n_base))
    base = np.column_stack([rb * np.cos(tb), rb * np.sin(tb),
                            -H_BODY / 2 + rng.normal(0, 0.004, n_base)])

    tt = rng.uniform(-0.62 * np.pi, 0.62 * np.pi, n_handle)
    tp = rng.uniform(0, 2 * np.pi, n_handle)
    rr = HANDLE_T + rng.normal(0, 0.004, n_handle)
    hx = HANDLE_C[0] + (HANDLE_R + rr * np.cos(tp)) * np.cos(tt)
    hz = HANDLE_C[2] + (HANDLE_R + rr * np.cos(tp)) * np.sin(tt)
    hy = rr * np.sin(tp)
    handle = np.column_stack([hx, hy, hz])
    return np.vstack([side, base, handle])


# ---- candidate grasps ------------------------------------------------------
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def side_grasp(az, z, score):
    a = np.array([-np.cos(az), -np.sin(az), 0.0])
    b = np.array([-np.sin(az), np.cos(az), 0.0])
    return dict(c=np.array([0.0, 0.0, z]), a=a, b=b, w=0.70, l=0.21,
                s=score, kind="side")


def rim_grasp(az, score):
    c = np.array([R_BODY * np.cos(az), R_BODY * np.sin(az), H_BODY / 2])
    return dict(c=c, a=np.array([0.0, 0.0, -1.0]),
                b=np.array([np.cos(az), np.sin(az), 0.0]), w=0.17, l=0.19,
                s=score, kind="rim")


def handle_grasp(ang, score):
    c = HANDLE_C + HANDLE_R * np.array([np.cos(ang), 0.0, np.sin(ang)])
    a = -unit([np.cos(ang), 0.0, np.sin(ang)])
    return dict(c=c, a=a, b=np.array([0.0, 1.0, 0.0]), w=0.16, l=0.18,
                s=score, kind="handle")


CANDIDATES = [
    side_grasp(np.deg2rad(-38), -0.02, 0.93),
    side_grasp(np.deg2rad(28), 0.09, 0.90),
    side_grasp(np.deg2rad(96), -0.09, 0.86),
    side_grasp(np.deg2rad(150), 0.05, 0.81),
    side_grasp(np.deg2rad(-104), 0.16, 0.70),
    side_grasp(np.deg2rad(-160), -0.19, 0.62),
    handle_grasp(np.deg2rad(6), 0.77),
    handle_grasp(np.deg2rad(74), 0.66),
    handle_grasp(np.deg2rad(-66), 0.34),
    rim_grasp(np.deg2rad(-14), 0.56),
    rim_grasp(np.deg2rad(118), 0.47),
    rim_grasp(np.deg2rad(-132), 0.40),
]
for i, g in enumerate(CANDIDATES):
    g["id"] = f"G{i + 1:02d}"


def gripper_segments(g):
    """Two fingers, the palm bar between them and the approach shaft."""
    c, a, b = g["c"], g["a"], g["b"]
    half = 0.5 * g["w"] * b
    tip_p, tip_m = c + half, c - half
    back_p, back_m = tip_p - a * g["l"], tip_m - a * g["l"]
    palm = c - a * g["l"]
    return [np.array([tip_p, back_p]), np.array([tip_m, back_m]),
            np.array([back_p, back_m]),
            np.array([palm, palm - a * 0.10])]


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    pts = mug_cloud(rng)

    scores = np.array([g["s"] for g in CANDIDATES])
    norm = mpl.colors.Normalize(vmin=scores.min(), vmax=scores.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.40], left=0.015,
                          right=0.955, top=0.885, bottom=0.105, wspace=0.02)
    right = gs[0, 1].subgridspec(3, 1, height_ratios=[0.10, 1.0, 0.14],
                                 hspace=0.0)

    # ---- (a) point cloud with candidate grippers --------------------------
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    lim = 0.60
    keep = ((np.abs(pts[:, 0]) < lim) & (np.abs(pts[:, 1]) < lim)
            & (np.abs(pts[:, 2]) < lim))          # 12.6: clip by hand
    p = pts[keep]
    depth = (p[:, 0] * 0.55 + p[:, 1] * 0.62 - p[:, 2] * 0.30)
    shade = 0.55 + 0.45 * (depth - depth.min()) / np.ptp(depth)
    rgb = np.array(mpl.colors.to_rgb(CLOUD))[None, :] * shade[:, None]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.6, c=np.clip(rgb, 0, 1),
               depthshade=False, linewidths=0, zorder=2)

    order = np.argsort(scores)                    # weakest first, best on top
    for idx in order:
        g = CANDIDATES[idx]
        col = CMAP(norm(g["s"]))
        best = idx == int(np.argmax(scores))
        for seg in gripper_segments(g):
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=col,
                    lw=2.2 if best else 1.35, solid_capstyle="round",
                    zorder=4 if best else 3, alpha=1.0 if best else 0.92)

    top = CANDIDATES[int(np.argmax(scores))]

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=1.42)
    ax.view_init(elev=20, azim=-56)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)                       # 12.7
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)

    # ground reference so the cloud does not float
    gx = np.linspace(-0.50, 0.72, 2)
    gy = np.linspace(-0.50, 0.50, 2)
    GX, GY = np.meshgrid(gx, gy)
    ax.plot_surface(GX, GY, np.full_like(GX, -H_BODY / 2 - 0.012),
                    color="#eef0f2", edgecolor="none", alpha=0.55,
                    shade=False, zorder=0)

    plate = dict(boxstyle="round,pad=0.22", facecolor="white",
                 edgecolor="none", alpha=0.88)
    ax.text2D(0.055, 0.965, "Candidate grasps on a single-view point cloud",
              transform=ax.transAxes, fontsize=9.0, fontweight="bold",
              color=INK, ha="left", va="top", zorder=8, bbox=plate)
    ax.text2D(0.055, 0.915, f"{len(p) / 1000:.1f} k points  ·  "
              f"{len(CANDIDATES)} sampled poses  ·  parallel-jaw, 85 mm "
              "stroke", transform=ax.transAxes, fontsize=7.0, color=MUTED,
              ha="left", va="top", zorder=8, bbox=plate)
    ax.text2D(0.055, 0.075, f"{top['id']}  ·  best candidate  ·  "
              f"{top['kind']} grasp  ·  score {top['s']:.2f}",
              transform=ax.transAxes, fontsize=7.4, fontweight="bold",
              color=ACCENT, ha="left", va="bottom", zorder=8, bbox=plate)

    cax = fig.add_axes([0.075, 0.088, 0.30, 0.020])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.set_label("predicted grasp score (-)", fontsize=7.0, color=INK,
                 labelpad=1.5)
    cb.ax.tick_params(labelsize=6.6, width=0.7, length=2.4)

    # ---- (b) ranked scores -------------------------------------------------
    axb = fig.add_subplot(right[1, 0])
    axb.add_patch(FancyBboxPatch(
        (-0.115, -0.055), 1.20, 1.135,
        boxstyle="round,pad=0.012,rounding_size=0.03", facecolor="#f5f5f5",
        edgecolor=HAIR, linewidth=0.7, transform=axb.transAxes,
        clip_on=False, zorder=-2))

    rank = np.argsort(scores)[::-1]
    ypos = np.arange(len(rank))[::-1]
    axb.barh(ypos, scores[rank], height=0.64,
             color=[CMAP(norm(s)) for s in scores[rank]], edgecolor="white",
             linewidth=0.5, zorder=3)
    for y, idx in zip(ypos, rank):
        axb.text(scores[idx] + 0.018, y, f"{scores[idx]:.2f}", va="center",
                 ha="left", fontsize=6.4, color=SUB, zorder=4)
    axb.set_yticks(ypos)
    axb.set_yticklabels([f"{CANDIDATES[i]['id']}  {CANDIDATES[i]['kind']}"
                         for i in rank], fontsize=6.6)
    axb.get_yticklabels()[0].set_fontweight("bold")
    axb.get_yticklabels()[0].set_color(ACCENT)
    axb.set_xlim(0.0, 1.10)
    axb.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axb.tick_params(axis="x", labelsize=6.6, length=2.2)
    axb.tick_params(axis="y", length=0)
    axb.set_xlabel("grasp score (-)", fontsize=7.0, labelpad=1.5)
    axb.set_title("Candidates ranked", fontsize=8.5, fontweight="bold",
                  loc="left", pad=4.0)
    axb.axvline(0.60, color="#8a8a8a", lw=0.7, ls=(0, (3.0, 2.0)), zorder=2)
    axb.text(0.60, len(rank) - 0.28, " execution threshold", fontsize=6.2,
             color=MUTED, va="center", ha="left")
    style_axes(axb)
    for side in ("left", "bottom"):
        axb.spines[side].set_linewidth(0.6)

    panel_label_inches(fig, ax, "(a)", dx=0.05, dy=-0.02)
    panel_label_inches(fig, axb, "(b)", dx=-0.52, dy=0.20)

    savefig(fig, "1125_wm_emb_grasp_poses.png")


if __name__ == "__main__":
    main()
