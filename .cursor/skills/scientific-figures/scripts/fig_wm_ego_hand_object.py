"""Figure 1160 - Hand-object interaction: five keyframes and one contact trace.

One egocentric clip on an assembly bench, sampled at five moments.  The strip
and the trace underneath are two views of a single array:

    CONTACT_P(t)  ->  hand pose (finger curl, wrist proximity)
                  ->  interaction state chip on each tile
                  ->  the curve, its shaded grasp interval and its duration

Change the trace and the hands close differently, the chips relabel
themselves and the shaded span moves.  Nothing here is typed twice, so the
panels cannot drift apart - the failure mode BRIEF_WM section 1.7 warns
about.

The keypoint overlay is not a guess either: the scene generator returns the
21 MANO joints it used to paint the hand, and the overlay is drawn through
exactly those coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pubstyle import set_pub_style, savefig, style_axes
from figkit import image_axes
from _wm_scenes import build, HAND_BONES, HAND_TIPS

H, W = 145, 210
INK, SUB, MUTED = "#1a1a1a", "#555555", "#777777"
RULE, HAIR = "#333333", "#d5d5d5"

# four interaction states, four tints of the house sequential blue
STATES = {
    "Approach": ("#dbe8f2", "#20455f"),
    "Contact": ("#9cc3dd", "#12354e"),
    "Grasp": ("#2e74a8", "#ffffff"),
    "Release": ("#6ba3cb", "#0f2c41"),
}
BONE_C, JOINT_C = "#bcd7e8", "#ffffff"
BOX_C = "#2e74a8"

T = np.linspace(0.0, 4.20, 260)
_rise = 1.0 / (1.0 + np.exp(-(T - 1.12) / 0.185))
_fall = 1.0 / (1.0 + np.exp((T - 3.28) / 0.150))
CONTACT_P = np.clip(_rise * _fall, 0.0, 1.0)
DP = np.gradient(CONTACT_P, T)
GRASP_TH = 0.72

KEYS = [0.52, 1.18, 1.90, 2.80, 3.44]           # the five sampled moments


def sample(t):
    return float(np.interp(t, T, CONTACT_P)), float(np.interp(t, T, DP))


def state_of(p, dp):
    """The chip label; one function, used for the tiles and for the trace."""
    if p >= GRASP_TH:
        return "Grasp"
    if p >= 0.20:
        return "Contact" if dp >= 0 else "Release"
    return "Approach"


def transport(t):
    """How far the tool has been carried, 0..1 - what separates (c) from (d)."""
    return float(np.clip((t - 2.25) / 1.05, 0.0, 1.0))


def props_at(u_tool):
    return [("tool_body", u_tool, 0.56, "tool", "#414852", 1.45),
            ("module", 0.180, 0.44, "box", "#69717b", 1.10),
            ("beaker", 0.885, 0.34, "cyl", "#7c8b95", 0.90),
            ("dish", 0.360, 0.74, "disc", "#8d9299", 0.78),
            ("tin", 0.955, 0.64, "cyl", "#5d686e", 0.88)]


def frame(t):
    """One moment of the clip, with the hand posed by the contact trace."""
    p, dp = sample(t)
    prox = float(np.clip(p / GRASP_TH, 0.0, 1.0))
    car = transport(t)
    u_tool = 0.620 - 0.170 * car
    hand = dict(u=0.775 - 0.150 * prox - 0.170 * car,
                v=1.075 - 0.075 * prox, s=0.140,
                ang=0.26 - 0.52 * prox, curl=0.10 + 0.62 * p, side="r")
    sc = build("ego", seed=6501, shape=(H, W), room="lab",
               props=props_at(u_tool), hands=[hand],
               cam=dict(exposure=1.03, wb=(0.960, 1.00, 1.082), dof=1.6,
                        contrast=1.12, vignette=0.30))
    return sc, p, dp, state_of(p, dp)


def skeleton(ax, J):
    for a, b in HAND_BONES:
        ax.plot([J[a, 0], J[b, 0]], [J[a, 1], J[b, 1]], color="#14202c",
                lw=2.2, solid_capstyle="round", zorder=5)
        ax.plot([J[a, 0], J[b, 0]], [J[a, 1], J[b, 1]], color=BONE_C,
                lw=1.1, solid_capstyle="round", zorder=6)
    ax.scatter(J[1:, 0], J[1:, 1], s=6.5, facecolors=JOINT_C,
               edgecolors="#14202c", linewidths=0.45, zorder=7)
    ax.scatter(J[list(HAND_TIPS), 0], J[list(HAND_TIPS), 1], s=13.0,
               facecolors=BOX_C, edgecolors="#ffffff", linewidths=0.55,
               zorder=8)
    ax.scatter(J[:1, 0], J[:1, 1], s=20.0, facecolors="#ffffff",
               edgecolors="#14202c", linewidths=0.8, zorder=8)


def main():
    set_pub_style(base_font=8)
    frames = [frame(t) for t in KEYS]

    lo = float(T[CONTACT_P >= GRASP_TH][0])
    hi = float(T[CONTACT_P >= GRASP_TH][-1])

    fig = plt.figure(figsize=(11.0, 3.8))
    gtop = fig.add_gridspec(1, 5, left=0.038, right=0.994, top=0.800,
                            bottom=0.430, wspace=0.026)
    gbot = fig.add_gridspec(1, 1, left=0.038, right=0.994, top=0.335,
                            bottom=0.175)

    for k, (sc, p, dp, st) in enumerate(frames):
        ax = fig.add_subplot(gtop[0, k])
        image_axes(ax, color="#9a9a9a", lw=0.55)
        ax.set_aspect("auto")
        ax.imshow(sc["rgb"], interpolation="bilinear", aspect="auto")

        obj = [a for a in sc["anchors"] if a["name"] == "tool_body"][0]
        touching = p >= 0.20
        bx, by = obj["cx"] - 1.08 * obj["rx"], obj["cy"] - 1.35 * obj["ry"]
        bw, bh = 2.16 * obj["rx"], 2.70 * obj["ry"]
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="none",
                               edgecolor=BOX_C if touching else "#a8bccd",
                               lw=1.25 if touching else 0.95,
                               ls="-" if touching else (0, (3.0, 2.0)),
                               zorder=4))
        ax.text(bx + 1.0, by - 2.0, f"power tool   p = {p:.2f}", fontsize=6.1,
                color=BOX_C if touching else "#5d7385", ha="left",
                va="bottom", zorder=5,
                bbox=dict(boxstyle="square,pad=0.20", facecolor="#ffffff",
                          edgecolor="none", alpha=0.82))

        for a in sc["anchors"]:
            if a["name"].startswith("hand_"):
                skeleton(ax, a["joints"])

        fill, tint = STATES[st]
        ax.text(0.030, 0.945, f"({'abcde'[k]})  {st}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.0, fontweight="bold",
                color=tint, zorder=9,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=fill,
                          edgecolor="#14202c", linewidth=0.55))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xlabel(f"t = {KEYS[k]:.2f} s", fontsize=7.0, color=SUB,
                      labelpad=2.6)

    # ---- the trace the strip was posed from -------------------------------
    ax = fig.add_subplot(gbot[0, 0])
    ax.axvspan(lo, hi, facecolor="#dfeaf3", edgecolor="none", zorder=0)
    ax.plot(T, CONTACT_P, color="#2e74a8", lw=1.9, zorder=4)
    ax.axhline(GRASP_TH, color="#8a8a8a", lw=0.8, ls=(0, (4.0, 2.4)), zorder=2)
    ax.text(0.030, GRASP_TH + 0.04, f"grasp threshold {GRASP_TH:.2f}",
            fontsize=6.4, color=MUTED, ha="left", va="bottom")
    ax.text(0.5 * (lo + hi), 0.28, f"grasp held {hi - lo:.2f} s",
            ha="center", va="center", fontsize=6.8, color="#20455f")

    for k, (t, (_, p, _, st)) in enumerate(zip(KEYS, frames)):
        fill, tint = STATES[st]
        ax.plot([t, t], [p, 1.20], color="#8a8a8a", lw=0.7,
                ls=(0, (2.2, 1.8)), zorder=3)
        ax.scatter([t], [p], s=46, facecolors=fill, edgecolors="#14202c",
                   linewidths=0.8, zorder=6)
        ax.text(t, 1.24, f"({'abcde'[k]})", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold", color=INK)

    ax.set_xlim(T[0], T[-1])
    ax.set_ylim(-0.06, 1.30)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("contact\nprobability", fontsize=7.2, labelpad=2.0,
                  linespacing=1.25)
    ax.set_xlabel("time since the reach began (s)", fontsize=7.4,
                  labelpad=2.6)
    ax.tick_params(labelsize=6.8)
    style_axes(ax)

    savefig(fig, "1160_wm_ego_hand_object.png")


if __name__ == "__main__":
    main()
