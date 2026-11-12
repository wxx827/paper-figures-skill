"""Three Bloch spheres: basis state, equal superposition, phased state.

All three panels share one wireframe generator, one camera and one radius, so
the only thing that differs between them is the state itself.  Every quoted
number - the polar and azimuthal angles, the two amplitudes, the population
of |1> - is computed from the same (theta, phi) pair that positions the
arrow, so changing a state changes its caption automatically.  The third
sphere additionally carries a driven-evolution track that spirals from the
north pole down to the plotted state, which is what gives the phase angle a
visible meaning.
"""
import numpy as np
import matplotlib.pyplot as plt

from pubstyle import set_pub_style, savefig
from figkit import panel_labels_inches

BLUE = "#3b5b92"        # state vector
PURPLE = "#8e6bbf"      # evolution track
INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
WIRE = "#cccccc"
EQ = "#8a8a8a"
GUIDE = "#808080"

# Orthographic projection matters here: under the default perspective camera
# a curve lying exactly on the unit sphere projects *outside* the silhouette
# when it passes near the viewer, which reads as an error.
R_LIM = 1.15
ELEV, AZIM = 20.0, 34.0

STATES = [
    (0.0, 0.0, "Computational basis state", False),
    (90.0, 0.0, "Equal superposition", False),
    (62.0, 128.0, "Relative phase with drive track", True),
]


def bloch_vec(theta_deg, phi_deg):
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def silhouette():
    """Unit circle in the plane facing the camera - the sphere's outline.

    Without it the wireframe thins out near the poles and a curve lying on the
    far side of the surface looks as if it had escaped the sphere.
    """
    e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u = np.cross(d, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u)
    w = np.cross(d, u)
    t = np.linspace(0.0, 2.0 * np.pi, 241)
    return np.outer(np.cos(t), u) + np.outer(np.sin(t), w)


def draw_frame(ax):
    ring = silhouette()
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=EQ, lw=0.9, zorder=1)
    tt = np.linspace(0.0, 2.0 * np.pi, 181)
    for ph in np.linspace(0.0, np.pi, 7)[:-1]:
        ax.plot(np.sin(tt) * np.cos(ph), np.sin(tt) * np.sin(ph), np.cos(tt),
                color=WIRE, lw=0.4, zorder=1)
    for th in np.deg2rad([30.0, 60.0, 90.0, 120.0, 150.0]):
        eq = abs(th - 0.5 * np.pi) < 1e-9
        ax.plot(np.sin(th) * np.cos(tt), np.sin(th) * np.sin(tt),
                np.full_like(tt, np.cos(th)), color=EQ if eq else WIRE,
                lw=0.9 if eq else 0.4, zorder=1)
    for vec in np.eye(3):
        ax.plot(*[[-1.06 * c, 1.06 * c] for c in vec], color=GUIDE, lw=0.6,
                ls=(0, (3, 2.4)), zorder=2)

    ax.text(0.0, 0.26, 1.08, r"$|0\rangle$", ha="left", va="center",
            fontsize=8.0, color=INK)
    ax.text(0.0, 0.26, -1.08, r"$|1\rangle$", ha="left", va="center",
            fontsize=8.0, color=INK)
    ax.text(1.20, 0.0, -0.08, "$x$", ha="center", va="center", fontsize=7.4,
            color=MUTED)
    ax.text(0.0, 1.20, -0.08, "$y$", ha="center", va="center", fontsize=7.4,
            color=MUTED)


def draw_angles(ax, theta_deg, phi_deg):
    """Polar arc from +z to the state, azimuth arc in the equatorial plane."""
    if theta_deg > 1.0:
        a = np.linspace(0.0, np.deg2rad(theta_deg), 60)
        p = np.deg2rad(phi_deg)
        ax.plot(0.40 * np.sin(a) * np.cos(p), 0.40 * np.sin(a) * np.sin(p),
                0.40 * np.cos(a), color=INK, lw=0.8, zorder=6)
        am = np.deg2rad(theta_deg * 0.55)
        ax.text(0.52 * np.sin(am) * np.cos(p), 0.52 * np.sin(am) * np.sin(p),
                0.52 * np.cos(am), r"$\theta$", ha="center", va="center",
                fontsize=7.6, color=INK, zorder=7)
    if phi_deg > 1.0:
        a = np.linspace(0.0, np.deg2rad(phi_deg), 60)
        ax.plot(0.46 * np.cos(a), 0.46 * np.sin(a), np.zeros_like(a),
                color=INK, lw=0.8, zorder=6)
        am = np.deg2rad(phi_deg * 0.55)
        ax.text(0.62 * np.cos(am), 0.62 * np.sin(am), 0.03, r"$\varphi$",
                ha="center", va="center", fontsize=7.6, color=INK, zorder=7)


def draw_state(ax, theta_deg, phi_deg):
    v = bloch_vec(theta_deg, phi_deg)
    ax.quiver(0.0, 0.0, 0.0, v[0], v[1], v[2], color=BLUE, lw=1.8,
              arrow_length_ratio=0.15, zorder=8)
    if theta_deg > 1.0:
        ax.plot([v[0], v[0]], [v[1], v[1]], [v[2], 0.0], color=BLUE, lw=0.7,
                ls=(0, (2.4, 2.0)), zorder=7)
        ax.plot([0.0, v[0]], [0.0, v[1]], [0.0, 0.0], color=BLUE, lw=0.7,
                ls=(0, (2.4, 2.0)), zorder=7)
    ax.scatter([v[0]], [v[1]], [v[2]], s=14, color=BLUE, depthshade=False,
               zorder=9)
    return v


def draw_track(ax, theta_deg, phi_deg, turns=2.0, n=500):
    """Surface spiral from |0> to the plotted state (driven Rabi evolution)."""
    s = np.linspace(0.0, 1.0, n)
    th = np.deg2rad(theta_deg) * s
    ph = np.deg2rad(phi_deg) + 2.0 * np.pi * turns * (s - 1.0)
    ax.plot(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th),
            color=PURPLE, lw=1.1, alpha=0.9, zorder=5)


def caption(theta_deg, phi_deg):
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    head = (rf"$|\psi\rangle = {c:.2f}\,|0\rangle + {s:.2f}\,"
            rf"e^{{i{phi_deg:.0f}^\circ}}|1\rangle$")
    tail = (rf"$\theta$ = {theta_deg:.0f}$^\circ$,  "
            rf"$\varphi$ = {phi_deg:.0f}$^\circ$,  "
            rf"$P(1)$ = {s ** 2 * 100:.0f} %")
    return head, tail


def main():
    set_pub_style()
    fig = plt.figure(figsize=(10.4, 4.5))
    axes = [fig.add_subplot(1, 3, k + 1, projection="3d")
            for k in range(len(STATES))]

    for ax, (theta, phi, title, track) in zip(axes, STATES):
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_visible(False)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.set_proj_type("ortho")
        ax.view_init(elev=ELEV, azim=AZIM)
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-R_LIM, R_LIM)

        draw_frame(ax)
        draw_angles(ax, theta, phi)
        if track:
            draw_track(ax, theta, phi)
        draw_state(ax, theta, phi)

        ax.text2D(0.5, 0.99, title, transform=ax.transAxes, ha="center",
                  va="top", fontsize=8.8, fontweight="bold", color=INK)
        head, tail = caption(theta, phi)
        ax.text2D(0.5, 0.075, head, transform=ax.transAxes, ha="center",
                  va="center", fontsize=7.6, color=INK)
        ax.text2D(0.5, 0.015, tail, transform=ax.transAxes, ha="center",
                  va="center", fontsize=6.8, color=SUB)

    axes[2].text2D(0.5, 0.135, "purple track: driven evolution from "
                   r"$|0\rangle$", transform=axes[2].transAxes, ha="center",
                   va="center", fontsize=6.6, color=PURPLE)

    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.subplots_adjust(wspace=0.02)
    panel_labels_inches(fig, axes, dx=0.52, dy=-0.62)
    savefig(fig, "493_qc_bloch_sphere.png")


if __name__ == "__main__":
    main()
