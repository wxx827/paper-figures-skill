"""Simulation-environment gallery: 16 representative frames in four families.

Every tile is rendered by ``_wm_scenes.build`` with a *different* skeleton or a
different room, camera preset, focal length and white balance - eight scene
kinds across sixteen tiles, never one kind re-seeded sixteen times.  The four
family bands (navigation, manipulation, cooperation, competition) are separated
by hairlines with the family name in the right-hand margin, the way a dataset
figure labels its rows.

Each tile carries the environment name and its observation shape; the footer
counts the families straight off ``ENVS``, so the tally can never drift from
the grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from pubstyle import COLORS, savefig, set_pub_style
from _wm_scenes import body_radii, build, human_pose, project

BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"

H, W = 118, 164
DZ = 3.55
PROJ = dict(f=1.30, cam_y=1.02, cam_z=-DZ, cy=0.520, aspect=W / H)

# A cool, low-chroma house camera: the batch caps warm area at 8 percent and a
# room lit by tungsten would blow straight through it.
COOL = dict(wb=(0.955, 1.000, 1.045), sat=0.74)


def cam(**kw):
    c = dict(COOL)
    c.update(kw)
    return c


FAMILIES = ["Navigation", "Manipulation", "Cooperation", "Competition"]

ENVS = [
    # ---- navigation ------------------------------------------------------
    dict(fam=0, name="UrbanDrive-v2", obs="3 x 128 x 128", kind="road",
         seed=201, t=0.30, cam=cam(exposure=1.00, contrast=1.10, dof=1.3),
         par=dict(horizon=0.455, vpoint=0.50, block=13)),
    dict(fam=0, name="AerialSurvey-8", obs="3 x 96 x 96", kind="aerial",
         seed=202, t=0.22, cam=cam(exposure=1.05, contrast=1.14, dof=1.0),
         par=dict(pan=34.0, plan=7)),
    dict(fam=0, name="HomeNav-XL", obs="3 x 84 x 84 + 12-d", kind="indoor",
         seed=203, t=0.00, cam=cam(exposure=1.07, contrast=1.10, dof=1.1,
                                   sharpen=1.25, vignette=0.22),
         par=dict(horizon=0.545)),
    dict(fam=0, name="MarketWalk-v1", obs="3 x 112 x 112", kind="ego",
         seed=204, t=0.00, cam=cam(exposure=1.18, contrast=1.12, dof=1.2,
                                   sharpen=1.30, vignette=0.20),
         par=dict(room="store")),
    # ---- manipulation ----------------------------------------------------
    dict(fam=1, name="TableTop-7", obs="3 x 128 x 128", kind="tabletop",
         seed=211, t=0.00, cam=cam(exposure=1.06, contrast=1.16, dof=1.6,
                                   sharpen=1.30),
         par=dict(horizon=0.340)),
    dict(fam=1, name="LinePick-v1", obs="3 x 120 x 90", kind="conveyor",
         seed=212, t=0.35, cam=cam(exposure=1.05, contrast=1.12, dof=1.1),
         par=dict()),
    dict(fam=1, name="BenchAssembly", obs="3 x 112 x 112 + 7-d", kind="ego",
         seed=213, t=0.00, cam=cam(exposure=1.10, contrast=1.12, dof=1.3,
                                   sharpen=1.25, vignette=0.24),
         par=dict(room="workbench")),
    dict(fam=1, name="ShelfSort-3", obs="3 x 96 x 96", kind="shelf",
         seed=214, t=0.00, cam=cam(exposure=1.28, contrast=1.08, dof=1.4,
                                   sharpen=1.20, vignette=0.18),
         par=dict(shelves=3, plan=3)),
    # ---- cooperation -----------------------------------------------------
    dict(fam=2, name="TeamCarry-2", obs="3 x 128 x 128 + 18-d", kind="studio",
         seed=221, t=0.00, cam=cam(exposure=1.02, contrast=1.09, dof=1.4),
         par=dict(horizon=0.420, pose=("walk", 0.18, 0.30),
                  shirt="#4c5967", trousers="#3a414b")),
    dict(fam=2, name="CookAssist-v2", obs="3 x 112 x 112", kind="ego",
         seed=222, t=0.00, cam=cam(exposure=1.12, contrast=1.11, dof=1.3,
                                   sharpen=1.25, vignette=0.24),
         par=dict(room="kitchen")),
    dict(fam=2, name="SwarmRelay-16", obs="3 x 96 x 96 + 64-d", kind="aerial",
         seed=223, t=0.62, cam=cam(exposure=1.08, contrast=1.10, dof=1.4),
         par=dict(pan=62.0, plan=19)),
    dict(fam=2, name="RoomTidy-4", obs="3 x 84 x 84", kind="ego",
         seed=224, t=0.00, cam=cam(exposure=1.12, contrast=1.08, dof=1.3,
                                   sharpen=1.20, vignette=0.20),
         par=dict(room="living")),
    # ---- competition -----------------------------------------------------
    dict(fam=3, name="RaceMerge-v3", obs="3 x 128 x 128", kind="road",
         seed=231, t=0.72, cam=cam(exposure=1.00, contrast=1.16, dof=1.4,
                                   vignette=0.30),
         par=dict(horizon=0.395, vpoint=0.62, block=9)),
    dict(fam=3, name="ArenaDuel-2", obs="3 x 128 x 128", kind="studio",
         seed=232, t=0.00, cam=cam(exposure=1.04, contrast=1.14, dof=1.2,
                                   sharpen=1.20),
         par=dict(horizon=0.560, pose=("turn", 0.42, -0.20),
                  shirt="#41505f", trousers="#333a44")),
    dict(fam=3, name="TokenGrab-9", obs="3 x 96 x 96 + 9-d", kind="tabletop",
         seed=233, t=0.00, cam=cam(exposure=1.05, contrast=1.20, dof=1.6,
                                   sharpen=1.30),
         par=dict(horizon=0.265)),
    dict(fam=3, name="LabRivals-v1", obs="3 x 112 x 112", kind="ego",
         seed=234, t=0.00, cam=cam(exposure=1.16, contrast=1.13, dof=1.3,
                                   sharpen=1.25, vignette=0.22),
         par=dict(room="lab")),
]


FAM_COL = [BLUE[0], BLUE[1], BLUE[2], BLUE[3]]


def obs_size(text):
    """Flattened observation size implied by the string printed on the tile."""
    total = 0
    for part in text.split("+"):
        part = part.strip()
        if part.endswith("-d"):
            total += int(part[:-2])
        else:
            v = 1
            for f in part.split("x"):
                v *= int(f.strip())
            total += v
    return total


def render(spec):
    par = dict(spec["par"])
    pose = par.pop("pose", None)
    if pose is not None:
        kind, phase, yaw = pose
        pts, _ = project(human_pose(kind, phase=phase), yaw=yaw, **PROJ)
        par.update(body=pts, radii=body_radii(DZ),
                   head_r=0.108 * PROJ["f"] / DZ)
    return build(spec["kind"], seed=spec["seed"], shape=(H, W), t=spec["t"],
                 cam=spec["cam"], **par)


def main():
    set_pub_style(base_font=8)

    counts = [sum(1 for e in ENVS if e["fam"] == f) for f in range(4)]
    assert sum(counts) == len(ENVS) == 16 and set(counts) == {4}
    kinds = sorted({e["kind"] for e in ENVS})

    fig = plt.figure(figsize=(9.6, 7.6))
    x0, tw, gx = 0.034, 0.1600, 0.0125
    th = tw * 9.6 / 7.6 * 0.720
    gy, top = 0.0270, 0.8860
    pitch = th + gy
    x_end = x0 + 4 * tw + 3 * gx

    for k, spec in enumerate(ENVS):
        r, c = divmod(k, 4)
        y = top - r * pitch - th
        ax = fig.add_axes([x0 + c * (tw + gx), y, tw, th])
        ax.imshow(render(spec)["rgb"], interpolation="bilinear", zorder=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BLUE[1])
            s.set_linewidth(0.7)
        ax.add_patch(FancyBboxPatch(
            (0.325, 0.035), 0.645, 0.275, transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor="#0f1720", edgecolor="none", alpha=0.62, zorder=4))
        ax.text(0.955, 0.185, spec["name"], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.4, fontweight="bold",
                color="white", zorder=5)
        ax.text(0.955, 0.062, "obs " + spec["obs"], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.0, color=BLUE[5],
                zorder=5)

    # ---- family bands: hairline between rows, name in the right margin ----
    for f in range(4):
        yc = top - f * pitch - th / 2.0
        fig.text(x_end + 0.020, yc + 0.020, FAMILIES[f], ha="left",
                 va="center", fontsize=9.0, fontweight="bold",
                 color=FAM_COL[f])
        fig.text(x_end + 0.020, yc - 0.014,
                 f"{counts[f]:d} environments", ha="left", va="center",
                 fontsize=6.8, color=COLORS["sub"])
        if f:                                    # separator above each band
            ys = top - f * pitch + gy / 2.0
            fig.add_artist(Line2D([x0, 0.988], [ys, ys], color="#c8d2db",
                                  lw=0.7, zorder=1))


    # ---- how big an observation is, per family ----------------------------
    # Every bar is parsed back out of the string printed on the tiles, so the
    # summary cannot drift away from the grid it summarises.
    sizes = np.array([obs_size(e["obs"]) for e in ENVS], dtype=float)
    assert obs_size("3 x 128 x 128 + 18-d") == 3 * 128 * 128 + 18
    means = np.array([sizes[[e["fam"] == f for e in ENVS]].mean()
                      for f in range(4)])

    axb = fig.add_axes([0.122, 0.062, 0.225, 0.112])
    ypos = np.arange(4)
    axb.barh(ypos, means / 1000.0, height=0.62, color=FAM_COL,
             edgecolor="white", linewidth=0.5, zorder=3)
    for f in range(4):
        axb.text(means[f] / 1000.0 + 0.9, f, f"{means[f] / 1000.0:.1f}k",
                 ha="left", va="center", fontsize=6.4, color=INK, zorder=4)
    axb.set_yticks(ypos)
    axb.set_yticklabels(FAMILIES, fontsize=6.8, color=INK)
    axb.invert_yaxis()
    axb.set_xlim(0, max(means) / 1000.0 * 1.30)
    axb.set_xticks([0, 20, 40])
    axb.tick_params(axis="x", labelsize=6.4, length=2.2)
    axb.tick_params(axis="y", length=0, pad=2.5)
    axb.grid(axis="x", color="#e6eaee", lw=0.6)
    axb.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axb.spines[side].set_visible(False)
    axb.set_xlabel("mean flattened observation (thousand floats)",
                   fontsize=6.6, color=COLORS["sub"], labelpad=1.5)
    fig.text(0.034, 0.196, "Observation budget per family", ha="left",
             va="bottom", fontsize=7.4, fontweight="bold", color=INK)

    key = [Line2D([], [], marker="s", ls="none", ms=6.0, mfc="#0f1720",
                  mec="none", label="name + observation shape overlay"),
           Line2D([], [], color="#c8d2db", lw=0.9,
                  label="family separator hairline")]
    leg = fig.legend(handles=key, loc="upper left",
                     bbox_to_anchor=(0.435, 0.200), ncol=1, fontsize=7.0,
                     frameon=False, handletextpad=0.55, labelspacing=0.55)
    for txt in leg.get_texts():
        txt.set_color(COLORS["sub"])

    fig.text(0.435, 0.108,
             f"{len(kinds):d} scene skeletons ({', '.join(kinds)}) - a "
             f"different camera preset per tile, so exposure,\nfocal length, "
             f"white balance and depth of field all vary between "
             f"environments.  Sizes run\n"
             f"{sizes.min() / 1000.0:.1f}k to {sizes.max() / 1000.0:.1f}k "
             f"floats per observation across the {len(ENVS):d} tiles.",
             ha="left", va="top", fontsize=6.6, color=COLORS["muted"],
             linespacing=1.55)

    savefig(fig, "1184_wm_mas_env_gallery.png")


if __name__ == "__main__":
    main()
