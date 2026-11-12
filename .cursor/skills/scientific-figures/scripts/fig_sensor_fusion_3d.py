"""Camera / LiDAR / radar fusion on a single frame.

2 x 2 panel: (a) the camera image with the LiDAR sweep projected onto it and
coloured by range, (b) the raw 3-D LiDAR cloud with detections, (c) sparse
radar returns carrying Doppler velocity vectors, (d) the fused BEV output.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from pubstyle import set_pub_style, savefig

SEED = 2077
CAM = np.array([0.0, 0.0, 1.62])
LID = np.array([0.0, 0.0, 1.92])
DMAX = 52.0
SKY = np.array([0.79, 0.86, 0.93])
IMW, IMH = 300, 226
FOC = 0.90 * IMW
CX, CY = IMW / 2.0, IMH * 0.46

SEM = {"vehicle": "#3b5b92", "pedestrian": "#c9a227", "other": "#8e6bbf"}
FWD, LAT = (-4.0, 44.0), 20.0


def build_scene():
    boxes, spheres, tgt = [], [], []

    def box(cx, cy, cz, sx, sy, sz, yaw, col, kind="box"):
        boxes.append(dict(c=np.array([cx, cy, cz]), s=np.array([sx, sy, sz]),
                          yaw=yaw, col=np.array(to_rgb(col)), kind=kind))

    for cx, cy, sx, sy, h, c in [(10, 18.0, 18, 11, 10.0, "#c3a684"),
                                 (32, 18.5, 18, 12, 13.0, "#b09a86"),
                                 (8, -18.0, 18, 11, 11.5, "#c9b49b"),
                                 (30, -18.5, 18, 12, 8.5, "#a89279"),
                                 (60, 3.0, 14, 44, 12.0, "#b6a893")]:
        box(cx, cy, h / 2, sx, sy, h, 0.0, c)

    cars = [(11.0, 3.3, 0.03, "#cbced3", 9.6), (22.5, 3.1, -0.03, "#9b4046", 11.4),
            (34.0, 3.4, 0.02, "#6b7280", 8.2), (13.0, -3.4, 3.12, "#e8e8e8", -7.8),
            (25.0, -3.2, 3.14, "#5c6f8c", -10.5), (38.0, -3.5, 3.11, "#a7adb3", -9.1)]
    for cx, cy, yaw, col, vx in cars:
        box(cx, cy, 0.79, 4.5, 1.95, 1.58, yaw, col, kind="car")
        tgt.append(dict(c=np.array([cx, cy]), s=np.array([4.7, 2.1]), yaw=yaw,
                        cls="vehicle", v=vx))
    for cx, cy in [(9.5, 8.6), (10.7, 9.2), (27.0, -8.5)]:
        box(cx, cy, 0.88, 0.62, 0.55, 1.76, 0.0, "#d98b6a", kind="ped")
        tgt.append(dict(c=np.array([cx, cy]), s=np.array([0.9, 0.85]), yaw=0.0,
                        cls="pedestrian", v=1.3))
    for cx, cy in [(17.0, 9.0)]:
        box(cx, cy, 0.72, 1.7, 0.55, 1.45, 0.0, "#7d8ea3", kind="cyc")
        tgt.append(dict(c=np.array([cx, cy]), s=np.array([1.9, 0.9]), yaw=0.0,
                        cls="other", v=4.6))
    for cx, cy in [(5, 9.8), (17, 9.8), (29, 9.8), (41, 9.8),
                   (3, -9.8), (15, -9.8), (27, -9.8), (39, -9.8)]:
        box(cx, cy, 1.6, 0.38, 0.38, 3.2, 0.0, "#6b4b32")
        spheres.append(dict(c=np.array([cx, cy, 4.8]), r=2.15,
                            col=np.array([0.29, 0.50, 0.28])))
    return boxes, spheres, tgt


def _slab(o, D, b):
    c, s = np.cos(-b["yaw"]), np.sin(-b["yaw"])
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    oo, DD = R @ (o - b["c"]), D @ R.T
    DD = np.where(np.abs(DD) < 1e-9, 1e-9, DD)
    h = b["s"] / 2.0
    t1, t2 = (-h - oo) / DD, (h - oo) / DD
    tmin = np.max(np.minimum(t1, t2), axis=1)
    tmax = np.min(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax > 0.0)
    t = np.where(tmin > 0.0, tmin, tmax)
    return np.where(hit & (t > 0.0), t, np.inf), R


def trace(o, D, boxes, spheres, rng):
    t = np.full(D.shape[0], np.inf)
    prim = np.full(D.shape[0], -1)
    dz = np.where(np.abs(D[:, 2]) < 1e-9, 1e-9, D[:, 2])
    tg = (0.0 - o[2]) / dz
    tg = np.where((D[:, 2] < -1e-4) & (tg > 0), tg, np.inf)
    u = tg < t
    t[u], prim[u] = tg[u], 0
    rots = []
    for i, b in enumerate(boxes):
        tb, R = _slab(o, D, b)
        rots.append(R)
        u = tb < t
        t[u], prim[u] = tb[u], i + 1
    off = len(boxes) + 1
    for j, sp in enumerate(spheres):
        L = o - sp["c"]
        bq = 2.0 * (D @ L)
        disc = bq * bq - 4.0 * (L @ L - sp["r"] ** 2)
        ts = (-bq - np.sqrt(np.maximum(disc, 0.0))) / 2.0
        ts = np.where((disc > 0) & (ts > 0), ts, np.inf)
        ts = ts * (1.0 + rng.normal(0, 0.012, ts.size))
        u = ts < t
        t[u], prim[u] = ts[u], off + j
    return t, prim, rots, off


def render(boxes, spheres, rng):
    u = np.arange(IMW) + 0.5 - CX
    v = np.arange(IMH) + 0.5 - CY
    U, V = np.meshgrid(u, v)
    D = np.stack([np.full(U.shape, FOC), -U, -V], axis=-1)
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    Df = D.reshape(-1, 3)
    t, prim, rots, off = trace(CAM, Df, boxes, spheres, rng)
    valid = np.isfinite(t) & (t < DMAX)
    P = CAM + Df * np.where(valid, t, DMAX)[:, None]

    col = np.zeros((Df.shape[0], 3))
    nrm = np.zeros((Df.shape[0], 3))
    g = prim == 0
    road = np.abs(P[:, 1]) < 8.0
    base = np.where(road[:, None], np.array([0.45, 0.47, 0.49]),
                    np.array([0.61, 0.60, 0.56]))
    lane = ((np.abs(np.abs(P[:, 1]) - 4.0) < 0.16) | (np.abs(P[:, 1]) < 0.18))
    base = np.where((lane & road)[:, None], np.array([0.85, 0.83, 0.61]), base)
    col[g], nrm[g] = base[g], np.array([0.0, 0.0, 1.0])
    for i, b in enumerate(boxes):
        m = prim == i + 1
        if not m.any():
            continue
        q = (P[m] - b["c"]) @ rots[i].T
        f = np.abs(q) / (b["s"] / 2.0)
        a_ = np.argmax(f, axis=1)
        n = np.zeros_like(q)
        n[np.arange(q.shape[0]), a_] = np.sign(q[np.arange(q.shape[0]), a_])
        nrm[m] = n @ rots[i]
        c = np.tile(b["col"], (m.sum(), 1))
        if b["kind"] == "box":
            band = ((np.floor(P[m][:, 2] / 1.9) % 2) == 1) & (a_ != 2)
            c = np.where(band[:, None], c * 0.74, c)
        col[m] = c
    for j, sp in enumerate(spheres):
        m = prim == off + j
        if not m.any():
            continue
        n = P[m] - sp["c"]
        nrm[m] = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        col[m] = sp["col"] * (0.80 + 0.40 * rng.random((m.sum(), 1)))
    light = np.array([0.35, 0.42, 0.84])
    light /= np.linalg.norm(light)
    lam = 0.66 + 0.40 * np.clip(nrm @ light, 0.0, 1.0)
    rgb = np.clip(col * lam[:, None], 0, 1)
    fog = np.clip((t - 18.0) / 70.0, 0.0, 0.48)[:, None]
    rgb = rgb * (1 - fog) + SKY * fog
    rgb[~valid] = SKY
    return rgb.reshape(IMH, IMW, 3)


def scan(boxes, spheres, rng, n_ch=48, n_az=1250):
    el = np.deg2rad(np.linspace(-24.0, 14.0, n_ch))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    E, A = E.ravel(), A.ravel()
    D = np.column_stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                         np.sin(E)])
    ch = np.repeat(np.arange(n_ch), n_az)
    t, prim, _, _ = trace(LID, D, boxes, spheres, rng)
    ok = np.isfinite(t) & (t < DMAX)
    P = LID + D[ok] * t[ok][:, None] + rng.normal(0, 0.022, (ok.sum(), 3))
    return P, t[ok], ch[ok]


def corners2d(c, s, yaw):
    h = np.asarray(s) / 2.0
    k = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * h
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return k @ R.T + np.asarray(c)


def bev_frame(ax, title):
    ax.set_xlim(LAT, -LAT)
    ax.set_ylim(*FWD)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#8a8a8a")
    ax.set_title(title, fontsize=8.2, pad=2.6)
    for r in (10, 20, 30, 40):
        th = np.linspace(-np.pi / 2, np.pi / 2, 120)
        ax.plot(r * np.sin(th), r * np.cos(th), color="#d0d0d0", lw=0.4,
                ls=(0, (3, 2)), zorder=1)
    ax.plot([0], [0], marker="^", ms=4.5, mfc="#b23a48", mec="#5c1113",
            mew=0.6, zorder=9)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(SEED)
    boxes, spheres, tgt = build_scene()
    rgb = render(boxes, spheres, rng)
    P, rr, ch = scan(boxes, spheres, rng)

    dcm = LinearSegmentedColormap.from_list(
        "d", ["#f6e27a", "#f0932b", "#c9432f", "#7b2a6b", "#25235c"])

    fig = plt.figure(figsize=(7.3, 5.35))
    W, H = 0.437, 0.383
    X = [0.052, 0.518]
    Y = [0.545, 0.108]

    # ---- (a) camera + projected LiDAR ------------------------------------
    ax = fig.add_axes([X[0], Y[0], W, H])
    ax.imshow(rgb, interpolation="bilinear", extent=[0, IMW, IMH, 0])
    q = P - CAM
    fr = (q[:, 0] > 1.5) & (ch % 3 == 0)          # keep every 3rd ring
    uu = CX - FOC * q[fr, 1] / q[fr, 0]
    vv = CY - FOC * q[fr, 2] / q[fr, 0]
    dd = np.linalg.norm(q[fr], axis=1)
    m = (uu > 0) & (uu < IMW) & (vv > 0) & (vv < IMH)
    ax.scatter(uu[m], vv[m], s=1.1, c=dd[m], cmap=dcm, vmin=4, vmax=42,
               marker=".", linewidths=0, alpha=0.92)
    ax.set_xlim(0, IMW)
    ax.set_ylim(IMH, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
        sp.set_color("#8a8a8a")
    ax.set_title("Camera image with projected LiDAR", fontsize=8.2, pad=2.6)

    cax = fig.add_axes([X[0] + W / 2 - 0.058, Y[0] + H - 0.050, 0.115, 0.012])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(4, 42), cmap=dcm),
                      cax=cax, orientation="horizontal", ticks=[5, 40])
    cb.set_label("LiDAR range (m)", fontsize=7.0, color="#1a1a1a", labelpad=1.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    cb.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, pad=1.0,
                      color="#333333")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("bottom")

    # ---- (b) raw LiDAR cloud ---------------------------------------------
    axb = fig.add_axes([X[1] - 0.030, Y[0] - 0.048, W + 0.060, H + 0.096],
                       projection="3d")
    axb.patch.set_alpha(0.0)
    keep = ((P[:, 0] > FWD[0]) & (P[:, 0] < FWD[1]) & (np.abs(P[:, 1]) < LAT)
            & (P[:, 2] < 14.0))
    Q, qr = P[keep], rr[keep]
    o = np.argsort(-qr)
    Q, qr = Q[o], qr[o]
    rgba = plt.get_cmap("viridis")(Normalize(-1, 12)(np.clip(Q[:, 2], 0, 12)))
    rgba[:, 3] = np.clip(0.95 - 0.007 * qr, 0.5, 0.95)
    axb.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=np.clip(2.6 - 0.04 * qr, 0.5, 2.6),
                c=rgba, marker=".", linewidths=0, depthshade=False,
                rasterized=True)
    for t_ in tgt:
        if not (FWD[0] < t_["c"][0] < FWD[1]):
            continue
        v = corners2d(t_["c"], t_["s"], t_["yaw"])
        hz = {"vehicle": 1.8, "pedestrian": 1.85, "other": 1.6}[t_["cls"]]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            for z in (0.03, hz):
                axb.plot([v[a, 0], v[b, 0]], [v[a, 1], v[b, 1]], [z, z],
                         color=SEM[t_["cls"]], lw=0.9, zorder=40)
        for a in range(4):
            axb.plot([v[a, 0]] * 2, [v[a, 1]] * 2, [0.03, hz],
                     color=SEM[t_["cls"]], lw=0.9, zorder=40)
    axb.set_xlim(*FWD)
    axb.set_ylim(-LAT, LAT)
    axb.set_zlim(0, 14)
    axb.set_box_aspect((1.0, 0.82, 0.44))
    axb.view_init(elev=25, azim=-64)
    axb.set_xticklabels([])
    axb.set_yticklabels([])
    axb.set_zticklabels([])
    axb.tick_params(length=0)
    for a in (axb.xaxis, axb.yaxis, axb.zaxis):
        a.pane.fill = False
        a.pane.set_edgecolor("#d8d8d8")
        a.pane.set_linewidth(0.45)
        a._axinfo["grid"].update(color="#ececec", linewidth=0.4)
    fig.text(X[1] + W / 2, Y[0] + H + 0.017, "LiDAR point cloud + 3-D boxes",
             fontsize=8.2, ha="center", va="center")

    # ---- (c) radar returns ------------------------------------------------
    axr = fig.add_axes([X[0], Y[1], W, H])
    bev_frame(axr, "Radar returns with Doppler velocity")
    for yy in (-8.0, 8.0):                        # faint road / kerb context
        axr.plot([yy, yy], [FWD[0], FWD[1]], color="#dcdcdc", lw=0.8, zorder=1)
    for cy, sy in ((18.0, 11), (-18.0, 11)):
        axr.add_patch(plt.Rectangle((cy - sy / 2, 2.0), sy, 30.0,
                                    facecolor="#f0ece7", edgecolor="#e0dbd4",
                                    linewidth=0.5, zorder=0))
    rad, radv = [], []
    for t_ in tgt:
        n = 3 if t_["cls"] == "vehicle" else 1
        for _ in range(n):
            p = t_["c"] + rng.normal(0, 0.55, 2)
            rad.append(p)
            radv.append(t_["v"])
    rad, radv = np.array(rad), np.array(radv)
    clut = np.column_stack([rng.uniform(2, 42, 22), rng.uniform(-18, 18, 22)])
    nv = Normalize(-12, 12)
    axr.scatter(clut[:, 1], clut[:, 0], s=9, facecolors="none",
                edgecolors="#a0a0a0", linewidths=0.6, marker="D", zorder=5)
    axr.scatter(rad[:, 1], rad[:, 0], s=20, c=radv, cmap="coolwarm", norm=nv,
                marker="D", edgecolors="#333333", linewidths=0.45, zorder=6)
    for p, v in zip(rad, radv):
        if abs(v) < 1.2:
            continue
        axr.annotate("", xy=(p[1], p[0] + v * 0.42), xytext=(p[1], p[0]),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, mutation_scale=6,
                                     color="#b23a48" if v > 0 else "#2166ac"),
                     zorder=7)
    axr.text(0.03, 0.02, f"{len(rad)} target returns  |  22 clutter",
             transform=axr.transAxes, fontsize=6.0, color="#555555",
             ha="left", va="bottom")
    cax2 = fig.add_axes([X[0] + W - 0.132, Y[1] + 0.030, 0.118, 0.013])
    cb2 = fig.colorbar(ScalarMappable(norm=nv, cmap="coolwarm"), cax=cax2,
                       orientation="horizontal", ticks=[-10, 0, 10])
    cb2.set_label("radial velocity (m s$^{-1}$)", fontsize=7.0,
                  color="#1a1a1a", labelpad=1.0)
    cb2.outline.set_linewidth(0.6)
    cb2.outline.set_edgecolor("#333333")
    cb2.ax.tick_params(labelsize=7.0, width=0.7, length=2.5, pad=1.0,
                       color="#333333")

    # ---- (d) fused BEV ----------------------------------------------------
    axf = fig.add_axes([X[1], Y[1], W, H])
    ny = 250
    Hh, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=[ny, ny],
                              range=[[FWD[0], FWD[1]], [-LAT, LAT]])
    axf.imshow(np.log1p(Hh), origin="lower", cmap="bone_r",
               extent=[LAT, -LAT, FWD[0], FWD[1]], vmin=0.0,
               vmax=np.percentile(np.log1p(Hh[Hh > 0]), 99.6), zorder=2,
               interpolation="nearest", alpha=0.80)
    bev_frame(axf, "Fused BEV detections")
    axf.scatter(rad[:, 1], rad[:, 0], s=9, c="#b23a48", marker="D",
                edgecolors="none", alpha=0.55, zorder=5)
    for t_ in tgt:
        v = corners2d(t_["c"], t_["s"], t_["yaw"])
        col = SEM[t_["cls"]]
        axf.add_patch(Polygon(v[:, ::-1], closed=True, facecolor=col,
                              alpha=0.18, edgecolor="none", zorder=6))
        axf.add_patch(Polygon(v[:, ::-1], closed=True, facecolor="none",
                              edgecolor=col, linewidth=1.1, zorder=7))
        d = np.array([np.cos(t_["yaw"]), np.sin(t_["yaw"])])
        tip = t_["c"] + d * (abs(t_["v"]) * 0.32 + 1.0) * np.sign(t_["v"] or 1)
        axf.annotate("", xy=(tip[1], tip[0]), xytext=(t_["c"][1], t_["c"][0]),
                     arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                     mutation_scale=6, color=col), zorder=8)
        axf.text(t_["c"][1] - 1.1, t_["c"][0] + 1.5, f"{abs(t_['v']):.0f}",
                 fontsize=6.0, color=col, ha="right", va="bottom", zorder=9)

    handles = [Line2D([], [], color=SEM["vehicle"], lw=1.5, label="Car"),
               Line2D([], [], color=SEM["pedestrian"], lw=1.5,
                      label="Pedestrian"),
               Line2D([], [], color=SEM["other"], lw=1.5, label="Cyclist"),
               Line2D([], [], color="#b23a48", marker="D", ls="none", ms=3.2,
                      label="Radar return")]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, 0.014), ncol=4, frameon=False,
                     fontsize=7.0, handlelength=1.6, columnspacing=1.4,
                     handletextpad=0.45)
    leg.set_zorder(20)

    for lab, x0, y0 in [("(a)", X[0], Y[0] + H), ("(b)", X[1], Y[0] + H),
                        ("(c)", X[0], Y[1] + H), ("(d)", X[1], Y[1] + H)]:
        fig.text(x0 - 0.030, y0 + 0.016, lab, fontsize=9.5, fontweight="bold",
                 ha="left", va="center")


    savefig(fig, "290_sensor_fusion_3d.png")


if __name__ == "__main__":
    main()
