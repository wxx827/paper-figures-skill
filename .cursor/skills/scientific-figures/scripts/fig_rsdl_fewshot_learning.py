"""Figure 339 - Few-shot remote-sensing scene classification.

A 5-way 5-shot episode: the support set on the left with class-coloured
chip borders, the query set on the right with green / red verdict borders,
and two subordinate panels (58% height) showing how accuracy grows with the
number of shots and how it splits across classes.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, add_panel_label, style_axes

CLASSES = ["Farmland", "Forest", "River", "Dense residential", "Bare land"]
SHORT = ["Farm", "Forest", "River", "Resid.", "Bare"]
CLASS_COLORS = ["#8fbc4a", "#1d6b3f", "#3b7ea1", "#b23a48", "#d9a86c"]
OK_GREEN = "#2e7d5b"
BAD_RED = "#b23a48"
CHIP = 44


def _noise(rng, n, sigma):
    a = gaussian_filter(rng.standard_normal((n, n)), sigma, mode="wrap")
    return a / (a.std() + 1e-9)


def _fbm(rng, n, scales=(2, 5, 12), weights=(0.5, 0.32, 0.18)):
    out = sum(w * _noise(rng, n, s) for s, w in zip(scales, weights))
    return out / (out.std() + 1e-9)


def _rot(xx, yy, ang):
    th = np.deg2rad(ang)
    return xx * np.cos(th) + yy * np.sin(th), -xx * np.sin(th) + yy * np.cos(th)


def chip(cls, seed, n=CHIP):
    """One synthetic scene chip for the requested class."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    bands = np.zeros((3, n, n))

    if cls == 0:                                    # farmland parcels
        u, v = _rot(xx, yy, rng.uniform(-60, 60))
        u = u + 1.5 * _noise(rng, n, 6)
        pid = (np.floor(u / rng.uniform(6, 10)).astype(int) * 173
               + np.floor(v / rng.uniform(9, 16)).astype(int) * 61)
        vig = rng.random(2048)[np.abs(pid) % 2048] ** 0.55
        for k, (vg, sl) in enumerate([(0.036, 0.200), (0.084, 0.180),
                                      (0.034, 0.152)]):
            bands[k] = vig * vg + (1 - vig) * sl
        amp = 0.07
    elif cls == 1:                                  # closed forest canopy
        f = _fbm(rng, n, (1.2, 3.0, 7.0))
        for k, base in enumerate([0.036, 0.070, 0.040]):
            bands[k] = base * (1 + 0.55 * f)
        amp = 0.26
    elif cls == 2:                                  # river through vegetation
        t = np.linspace(0, n - 1, 120)
        cx = 0.5 * n + rng.uniform(0.08, 0.16) * n * np.sin(
            2 * np.pi * t / n * rng.uniform(0.8, 1.4) + rng.uniform(0, 6))
        d = np.min(np.abs(xx[..., None] - np.interp(yy, t, cx)[..., None]),
                   axis=-1)
        water = d < rng.uniform(4.5, 7.5) + 1.2 * _noise(rng, n, 5)
        veg = _fbm(rng, n, (2, 5, 11))
        for k, (vg, sl) in enumerate([(0.040, 0.170), (0.086, 0.160),
                                      (0.038, 0.135)]):
            m = 0.5 + 0.35 * veg
            bands[k] = np.clip(m, 0, 1) * vg + (1 - np.clip(m, 0, 1)) * sl
        for k, val in enumerate((0.028, 0.046, 0.062)):
            bands[k] = np.where(water, val, bands[k])
        amp = 0.10
    elif cls == 3:                                  # dense residential
        ang = rng.uniform(-40, 40)
        u, v = _rot(xx, yy, ang)
        cu, pv = rng.uniform(8, 11), rng.uniform(5.5, 7.5)
        fu, fv = np.mod(u, cu), np.mod(v, pv)
        street = fu < 2.1
        bid = (np.floor(u / cu).astype(int) * 331
               + np.floor(v / pv).astype(int) * 17)
        key = np.abs(bid) % 2048
        keep = rng.random(2048)[key] > 0.14
        build = (fu > 2.6) & (fu < cu - 1.2) & (fv > 1.0) & (fv < pv - 1.0) \
            & keep & ~street
        for k, base in enumerate([0.086, 0.126, 0.084]):
            bands[k] = base
        for k, val in enumerate((0.128, 0.132, 0.138)):
            bands[k] = np.where(street, val, bands[k])
        bright = 0.13 + 0.24 * rng.random(2048)[key]
        terra = rng.random(2048)[key] > 0.68
        for k, f in enumerate((1.0 + 0.42 * terra, 1.0 - 0.14 * terra,
                               1.0 - 0.34 * terra)):
            bands[k] = np.where(build, bright * f, bands[k])
        sh = np.zeros((n, n), bool)
        for d in range(1, 3):
            sh |= np.roll(np.roll(build, d, 0), d, 1)
        sh &= ~build
        for k in range(3):
            bands[k] = np.where(sh, bands[k] * 0.45, bands[k])
        amp = 0.14
    else:                                           # bare land / dunes
        u, _v = _rot(xx, yy, rng.uniform(0, 180))
        ripple = 0.5 + 0.5 * np.sin(u / rng.uniform(3.5, 6.0)
                                    + 2.0 * _noise(rng, n, 7))
        f = _fbm(rng, n, (2, 6, 14))
        for k, base in enumerate([0.235, 0.205, 0.168]):
            bands[k] = base * (1 + 0.20 * f + 0.14 * ripple)
        amp = 0.09

    tex = 1 + amp * (0.6 * _noise(rng, n, 0.8) + 0.4 * _noise(rng, n, 2.2))
    for k in range(3):
        bands[k] = bands[k] * tex + 0.003 * rng.standard_normal((n, n))
    # One radiometric scale for every chip: per-chip stretching would wash
    # dark forest up to the brightness of bare land and destroy the classes.
    lo, hi = 0.012, 0.300
    return np.dstack([np.clip((bands[i] - lo) / (hi - lo), 0, 1) ** 0.74
                      for i in range(3)])


def chip_axes(ax, color, lw=1.3):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(lw)
        s.set_color(color)


def main():
    set_pub_style(base_font=8)
    rng = np.random.default_rng(77)

    fig = plt.figure(figsize=(10.0, 6.0))
    gs_s = fig.add_gridspec(5, 5, left=0.098, right=0.452, top=0.845,
                            bottom=0.375, wspace=0.075, hspace=0.075)
    gs_q = fig.add_gridspec(3, 6, left=0.520, right=0.988, top=0.845,
                            bottom=0.470, wspace=0.075, hspace=0.30)
    gs_b = fig.add_gridspec(1, 2, left=0.098, right=0.978, top=0.290,
                            bottom=0.088, wspace=0.24)

    # ---- (a) support set --------------------------------------------------
    for r in range(5):
        for c in range(5):
            ax = fig.add_subplot(gs_s[r, c])
            chip_axes(ax, CLASS_COLORS[r])
            ax.imshow(chip(r, 1000 + 37 * r + c), interpolation="nearest")
            if c == 0:
                ax.set_ylabel(CLASSES[r], fontsize=6.8, fontweight="bold",
                              labelpad=3, color=CLASS_COLORS[r],
                              rotation=0, ha="right", va="center")
            if r == 0:
                ax.set_title(f"shot {c + 1}", fontsize=6.4, pad=2.2,
                             color="#555555")
            if r == 0 and c == 0:
                add_panel_label(ax, "(a)", x=-2.35, y=1.32, size=11)

    ax0 = fig.axes[0].get_position()
    ax4 = fig.axes[24].get_position()
    fig.text(0.5 * (ax0.x0 + ax4.x1), ax0.y1 + 0.075,
             "Support set  -  5-way 5-shot episode", ha="center", va="bottom",
             fontsize=8.2, fontweight="bold")

    # ---- (b) query set ----------------------------------------------------
    truth = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 2, 3, 1])
    pred = truth.copy()
    pred[4] = 0        # bare land read as farmland
    pred[9] = 3        # bare land read as residential
    pred[13] = 4       # residential read as bare land
    qaxes = []
    for i in range(18):
        ax = fig.add_subplot(gs_q[i // 6, i % 6])
        qaxes.append(ax)
        good = pred[i] == truth[i]
        chip_axes(ax, OK_GREEN if good else BAD_RED, lw=1.5)
        ax.imshow(chip(truth[i], 5000 + 53 * i), interpolation="nearest")
        ax.set_xlabel(SHORT[pred[i]], fontsize=6.4,
                      color=OK_GREEN if good else BAD_RED,
                      fontweight="bold" if not good else "normal",
                      labelpad=1.6)
        if i == 0:
            add_panel_label(ax, "(b)", x=-0.30, y=1.30, size=11)

    q0 = qaxes[0].get_position()
    q5 = qaxes[5].get_position()
    acc_q = float(np.mean(pred == truth))
    fig.text(0.5 * (q0.x0 + q5.x1), q0.y1 + 0.075,
             f"Query set  -  {int(acc_q * 18)}/18 correct "
             f"({100 * acc_q:.1f}%)", ha="center", va="bottom",
             fontsize=8.2, fontweight="bold")

    # ---- (c) accuracy versus shots ---------------------------------------
    ax_c = fig.add_subplot(gs_b[0, 0])
    shots = np.array([1, 5, 10, 20])
    curves = {"ProtoNet": np.array([0.521, 0.679, 0.742, 0.788]),
              "RelationNet": np.array([0.548, 0.708, 0.771, 0.816]),
              "Ours (Meta-RS)": np.array([0.612, 0.776, 0.828, 0.865])}
    err = {"ProtoNet": 0.036, "RelationNet": 0.031, "Ours (Meta-RS)": 0.024}
    cols = ["#9aa7b1", "#7d8f9e", "#b23a48"]
    for (nm, v), cc, mk in zip(curves.items(), cols, ("s", "^", "o")):
        ax_c.fill_between(shots, v - err[nm], v + err[nm], color=cc,
                          alpha=0.16, linewidth=0)
        ax_c.plot(shots, v, color=cc, lw=1.6, marker=mk, ms=4.0,
                  mec="white", mew=0.6, label=nm)
    for s, v in zip(shots, curves["Ours (Meta-RS)"]):
        ax_c.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=6.2,
                      color="#7a1f2b")
    ax_c.set_xscale("log")
    ax_c.set_xticks(shots)
    ax_c.set_xticklabels([f"{s}-shot" for s in shots], fontsize=6.8)
    ax_c.minorticks_off()
    ax_c.set_ylim(0.44, 0.94)
    ax_c.set_xlabel("labelled samples per class (-)", fontsize=7.4,
                    labelpad=1.5)
    ax_c.set_ylabel("5-way accuracy (-)", fontsize=7.4, labelpad=2)
    ax_c.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_c.grid(linewidth=0.4, alpha=0.6)
    ax_c.legend(frameon=False, fontsize=6.6, loc="lower right",
                handlelength=1.4)
    style_axes(ax_c)
    ax_c.set_title("Accuracy grows with the number of shots", fontsize=7.8,
                   fontweight="bold", pad=3.0)
    add_panel_label(ax_c, "(c)", x=-0.115, y=1.05, size=11)

    # ---- (d) per-class accuracy at 5 shots --------------------------------
    ax_d = fig.add_subplot(gs_b[0, 1])
    per_cls = np.array([0.812, 0.851, 0.867, 0.744, 0.706])
    base_cls = np.array([0.716, 0.769, 0.783, 0.641, 0.585])
    y = np.arange(5)[::-1]
    ax_d.barh(y + 0.19, per_cls, 0.34, color=CLASS_COLORS,
              edgecolor="#333333", linewidth=0.5, label="Ours (Meta-RS)")
    ax_d.barh(y - 0.19, base_cls, 0.34, color="#c6ccd1",
              edgecolor="#333333", linewidth=0.5, label="ProtoNet")
    for yi, a, b in zip(y, per_cls, base_cls):
        ax_d.text(a + 0.012, yi + 0.19, f"{a:.3f}", va="center",
                  fontsize=6.2, color="#333333")
        ax_d.text(b + 0.012, yi - 0.19, f"{b:.3f}", va="center",
                  fontsize=6.2, color="#777777")
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(CLASSES, fontsize=6.8)
    for t, c in zip(ax_d.get_yticklabels(), CLASS_COLORS):
        t.set_color(c if c != "#d9a86c" else "#a9743d")
    ax_d.set_xlim(0, 1.06)
    ax_d.set_xlabel("5-shot class accuracy (-)", fontsize=7.4, labelpad=1.5)
    ax_d.tick_params(labelsize=6.8, length=2, pad=1.5)
    ax_d.grid(axis="x", linewidth=0.4, alpha=0.6)
    ax_d.legend(frameon=False, fontsize=6.6, loc="lower right",
                handlelength=1.3)
    style_axes(ax_d)
    ax_d.set_title("Bare land stays the hardest class", fontsize=7.8,
                   fontweight="bold", pad=3.0)
    add_panel_label(ax_d, "(d)", x=-0.185, y=1.05, size=11)

    handles = [Line2D([], [], color=OK_GREEN, lw=1.8,
                      label="Query classified correctly"),
               Line2D([], [], color=BAD_RED, lw=1.8,
                      label="Query misclassified"),
               Patch(facecolor="none", edgecolor="#8fbc4a", linewidth=1.4,
                     label="Support chip border = class colour")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                     frameon=False, fontsize=6.8, handlelength=1.5,
                     bbox_to_anchor=(0.54, 0.004))
    leg.set_clip_on(False)


    savefig(fig, "339_rsdl_fewshot_learning.png")


if __name__ == "__main__":
    main()
