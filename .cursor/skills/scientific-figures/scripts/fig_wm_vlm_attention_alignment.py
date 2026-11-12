"""Figure 1176 - Token-by-patch cross attention, with the peaks named.

The frame is cut into a 3 x 3 patch grid; the key in the top-left corner shows
the cut, and the strip down the left side shows the nine crops in the same
order as the matrix rows.  Columns are the ten tokens of the caption.

The alignment prior is read off the scene, not typed in: the patch a content
word should look at is found by asking which cell of the grid the generator's
anchor for that object falls into.  Every column is then normalised, so each
token spends exactly one unit of attention over the nine patches.

The three ringed cells come from ``argsort`` over the finished matrix, taken
greedily with no token and no patch used twice - so a call-out can never point
at a cell that is not a real local peak, and three rings can never pile onto
one hot patch.  It is a constrained maximum, not the unconstrained top three:
the second-strongest cell in the "hand" column is skipped because that patch
is already spoken for.  The card underneath says so and prints each weight
next to the column total it came out of.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

from pubstyle import set_pub_style, savefig, get_cmap
from _wm_scenes import build

INK = "#1a1a1a"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#d5d5d5"
DEEP = "#2f4a75"
TEAL = "#2a7f7a"
VIOLET = "#7b3294"
STEEL = "#3d6b8e"
PEAK_COLS = [DEEP, TEAL, VIOLET]

TOKENS = ["A", "hand", "reaches", "for", "the", "carton", "on", "the",
          "middle", "shelf"]
# token index -> the object whose anchor decides which patch it looks at,
# or a row of the grid ("row0" is the top row)
LOOKS_AT = {1: "hand_r", 2: "hand_r", 5: "carton", 8: "row1", 9: "row0"}

GR = 3                                   # 3 x 3 patch grid -> 9 rows
W, H = 8.8, 6.4
PH, PW = 183, 300                        # frame raster (3 | 3 divisible)

MX0, MW, MY0, MH = 1.76, 5.60, 1.62, 3.60
CELL_W, CELL_H = MW / len(TOKENS), MH / (GR * GR)
STRIP_R = 1.66                           # right edge of the patch strip
KEY = (0.34, 5.26, 1.30, 0.79)           # x, y, w, h of the grid key
CB = (7.54, 0.20)                        # colour bar x, width


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")
    return fig, ax


def box(ax, x0, y0, w, h, fc="white", ec=HAIR, lw=0.8, r=0.05, zorder=2):
    p = FancyBboxPatch((x0, y0), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)
    return p


def halo(ax, xs, ys, color, lw=0.9, zorder=14):
    ax.plot(xs, ys, color="white", lw=lw + 1.6, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder, alpha=0.90)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round",
            solid_joinstyle="round", zorder=zorder + 1)


def patch_of(anchor):
    """Which grid cell an anchor falls into, as a flat row-major index."""
    c = int(np.clip(anchor["cx"] / PW * GR, 0, GR - 1))
    r = int(np.clip(anchor["cy"] / PH * GR, 0, GR - 1))
    return r * GR + c


def build_attention(anc, rng):
    """A prior over patches per token, then a column-wise softmax-like norm."""
    npat = GR * GR
    A = 0.055 + 0.030 * rng.random((npat, len(TOKENS)))
    rr = np.repeat(np.arange(GR), GR)
    cc = np.tile(np.arange(GR), GR)
    for t, target in LOOKS_AT.items():
        if target.startswith("row"):
            want = int(target[3:])
            bump = np.exp(-0.5 * ((rr - want) / 0.55) ** 2)
        else:
            p = patch_of(anc[target])
            d = np.hypot(rr - p // GR, cc - p % GR)
            bump = np.exp(-0.5 * (d / 0.62) ** 2)
        A[:, t] += (0.92 if t in (5, 1) else 0.62) * bump
    # function words spread over whatever the neighbouring content word saw
    for t in (0, 3, 4, 6, 7):
        A[:, t] += 0.16 * A[:, max(t - 1, 0)]
    return A / A.sum(axis=0, keepdims=True)


def main():
    set_pub_style()
    fig, ax = canvas(W, H)
    rng = np.random.default_rng(1581)

    sc = build("ego", seed=1581, shape=(PH, PW), room="store",
               cam=dict(exposure=1.02, contrast=1.14, sat=0.76,
                        wb=(0.958, 1.000, 1.068), dof=1.7, sharpen=1.15))
    rgbf = np.clip(sc["rgb"], 0, 1)
    anc = {a["name"]: a for a in sc["anchors"]}

    A = build_attention(anc, rng)
    npat = GR * GR
    assert np.allclose(A.sum(axis=0), 1.0)

    # The three largest weights, restricted to distinct rows and columns so
    # the three call-outs name three different pairings rather than circling
    # one hot patch three times.
    peaks, seen_t, seen_p = [], set(), set()
    for flat in np.argsort(A, axis=None)[::-1]:
        p, t = int(flat // len(TOKENS)), int(flat % len(TOKENS))
        if t in seen_t or p in seen_p:
            continue
        seen_t.add(t)
        seen_p.add(p)
        peaks.append((p, t, float(A[p, t])))
        if len(peaks) == 3:
            break

    # ---- grid key ---------------------------------------------------------
    kx, ky, kw, kh = KEY
    ax.imshow(rgbf, extent=[kx, kx + kw, ky, ky + kh], zorder=3,
              interpolation="bilinear")
    ax.add_patch(Rectangle((kx, ky), kw, kh, facecolor="none",
                           edgecolor="#9a9a9a", lw=0.7, zorder=6))
    for g in range(1, GR):
        ax.plot([kx + g * kw / GR] * 2, [ky, ky + kh], color="white",
                lw=0.7, alpha=0.85, zorder=5)
        ax.plot([kx, kx + kw], [ky + g * kh / GR] * 2, color="white",
                lw=0.7, alpha=0.85, zorder=5)
    for p in range(npat):
        r, c = p // GR, p % GR
        ax.text(kx + c * kw / GR + 0.045, ky + kh - r * kh / GR - 0.040,
                f"{p + 1}", ha="left", va="top", fontsize=6.0,
                color="white", fontweight="bold", zorder=7)
    ax.text(kx + kw - 0.05, ky + 0.05, "input frame  ·  3 x 3 patches",
            ha="right", va="bottom", fontsize=6.0, color="#f0f0f0", zorder=8,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#2b2f33",
                      edgecolor="none", alpha=0.80))

    # ---- patch strip ------------------------------------------------------
    pw_pix, ph_pix = PW // GR, PH // GR
    tw = CELL_H * 0.86 / (ph_pix / pw_pix)
    strip_x = STRIP_R - tw
    prow_y = []
    for p in range(npat):
        r, c = p // GR, p % GR
        crop = rgbf[r * ph_pix:(r + 1) * ph_pix, c * pw_pix:(c + 1) * pw_pix]
        cy = MY0 + MH - (p + 0.5) * CELL_H
        prow_y.append(cy)
        th = CELL_H * 0.86
        ax.imshow(crop, extent=[strip_x, STRIP_R, cy - th / 2, cy + th / 2],
                  zorder=4, interpolation="bilinear")
        ax.add_patch(Rectangle((strip_x, cy - th / 2), tw, th,
                               facecolor="none", edgecolor="#b5b5b5",
                               lw=0.6, zorder=6))
        ax.text(strip_x - 0.07, cy, f"P{p + 1}", ha="right", va="center",
                fontsize=6.4, color=SUB)

    # ---- matrix -----------------------------------------------------------
    cmap = get_cmap("cb_sequential_blue")
    vmax = float(A.max())
    im = ax.imshow(A, extent=[MX0, MX0 + MW, MY0, MY0 + MH], cmap=cmap,
                   vmin=0.0, vmax=vmax, zorder=3, aspect="auto",
                   interpolation="nearest")
    ax.add_patch(Rectangle((MX0, MY0), MW, MH, facecolor="none",
                           edgecolor="#8a8a8a", lw=0.8, zorder=8))
    for g in range(1, len(TOKENS)):
        ax.plot([MX0 + g * CELL_W] * 2, [MY0, MY0 + MH], color="white",
                lw=0.45, alpha=0.55, zorder=6)
    for g in range(1, npat):
        ax.plot([MX0, MX0 + MW], [MY0 + g * CELL_H] * 2, color="white",
                lw=0.45, alpha=0.55, zorder=6)

    # ---- token chips ------------------------------------------------------
    chip_y, chip_h = 5.28, 0.30
    chip_x = []
    for t, tok in enumerate(TOKENS):
        cx = MX0 + (t + 0.5) * CELL_W
        chip_x.append(cx)
        hit = [k for k, (p, tt, v) in enumerate(peaks) if tt == t]
        col = PEAK_COLS[hit[0]] if hit else "#c2c2c2"
        box(ax, cx - CELL_W * 0.45, chip_y, CELL_W * 0.90, chip_h,
            fc="#eef2f7" if hit else "#f5f5f5", ec=col,
            lw=1.4 if hit else 0.7, r=0.045, zorder=4)
        ax.text(cx, chip_y + chip_h / 2, tok, ha="center", va="center",
                fontsize=6.6, color=INK if hit else SUB,
                fontweight="bold" if hit else "normal", zorder=5)
    ax.text(MX0, chip_y + chip_h + 0.13, "caption tokens", ha="left",
            va="center", fontsize=6.8, color=MUTED)
    ax.text(MX0 + MW, chip_y + chip_h + 0.13,
            "each column sums to 1.00 over the nine patches", ha="right",
            va="center", fontsize=6.6, color=MUTED)

    # ---- peak call-outs ---------------------------------------------------
    for k, (p, t, v) in enumerate(peaks):
        col = PEAK_COLS[k]
        cx = MX0 + (t + 0.5) * CELL_W
        cy = MY0 + MH - (p + 0.5) * CELL_H
        ax.add_patch(Rectangle((cx - CELL_W / 2, cy - CELL_H / 2), CELL_W,
                               CELL_H, facecolor="none", edgecolor="white",
                               lw=2.2, zorder=11))
        ax.add_patch(Rectangle((cx - CELL_W / 2, cy - CELL_H / 2), CELL_W,
                               CELL_H, facecolor="none", edgecolor=col,
                               lw=1.3, zorder=12))
        halo(ax, [cx, cx], [cy + CELL_H / 2, chip_y - 0.04], col)
        ax.add_patch(Circle((cx, chip_y - 0.04), 0.034, facecolor=col,
                            edgecolor="white", linewidth=0.7, zorder=16))
        halo(ax, [cx - CELL_W / 2, STRIP_R + 0.03], [cy, cy], col)
        ax.add_patch(Circle((STRIP_R + 0.03, cy), 0.034, facecolor=col,
                            edgecolor="white", linewidth=0.7, zorder=16))
        ax.add_patch(Rectangle((strip_x, cy - CELL_H * 0.43), tw,
                               CELL_H * 0.86, facecolor="none",
                               edgecolor=col, lw=1.5, zorder=9))

    # ---- colour bar (drawn, so it lands exactly beside the matrix) --------
    cbx, cbw = CB
    ramp = np.linspace(1, 0, 256)[:, None]
    ax.imshow(ramp, extent=[cbx, cbx + cbw, MY0, MY0 + MH], cmap=cmap,
              vmin=0, vmax=1, zorder=3, aspect="auto",
              interpolation="bilinear")
    ax.add_patch(Rectangle((cbx, MY0), cbw, MH, facecolor="none",
                           edgecolor="#8a8a8a", lw=0.7, zorder=6))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = MY0 + frac * MH
        ax.plot([cbx + cbw, cbx + cbw + 0.05], [yy, yy], color="#8a8a8a",
                lw=0.7, zorder=6)
        ax.text(cbx + cbw + 0.09, yy, f"{frac * vmax:.2f}", ha="left",
                va="center", fontsize=6.4, color=SUB)
    ax.text(cbx + cbw / 2, MY0 + MH + 0.12, "attention", ha="center",
            va="center", fontsize=6.6, color=SUB)
    ax.text(cbx + cbw / 2, MY0 - 0.14, "weight (-)", ha="center",
            va="center", fontsize=6.6, color=SUB)

    # ---- peak card --------------------------------------------------------
    cx0, cy0, cw, ch = 0.34, 0.40, 8.12, 0.94
    box(ax, cx0, cy0, cw, ch, fc="#f5f5f5", ec="#e4e4e4", lw=0.8, r=0.07,
        zorder=1)
    ax.text(cx0 + 0.18, cy0 + ch - 0.20,
            "the three strongest pairings, no token or patch used twice",
            ha="left", va="center", fontsize=7.4, fontweight="bold",
            color=INK, zorder=3)
    lx = cx0 + 0.18
    for k, (p, t, v) in enumerate(peaks):
        col = PEAK_COLS[k]
        ax.add_patch(Circle((lx + 0.07, cy0 + 0.32), 0.075, facecolor=col,
                            edgecolor="none", zorder=3))
        ax.text(lx + 0.07, cy0 + 0.32, str(k + 1), ha="center", va="center",
                fontsize=6.0, fontweight="bold", color="white", zorder=4)
        ax.text(lx + 0.20, cy0 + 0.32,
                f"\u201c{TOKENS[t]}\u201d \u2192 P{p + 1}    weight {v:.2f}",
                ha="left", va="center", fontsize=7.2, color=SUB, zorder=3)
        lx += 1.92
    ax.text(cx0 + 0.18 + 3 * 1.92, cy0 + 0.32,
            f"a flat column would give {1.0 / npat:.3f} to every patch",
            ha="left", va="center", fontsize=6.8, color=MUTED, zorder=3)
    ax.text(cx0 + cw - 0.18, cy0 + ch - 0.20,
            f"peak {peaks[0][2]:.2f} / {1.0 / npat:.3f} = "
            f"{peaks[0][2] / (1.0 / npat):.1f}x the flat weight", ha="right",
            va="center", fontsize=6.6, color=MUTED, zorder=3)


    savefig(fig, "1176_wm_vlm_attention_alignment.png")


if __name__ == "__main__":
    main()
