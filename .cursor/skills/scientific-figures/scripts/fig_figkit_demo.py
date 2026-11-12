"""Figure 353 - capability sheet for the shared helper module ``figkit``.

Six panels, one per helper group.  Every panel is drawn only with figkit plus
pubstyle, so the sheet doubles as a regression test: if a helper regresses,
this figure shows it.

(a) colour-boxed zoom insets with aspect-matched crops
(b) elbow leader labels that can neither overlap nor cross
(c) inch canvas + architecture primitives
(d) shared colour scale, before and after
(e) rounded cards, grouping cards and halo text
(f) local-linear smoother with a live-computed annotation

The whole figure is laid out on an inch grid via ``axes_inches`` / ``inch_axes``
rather than a GridSpec, so panel edges and the panel-letter column line up by
construction.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from pubstyle import set_pub_style, savefig, get_cmap, COLORS, CATEGORICAL
from figkit import (ZOOM, ZOOM_ALT, add_scale_bar, add_zoom_inset,
                    axes_inches, background_card, block, connect,
                    corner_stamp, feature_map, halo_text, image_axes,
                    inch_axes, lowess, op_circle, panel_card,
                    panel_label_inches, place_outside_labels, route,
                    safe_colorbar, shared_norm, tensor_label, text_card)

W, H = 11.4, 7.0
INK, SUB, MUTED = COLORS["ink"], COLORS["sub"], COLORS["muted"]

# One inch grid for the whole sheet.  Both rows end on a common top edge so
# the panel letters form a straight column (see DESIGN_SYSTEM section 13.3).
ROW1_TOP, ROW2_TOP = 6.28, 3.34
R_ZOOM = (0.55, ROW1_TOP - 2.42, 2.42, 2.42)
R_PIE = (3.40, ROW1_TOP - 2.42, 3.05, 2.42)
R_DIAG = (6.88, ROW1_TOP - 2.42, 4.20, 2.42)
R_CARDS = (4.50, ROW2_TOP - 2.62, 2.80, 2.62)
R_SMOOTH = (8.05, ROW2_TOP - 2.36, 2.95, 2.36)
CARD_D = (0.42, 0.58, 3.63, 2.76)          # background card of panel (d)


# ---------------------------------------------------------------------------
# Synthetic content
# ---------------------------------------------------------------------------
def _fbm(rng, n, sigmas=(18, 7, 2.5), weights=(1.0, 0.55, 0.28)):
    out = np.zeros((n, n))
    for s, w in zip(sigmas, weights):
        out += w * gaussian_filter(rng.standard_normal((n, n)), s,
                                   mode="reflect")
    return (out - out.min()) / (np.ptp(out) + 1e-12)


def texture_scene(n=220, seed=7):
    """Terrain-like RGB tile with fine detail worth magnifying."""
    rng = np.random.default_rng(seed)
    elev = _fbm(rng, n)
    yy, xx = np.mgrid[0:n, 0:n] / n
    ridge = np.exp(-((yy - 0.34 - 0.10 * np.sin(6 * xx)) ** 2) / 0.0016)
    field = np.clip(elev + 0.32 * ridge, 0, 1)
    field = np.clip(field + 0.07 * gaussian_filter(
        rng.standard_normal((n, n)), 0.55), 0, 1)
    rgb = get_cmap("cb_sequential_green")(field)[..., :3]
    rgb[field < 0.24] = np.array([0.17, 0.33, 0.42])
    return rgb


def response_fields(n=64, seed=3):
    """Three fields of the *same* quantity with genuinely different strength."""
    base = _fbm(np.random.default_rng(seed), n, (9, 4), (1.0, 0.5)) - 0.5
    out = []
    for k, amp in enumerate((1.00, 0.46, 0.13)):
        detail = _fbm(np.random.default_rng(seed + 1 + k), n, (5,), (1.0,))
        f = amp * (base + 0.30 * (detail - 0.5))
        out.append(f - f.mean())
    return out


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_zoom(fig, rect):
    ax = axes_inches(fig, rect)
    img = texture_scene()
    ax.imshow(img, interpolation="nearest")
    image_axes(ax, color="#b0b0b0", lw=0.7)

    # Both crops are deliberately non-square; match_aspect grows them to the
    # inset's on-paper aspect, so nothing is letterboxed and nothing stretched.
    add_zoom_inset(ax, img, (26, 58, 34, 22), (0.020, 0.020, 0.335, 0.335),
                   color=ZOOM, match_aspect=True)
    add_zoom_inset(ax, img, (142, 64, 24, 32), (0.645, 0.020, 0.335, 0.335),
                   color=ZOOM_ALT, match_aspect=True)
    add_scale_bar(ax, 44, "220 m", loc=(0.400, 0.045), height=0.020)
    corner_stamp(ax, "match_aspect: 34x22 px and 24x32 px\n"
                     "crops both squared to fill their frames",
                 loc="upper left", fontsize=6.0, color=INK, halo="white")
    return ax


def panel_labels(fig, rect):
    ax = axes_inches(fig, rect)
    shares = [26.4, 18.1, 13.7, 11.2, 9.6, 8.3, 6.4, 3.9, 2.4]
    names = ["Vegetation", "Building", "Road", "Vehicle", "Sky",
             "Sidewalk", "Pedestrian", "Sign", "Rider"]
    cols = (CATEGORICAL + CATEGORICAL[:4])[:len(shares)]
    total = 96540

    wedges, _ = ax.pie(shares, radius=1.0, colors=cols, startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.34, edgecolor="white",
                                       linewidth=1.6))
    items = []
    for w, nm, v, c in zip(wedges, names, shares, cols):
        ang = (w.theta1 + w.theta2) / 2.0
        items.append((ang, nm, f"{v:.1f}%  ({int(v / 100 * total):,})", c))
    place_outside_labels(ax, items, min_gap=0.250, y_span=1.18)

    ax.text(0, 0.09, f"{total:,}", ha="center", va="center", fontsize=11.0,
            fontweight="bold", color=INK)
    ax.text(0, -0.10, "instances", ha="center", va="center", fontsize=6.4,
            color=MUTED)
    ax.text(0, -1.40, "9 wedges, 2 columns, 0 crossings", ha="center",
            va="center", fontsize=6.2, color=MUTED)
    ax.set_xlim(-1.98, 1.98)
    ax.set_ylim(-1.46, 1.34)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def panel_diagram(fig, rect):
    ax = inch_axes(fig, rect)          # data units are inches from here on
    w = rect[2]
    y = 1.34

    panel_card(ax, 2.24, y, 2.66, 1.24, fc="#f7f8f9", ec=COLORS["grid"],
               rounding=0.07, label="shared trunk", label_fs=6.0, zorder=1)

    fm_in = feature_map(ax, 0.55, y, 0.42, 0.56, n_layers=3,
                        color=COLORS["baseline"], depth=0.05)
    enc = block(ax, 1.42, y, 0.80, 0.62, "Encoder", sub="ResNet-50",
                color=CATEGORICAL[1], n_stack=3, stack_off=0.05)
    op = op_circle(ax, 2.26, y, "+", r=0.125)
    dec = block(ax, 3.06, y, 0.80, 0.62, "Decoder", sub="FPN",
                color=CATEGORICAL[2])
    fm_out = feature_map(ax, 3.80, y, 0.36, 0.48, n_layers=2,
                         color=COLORS["baseline"], depth=0.05)

    for a, b in ((fm_in, enc), (enc, op), (op, dec), (dec, fm_out)):
        connect(ax, a, b, gap=0.075)
    route(ax, [enc["t"], (enc["x"], y + 0.80), (op["x"], y + 0.80), op["t"]],
          ls=(0, (3.4, 2.2)), color=MUTED, lw=0.9)
    ax.text((enc["x"] + op["x"]) / 2, y + 0.84, "residual", ha="center",
            va="bottom", fontsize=6.0, color=MUTED)

    for cx, txt in ((0.55, "3x512x512"), (1.42, "256x64x64"),
                    (3.06, "128x128x128"), (3.80, "Kx512x512")):
        tensor_label(ax, cx, y - 0.46, txt)
    ax.text(w / 2, 0.30,
            "inch_axes: 1 data unit = 1 inch, so arrow(gap=0.075) leaves the\n"
            "same clearance at every box edge, on any canvas size",
            ha="center", va="center", fontsize=6.2, color=MUTED,
            linespacing=1.45)
    return ax


def panel_shared_scale(fig):
    """Two rows of the same three fields: autoscaled vs one shared norm."""
    fields = response_fields()
    peaks = [float(np.abs(f).max()) for f in fields]
    cmap = get_cmap("cb_diverging")
    norm = shared_norm(fields, symmetric=True)

    x0, size, gap = 0.66, 0.80, 0.13
    y_top, y_bot = 2.12, 1.02
    fw, fh = fig.get_size_inches()

    for i, f in enumerate(fields):
        x = x0 + i * (size + gap)
        at = axes_inches(fig, (x, y_top, size, size))
        at.imshow(f, cmap=cmap, interpolation="nearest")        # autoscaled
        image_axes(at, color="#b0b0b0", lw=0.6)

        ab = axes_inches(fig, (x, y_bot, size, size))
        im = ab.imshow(f, cmap=cmap, norm=norm, interpolation="nearest")
        image_axes(ab, color="#b0b0b0", lw=0.6)

        for a in (at, ab):
            corner_stamp(a, f"peak {peaks[i]:.2f}", loc="lower left",
                         fontsize=6.0, color=INK, halo="white")
        fig.text((x + size / 2) / fw, 3.00 / fh,
                 ["strong", "medium (0.46x)", "weak (0.13x)"][i],
                 fontsize=6.2, color=SUB, ha="center", va="bottom")
    safe_colorbar(fig, im, ab, label="response (a.u.)", width=0.085,
                  pad=0.055, labelsize=6.0)

    for y, txt, col in (
            (3.16, "per-panel autoscale  -  all three look identical",
             COLORS["accent"]),
            (1.90, "shared_norm()  -  the 8x spread becomes visible",
             CATEGORICAL[2])):
        fig.text(x0 / fw, y / fh, txt, fontsize=6.5, color=col,
                 fontweight="bold", ha="left", va="bottom")
    fig.text(x0 / fw, 0.80 / fh,
             "same data, same colormap, one shared colorbar - only the norm "
             "differs", fontsize=6.0, color=MUTED, ha="left", va="bottom")


def panel_cards(fig, rect):
    ax = inch_axes(fig, rect)
    w = rect[2]

    text_card(ax, 0.12, 1.88, w - 0.24, 0.62,
              "zorder=0 is a tie, and ties are lost:\n"
              "fig.patches draws after fig.axes.",
              title="background_card()", fc="#f7f8f9", fs=6.3, pad=0.11,
              rounding=0.045, title_fs=7.0)
    text_card(ax, 0.12, 1.10, w - 0.24, 0.62,
              "a second legend inherits the axes clip\n"
              "and is dropped from the tight bbox.",
              title="unclip_legend()", fc="#f7f8f9", fs=6.3, pad=0.11,
              rounding=0.045, title_fs=7.0)

    axs = ax.inset_axes([0.12 / w, 0.32 / rect[3], (w - 0.24) / w,
                         0.54 / rect[3]])
    axs.imshow(np.linspace(0, 1, 256)[None, :],
               cmap=get_cmap("cb_sequential_purple"), aspect="auto")
    axs.set_xticks([])
    axs.set_yticks([])
    for s in axs.spines.values():
        s.set_visible(False)
    axs.set_zorder(4)
    halo_text(axs, 0.5, 0.5, "halo_text() stays legible on any background",
              transform=axs.transAxes, ha="center", va="center",
              fontsize=6.4, halo_lw=2.2, zorder=6)
    return ax


def panel_smoother(fig, rect):
    ax = axes_inches(fig, rect)
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 10, 220))
    y = 1.8 + 0.62 * x - 0.052 * x ** 2 + rng.normal(0, 0.55, x.size)

    ax.scatter(x, y, s=5, color=CATEGORICAL[1], alpha=0.40, linewidths=0,
               zorder=2)
    xs, ys, sd = lowess(x, y, frac=0.35, return_scatter=True)
    ax.fill_between(xs, ys - sd, ys + sd, color=CATEGORICAL[0], alpha=0.15,
                    linewidth=0, zorder=3)
    ax.plot(xs, ys, color=CATEGORICAL[0], lw=1.7, zorder=4,
            label="lowess(frac=0.35)")

    # Rule 13.2: every number printed below is derived from the arrays above.
    i = int(np.argmax(ys))
    ax.plot([xs[i]], [ys[i]], marker="o", ms=4.5, mfc="white",
            mec=CATEGORICAL[0], mew=1.2, zorder=5)
    ax.annotate(f"peak {ys[i]:.2f} at x = {xs[i]:.1f}", xy=(xs[i], ys[i]),
                xytext=(0, 17), textcoords="offset points", fontsize=6.4,
                color=INK, ha="center")
    ax.text(0.035, 0.055, f"n = {x.size}    mean residual sd = {sd.mean():.2f}",
            transform=ax.transAxes, fontsize=6.2, color=MUTED, va="bottom")

    ax.set_xlabel("predictor (-)", fontsize=8)
    ax.set_ylabel("response (a.u.)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right",
              borderaxespad=0.3)
    return ax


# ---------------------------------------------------------------------------
def main():
    set_pub_style(base_font=8)
    fig = plt.figure(figsize=(W, H))

    # Subordinate group: the card says "these six tiles are one exhibit",
    # which six equal-sized panels could never communicate on their own.
    background_card(fig, (CARD_D[0] / W, CARD_D[1] / H,
                          CARD_D[2] / W, CARD_D[3] / H),
                    facecolor="#f2f3f4", zorder=-5, radius=0.010)

    ax_a = panel_zoom(fig, R_ZOOM)
    ax_b = panel_labels(fig, R_PIE)
    ax_c = panel_diagram(fig, R_DIAG)
    panel_shared_scale(fig)
    ax_e = panel_cards(fig, R_CARDS)
    ax_f = panel_smoother(fig, R_SMOOTH)

    # Letter + group name on one line: two stacked headings above a panel
    # waste a band of canvas and never align across panels of unequal height.
    for ax, lab in ((ax_a, "(a)  Zoom & inset"),
                    (ax_b, "(b)  Labels & annotation"),
                    (ax_c, "(c)  Cards & diagram primitives"),
                    (ax_e, "(e)  Layout safety"),
                    (ax_f, "(f)  Data")):
        panel_label_inches(fig, ax, lab, dx=-0.30, dy=0.13, size=9.0)
    fig.text((CARD_D[0] - 0.06) / W, (CARD_D[1] + CARD_D[3] + 0.13) / H,
             "(d)  Scales", fontsize=9.0, fontweight="bold", color=INK,
             ha="left", va="bottom")


    savefig(fig, "353_figkit_demo.png")


if __name__ == "__main__":
    main()
