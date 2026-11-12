"""Wm blue error grid: 8 scenes x 5 methods of rollout error maps.

Composition follows the dense error-map grids common in novel-view synthesis
papers -- one scene per row, one method per column, a slim colourbar at the end
of every row, scene names set vertically down the left edge.  The reference
sheet uses ``jet``; house rule bans it, so the ramp here goes dark navy -> pale
blue, which shows the sparse high-error silhouettes just as clearly.

Synthetic data lives in ``gen_data`` only -- swap that one function for a
``pandas.read_csv`` loader to plot real data and leave the styling untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pubstyle import set_pub_style, savefig

# WM blue batch ladder (dark -> near white); see BRIEF_WM_BLUE.md section 2.
BLUE = ["#12314f", "#1f4e79", "#2f6fa8", "#4a8cc7", "#7fb0dc", "#b3d0ea",
        "#dce9f5", "#f2f7fb"]
INK = "#22272b"
# Low error = dark base, high error = bright line, i.e. the ladder as given.
ERR_CMAP = LinearSegmentedColormap.from_list("wm_blue_dark", BLUE)

SCENES = ["bouncingballs", "hellwarrior", "hook", "jumpingjacks",
          "lego", "mutant", "standup", "trex"]
METHODS = ["D-NeRF", "TiNeuVox", "4DGS", "Deform-GS", "Ours"]
H, W = 96, 128


def _blob_field(rng, n_blobs, sigma):
    """Smooth random scalar field built from a few Gaussian bumps."""
    yy, xx = np.mgrid[0:H, 0:W]
    field = np.zeros((H, W))
    for _ in range(n_blobs):
        cy = rng.uniform(0.18, 0.82) * H
        cx = rng.uniform(0.18, 0.82) * W
        sy = sigma * rng.uniform(0.7, 1.4) * H
        sx = sigma * rng.uniform(0.7, 1.4) * W
        field += rng.uniform(0.6, 1.0) * np.exp(
            -(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    return field


def gen_data(seed=1511):
    """Synthetic stand-in data. REPLACE THIS FUNCTION with your real loader.

    Example::

        import numpy as np
        maps = np.load("data/wm_blue_error_grid.npy")   # (scene, method, H, W)
        return maps
    """
    rng = np.random.default_rng(seed)
    maps = np.zeros((len(SCENES), len(METHODS), H, W))
    for s in range(len(SCENES)):
        field = _blob_field(rng, rng.integers(3, 6), 0.11)
        field /= field.max()
        # Nested level sets: outer silhouette bright, inner creases fainter.
        edge = np.zeros((H, W))
        for q, w in ((0.80, 1.00), (0.87, 0.45), (0.94, 0.28)):
            mask = (field > np.quantile(field, q)).astype(float)
            gy, gx = np.gradient(mask)
            edge += w * np.hypot(gy, gx)
        edge /= edge.max()

        body = (field > np.quantile(field, 0.80)).astype(float)
        interior = body * 0.05
        # Sparse hot pixels where the reconstruction pops (specular detail).
        speck = np.zeros((H, W))
        idx = np.flatnonzero(body)
        hot = rng.choice(idx, size=min(45, idx.size), replace=False)
        speck.ravel()[hot] = rng.uniform(0.35, 0.85, hot.size)

        # Ground-contact band, placed a little differently in every scene.
        band = np.zeros((H, W))
        y0 = int(rng.uniform(0.66, 0.80) * H)
        band[y0:y0 + 2, int(rng.uniform(0.06, 0.16) * W):
             int(rng.uniform(0.84, 0.96) * W)] = rng.uniform(0.30, 0.50)

        scale = np.linspace(1.0, 0.34, len(METHODS)) * rng.uniform(0.85, 1.15)
        for m in range(len(METHODS)):
            err = scale[m] * (edge + band + interior
                              + speck * rng.uniform(0.5, 1.0))
            err += rng.normal(0, 0.010, (H, W)) * scale[m]
            maps[s, m] = np.clip(err, 0.0, None)
    return maps


def main():
    set_pub_style(base_font=8)
    maps = gen_data()
    n_rows, n_cols = maps.shape[:2]

    # One shared scale for every panel in the sheet (house rule).
    vmax = float(np.quantile(maps, 0.998))
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(7.0, 9.6))
    gs = fig.add_gridspec(n_rows, n_cols + 1, left=0.075, right=0.925,
                          top=0.925, bottom=0.030,
                          width_ratios=[1.0] * n_cols + [0.075],
                          wspace=0.035, hspace=0.045)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(maps[r, c], cmap=ERR_CMAP, norm=norm,
                           interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.5)
                s.set_edgecolor(INK)
            if r == 0:
                ax.set_title(METHODS[c], fontsize=7.2, color=INK, pad=3)
            if c == 0:
                ax.set_ylabel(SCENES[r], fontsize=6.8, color=INK, labelpad=4)

        cax = fig.add_subplot(gs[r, n_cols])
        # Shrink inside its cell so neighbouring rows' end labels never touch.
        p = cax.get_position()
        cax.set_position([p.x0, p.y0 + 0.14 * p.height,
                          p.width * 0.62, p.height * 0.72])
        cb = fig.colorbar(im, cax=cax, ticks=[0.0, vmax])
        cb.ax.set_yticklabels(["0", "1"])
        for lab, va in zip(cb.ax.get_yticklabels(), ("bottom", "top")):
            lab.set_verticalalignment(va)
        cb.ax.tick_params(labelsize=6.2, length=1.6, width=0.5, color=INK,
                          labelcolor=INK, pad=1.5)
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(INK)


    savefig(fig, "1106_wm_blue_error_grid.png")


if __name__ == "__main__":
    main()
