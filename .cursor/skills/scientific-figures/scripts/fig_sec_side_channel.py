"""Recovering one key byte of a block cipher from its power consumption.

A simulated device leaks the Hamming weight of a substitution-table output
into its supply current, buried under noise several times larger than the
leak itself.  The top panel shows why a single trace is useless and why an
average over many is not: the two Hamming-weight groups only separate inside
one narrow window.  The middle panel is the classic difference-of-means DPA
attack over all 256 key hypotheses, and the bottom panel is the full
correlation matrix, hypothesis by sample.  Everything is computed from the
same trace array, including the number of traces the attack actually needs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import set_pub_style, savefig, get_cmap, add_colorbar
from figkit import panel_labels_inches

RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
ACCENT = "#b23a48"      # house red: the correct key hypothesis only
BLUE = "#3b5b92"
GOLD = "#c9a227"

N_TRACES = 1500
N_SAMP = 300
DT = 2.0                # ns per sample
KEY = 0xB7              # the byte the attack has to find
LEAK_S = 118            # sample index where the substitution output is used
NOISE = 8.0             # mV, still larger than the leak on a single trace
LEAK_MV = 3.0           # mV per Hamming-weight bit


def hamming(x):
    return np.array([bin(int(v)).count("1") for v in np.ravel(x)]).reshape(
        np.shape(x))


def simulate(rng):
    """Traces, plaintext bytes and the substitution table used by the device."""
    sbox = rng.permutation(256)
    plain = rng.integers(0, 256, N_TRACES)
    t = np.arange(N_SAMP) * DT

    base = (42.0 + 4.0 * np.sin(2 * np.pi * t / 62.0)
            + 2.2 * np.sin(2 * np.pi * t / 17.0)
            + 6.0 * np.exp(-0.5 * ((t - 90.0) / 26.0) ** 2))
    window = np.exp(-0.5 * ((np.arange(N_SAMP) - LEAK_S) / 5.5) ** 2)
    hw = hamming(sbox[plain ^ KEY]).astype(float)

    traces = (base[None, :] + LEAK_MV * hw[:, None] * window[None, :]
              + rng.normal(0.0, NOISE, (N_TRACES, N_SAMP)))
    return t, traces, plain, sbox, hw


def correlate(hyp_hw, traces):
    """Pearson rho between every key hypothesis and every sample."""
    hc = hyp_hw - hyp_hw.mean(axis=0, keepdims=True)
    tc = traces - traces.mean(axis=0, keepdims=True)
    num = hc.T @ tc
    den = np.sqrt(np.sum(hc ** 2, axis=0)[:, None] *
                  np.sum(tc ** 2, axis=0)[None, :])
    return num / np.maximum(den, 1e-12)


def main():
    set_pub_style()
    rng = np.random.default_rng(578)
    t, traces, plain, sbox, hw = simulate(rng)

    keys = np.arange(256)
    hyp_out = sbox[plain[:, None] ^ keys[None, :]]        # N x 256
    hyp_hw = hamming(hyp_out).astype(float)

    rho = correlate(hyp_hw, traces)
    peak_s = int(np.argmax(np.abs(rho[KEY])))
    ranked = np.argsort(-np.abs(rho[:, peak_s]))
    runner_up = int(ranked[1] if ranked[0] == KEY else ranked[0])

    # difference of means on the lowest output bit - the textbook DPA
    bit = (hyp_out & 1).astype(float)                      # N x 256
    n1 = bit.sum(axis=0)
    dpa = ((bit.T @ traces) / np.maximum(n1, 1)[:, None] -
           ((1.0 - bit).T @ traces) /
           np.maximum(N_TRACES - n1, 1)[:, None])

    # how many traces until the right hypothesis is ranked first and stays
    grid = np.arange(10, N_TRACES + 1, 10)
    top = []
    for n in grid:
        r = correlate(hyp_hw[:n], traces[:n, peak_s:peak_s + 1])[:, 0]
        top.append(int(np.argmax(np.abs(r))))
    top = np.array(top)
    ok = top == KEY
    need = int(grid[next(i for i in range(len(grid)) if ok[i:].all())])

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.80, 0.86],
                          left=0.075, right=0.905, top=0.895, bottom=0.095,
                          hspace=0.46)
    ax_tr = fig.add_subplot(gs[0])
    ax_dp = fig.add_subplot(gs[1])
    ax_rh = fig.add_subplot(gs[2])

    # ---- (a) raw traces and the two Hamming-weight groups -----------------
    lo, hi = hw < 4.0, hw >= 4.0
    for i in rng.choice(N_TRACES, 40, replace=False):
        ax_tr.plot(t, traces[i], color="#9cc3dd", lw=0.32, alpha=0.28,
                   zorder=1)
    ax_tr.plot(t, traces[lo].mean(axis=0), color=BLUE, lw=1.3, zorder=4,
               label=f"Mean of {int(lo.sum())} traces, HW < 4")
    ax_tr.plot(t, traces[hi].mean(axis=0), color=GOLD, lw=1.3, zorder=4,
               label=f"Mean of {int(hi.sum())} traces, HW $\\geq$ 4")
    ax_tr.axvspan(t[LEAK_S] - 14.0, t[LEAK_S] + 14.0, color=ACCENT,
                  alpha=0.11, lw=0, zorder=0)
    sep = (traces[hi].mean(axis=0) - traces[lo].mean(axis=0))[peak_s]
    ax_tr.annotate(f"substitution output loaded here:\n"
                   f"group means differ by {sep:.2f} mV",
                   xy=(t[peak_s], traces[hi].mean(axis=0)[peak_s]),
                   xytext=(t[peak_s] + 62.0, 70.0), fontsize=6.6,
                   color=ACCENT, linespacing=1.3,
                   arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
    ax_tr.set_ylabel("supply current (mV)", fontsize=8.0)
    ax_tr.set_xlabel("time within the encryption round (ns)", fontsize=8.0)
    ax_tr.set_ylim(14.0, 82.0)
    ax_tr.legend(loc="upper right", fontsize=6.8, frameon=False, ncol=1,
                 handlelength=1.6, labelspacing=0.35)
    ax_tr.set_title(f"One leak of {LEAK_MV:.1f} mV per bit under "
                    f"{NOISE:.0f} mV of noise", fontsize=9.0,
                    fontweight="bold")

    # ---- (b) difference-of-means DPA over all 256 hypotheses --------------
    for k in range(256):
        if k == KEY:
            continue
        ax_dp.plot(t, dpa[k], color="#cccccc", lw=0.30, alpha=0.65, zorder=1)
    ax_dp.plot(t, dpa[KEY], color=ACCENT, lw=1.25, zorder=5)
    peak_dpa = dpa[KEY][np.argmax(np.abs(dpa[KEY]))]
    wrong = np.abs(np.delete(dpa, KEY, axis=0)).max()
    ax_dp.annotate(f"correct hypothesis 0x{KEY:02X}: {abs(peak_dpa):.2f} mV, "
                   f"{abs(peak_dpa) / wrong:.1f}x the best wrong guess",
                   xy=(t[int(np.argmax(np.abs(dpa[KEY])))], peak_dpa),
                   xytext=(t[peak_s] + 58.0, peak_dpa * 0.92), fontsize=6.8,
                   color=ACCENT, va="center",
                   arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
    ax_dp.axhline(0.0, color="#808080", lw=0.6, ls="--")
    ax_dp.set_ylabel("difference of means (mV)", fontsize=8.0)
    ax_dp.set_xlabel("time within the encryption round (ns)", fontsize=8.0)
    ax_dp.set_title("Differential power analysis, one curve per key "
                    "hypothesis", fontsize=8.5, fontweight="bold")

    for ax in (ax_tr, ax_dp):
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=7.0, length=2.6)
        ax.grid(axis="x", lw=0.4, alpha=0.32)
        ax.set_axisbelow(True)

    # ---- (c) correlation, hypothesis by sample ----------------------------
    im = ax_rh.imshow(np.abs(rho), aspect="auto", origin="lower",
                      cmap=get_cmap("cb_sequential_blue"),
                      extent=[t[0], t[-1], -0.5, 255.5], vmin=0.0,
                      vmax=float(np.abs(rho).max()))
    ax_rh.plot([t[0]], [KEY], marker=">", ms=5.0, color=ACCENT,
               clip_on=False, zorder=6)
    ax_rh.annotate(f"hypothesis 0x{KEY:02X}", xy=(t[peak_s] + 15.0, KEY + 4.0),
                   xytext=(t[peak_s] + 52.0, KEY + 56.0), fontsize=6.8,
                   color=ACCENT,
                   arrowprops=dict(arrowstyle="-", lw=0.7, color=ACCENT))
    ax_rh.set_yticks([0, 64, 128, 192, 255])
    ax_rh.set_xlabel("time within the encryption round (ns)", fontsize=8.0)
    ax_rh.set_ylabel("key hypothesis (0-255)", fontsize=8.0)
    ax_rh.tick_params(labelsize=7.0, length=2.6)
    ax_rh.set_title("Correlation of the Hamming-weight model with every "
                    "sample", fontsize=8.5, fontweight="bold")
    add_colorbar(fig, im, ax_rh, label="|correlation| (-)", aspect=10,
                 fraction=0.030, pad=0.015)


    panel_labels_inches(fig, [ax_tr, ax_dp, ax_rh], dx=-0.52, dy=0.06)
    savefig(fig, "578_sec_side_channel.png")


if __name__ == "__main__":
    main()
