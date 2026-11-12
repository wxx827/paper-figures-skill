"""Swim-lane sequence of an authenticated TLS-style handshake and first request.

Time runs downward, one lane per principal, and every message is drawn as a
sloped arrow so that its transit time is visible rather than implied.  Colour
and dash pattern carry the confidentiality state - plaintext, CA-signed, or
encrypted under a session key - which is the property a reviewer of a protocol
trace actually looks for.  The phase bands and the duration bars on the right
share the same time axis, and both are measured from the message table, so a
change to one message moves the band, the bar and the percentage together.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pubstyle import set_pub_style, savefig
from figkit import arrow, panel_labels_inches

RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
BLUE = "#3b5b92"
GREEN = "#2e7d5b"
GOLD = "#c9a227"

LANES = ["Client", "Server", "Certificate authority"]
LANE_X = [0.85, 3.05, 5.25]

KIND = {"plain": (MUTED, (0, (3.2, 1.8)), "Plaintext"),
        "signed": (BLUE, "-", "CA-signed, not confidential"),
        "enc": (GREEN, "-", "Encrypted under a session key")}

# (phase, t0 in ms, source lane, target lane, transit ms, label, kind)
MESSAGES = [
    (0, 0.40, 0, 1, 1.60, "ClientHello + key_share", "plain"),
    (0, 2.60, 1, 2, 1.80, "OCSP status request", "plain"),
    (0, 6.00, 2, 1, 1.80, "OCSP response (signed)", "signed"),
    (0, 9.00, 1, 0, 1.60, "ServerHello + key_share", "plain"),
    (1, 11.40, 1, 0, 1.60, "Certificate chain", "enc"),
    (1, 13.60, 1, 0, 1.60, "CertificateVerify + Finished", "enc"),
    (1, 18.40, 0, 1, 1.60, "Finished (client)", "enc"),
    (2, 20.80, 0, 1, 1.60, "GET /account", "enc"),
    (2, 23.40, 1, 0, 1.60, "200 OK + application records", "enc"),
    (2, 26.00, 0, 1, 1.60, "close_notify", "enc"),
]
LOCAL = (1, 15.60, 0, 2.20, "Verify chain to CA root")
PHASES = ["Key exchange", "Authentication", "Application data"]
BAND = ["#f5f5f5", "#f0f0f0", "#f5f5f5"]


def phase_spans():
    """Start / end / duration of each phase, measured from the messages."""
    out = []
    for p in range(len(PHASES)):
        ts = [m[1] for m in MESSAGES if m[0] == p]
        te = [m[1] + m[4] for m in MESSAGES if m[0] == p]
        if LOCAL[0] == p:
            ts.append(LOCAL[1])
            te.append(LOCAL[1] + LOCAL[3])
        out.append((min(ts), max(te), max(te) - min(ts)))
    return out


def main():
    set_pub_style()
    np.random.default_rng(579)          # fixed seed; the trace is scripted
    spans = phase_spans()
    t_end = spans[-1][1]
    total = t_end - spans[0][0]

    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.55], wspace=0.16,
                          left=0.055, right=0.90, top=0.865, bottom=0.105)
    ax = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])

    # ---- (a) swim lanes ---------------------------------------------------
    for (t0, t1, _d), col, name in zip(spans, BAND, PHASES):
        ax.axhspan(t0 - 0.30, t1 + 0.30, color=col, lw=0, zorder=0)
        ax_d.axhspan(t0 - 0.30, t1 + 0.30, color=col, lw=0, zorder=0)

    for x, name in zip(LANE_X, LANES):
        ax.plot([x, x], [-0.6, t_end + 1.0], color="#cccccc", lw=1.0,
                zorder=1)
        ax.text(x, -1.5, name, ha="center", va="center", fontsize=8.0,
                color="#1a1a1a", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                          edgecolor="#cccccc", linewidth=0.8))

    for _p, t0, s, d, dur, text, kind in MESSAGES:
        col, ls, _lab = KIND[kind]
        x0, x1 = LANE_X[s], LANE_X[d]
        arrow(ax, (x0, t0), (x1, t0 + dur), color=col, ls=ls, lw=1.05,
              ms=10, zorder=5)
        ax.text(0.5 * (x0 + x1), t0 + 0.5 * dur - 0.55, text, ha="center",
                va="bottom", fontsize=6.6, color=RULE, zorder=6)

    _p, lt0, llane, ldur, ltext = LOCAL
    lx = LANE_X[llane]
    ax.add_patch(Rectangle((lx - 0.13, lt0), 0.26, ldur, facecolor=BLUE,
                           edgecolor="white", linewidth=0.8, zorder=5))
    ax.text(lx + 0.24, lt0 + 0.5 * ldur, f"{ltext}\nlocal, {ldur:.1f} ms",
            ha="left", va="center", fontsize=6.6, color=BLUE,
            linespacing=1.25, zorder=6)

    ax.set_xlim(0.0, 6.35)
    ax.set_ylim(t_end + 1.2, -2.6)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("elapsed time since first byte (ms)", fontsize=8.0)
    ax.set_yticks(np.arange(0, 30, 5))
    ax.tick_params(labelsize=7.0, length=2.6)
    ax.set_title("Every message, its direction and its confidentiality state",
                 fontsize=9.0, fontweight="bold", pad=8)

    # The lower right of the lane panel is empty once the CA drops out of the
    # exchange, so the legend costs no space of its own.
    handles = [Line2D([], [], color=c, ls=s, lw=1.2, label=l)
               for c, s, l in KIND.values()]
    handles.append(Line2D([], [], color=BLUE, lw=5.0,
                          label="Local verification"))
    ax.legend(handles=handles, loc="lower right",
              bbox_to_anchor=(1.005, 0.005), ncol=1, fontsize=7.0,
              frameon=False, handlelength=2.0, labelspacing=0.45)

    # ---- (b) time budget per phase ---------------------------------------
    mids = [0.5 * (t0 + t1) for t0, t1, _d in spans]
    durs = [d for _t0, _t1, d in spans]
    heights = [(t1 - t0) * 0.62 for t0, t1, _d in spans]
    cols = [GOLD if d == max(durs) else BLUE for d in durs]
    ax_d.barh(mids, durs, height=heights, color=cols, edgecolor="white",
              linewidth=0.8, zorder=3)
    for m, d in zip(mids, durs):
        ax_d.text(d + 0.25, m, f"{d:.1f} ms  ({d / total:.0%})", ha="left",
                  va="center", fontsize=6.8, color=SUB, zorder=4)
    ax_d.set_xlim(0.0, max(durs) * 1.62)
    ax_d.set_ylim(*ax.get_ylim())
    ax_d.set_yticks(mids)
    ax_d.set_yticklabels(PHASES, fontsize=7.2)
    ax_d.tick_params(labelsize=6.8, length=2.4, labelleft=False,
                     labelright=True, left=False, right=True)
    ax_d.yaxis.set_label_position("right")
    ax_d.spines["right"].set_visible(True)
    ax_d.spines["left"].set_visible(False)
    ax_d.set_xlabel("phase duration (ms)", fontsize=7.6)
    ax_d.set_title("Where the round trips go", fontsize=8.5,
                   fontweight="bold", pad=8)
    ax_d.grid(axis="x", lw=0.45, alpha=0.45)
    ax_d.set_axisbelow(True)

    rtt = MESSAGES[3][1] + MESSAGES[3][4] - MESSAGES[0][1]

    panel_labels_inches(fig, [ax, ax_d], dx=-0.42, dy=0.30)
    savefig(fig, "579_sec_protocol_sequence.png")


if __name__ == "__main__":
    main()
