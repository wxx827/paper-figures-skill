"""Five-qubit variational circuit drawn gate by gate on an inch canvas.

The schedule lives in three plain lists (single-qubit gates, two-qubit
entanglers, measurements) keyed by time step, so every number quoted at the
bottom of the figure - depth, gate counts, two-qubit layer count, the
estimated wall time built from per-class gate durations - is derived from the
same lists that draw the picture.  Four stages are separated by barriers and
alternating tint bands: state preparation, the CNOT entangling ladder, the
variational rotation block with one controlled-Rz, and the readout layer.
Right-hand brackets say what each register is for; the classical bits leave
the measurement dials on the usual double wire.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle

from pubstyle import set_pub_style, savefig
from figkit import block

BLUE = "#3b5b92"        # single-qubit rotations / state manipulation
PURPLE = "#8e6bbf"      # entangling (two-qubit) gates
DARK = "#3a3a3a"        # measurement
INK = "#1a1a1a"
RULE = "#333333"
SUB = "#555555"
MUTED = "#777777"
HAIR = "#cccccc"
SOFT = "#f5f5f5"

W, H = 10.4, 6.0
NQ = 5
X0, DX = 1.32, 0.545                      # first column, column pitch
Y0, DY = 4.52, 0.55                       # top wire, wire pitch
WIRE_X0 = 0.92
BAND_T, BAND_B = 4.82, 2.06

# (column, qubit, label, sub-label)
SINGLE = [
    (0, 0, "$H$", None), (0, 1, "$H$", None), (0, 2, "$H$", None),
    (0, 3, "$H$", None), (0, 4, "$H$", None),
    (5, 0, "$R_z$", r"$\theta_1$"),
    (5, 2, "$R_z$", r"$\theta_2$"),
    (5, 4, "$R_z$", r"$\theta_3$"),
    (6, 1, "$X$", None),
    (6, 3, "$R_z$", r"$\theta_4$"),
    (8, 1, "$R_z$", r"$\theta_5$"),
    (8, 3, "$R_z$", r"$\theta_6$"),
    (10, 0, "$H$", None), (10, 2, "$H$", None), (10, 4, "$H$", None),
]
# (column, control qubit, target qubit, kind)
TWOQ = [
    (1, 0, 1, "cx"), (2, 1, 2, "cx"), (3, 2, 3, "cx"), (4, 3, 4, "cx"),
    (7, 0, 2, "crz"),
    (9, 4, 3, "cx"),
]
MEAS = [(11, q) for q in range(NQ)]

# Calibrated gate durations (microseconds) for a superconducting register.
DUR = {"1q": 0.035, "2q": 0.300, "meas": 1.200}

STAGES = [
    (0, 0, "State preparation"),
    (1, 4, "Entangling ladder"),
    (5, 9, "Variational rotations"),
    (10, 11, "Readout"),
]


def cx(col):
    return X0 + col * DX


def wy(q):
    return Y0 - q * DY


def wire(ax, x1):
    for q in range(NQ):
        y = wy(q)
        ax.plot([WIRE_X0, x1], [y, y], color=RULE, lw=0.9, zorder=2)
        ax.text(WIRE_X0 - 0.08, y, rf"$q_{q}\!:|0\rangle$", ha="right",
                va="center", fontsize=7.4, color=INK)


def control_dot(ax, x, y, color=PURPLE):
    ax.add_patch(Circle((x, y), 0.052, facecolor=color, edgecolor=color,
                        lw=0.8, zorder=6))


def target_xor(ax, x, y, color=PURPLE, r=0.105):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color,
                        lw=1.1, zorder=6))
    ax.plot([x - r, x + r], [y, y], color=color, lw=1.0, zorder=7)
    ax.plot([x, x], [y - r, y + r], color=color, lw=1.0, zorder=7)


def meter(ax, x, y, w=0.40, h=0.42):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.055", facecolor="white",
        edgecolor=DARK, linewidth=0.9, mutation_aspect=1.0, zorder=5))
    ax.add_patch(Arc((x, y - 0.10), 0.24, 0.21, theta1=0.0, theta2=180.0,
                     edgecolor=DARK, lw=0.9, zorder=6))
    ax.plot([x, x + 0.085], [y - 0.10, y + 0.07], color=DARK, lw=0.9,
            solid_capstyle="round", zorder=6)


def classical_out(ax, x, y, bit):
    for off in (-0.022, 0.022):
        ax.plot([x, x + 0.18], [y + off, y + off], color=DARK, lw=0.7,
                zorder=2)
    ax.text(x + 0.23, y, rf"$c_{bit}$", ha="left", va="center", fontsize=6.8,
            color=SUB)


def reg_bracket(ax, x, y_lo, y_hi, title, note, tip=0.08):
    ax.plot([x - tip, x, x, x - tip], [y_hi, y_hi, y_lo, y_lo], color=SUB,
            lw=0.8, solid_joinstyle="miter")
    ym = 0.5 * (y_lo + y_hi)
    ax.text(x + 0.10, ym + 0.09, title, ha="left", va="center", fontsize=7.0,
            color=INK)
    ax.text(x + 0.10, ym - 0.10, note, ha="left", va="center", fontsize=6.4,
            color=MUTED)


def draw_stages(ax):
    x_stop = cx(STAGES[-1][1]) + 0.20
    for k, (c_lo, c_hi, name) in enumerate(STAGES):
        xa = cx(c_lo) - 0.5 * DX
        xb = min(cx(c_hi) + 0.5 * DX, x_stop)
        if k % 2 == 1:
            ax.add_patch(Rectangle((xa, BAND_B), xb - xa, BAND_T - BAND_B,
                                   facecolor=SOFT, edgecolor="none",
                                   zorder=0))
        if k:
            ax.plot([xa, xa], [BAND_B, BAND_T], color=HAIR, lw=0.8,
                    ls=(0, (3, 2.4)), zorder=1)
        ax.text(0.5 * (xa + xb), BAND_T + 0.14, name, ha="center",
                va="bottom", fontsize=7.0, color=SUB)


def draw_gates(ax):
    for col, ctrl, targ, kind in TWOQ:
        x = cx(col)
        yc, yt = wy(ctrl), wy(targ)
        ax.plot([x, x], [yc, yt], color=PURPLE, lw=1.1, zorder=3)
        control_dot(ax, x, yc)
        if kind == "cx":
            target_xor(ax, x, yt)
        else:
            block(ax, x, yt, 0.46, 0.42, "$R_z$", color=PURPLE,
                  sub=r"$\theta_7$", fontsize=7.4, zorder=5)

    for col, q, label, sub in SINGLE:
        w = 0.46 if sub else 0.40
        block(ax, cx(col), wy(q), w, 0.42, label, color=BLUE, sub=sub,
              fontsize=7.6, zorder=5)

    for col, q in MEAS:
        meter(ax, cx(col), wy(q))


def stats_text():
    depth = max([c for c, *_ in SINGLE] + [c for c, *_ in TWOQ] +
                [c for c, _ in MEAS]) + 1
    n1, n2, nm = len(SINGLE), len(TWOQ), len(MEAS)
    two_layers = len({c for c, *_ in TWOQ})
    per_col = {}
    for col, *_ in SINGLE:
        per_col[col] = max(per_col.get(col, 0.0), DUR["1q"])
    for col, *_ in TWOQ:
        per_col[col] = max(per_col.get(col, 0.0), DUR["2q"])
    for col, _ in MEAS:
        per_col[col] = max(per_col.get(col, 0.0), DUR["meas"])
    wall = sum(per_col.values())
    return depth, n1, n2, nm, two_layers, wall


def legend_row(ax, y):
    items = [(BLUE, r"Single-qubit gate:  $H$,  $X$,  $R_z(\theta)$"),
             (PURPLE, r"Two-qubit entangler:  CNOT,  controlled-$R_z$"),
             (DARK, "Projective measurement in the $Z$ basis")]
    x = 0.44
    for color, label in items:
        ax.add_patch(FancyBboxPatch(
            (x, y - 0.075), 0.20, 0.15,
            boxstyle="round,pad=0,rounding_size=0.04", facecolor=color,
            edgecolor="white", linewidth=0.7, mutation_aspect=1.0, zorder=4))
        ax.text(x + 0.28, y, label, ha="left", va="center", fontsize=6.8,
                color=INK, zorder=4)
        x += 3.18


def main():
    set_pub_style()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    depth, n1, n2, nm, two_layers, wall = stats_text()

    ax.text(0.30, 5.62,
            "Hardware-efficient five-qubit ansatz: entangling ladder feeds a "
            "six-angle rotation block",
            ha="left", va="center", fontsize=11.0, fontweight="bold",
            color=INK)
    ax.text(0.30, 5.32,
            "Time flows left to right; every column is one scheduled layer, "
            "and the barriers mark the four stages of the circuit",
            ha="left", va="center", fontsize=7.6, color=SUB)

    draw_stages(ax)
    x_end = cx(MEAS[0][0]) + 0.20
    wire(ax, x_end)
    draw_gates(ax)

    for col, q in MEAS:
        classical_out(ax, x_end + 0.02, wy(q), q)

    reg_bracket(ax, 8.32, wy(1) - 0.30, wy(0) + 0.30, "Data register",
                "2 qubits, problem state")
    reg_bracket(ax, 8.32, wy(3) - 0.30, wy(2) + 0.30, "Ancilla register",
                "entanglement mediators")
    reg_bracket(ax, 8.32, wy(4) - 0.24, wy(4) + 0.24, "Syndrome flag",
                "parity readout bit")

    ax.add_patch(FancyBboxPatch(
        (0.30, 0.56), 9.80, 1.14,
        boxstyle="round,pad=0,rounding_size=0.05", facecolor="white",
        edgecolor=HAIR, linewidth=0.7, mutation_aspect=1.0, zorder=1))
    legend_row(ax, 1.42)
    ax.text(0.44, 1.02,
            f"Circuit depth {depth} layers   ·   {n1} single-qubit gates   "
            f"·   {n2} two-qubit gates in {two_layers} entangling layers   "
            f"·   {nm} measurements   ·   {n1 + n2} gates total",
            ha="left", va="center", fontsize=7.0, color=INK)
    ax.text(0.44, 0.76,
            f"Scheduled wall time {wall:.2f} $\\mu$s at "
            f"{DUR['1q'] * 1e3:.0f} ns per single-qubit gate, "
            f"{DUR['2q'] * 1e3:.0f} ns per CNOT and "
            f"{DUR['meas'] * 1e3:.0f} ns per readout",
            ha="left", va="center", fontsize=6.6, color=MUTED)

    savefig(fig, "492_qc_circuit_diagram.png")


if __name__ == "__main__":
    main()
