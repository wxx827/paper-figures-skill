"""CONSORT 2010 participant flow diagram for a two-arm randomised trial."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from pubstyle import set_pub_style, savefig

INK = "#333333"
ARM_A = {"fc": "#dce6f2", "ec": "#3b5b92"}
ARM_B = {"fc": "#ededed", "ec": "#6b6b6b"}
PLAIN = {"fc": "#ffffff", "ec": INK}
STAGE_FC = "#3b5b92"

TITLE_FS = 7.2
BODY_FS = 6.3
TITLE_GAP = 3.0        # data units between the bold title and the body block


def box(ax, x0, x1, y0, y1, title, lines=None, style=PLAIN, dashed=False):
    """Rounded flow-chart box: bold title, optional bulleted detail lines."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=style["fc"], edgecolor=style["ec"],
        linewidth=0.9, linestyle="--" if dashed else "-", zorder=2))
    if not lines:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, title, ha="center", va="center",
                fontsize=TITLE_FS, fontweight="bold", color="#111111", zorder=3)
        return
    ax.text(x0 + 1.8, y1 - 1.6, title, ha="left", va="top",
            fontsize=TITLE_FS, fontweight="bold", color="#111111", zorder=3)
    ax.text(x0 + 1.8, y1 - 1.6 - TITLE_GAP, "\n".join(lines), ha="left",
            va="top", fontsize=BODY_FS, color="#1a1a1a", linespacing=1.55,
            zorder=3)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
        linewidth=0.9, color="#444444", shrinkA=0, shrinkB=0, zorder=1))


def line(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], lw=0.9, color="#444444", zorder=1,
            solid_capstyle="butt")


def stage(ax, label, y0, y1):
    ax.add_patch(Rectangle((-13.5, y0), 6.5, y1 - y0, facecolor=STAGE_FC,
                           edgecolor="none", alpha=0.9, zorder=2))
    ax.text(-10.25, (y0 + y1) / 2, label, ha="center", va="center", rotation=90,
            fontsize=7.4, fontweight="bold", color="white", zorder=3)


def main():
    set_pub_style()
    fig, ax = plt.subplots(figsize=(7.5, 8.4))
    ax.set_xlim(-15, 106)
    ax.set_ylim(-9, 101)
    ax.axis("off")

    stage(ax, "ENROLMENT", 69.0, 100.0)
    stage(ax, "ALLOCATION", 48.0, 64.0)
    stage(ax, "FOLLOW-UP", 26.0, 43.5)
    stage(ax, "ANALYSIS", 4.0, 21.5)

    # ---- enrolment ---------------------------------------------------------
    box(ax, 30, 66, 93, 100, "Assessed for eligibility  (n = 1,284)")
    box(ax, 68, 106, 76.5, 92.0,
        "Excluded  (n = 372)",
        ["\u2022  Not meeting inclusion criteria (n = 214)",
         "\u2022  Declined to participate (n = 118)",
         "\u2022  Other reasons (n = 40)",
         "     \u2013 logistic / transport barrier (n = 23)",
         "     \u2013 enrolled in another trial (n = 17)"],
        dashed=True)
    line(ax, 48, 93, 48, 85.0)
    arrow(ax, 48, 85.0, 67.6, 85.0)
    arrow(ax, 48, 93, 48, 76.2)
    box(ax, 30, 66, 69.0, 76.0, "Randomised  (n = 912)")

    # ---- allocation --------------------------------------------------------
    line(ax, 48, 69.0, 48, 66.0)
    line(ax, 25, 66.0, 74, 66.0)
    arrow(ax, 25, 66.0, 25, 64.3)
    arrow(ax, 74, 66.0, 74, 64.3)

    box(ax, 1.5, 48.5, 48.0, 64.0,
        "Allocated to intervention  (n = 456)",
        ["\u2022  Received allocated intervention (n = 441)",
         "\u2022  Did not receive intervention (n = 15)",
         "     \u2013 withdrew consent before dose 1 (n = 9)",
         "     \u2013 clinical deterioration (n = 6)"], style=ARM_A)
    box(ax, 50.5, 97.5, 48.0, 64.0,
        "Allocated to placebo  (n = 456)",
        ["\u2022  Received allocated placebo (n = 448)",
         "\u2022  Did not receive placebo (n = 8)",
         "     \u2013 withdrew consent before dose 1 (n = 5)",
         "     \u2013 clinical deterioration (n = 3)"], style=ARM_B)

    # ---- follow-up ---------------------------------------------------------
    arrow(ax, 25, 48.0, 25, 43.8)
    arrow(ax, 74, 48.0, 74, 43.8)
    box(ax, 1.5, 48.5, 26.0, 43.5,
        "Lost to follow-up  (n = 19)",
        ["\u2022  Withdrew consent (n = 11)",
         "\u2022  Lost to contact (n = 8)",
         "",
         "Discontinued intervention  (n = 34)",
         "\u2022  Adverse events (n = 22)",
         "\u2022  Physician decision (n = 12)"], style=ARM_A)
    box(ax, 50.5, 97.5, 26.0, 43.5,
        "Lost to follow-up  (n = 24)",
        ["\u2022  Withdrew consent (n = 14)",
         "\u2022  Lost to contact (n = 10)",
         "",
         "Discontinued placebo  (n = 41)",
         "\u2022  Adverse events (n = 15)",
         "\u2022  Physician decision (n = 26)"], style=ARM_B)

    # ---- analysis ----------------------------------------------------------
    arrow(ax, 25, 26.0, 25, 21.8)
    arrow(ax, 74, 26.0, 74, 21.8)
    box(ax, 1.5, 48.5, 4.0, 21.5,
        "Analysed \u2014 intention to treat  (n = 456)",
        ["\u2022  Primary endpoint at 24 months (n = 456)",
         "\u2022  Multiple imputation for missing data (n = 19)",
         "",
         "Per-protocol population  (n = 403)",
         "\u2022  Excluded from per-protocol set (n = 53)"], style=ARM_A)
    box(ax, 50.5, 97.5, 4.0, 21.5,
        "Analysed \u2014 intention to treat  (n = 456)",
        ["\u2022  Primary endpoint at 24 months (n = 456)",
         "\u2022  Multiple imputation for missing data (n = 24)",
         "",
         "Per-protocol population  (n = 391)",
         "\u2022  Excluded from per-protocol set (n = 65)"], style=ARM_B)

    ax.plot([-13.5, 106], [-1.5, -1.5], lw=0.7, color="#cccccc")
    ax.text(-13.5, -4.0,
            "Multicentre, double-blind, placebo-controlled trial \u2014 1:1 "
            "permuted-block randomisation stratified by site and baseline "
            "eGFR (mL/min/1.73 m$^2$).",
            ha="left", va="center", fontsize=6.5, color="#444444")
    ax.text(-13.5, -7.2,
            "Primary endpoint: composite of sustained eGFR decline "
            "\u2265 40%, end-stage kidney disease or renal death at 24 months. "
            "Analysis by intention to treat.",
            ha="left", va="center", fontsize=6.5, color="#444444")

    ax.set_title("Participant flow through the trial (CONSORT 2010)",
                 fontsize=10.5, fontweight="bold", pad=8, loc="left", x=-0.005)

    fig.tight_layout()
    savefig(fig, "111_consort_flow.png")


if __name__ == "__main__":
    main()
