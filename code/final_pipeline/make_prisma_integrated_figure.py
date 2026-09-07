from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("figures")


def box(ax, x, y, width, height, title, detail, color="#EDF3F8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#555555",
            facecolor=color,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.32,
        detail,
        ha="center",
        va="center",
        fontsize=6.5,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=0.8,
            color="#333333",
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(7.4, 8.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(
        ax,
        3.0,
        8.9,
        4.0,
        0.8,
        "Identification",
        "PubMed records identified (n = 324)",
        "#DEEBF7",
    )
    arrow(ax, 5.0, 8.9, 5.0, 8.2)

    box(
        ax,
        2.2,
        7.0,
        5.6,
        1.35,
        "Initial scripted screening",
        "192 title/abstract; 132 excluded\n"
        "154 rule-based; 144 strict inclusion\n"
        "Title/abstract exclusions: 88 not signature-like, 37 non-mRNA, 7 other",
        "#DEEBF7",
    )
    arrow(ax, 5.0, 7.0, 5.0, 6.35)

    box(
        ax,
        1.5,
        4.55,
        7.0,
        1.8,
        "Independent dual screening of all 324 records",
        "Reviewer 1: 159 include / 165 exclude\n"
        "Reviewer 2: 217 include / 107 exclude\n"
        "Agreement 244/324; Cohen's kappa = 0.509",
        "#FFF2CC",
    )
    arrow(ax, 5.0, 4.55, 5.0, 3.95)

    box(
        ax,
        2.4,
        2.75,
        5.2,
        1.2,
        "Consensus adjudication",
        "80 disagreements adjudicated\n"
        "213 screening-eligible; 111 excluded",
        "#DEEBF7",
    )
    arrow(ax, 5.0, 2.75, 5.0, 2.2)

    box(
        ax,
        1.4,
        0.35,
        7.2,
        1.85,
        "Analysis-freeze gene-list extraction and scoring",
        "203 complete gene lists from 213 consensus-eligible records\n"
        "TCGA-BLCA scoreable 200; GSE13507 scoreable 201\n"
        "10 consensus-eligible records had no extractable list",
        "#FFF2CC",
    )

    fig.tight_layout(pad=0.3)
    for ext in ["pdf", "png"]:
        fig.savefig(
            OUT / f"FigureS1.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("integrated PRISMA figure written")


if __name__ == "__main__":
    main()
