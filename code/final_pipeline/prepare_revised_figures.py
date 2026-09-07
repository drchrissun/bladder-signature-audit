# -*- coding: utf-8 -*-
"""Regenerate edge-safe figures and assemble the final submission figure folder."""

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "p01_full"
MARKER_COMPARISON = (
    ROOT / "data" / "processed" / "revision_gse13507_marker_group_comparison.csv"
)
OUT = ROOT / "figures"


def save(fig, name: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT / f"{name}.{extension}",
            dpi=300,
            facecolor="white",
        )


def build_figure2() -> None:
    mucosa = pd.read_csv(PROCESSED / "normal_mucosa_signature_summary.csv")
    marker = pd.read_csv(MARKER_COMPARISON)
    marker.loc[
        marker["cell_type"].eq("Normal_urothelial"), "cell_type"
    ] = "Normal-like component"
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    axes[0].hist(
        mucosa["normal_looking_minus_primary"],
        bins=35,
        color="#56B4E9",
        alpha=0.9,
        edgecolor="white",
        lw=0.4,
    )
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Normal-looking mucosa score minus primary tumor score", labelpad=12)
    axes[0].set_ylabel("Number of gene lists")
    axes[0].set_title("A: Gene-list score comparison", pad=4)
    sns.despine(ax=axes[0])

    order = [
        "Normal-like component",
        "Tumor_epithelial",
        "Fibroblast",
        "Myofibroblast",
        "Endothelial",
        "T_cell",
        "NK_cell",
        "B_plasma",
        "Myeloid",
        "Mast_cell",
    ]
    marker = marker.set_index("cell_type").loc[order].reset_index()
    display_names = {
        "Normal-like component": "Normal-like",
        "Tumor_epithelial": "Tumor epi.",
        "T_cell": "T cell",
        "NK_cell": "NK cell",
        "B_plasma": "B/plasma",
        "Mast_cell": "Mast",
    }
    marker["display"] = marker["cell_type"].map(
        lambda value: display_names.get(value, value.replace("_", " "))
    )
    colors = [
        "#0072B2" if value >= 0 else "#D55E00"
        for value in marker["mean_difference"]
    ]
    axes[1].barh(
        marker["display"],
        marker["mean_difference"],
        color=colors,
        alpha=0.85,
    )
    axes[1].axvline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Normal-minus-primary", labelpad=12)
    axes[1].set_ylabel("Marker-score cell type")
    axes[1].set_title("B: Cell-type marker-score comparison", pad=4)
    axes[1].invert_yaxis()
    axes[1].tick_params(axis="y", labelsize=7.5)
    sns.despine(ax=axes[1])
    fig.subplots_adjust(left=0.11, right=0.94, top=0.85, bottom=0.30, wspace=0.62)
    save(fig, "Figure2")
    plt.close(fig)


def build_figure_s2() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.8)
    ax.axis("off")
    nodes = {
        "stage": (0.4, 7.0, "Stage/grade", "#DEEBF7"),
        "tumor": (4.0, 7.0, "Tumor biology", "#DEEBF7"),
        "normal": (0.4, 4.0, "Normal-like\ncomponent score", "#FFF2CC"),
        "micro": (4.0, 4.0, "Immune/stromal\ncomposition", "#FFF2CC"),
        "normal_score": (0.4, 0.8, "Measured normal-like\nscore", "#E2EFDA"),
        "micro_score": (4.0, 0.8, "Measured immune/\nstromal score", "#E2EFDA"),
        "survival": (7.6, 4.0, "Overall survival", "#FCE4D6"),
        "platform": (7.6, 0.8, "Platform/sampling", "#EDEDED"),
    }
    for key, (x, y, label, color) in nodes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.0,
                1.05,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=0.8,
                edgecolor="#555555",
                facecolor=color,
                zorder=2,
            )
        )
        ax.text(
            x + 1.0,
            y + 0.525,
            label,
            ha="center",
            va="center",
            fontsize=7.4,
            linespacing=1.9,
            zorder=3,
        )

    def add_edge(start, end, sign, connection, label_offset, sign_offset=(0.18, 0.18)):
        sx, sy = start
        ex, ey = end
        ax.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (ex, ey),
                connectionstyle=connection,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=0.9,
                color="#333333",
                zorder=1,
            )
        )
        ax.text(
            label_offset[0],
            label_offset[1],
            sign,
            ha="center",
            va="center",
            fontsize=7.3,
            color="#CC0000",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=4,
        )

    add_edge(
        (1.4, 7.0),
        (1.4, 5.05),
        "+/-",
        "arc3,rad=0",
        (1.72, 5.82),
    )
    add_edge(
        (2.4, 7.52),
        (8.6, 4.52),
        "-",
        "arc3,rad=-0.52",
        (3.55, 7.62),
        sign_offset=(0.12, 0.12),
    )
    add_edge(
        (5.0, 7.0),
        (5.0, 5.05),
        "+",
        "arc3,rad=0",
        (5.28, 5.82),
    )
    add_edge(
        (6.0, 7.32),
        (7.6, 4.82),
        "-",
        "arc3,rad=-0.28",
        (6.85, 6.05),
    )
    add_edge(
        (1.4, 4.0),
        (1.4, 1.85),
        "+",
        "arc3,rad=0",
        (1.68, 2.72),
    )
    add_edge(
        (5.0, 4.0),
        (5.0, 1.85),
        "+",
        "arc3,rad=0",
        (5.28, 2.72),
    )
    add_edge(
        (6.0, 4.52),
        (7.6, 4.52),
        "+/-",
        "arc3,rad=0",
        (6.8, 4.78),
    )
    add_edge(
        (8.6, 0.8),
        (2.4, 1.32),
        "+/-",
        "arc3,rad=-0.62",
        (5.45, 0.12),
        sign_offset=(0, 0),
    )
    add_edge(
        (7.6, 1.32),
        (6.0, 1.32),
        "+/-",
        "arc3,rad=0",
        (6.8, 1.58),
    )
    ax.set_title("Conceptual DAG with assumed effect directions", fontsize=9.5, pad=0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    save(fig, "FigureS2")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_figure2()
    build_figure_s2()
    print("revised figure package written to", OUT)


if __name__ == "__main__":
    main()
