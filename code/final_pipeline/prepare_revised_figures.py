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
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    axes[0].hist(
        mucosa["normal_looking_minus_primary"],
        bins=35,
        color="#56B4E9",
        alpha=0.9,
        edgecolor="white",
        lw=0.4,
    )
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Normal-looking mucosa score minus primary tumor score")
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
    axes[1].bar(
        marker["display"],
        marker["mean_difference"],
        color=colors,
        alpha=0.85,
    )
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Marker-score cell type")
    axes[1].set_ylabel("Normal-minus-primary")
    axes[1].set_title("B: Cell-type marker-score comparison", pad=4)
    axes[1].tick_params(axis="x", rotation=20, labelsize=8)
    sns.despine(ax=axes[1])
    fig.subplots_adjust(left=0.12, right=0.93, top=0.86, bottom=0.26, wspace=0.58)
    save(fig, "Figure2")
    plt.close(fig)


def build_figure_s2() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    boxes = {
        "stage": (0.7, 5.3, "Stage/grade", "#DEEBF7"),
        "tumor": (4.3, 5.5, "Tumor biology", "#DEEBF7"),
        "normal": (0.7, 3.0, "Normal-like component score", "#FFF2CC"),
        "micro": (4.3, 3.0, "Immune/stromal composition", "#FFF2CC"),
        "normal_score": (0.7, 0.7, "Measured normal-like score", "#E2EFDA"),
        "micro_score": (4.3, 0.7, "Measured immune/stromal score", "#E2EFDA"),
        "survival": (7.8, 3.0, "Overall survival", "#FCE4D6"),
        "platform": (7.5, 5.5, "Platform/sampling", "#EDEDED"),
    }
    for key, (x, y, label, color) in boxes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.0,
                1.0,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=0.7,
                edgecolor="#555555",
                facecolor=color,
            )
        )
        ax.text(x + 1.0, y + 0.5, label, ha="center", va="center", fontsize=7)
    arrows = [
        ("stage", "normal", "+/-"),
        ("stage", "survival", "-"),
        ("tumor", "micro", "+"),
        ("tumor", "survival", "-"),
        ("normal", "normal_score", "+"),
        ("micro", "micro_score", "+"),
        ("micro", "survival", "+/-"),
        ("platform", "normal_score", "+/-"),
        ("platform", "micro_score", "+/-"),
    ]
    for start, end, sign in arrows:
        sx = boxes[start][0] + 1.0
        sy = boxes[start][1]
        ex = boxes[end][0] + 1.0
        ey = boxes[end][1] + 1.0
        ax.add_patch(
            FancyArrowPatch(
                (sx, sy),
                (ex, ey),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.7,
                color="#333333",
            )
        )
        ax.text((sx + ex) / 2, (sy + ey) / 2, sign, fontsize=7, color="#CC0000")
    ax.set_title("Conceptual DAG with assumed effect directions", fontsize=9, pad=0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    save(fig, "FigureS2")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_figure2()
    build_figure_s2()
    print("revised figure package written to", OUT)


if __name__ == "__main__":
    main()
