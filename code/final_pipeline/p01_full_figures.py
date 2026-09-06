from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROCESSED = Path("data/processed/p01_full")
MARKER_COMPARISON = Path(
    "data/processed/revision_gse13507_marker_group_comparison.csv"
)
OUT = Path("figures")


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=300,
            bbox_inches="tight",
        )


def load_nested():
    nested = pd.read_csv(PROCESSED / "nested_cox_standard.csv")
    pivot = nested.pivot(index="pmid", columns="model", values="coef")
    q = nested.pivot(index="pmid", columns="model", values="qvalue_global")
    return pivot, q


def figure1():
    pivot, q = load_nested()
    classification = pd.read_csv(
        PROCESSED / "classification_v6_standard.csv"
    ).set_index("pmid")["classification_v6_standard"]
    frame = pivot.copy()
    frame["classification"] = classification
    frame["lost"] = (q["m0_score"] < 0.05) & (q["m3_microenvironment"] >= 0.05)
    frame["gained"] = (q["m0_score"] >= 0.05) & (q["m3_microenvironment"] < 0.05)
    frame["fdr_significant"] = q["m3_microenvironment"] < 0.05
    frame["color"] = np.select(
        [
            frame["gained"],
            frame["lost"],
            frame["fdr_significant"],
        ],
        ["green", "orange", "blue"],
        default="gray",
    )

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    axes[0].scatter(
        frame["m0_score"],
        frame["m3_microenvironment"],
        s=7,
        color=frame["color"],
        alpha=0.75,
    )
    axes[0].set_xlabel("M0 log HR")
    axes[0].set_ylabel("M3 log HR")
    axes[0].set_title("A: M0 versus M3")

    corr = pd.read_csv(PROCESSED / "microenvironment_correlations_tcga.csv")
    micro = corr[corr["covariate"].isin(["immune_score", "stromal_score"])]
    top_series = (
        micro.groupby("pmid")["spearman_rho"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=False)
        .head(30)
    )
    axes[1].barh(
        range(len(top_series)),
        top_series.values,
        color="#0072B2",
    )
    axes[1].set_title("B: Top 30 microenvironment rho")

    mucosa = pd.read_csv(PROCESSED / "normal_mucosa_signature_summary.csv")
    axes[2].hist(
        mucosa["normal_looking_minus_primary"],
        bins=30,
        color="#9AA0A6",
        edgecolor="white",
    )
    axes[2].set_xlabel("Normal-looking minus primary")
    axes[2].set_title("C: GSE13507 score difference")

    counts = classification.value_counts().reindex(
        [
            "composition_adjustment_persistent_standard",
            "microenvironment_associated_attenuation_standard",
            "microenvironment_correlated_standard",
            "unclassified_standard",
        ]
    )
    axes[3].bar(
        ["Persistent", "Attenuation", "Correlated", "Unclassified"],
        counts,
        color="#0072B2",
    )
    axes[3].tick_params(axis="x", rotation=30)
    axes[3].set_title("D: Classification")

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "Figure1")


def figure2():
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


def figure_s3():
    bootstrap = pd.read_csv(PROCESSED / "revision_bootstrap_1000_summary.csv")
    nested = pd.read_csv(PROCESSED / "nested_cox_standard.csv")
    ids = nested[
        (nested["model"] == "m3_microenvironment")
        & (nested["qvalue_global"] < 0.05)
    ]["pmid"]
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.hist(
        bootstrap["significant_rate"],
        bins=25,
        color="#9AA0A6",
        edgecolor="white",
        label="All 200 signatures",
    )
    ax.hist(
        bootstrap[bootstrap["pmid"].isin(ids)]["significant_rate"],
        bins=20,
        color="#0072B2",
        edgecolor="white",
        alpha=0.75,
        label="31 FDR-significant signatures",
    )
    ax.set_xlabel("Bootstrap M3 significance rate, 1000 resamples")
    ax.set_ylabel("Number of signatures")
    ax.legend(frameon=False, fontsize=6)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "FigureS3")


def figure_s4():
    nested = pd.read_csv(PROCESSED / "nested_cox_standard.csv")
    ids = nested[
        (nested["model"] == "m3_microenvironment")
        & (nested["qvalue_global"] < 0.05)
    ]["pmid"]
    tcga = nested[nested["model"] == "m3_microenvironment"].set_index("pmid")["coef"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, name in zip(
        axes,
        [
            "gse13507_survival_validation_standard.csv",
            "gse48075_survival_validation.csv",
        ],
    ):
        ext = pd.read_csv(PROCESSED / name)
        ext = ext[(ext["model"] == "adjusted") & ext["pmid"].isin(ids)].merge(
            tcga.rename("tcga_coef"),
            on="pmid",
        )
        ax.scatter(ext["tcga_coef"], ext["coef"], s=10, color="#0072B2")
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.axvline(0, color="gray", lw=0.6, ls="--")
        ax.set_xlabel("TCGA-BLCA adjusted coefficient")
        ax.set_ylabel("External adjusted coefficient")
        ax.set_title("GSE13507" if "gse13507" in name else "GSE48075")
        sns.despine(ax=ax)
    fig.suptitle("31 M3 FDR-significant gene lists", fontsize=9)
    fig.tight_layout()
    save(fig, "FigureS4")


def figure_s5():
    decomposition = pd.read_csv(PROCESSED / "decomposition_models.csv")
    q = decomposition.pivot(index="pmid", columns="model", values="qvalue_global")
    counts = q[["m1_clinical", "m1_plus_tumor", "m1_plus_normal", "m1_plus_both"]].apply(
        lambda s: (s < 0.05).sum()
    )
    detail = pd.read_csv(PROCESSED / "gained_decomposition_detail.csv")
    detail["shift"] = detail["m1_plus_both_coef"] - detail["m1_clinical_coef"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    axes[0].bar(counts.index, counts.values, color="#0072B2")
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(
        ["M1", "M1+tumor", "M1+normal-like", "M1+both"], rotation=20
    )
    axes[0].set_ylabel("FDR-significant gene lists")
    axes[1].bar(range(len(detail)), detail.sort_values("shift")["shift"], color="#E69F00")
    axes[1].set_ylabel("M1+both coefficient minus M1")
    axes[1].set_xlabel("14 newly significant gene lists")
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "FigureS5")


def figure_s6():
    cross = pd.read_csv(
        "outputs/consensus/p01_consensus_cross_tab_classification.csv",
        index_col=0,
    )
    cross.index = [
        "Persistent",
        "Attenuation",
        "Microenvironment-correlated",
        "Unclassified",
    ]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    sns.heatmap(
        cross,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Signatures"},
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Audit class by modal Consensus subtype in high-score patients")
    ax.set_xlabel("Consensus subtype")
    ax.set_ylabel("Audit class")
    fig.tight_layout()
    save(fig, "FigureS6")


if __name__ == "__main__":
    figure1()
    figure2()
    figure_s3()
    figure_s4()
    figure_s5()
    figure_s6()
