from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "p01_full"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

PALETTE = {
    "blue": "#0F4D92",
    "blue_light": "#B4C0E4",
    "red": "#B64342",
    "orange": "#D55E00",
    "green": "#2E9E44",
    "grey": "#767676",
    "grey_light": "#D8D8D8",
    "black": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

CLASS_COLORS = {
    "composition_adjustment_persistent": "#0F4D92",
    "microenvironment_associated_attenuation": "#D55E00",
    "microenvironment_correlated": "#42949E",
    "unclassified": "#8F8F8F",
}


def save_figure(fig, name):
    stem = OUT / name
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_panel(ax, label, x=-0.05, y=1.02):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def load_main_tables():
    nested = pd.read_csv(PROCESSED / "nested_cox_standard.csv", dtype={"pmid": str})
    gse13507 = pd.read_csv(
        PROCESSED / "gse13507_survival_validation_standard.csv",
        dtype={"pmid": str},
    )
    gse48075 = pd.read_csv(
        PROCESSED / "gse48075_survival_validation.csv",
        dtype={"pmid": str},
    )
    classification = pd.read_csv(
        PROCESSED / "classification_v6_standard.csv", dtype={"pmid": str}
    )
    corr = pd.read_csv(
        PROCESSED / "microenvironment_correlations_tcga.csv",
        dtype={"pmid": str},
    )
    mucosa = pd.read_csv(
        PROCESSED / "normal_mucosa_signature_summary.csv",
        dtype={"pmid": str},
    )
    return nested, gse13507, gse48075, classification, corr, mucosa


def figure_external_forest(nested, gse13507, gse48075):
    ids = nested[
        (nested["model"] == "m3_microenvironment")
        & (nested["qvalue_global"] < 0.05)
    ]["pmid"]

    tcga = nested[nested["model"] == "m3_microenvironment"].set_index("pmid")
    tcga = tcga.loc[ids, ["coef", "hr", "pvalue"]]

    gse1 = gse13507[gse13507["model"] == "adjusted"].set_index("pmid")
    gse1 = gse1.loc[ids, ["coef", "hr", "pvalue"]]

    gse2 = gse48075[gse48075["model"] == "adjusted"].set_index("pmid")
    gse2 = gse2.loc[ids, ["coef", "hr", "pvalue"]]

    frame = pd.DataFrame(
        {
            "tcga_hr": tcga["hr"],
            "gse13507_hr": gse1["hr"],
            "gse48075_hr": gse2["hr"],
            "gse13507_p": gse1["pvalue"],
            "gse48075_p": gse2["pvalue"],
        }
    )
    frame = frame.sort_values("gse48075_hr", ascending=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 5.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1, 1]},
    )
    y = np.arange(len(frame))[::-1]

    for ax, column, pcolumn, title, color in (
        (
            axes[0],
            "tcga_hr",
            None,
            "TCGA-BLCA M3",
            PALETTE["black"],
        ),
        (
            axes[1],
            "gse13507_hr",
            "gse13507_p",
            "GSE13507 adjusted",
            PALETTE["blue"],
        ),
        (
            axes[2],
            "gse48075_hr",
            "gse48075_p",
            "GSE48075 adjusted",
            PALETTE["orange"],
        ),
    ):
        hr = frame[column].to_numpy()
        nominal = (
            frame[pcolumn].to_numpy() < 0.05 if pcolumn is not None else np.zeros(len(frame), dtype=bool)
        )
        point_colors = []
        for value, significant in zip(hr, nominal):
            point_colors.append(
                "#0072B2" if value < 1 else "#D55E00"
            )
        ax.scatter(
            hr,
            y,
            s=18,
            c=point_colors,
            edgecolors="black" if column == "tcga_hr" else (
                np.where(nominal, PALETTE["black"], PALETTE["grey"])
            ),
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )
        ax.axvline(1, color=PALETTE["grey"], linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.25, 4.2)
        ax.set_xticks([0.5, 1, 2, 4])
        ax.minorticks_off()
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.tick_params(labelsize=6)
        ax.set_title(title, fontsize=7, pad=4)
        ax.tick_params(labelsize=6)
        ax.set_facecolor("white")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(frame.index, fontsize=5)
    axes[0].set_ylabel("PMID", fontsize=7)
    axes[0].set_xlabel("HR", fontsize=7)
    axes[1].set_xlabel("HR", fontsize=7)
    axes[2].set_xlabel("HR", fontsize=7)
    axes[0].set_ylim(-0.7, len(frame) - 0.3)

    for ax, label in zip(axes, "abc"):
        add_panel(ax, label)

    fig.subplots_adjust(wspace=0.28, left=0.16, right=0.97, top=0.92, bottom=0.10)
    save_figure(fig, "NatureCandidate_ExternalValidationDotPlot")


def figure_classification_atlas(classification, corr, mucosa, nested):
    rho = corr.pivot(index="pmid", columns="covariate", values="spearman_rho")
    rho = rho.rename(
        columns={
            "normal_urothelial_fraction": "normal_rho",
            "tumor_epithelial_fraction": "tumor_rho",
            "immune_score": "immune_rho",
            "stromal_score": "stromal_rho",
        }
    )
    hr = nested.pivot(index="pmid", columns="model", values="hr")
    log2hr = np.log2(hr)
    frame = classification.merge(
        rho, left_on="pmid", right_index=True, how="inner"
    )
    frame = frame.merge(
        log2hr[["m0_score", "m3_microenvironment"]],
        left_on="pmid",
        right_index=True,
        how="inner",
    )
    mucosa = mucosa.set_index("pmid")["normal_looking_minus_primary"]
    frame["normal_mucosa_delta"] = frame["pmid"].map(mucosa)

    order = [
        "composition_adjustment_persistent_standard",
        "microenvironment_associated_attenuation_standard",
        "microenvironment_correlated_standard",
        "unclassified_standard",
    ]
    frame["class_order"] = frame["classification_v6_standard"].map(order.index)
    frame = frame.sort_values(
        ["class_order", "max_micro_abs_rho"],
        ascending=[True, False],
    )

    columns = [
        "m0_score",
        "m3_microenvironment",
        "immune_rho",
        "stromal_rho",
        "normal_rho",
        "tumor_rho",
        "normal_mucosa_delta",
    ]
    matrix = frame[columns].to_numpy(dtype=float)
    matrix = (matrix - np.nanmean(matrix, axis=0)) / np.nanstd(matrix, axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-2.5,
        vmax=2.5,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [
            "M0 log2 HR",
            "M3 log2 HR",
            "Immune rho",
            "Stromal rho",
            "Normal-like rho",
            "Tumor rho",
            "Normal mucosa delta",
        ],
        rotation=30,
        ha="right",
        fontsize=6,
    )
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(columns) - 0.5)

    class_order = frame["classification_v6_standard"].to_numpy()
    class_ids = [order.index(value) for value in class_order]
    boundary_indices = [
        int(index)
        for index in range(1, len(class_ids))
        if class_ids[index] != class_ids[index - 1]
    ]
    for boundary in boundary_indices:
        ax.axhline(boundary - 0.5, color="black", linewidth=0.45, alpha=0.55)

    class_side = np.column_stack(
        [
            np.array(class_ids, dtype=float),
            np.array(class_ids, dtype=float),
            np.array(class_ids, dtype=float),
        ]
    )
    class_keys = [
        "composition_adjustment_persistent",
        "microenvironment_associated_attenuation",
        "microenvironment_correlated",
        "unclassified",
    ]
    ax_side = fig.add_axes([0.86, 0.125, 0.015, 0.77])
    ax_side.imshow(
        class_side,
        cmap=ListedColormap(
            [CLASS_COLORS[value] for value in class_keys]
        ),
        aspect="auto",
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    ax_side.set_ylabel("Class", fontsize=6)
    ax_side.yaxis.set_label_position("right")

    cax = fig.add_axes([0.895, 0.125, 0.012, 0.77])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Column z-score", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax.set_title(
        "200 signatures ordered by audit class and maximum immune/stromal correlation",
        fontsize=7,
        pad=5,
    )
    add_panel(ax, "a", x=-0.02, y=1.04)
    fig.legend(
        handles=[
            Patch(facecolor=CLASS_COLORS[value], edgecolor="none", label=value.replace("_", " "))
            for value in class_keys
        ],
        loc="lower center",
        bbox_to_anchor=(0.44, -0.015),
        ncol=4,
        fontsize=5,
        handlelength=1.2,
        columnspacing=0.8,
    )
    save_figure(fig, "NatureCandidate_SignatureClassificationAtlas")


def figure_sensitivity_grid():
    rho_counts = pd.read_csv(PROCESSED / "p13_rho_counts.csv")
    pivot = rho_counts.pivot(index="threshold", columns="class", values="count")
    class_names = [
        "composition_adjustment_persistent",
        "microenvironment_associated_attenuation",
        "microenvironment_correlated",
        "unclassified",
    ]

    bootstrap = pd.read_csv(PROCESSED / "revision_bootstrap_1000_summary.csv")
    nested = pd.read_csv(PROCESSED / "nested_cox_standard.csv", dtype={"pmid": str})
    significant_ids = set(
        nested[
            (nested["model"] == "m3_microenvironment")
            & (nested["qvalue_global"] < 0.05)
        ]["pmid"]
    )
    bootstrap["significant"] = bootstrap["pmid"].astype(str).isin(significant_ids)

    ocr_q = pd.read_csv(PROCESSED / "p16_ocr_q_counts.csv")
    model_order = ["m0_score", "m1_clinical", "m2_purity", "m3_microenvironment"]
    ocr_full = ocr_q[ocr_q["analysis_set"] == "full_200"].set_index("model")[
        "fdr_significant"
    ]
    ocr_excl = ocr_q[
        ocr_q["analysis_set"] == "exclude_42493447_199"
    ].set_index("model")["fdr_significant"]

    corr = pd.read_csv(PROCESSED / "p15_covariate_correlations.csv")
    cov_names = [
        "age",
        "sex_female",
        "stage_num",
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
        "immune_score",
        "stromal_score",
    ]
    cov_short = ["Age", "Sex", "Stage", "Tumor", "Normal", "Immune", "Stromal"]
    corr_matrix = pd.DataFrame(
        np.eye(len(cov_names)),
        index=cov_names,
        columns=cov_names,
    )
    for row in corr.itertuples(index=False):
        corr_matrix.loc[row.covariate_1, row.covariate_2] = row.pearson_r
        corr_matrix.loc[row.covariate_2, row.covariate_1] = row.pearson_r
    corr_values = corr_matrix.loc[cov_names, cov_names].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5))

    for class_name, color in zip(
        class_names,
        [
            PALETTE["blue"],
            PALETTE["orange"],
            PALETTE["teal"],
            PALETTE["grey"],
        ],
    ):
        short_labels = {
            "composition_adjustment_persistent": "Persistent",
            "microenvironment_associated_attenuation": "Attenuation",
            "microenvironment_correlated": "Correlated",
            "unclassified": "Unclassified",
        }
        axes[0, 0].plot(
            pivot.index,
            pivot[class_name],
            marker="o",
            linewidth=1.3,
            markersize=3.5,
            color=color,
            label=short_labels[class_name],
        )
    axes[0, 0].set_xlabel("rho threshold", fontsize=6)
    axes[0, 0].set_ylabel("Signatures", fontsize=6)
    axes[0, 0].set_xticks([0.3, 0.4, 0.5])
    axes[0, 0].tick_params(labelsize=5)
    axes[0, 0].legend(fontsize=5, loc="upper left", borderaxespad=0)

    axes[0, 1].hist(
        bootstrap.loc[~bootstrap["significant"], "significant_rate"],
        bins=20,
        color=PALETTE["grey_light"],
        edgecolor="white",
        linewidth=0.3,
        label="All other signatures",
        alpha=0.9,
    )
    axes[0, 1].hist(
        bootstrap.loc[bootstrap["significant"], "significant_rate"],
        bins=20,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.3,
        label="31 FDR-significant signatures",
        alpha=0.85,
    )
    axes[0, 1].set_xlabel("Bootstrap M3 significance rate", fontsize=6)
    axes[0, 1].set_ylabel("Signatures", fontsize=6)
    axes[0, 1].tick_params(labelsize=5)
    axes[0, 1].legend(fontsize=5)

    x = np.arange(len(model_order))
    width = 0.38
    axes[1, 0].bar(
        x - width / 2,
        [ocr_full[model] for model in model_order],
        width,
        color=PALETTE["grey"],
        label="Full 200",
    )
    axes[1, 0].bar(
        x + width / 2,
        [ocr_excl[model] for model in model_order],
        width,
        color=PALETTE["blue"],
        label="Exclude OCR",
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(["M0", "M1", "M2", "M3"], fontsize=5)
    axes[1, 0].set_ylabel("FDR-significant lists", fontsize=6)
    axes[1, 0].tick_params(labelsize=5)
    axes[1, 0].legend(fontsize=5)

    im = axes[1, 1].imshow(
        corr_values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axes[1, 1].set_xticks(range(len(cov_short)))
    axes[1, 1].set_xticklabels(cov_short, rotation=35, ha="right", fontsize=5)
    axes[1, 1].set_yticks(range(len(cov_short)))
    axes[1, 1].set_yticklabels(cov_short, fontsize=5)
    axes[1, 1].set_frame_on(False)
    axes[1, 1].tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    for ax, label in zip(axes.ravel(), "abcd"):
        add_panel(ax, label)

    fig.subplots_adjust(wspace=0.55, hspace=0.48, left=0.08, right=0.94)
    save_figure(fig, "NatureCandidate_SensitivityDiagnostics")


def figure_pathway_bubble():
    pathway = pd.read_csv(PROCESSED / "p13_class_pathway_enrichment.csv")
    pathway = pathway[pathway["group_family"] == "class"].copy()
    pathway["pathway_short"] = (
        pathway["pathway"]
        .str.replace("HALLMARK_", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )
    pathway["neg_log10_q"] = -np.log10(pathway["qvalue"].clip(lower=1e-8))
    selected = (
        pathway.sort_values("qvalue")
        .groupby("group")
        .head(6)
        .sort_values(["group", "neg_log10_q"])
    )
    selected["y"] = range(len(selected))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    groups = [
        "composition_adjustment_persistent",
        "microenvironment_associated_attenuation",
        "microenvironment_correlated",
        "unclassified",
    ]
    colors = [CLASS_COLORS[group] for group in groups]
    for group, color in zip(groups, colors):
        sub = selected[selected["group"] == group]
        ax.scatter(
            sub["neg_log10_q"],
            sub["y"],
            s=sub["overlap"] * 24,
            color=color,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.5,
            label=group.replace("_", " "),
        )
    ax.set_yticks(selected["y"])
    ax.set_yticklabels(selected["pathway_short"], fontsize=5)
    ax.set_xlabel("-log10(q)", fontsize=6)
    ax.set_ylabel("Hallmark pathway", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=5, loc="lower right")
    ax.set_xlim(left=0)
    for size in (5, 10, 20):
        ax.scatter(
            [],
            [],
            s=size * 24,
            color=PALETTE["grey_light"],
            edgecolors="white",
            label=f"{size} overlapping genes",
        )
    add_panel(ax, "a")
    save_figure(fig, "NatureCandidate_PathwayBubble")


def main():
    nested, gse13507, gse48075, classification, corr, mucosa = load_main_tables()
    figure_external_forest(nested, gse13507, gse48075)
    figure_classification_atlas(classification, corr, mucosa, nested)
    figure_sensitivity_grid()
    figure_pathway_bubble()
    print("Wrote nature candidate figures to", OUT)


if __name__ == "__main__":
    main()
