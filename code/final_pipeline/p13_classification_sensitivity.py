from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"
GMT = Path("reference/h.all.v2024.1.Hs.symbols.gmt")
COMMON = pd.read_csv(PROCESSED / "common_gene_symbols.csv")["symbol"]
WORKBOOK = Path(
    "supplementary/Supplementary_Tables_S1_S72_UroOnc_revised.xlsx"
)

CLASS_ORDER = [
    "composition_adjustment_persistent",
    "microenvironment_associated_attenuation",
    "microenvironment_correlated",
    "unclassified",
]


def classify_at(frame, threshold):
    strong = frame["max_micro_abs_rho"] >= threshold
    adjusted = frame["adjusted_fdr_significant"]
    lost = frame["lost_fdr_significance"]
    normal = frame["normal_mucosa_elevation_significant"]
    return np.select(
        [
            adjusted & ~(strong | normal),
            lost,
            strong | normal,
        ],
        [
            "composition_adjustment_persistent",
            "microenvironment_associated_attenuation",
            "microenvironment_correlated",
        ],
        default="unclassified",
    )


def load_gene_sets():
    sets = {}
    for line in GMT.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        sets[fields[0]] = set(fields[2:])
    return sets


def class_transition_summary(classification):
    rows = []
    for label in CLASS_ORDER:
        base = classification["class_0.4"].eq(label)
        base_n = int(base.sum())
        for threshold in (0.3, 0.5):
            column = f"class_{threshold}"
            n = int(classification.loc[base, column].eq(label).sum())
            rows.append(
                {
                    "threshold": threshold,
                    "base_class": label,
                    "base_count": base_n,
                    "retained_count": n,
                    "changed_count": base_n - n,
                }
            )
    return pd.DataFrame(rows)


def pathway_enrichment(classification, genes_by_pmid):
    universe = set(COMMON)
    gene_sets = load_gene_sets()
    classification = classification.copy()
    classification["survival_group"] = "not_significant"
    classification.loc[
        (classification["m0_global_q"] < 0.05)
        & (classification["m3_global_q"] >= 0.05),
        "survival_group",
    ] = "lost_fdr"
    classification.loc[
        (classification["m0_global_q"] >= 0.05)
        & (classification["m3_global_q"] < 0.05),
        "survival_group",
    ] = "gained_fdr"
    classification.loc[
        (classification["m0_global_q"] < 0.05)
        & (classification["m3_global_q"] < 0.05),
        "survival_group",
    ] = "remained_fdr"

    groups = {"class_": {}, "survival_": {}}
    for label in CLASS_ORDER:
        pmids = classification.loc[
            classification["class_0.4"].eq(label), "pmid"
        ]
        group_genes = set()
        for pmid in pmids:
            group_genes.update(genes_by_pmid.get(str(pmid), []))
        groups["class_"][label] = group_genes & universe
    for label in (
        "lost_fdr",
        "gained_fdr",
        "remained_fdr",
        "not_significant",
    ):
        pmids = classification.loc[
            classification["survival_group"].eq(label), "pmid"
        ]
        group_genes = set()
        for pmid in pmids:
            group_genes.update(genes_by_pmid.get(str(pmid), []))
        groups["survival_"][label] = group_genes & universe

    rows = []
    for family, family_groups in groups.items():
        for group_name, group_genes in family_groups.items():
            if not group_genes:
                continue
            for pathway, pathway_genes in gene_sets.items():
                pathway_genes = pathway_genes & universe
                overlap = group_genes & pathway_genes
                if not overlap:
                    continue
                a = len(overlap)
                b = len(group_genes - pathway_genes)
                c = len(pathway_genes - group_genes)
                d = len(universe - group_genes - pathway_genes)
                odds_ratio, pvalue = fisher_exact(
                    [[a, b], [c, d]], alternative="greater"
                )
                rows.append(
                    {
                        "group_family": family.rstrip("_"),
                        "group": group_name,
                        "pathway": pathway,
                        "group_unique_genes": len(group_genes),
                        "pathway_genes": len(pathway_genes),
                        "overlap": a,
                        "odds_ratio": odds_ratio,
                        "pvalue": pvalue,
                    }
                )
    result = pd.DataFrame(rows)
    result["qvalue"] = multipletests(result["pvalue"], method="fdr_bh")[1]
    return result.sort_values(["group_family", "group", "qvalue"])


def main():
    classification = pd.read_csv(OUT / "classification_v6_standard.csv")
    coverage = pd.read_csv(OUT / "signature_coverage.csv")
    inventory = pd.read_csv(
        OUT / "gene_extraction_raw_full.csv", dtype={"pmid": str}
    )
    inventory = inventory[inventory["extraction_status"] == "extracted"]
    mucosa = pd.read_csv(OUT / "normal_mucosa_signature_summary.csv")

    classification["pmid"] = classification["pmid"].astype(str)
    coverage["pmid"] = coverage["pmid"].astype(str)
    mucosa["pmid"] = mucosa["pmid"].astype(str)

    for threshold in (0.3, 0.4, 0.5):
        classification[f"class_{threshold}"] = classify_at(
            classification, threshold
        )

    per_signature = classification.merge(
        coverage[["pmid", "raw_gene_count", "tcga_mapped_count"]],
        on="pmid",
        how="left",
    ).merge(
        mucosa[
            [
                "pmid",
                "normal_looking_minus_primary",
                "mannwhitney_pvalue",
            ]
        ],
        on="pmid",
        how="left",
    )
    per_signature = per_signature[
        [
            "pmid",
            "max_micro_abs_rho",
            "normal_mucosa_elevation_significant",
            "m0_global_q",
            "m3_global_q",
            "raw_gene_count",
            "tcga_mapped_count",
            "normal_looking_minus_primary",
            "mannwhitney_pvalue",
            "class_0.3",
            "class_0.4",
            "class_0.5",
        ]
    ].sort_values(["class_0.4", "max_micro_abs_rho"], ascending=[True, False])
    per_signature.to_csv(OUT / "p13_rho_classification.csv", index=False)

    counts = []
    for threshold in (0.3, 0.4, 0.5):
        values = classification[f"class_{threshold}"].value_counts()
        for label in CLASS_ORDER:
            counts.append(
                {
                    "threshold": threshold,
                    "class": label,
                    "count": int(values.get(label, 0)),
                }
            )
    count_table = pd.DataFrame(counts)
    transition = class_transition_summary(classification)
    transition.to_csv(OUT / "p13_rho_transition_summary.csv", index=False)
    count_table.to_csv(OUT / "p13_rho_counts.csv", index=False)

    features = per_signature.merge(
        inventory[["pmid", "title", "year"]], on="pmid", how="left"
    )
    features["class_0.4"] = features["class_0.4"].astype("category")
    unclassified = features[features["class_0.4"].eq("unclassified")]
    unclassified.to_csv(OUT / "p13_unclassified_characteristics.csv", index=False)

    feature_rows = []
    for column in ("raw_gene_count", "tcga_mapped_count", "year"):
        for label in CLASS_ORDER:
            subset = features.loc[features["class_0.4"].eq(label), column].dropna()
            if subset.empty:
                continue
            feature_rows.append(
                {
                    "class": label,
                    "feature": column,
                    "n": len(subset),
                    "median": float(subset.median()),
                    "q1": float(subset.quantile(0.25)),
                    "q3": float(subset.quantile(0.75)),
                }
            )
        unclassified_values = unclassified[column].dropna()
        other_values = features.loc[
            ~features["class_0.4"].eq("unclassified"), column
        ].dropna()
        statistic, pvalue = mannwhitneyu(
            unclassified_values, other_values, alternative="two-sided"
        )
        feature_rows.append(
            {
                "class": "unclassified_vs_others",
                "feature": column,
                "n": len(unclassified_values),
                "median": float(unclassified_values.median()),
                "q1": float(unclassified_values.quantile(0.25)),
                "q3": float(unclassified_values.quantile(0.75)),
                "mannwhitney_pvalue": pvalue,
            }
        )
    feature_summary = pd.DataFrame(feature_rows)
    feature_summary.to_csv(OUT / "p13_unclassified_feature_summary.csv", index=False)

    genes_by_pmid = {}
    for row in inventory.itertuples(index=False):
        genes_by_pmid[row.pmid] = [
            gene.strip() for gene in row.genes.split("|") if gene.strip()
        ]
    pathway = pathway_enrichment(classification, genes_by_pmid)
    pathway.to_csv(OUT / "p13_class_pathway_enrichment.csv", index=False)
    class_pathway = pathway[pathway["group_family"].eq("class")]
    survival_pathway = pathway[pathway["group_family"].eq("survival")]
    class_pathway.to_csv(
        OUT / "p13_class_pathway_enrichment.csv", index=False
    )
    survival_pathway.to_csv(
        OUT / "p13_survival_pathway_enrichment.csv", index=False
    )
    unclassified_pathway = class_pathway[class_pathway["group"].eq("unclassified")]
    unclassified_pathway.to_csv(
        OUT / "p13_unclassified_pathway_enrichment.csv", index=False
    )

    print("\nClass counts")
    print(count_table.pivot(index="class", columns="threshold", values="count"))
    print("\nTransition summary")
    print(transition.to_string(index=False))
    print("\nUnclassified features")
    print(feature_summary.to_string(index=False))
    print("\nTop unclassified pathways")
    print(
        unclassified_pathway.sort_values("qvalue")
        .head(10)[
            [
                "pathway",
                "group_unique_genes",
                "pathway_genes",
                "overlap",
                "odds_ratio",
                "qvalue",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    not_significant_pathway = survival_pathway[
        survival_pathway["group"].eq("not_significant")
    ]
    print("\nTop not-significant survival-group pathways")
    print(
        not_significant_pathway.sort_values("qvalue")
        .head(8)[
            [
                "pathway",
                "group_unique_genes",
                "pathway_genes",
                "overlap",
                "odds_ratio",
                "qvalue",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    sheet_data = {
        "S58_rho_classes": per_signature,
        "S59_rho_counts": count_table,
        "S60_rho_transitions": transition,
        "S61_unclass_features": unclassified,
        "S62_unclass_feature_summary": feature_summary,
        "S63_unclass_pathways": unclassified_pathway,
        "S64_survival_pathways": survival_pathway,
    }
    obsolete = {
        "S58_rho_threshold_classification",
        "S59_rho_threshold_counts",
        "S60_rho_transition_summary",
        "S61_unclassified_characteristics",
        "S62_unclassified_feature_summary",
        "S63_unclassified_pathway_enrichment",
        "S64_survival_pathway_enrichment",
    }
    with pd.ExcelWriter(
        WORKBOOK,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        for sheet_name in obsolete:
            if sheet_name in writer.book.sheetnames:
                del writer.book[sheet_name]
        for sheet_name, frame in sheet_data.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\nAppended {len(sheet_data)} sheets to {WORKBOOK.name}")


if __name__ == "__main__":
    main()
