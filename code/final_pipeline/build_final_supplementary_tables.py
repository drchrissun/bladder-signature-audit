# -*- coding: utf-8 -*-
"""Build a single S1-S72 workbook aligned with the final 203/200/201 analysis set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
P01 = ROOT / "data" / "processed" / "p01_full"
OLD = ROOT / "data" / "processed"
INV = ROOT / "signature_inventory"
REV = ROOT / "data" / "processed"
OLD_WORKBOOK = ROOT / "supplementary" / "archive" / "review4_supplementary_tables.xlsx"
P01_WORKBOOK = ROOT / "supplementary" / "archive" / "review5_p01_full_supplementary_tables.xlsx"
OUT = ROOT / "supplementary" / "Supplementary_Tables_S1_S72_UroOnc_revised.xlsx"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_old_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(OLD_WORKBOOK, sheet_name=name)


def read_p01_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(P01_WORKBOOK, sheet_name=name)


def nested_counts() -> pd.DataFrame:
    nested = read_csv(P01 / "nested_cox_standard.csv")
    return (
        nested[nested.qvalue_global < 0.05]
        .groupby("model")
        .pmid.nunique()
        .reindex(
            ["m0_score", "m1_clinical", "m2_purity", "m3_microenvironment"]
        )
        .reset_index()
        .rename(columns={"pmid": "q<0.05"})
    )


def screening_counts() -> pd.DataFrame:
    screening = read_p01_sheet("S46_consensus_screening")
    include = int((screening.reviewer1_decision == "include").sum())
    exclude = int((screening.reviewer1_decision == "exclude").sum())
    return pd.DataFrame(
        [
            {"stage": "PubMed records identified", "records": 324, "note": "23 August 2026"},
            {"stage": "Independent dual screening", "records": 324, "note": "213 include / 111 exclude"},
            {"stage": "Consensus screening-eligible", "records": include, "note": "dual-review adjudicated"},
            {"stage": "Consensus excluded", "records": exclude, "note": "adjudicated exclusions"},
            {"stage": "Complete extractable gene lists", "records": 203, "note": "10 no extractable list"},
            {"stage": "TCGA-scoreable gene lists", "records": 200, "note": ">=2 mapped genes"},
            {"stage": "GSE13507-scoreable gene lists", "records": 201, "note": ">=2 mapped genes"},
        ]
    )


def vif_ph_table() -> pd.DataFrame:
    vif = read_csv(P01 / "p15_covariate_vif_condition.csv")
    vif["type"] = np.where(
        vif.covariate.eq("condition_number_intercept_plus_7"),
        "condition",
        "vif",
    )
    vif.loc[
        vif.covariate.eq("condition_number_intercept_plus_7"),
        "covariate",
    ] = "condition_number"
    ph = read_csv(P01 / "ph_complete.csv")
    ph = ph.rename(columns={"min_ph_pvalue": "value", "violated_covariates": "detail"})
    ph["type"] = "proportional_hazards"
    vif = vif.rename(columns={"vif": "value"})
    return pd.concat(
        [vif[["covariate", "value", "type"]], ph[["pmid", "value", "type", "detail"]]],
        ignore_index=True,
    )


def pathway_table() -> pd.DataFrame:
    class_path = read_csv(P01 / "p13_class_pathway_enrichment.csv")
    class_path["group_family"] = "classification"
    survival_path = read_csv(P01 / "p13_survival_pathway_enrichment.csv")
    survival_path["group_family"] = "survival"
    return pd.concat([class_path, survival_path], ignore_index=True)


def signature_characteristics() -> pd.DataFrame:
    extraction = read_csv(P01 / "gene_extraction_raw_full.csv").astype({"pmid": int})
    candidates = read_csv(INV / "primary_candidates.csv").astype({"pmid": int})
    candidates = candidates[
        ["pmid", "doi", "year", "journal", "title", "abstract", "pubtypes", "authors"]
    ]
    old_char = read_csv(OLD / "revision_signature_characteristics.csv").astype(
        {"pmid": int}
    )
    old_char = old_char[
        [
            "pmid",
            "endpoint_overall_survival",
            "endpoint_recurrence",
            "endpoint_progression",
            "endpoint_immunotherapy_response",
            "platform_rna_seq",
            "platform_microarray",
            "platform_tcga",
            "platform_geo",
            "sample_size_mentions",
            "immune_related_by_title_or_abstract",
        ]
    ]
    frame = extraction.merge(candidates, on="pmid", how="left").merge(
        old_char,
        on="pmid",
        how="left",
    )
    return frame


def included_studies() -> pd.DataFrame:
    extraction = read_csv(P01 / "gene_extraction_raw_full.csv").astype({"pmid": int})
    candidates = read_csv(INV / "primary_candidates.csv").astype({"pmid": int})
    classification = read_csv(P01 / "classification_v6_standard.csv").astype({"pmid": int})
    nested = read_csv(P01 / "nested_cox_standard.csv")
    m0_q = nested[nested.model == "m0_score"].set_index("pmid")["qvalue_global"]
    m3_q = nested[nested.model == "m3_microenvironment"].set_index("pmid")[
        "qvalue_global"
    ]
    frame = extraction.merge(
        candidates[["pmid", "doi", "year", "journal", "title", "authors"]],
        on="pmid",
        how="left",
    )
    frame = frame.merge(
        classification[
            ["pmid", "classification_v6_standard", "max_micro_abs_rho"]
        ],
        on="pmid",
        how="left",
    )
    frame["m0_standard_global_q"] = frame.pmid.map(m0_q)
    frame["m3_standard_global_q"] = frame.pmid.map(m3_q)
    return frame


def gained_suppression() -> pd.DataFrame:
    nested = read_csv(P01 / "nested_cox_standard.csv")
    pivot = nested.pivot_table(
        index="pmid",
        columns="model",
        values=["coef", "hr", "pvalue", "qvalue_global"],
    )
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reset_index()
    classification = read_csv(P01 / "classification_v6_standard.csv").astype(
        {"pmid": int}
    )
    gained = pivot[
        (pivot["m0_score_qvalue_global"] >= 0.05)
        & (pivot["m3_microenvironment_qvalue_global"] < 0.05)
    ]
    return gained.merge(classification, on="pmid", how="left")


def list_length() -> pd.DataFrame:
    coverage = read_csv(P01 / "signature_coverage.csv")
    nested = read_csv(P01 / "nested_cox_standard.csv")
    m3_q = nested[nested.model == "m3_microenvironment"].set_index("pmid")[
        "qvalue_global"
    ]
    scoreable = coverage[coverage.tcga_mapped_count >= 2].copy()
    scoreable["group"] = pd.cut(
        scoreable.tcga_mapped_count,
        [-1, 9, 50, np.inf],
        labels=["<10", "10-50", ">50"],
    )
    rows = []
    for group in ["<10", "10-50", ">50"]:
        ids = scoreable.loc[scoreable.group == group, "pmid"]
        rows.append(
            {
                "size_group": group,
                "n_lists": len(ids),
                "m0_q_lt_005": int((nested[(nested.model == "m0_score") & nested.pmid.isin(ids)].qvalue_global < 0.05).sum()),
                "m3_q_lt_005": int((m3_q.reindex(ids) < 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)


def normal_mucosa_fdr() -> pd.DataFrame:
    frame = read_csv(P01 / "normal_mucosa_signature_summary.csv")
    significant = frame[frame.qvalue < 0.05]
    summary = pd.DataFrame(
        [
            {
                "statistic": "FDR-significant gene-list comparisons",
                "value": len(significant),
            },
            {
                "statistic": "Higher in normal-looking mucosa",
                "value": int((significant.normal_looking_minus_primary > 0).sum()),
            },
        ]
    )
    return pd.concat(
        [summary, frame],
        ignore_index=True,
        sort=False,
    )


def external_class_cross() -> pd.DataFrame:
    nested = read_csv(P01 / "nested_cox_standard.csv")
    m3 = nested[nested.model == "m3_microenvironment"].set_index("pmid")
    ids = m3[m3.qvalue_global < 0.05].index
    class_frame = read_csv(P01 / "classification_v6_standard.csv").set_index("pmid")
    ext135 = read_csv(P01 / "gse13507_survival_validation_standard.csv")
    ext480 = read_csv(P01 / "gse48075_survival_validation.csv")
    ext135 = ext135[ext135.model == "adjusted"].set_index("pmid").loc[ids]
    ext480 = ext480[ext480.model == "adjusted"].set_index("pmid").loc[ids]
    concord135 = np.sign(ext135.coef) == np.sign(m3.loc[ids, "coef"])
    concord480 = np.sign(ext480.coef) == np.sign(m3.loc[ids, "coef"])
    rows = []
    for cls in [
        "composition_adjustment_persistent_standard",
        "microenvironment_associated_attenuation_standard",
        "microenvironment_correlated_standard",
        "unclassified_standard",
    ]:
        class_frame_for_ids = class_frame.loc[ids]
        cls_ids = class_frame_for_ids.index[
            class_frame_for_ids.classification_v6_standard == cls
        ]
        rows.append(
            {
                "classification": cls,
                "n_lists": len(cls_ids),
                "gse13507_nominal": int((ext135.loc[cls_ids].pvalue < 0.05).sum()),
                "gse13507_concordant": int(concord135.reindex(cls_ids).sum()),
                "gse48075_nominal": int((ext480.loc[cls_ids].pvalue < 0.05).sum()),
                "gse48075_concordant": int(concord480.reindex(cls_ids).sum()),
            }
        )
    return pd.DataFrame(rows)


def no_list_studies() -> pd.DataFrame:
    frame = read_csv(P01 / "gene_extraction_raw_full.csv")
    return frame[frame.extraction_status.eq("excluded")]


def inventory_year() -> pd.DataFrame:
    frame = read_csv(P01 / "gene_extraction_raw_full.csv")
    return (
        frame.year.astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "year", "year": "n_lists"})
    )


def main() -> None:
    old_sheets = {
        "S10_deconv_agreement": "S10_deconv_agreement",
        "S15_raw_normal_marker": "S15_raw_normal_marker",
        "S17_raw_score_sensitivity": "S17_raw_score_sensitivity",
        "S18_leave_one_out": "S18_leave_one_out",
        "S19_time_interaction": "S19_time_interaction",
        "S21_index_weights": "S21_index_weights",
        "S24_normalization": "S24_normalization",
        "S25_global_expression": "S25_global_expression",
        "S26_disease_proxy": "S26_disease_proxy",
        "S28_tcga_normal_markers": "S28_tcga_normal_markers",
        "S30_external_mechanism": "S30_external_mechanism",
        "S31_power": "S31_power",
        "S34_persistent_power": "S34_persistent_power",
        "S35_tcga_exclusions": "S35_tcga_exclusions",
        "S36_screening_examples": "S36_screening_examples",
        "S37_scRNA_composition": "S37_scRNA_composition",
        "S38_marker_panels": "S38_marker_panels",
        "S40_inventory_endpoint": "S40_inventory_endpoint",
        "S45_platform_info": "S45_platform_info",
    }

    tables: dict[str, pd.DataFrame] = {
        "S1_signature_inventory": read_csv(P01 / "gene_extraction_raw_full.csv"),
        "S2_prisma_counts": screening_counts(),
        "S3_standard_nested_cox": read_csv(P01 / "nested_cox_standard.csv"),
        "S4_micro_correlations": read_csv(P01 / "microenvironment_correlations_tcga.csv"),
        "S5_normal_mucosa_signatures": read_csv(P01 / "normal_mucosa_signature_summary.csv"),
        "S6_normal_mucosa_markers": read_csv(
            REV / "revision_gse13507_marker_group_comparison.csv"
        ),
        "S7_nested_counts": nested_counts(),
        "S8_vif_condition_ph": vif_ph_table(),
        "S9_bootstrap_1000": read_csv(P01 / "revision_bootstrap_1000_summary.csv"),
        "S11_formal_deconv": read_csv(
            P01 / "final_signature_sensitivity_formal_deconvolution.csv"
        ),
        "S12_classification_standard": read_csv(P01 / "classification_v6_standard.csv"),
        "S13_pathway_standard": pathway_table(),
        "S14_signature_characteristics": signature_characteristics(),
        "S16_penalty": read_csv(P01 / "penalty_sensitivity.csv"),
        "S20_external_standard": read_csv(P01 / "gse13507_survival_validation_standard.csv"),
        "S22_suppression": gained_suppression(),
        "S23_mediation": read_csv(REV / "p01_regmedint_survival_results.csv"),
        "S27_classification_bootstrap": read_csv(
            P01 / "final_classification_bootstrap_stability.csv"
        ),
        "S29_included_studies": included_studies(),
        "S32_m1_m2_suppression": read_csv(P01 / "gained_decomposition_detail.csv"),
        "S33_ocr_standard_sensitivity": pd.concat(
            [
                read_csv(P01 / "p16_ocr_q_counts.csv").assign(analysis="q_counts"),
                read_csv(P01 / "p16_ocr_classification_counts.csv").assign(
                    analysis="classification_counts"
                ),
                read_csv(P01 / "p16_ocr_external_summary.csv").assign(
                    analysis="external_summary"
                ),
            ],
            ignore_index=True,
        ),
        "S39_inventory_year": inventory_year(),
        "S41_external_class_cross": external_class_cross(),
        "S42_list_length": list_length(),
        "S43_normal_mucosa_fdr": normal_mucosa_fdr(),
        "S44_no_list_studies": no_list_studies(),
        "S72_gained_adjustment": read_csv(P01 / "p25_gained_reporting_table.csv"),
    }

    for final_name, old_name in old_sheets.items():
        frame = read_old_sheet(old_name)
        if final_name in {
            "S15_raw_normal_marker",
            "S17_raw_score_sensitivity",
            "S18_leave_one_out",
            "S19_time_interaction",
            "S21_index_weights",
            "S30_external_mechanism",
            "S34_persistent_power",
            "S40_inventory_endpoint",
        }:
            frame["scope"] = (
                "legacy exploratory analysis; not used for final 200-list inference"
            )
        tables[final_name] = frame

    p01_names = {
        "S46_consensus_screening",
        "S47_full_gene_extraction",
        "S48_full_bootstrap_1000",
        "S49_full_nested_cox",
        "S50_full_classification",
        "S51_full_external_validation",
        "S52_gse48075_validation",
        "S53_full_normal_mucosa",
        "S54_full_mediation",
        "S55_full_consensus_summary",
        "S56_full_consensus_tests",
        "S57_gained_decomposition_detail",
        "S58_rho_classes",
        "S59_rho_counts",
        "S60_rho_transitions",
        "S61_unclass_features",
        "S62_unclass_feature_summary",
        "S63_unclass_pathways",
        "S64_survival_pathways",
        "S65_covariate_scales",
        "S66_covariate_corr",
        "S67_covariate_vif",
        "S68_ridge_sensitivity",
        "S69_ocr_q_counts",
        "S70_ocr_class_counts",
        "S71_ocr_external",
    }
    for name in p01_names:
        tables[name] = read_p01_sheet(name)

    missing = [f"S{n}" for n in range(1, 73) if not any(k.startswith(f"S{n}_") for k in tables)]
    if missing:
        raise ValueError("missing sheets: " + ", ".join(missing))

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for name in sorted(tables, key=lambda x: int(x[1 : x.index("_", 1)])):
            tables[name].to_excel(writer, sheet_name=name[:31], index=False)

    print("saved", OUT)
    for name in sorted(tables, key=lambda x: int(x[1 : x.index("_", 1)])):
        print(name, tables[name].shape)


if __name__ == "__main__":
    main()
