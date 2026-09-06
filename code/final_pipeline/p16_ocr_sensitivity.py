from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"
WORKBOOK = Path(
    "supplementary/Supplementary_Tables_S1_S72_UroOnc_revised.xlsx"
)
OCR_PMID = "42493447"
CLASS_ORDER = [
    "composition_adjustment_persistent",
    "microenvironment_associated_attenuation",
    "microenvironment_correlated",
    "unclassified",
]


def classify_at(frame, threshold=0.4):
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


def recompute_q(nested):
    nested = nested.copy()
    nested["qvalue_global"] = multipletests(
        nested["pvalue"], method="fdr_bh"
    )[1]
    return nested


def build_reclassified(nested, base_flags, exclude_ocr):
    nested = recompute_q(nested)
    if exclude_ocr:
        nested = nested[nested["pmid"].ne(OCR_PMID)]

    m0 = nested[nested.model.eq("m0_score")].set_index("pmid")["qvalue_global"]
    m3 = nested[
        nested.model.eq("m3_microenvironment")
    ].set_index("pmid")["qvalue_global"]
    frame = base_flags.copy()
    frame["m0_global_q"] = frame["pmid"].map(m0)
    frame["m3_global_q"] = frame["pmid"].map(m3)
    frame = frame.dropna(subset=["m0_global_q", "m3_global_q"])
    frame["unadjusted_fdr_significant"] = frame["m0_global_q"] < 0.05
    frame["adjusted_fdr_significant"] = frame["m3_global_q"] < 0.05
    frame["lost_fdr_significance"] = (
        frame["unadjusted_fdr_significant"]
        & ~frame["adjusted_fdr_significant"]
    )
    frame["classification"] = classify_at(frame)
    return frame


def external_summary(significant_ids, gse13507, gse48075, tcga_coefs):
    rows = []
    for dataset, table in (
        ("GSE13507", gse13507),
        ("GSE48075", gse48075),
    ):
        adjusted = table[
            (table.model.eq("adjusted")) & (table.pmid.isin(significant_ids))
        ].set_index("pmid")
        merged = pd.DataFrame(
            {"tcga_coef": tcga_coefs.reindex(significant_ids)}
        ).join(adjusted[["coef", "pvalue"]], how="inner")
        nominal = int((merged.pvalue < 0.05).sum())
        concordant = int(
            (np.sign(merged.tcga_coef) == np.sign(merged.coef)).sum()
        )
        rho, pvalue = spearmanr(merged.tcga_coef, merged.coef)
        rows.append(
            {
                "dataset": dataset,
                "tested_lists": len(merged),
                "nominal_p05": nominal,
                "direction_concordant": concordant,
                "coefficient_spearman_rho": rho,
                "coefficient_spearman_p": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    gse13507_ids = set(
        gse13507[
            gse13507.model.eq("adjusted")
            & gse13507.pmid.isin(significant_ids)
            & gse13507.pvalue.lt(0.05)
        ].pmid
    )
    gse48075_ids = set(
        gse48075[
            gse48075.model.eq("adjusted")
            & gse48075.pmid.isin(significant_ids)
            & gse48075.pvalue.lt(0.05)
        ].pmid
    )
    result.loc[len(result)] = {
        "dataset": "both_nominal",
        "tested_lists": len(gse13507_ids & gse48075_ids),
        "nominal_p05": len(gse13507_ids & gse48075_ids),
        "direction_concordant": np.nan,
        "coefficient_spearman_rho": np.nan,
        "coefficient_spearman_p": np.nan,
    }
    return result


def main():
    nested = pd.read_csv(
        OUT / "nested_cox_standard.csv", dtype={"pmid": str}
    )
    base_flags = pd.read_csv(
        OUT / "classification_v6_standard.csv", dtype={"pmid": str}
    )[
        [
            "pmid",
            "max_micro_abs_rho",
            "normal_mucosa_elevation_significant",
        ]
    ]
    gse13507 = pd.read_csv(
        OUT / "gse13507_survival_validation_standard.csv",
        dtype={"pmid": str},
    )
    gse48075 = pd.read_csv(
        OUT / "gse48075_survival_validation.csv", dtype={"pmid": str}
    )

    full = build_reclassified(nested, base_flags, exclude_ocr=False)
    excluded = build_reclassified(nested, base_flags, exclude_ocr=True)

    q_rows = []
    for label, source_nested in (
        ("full_200", recompute_q(nested)),
        ("exclude_42493447_199", recompute_q(nested[nested.pmid.ne(OCR_PMID)])),
    ):
        for model in (
            "m0_score",
            "m1_clinical",
            "m2_purity",
            "m3_microenvironment",
        ):
            count = int(
                source_nested[
                    source_nested.model.eq(model)
                    & source_nested.qvalue_global.lt(0.05)
                ]
                .pmid.nunique()
            )
            q_rows.append(
                {"analysis_set": label, "model": model, "fdr_significant": count}
            )
    q_counts = pd.DataFrame(q_rows)
    q_counts.to_csv(OUT / "p16_ocr_q_counts.csv", index=False)

    class_rows = []
    for label, frame in (("full_200", full), ("exclude_42493447_199", excluded)):
        counts = frame.classification.value_counts()
        for class_name in CLASS_ORDER:
            class_rows.append(
                {
                    "analysis_set": label,
                    "classification": class_name,
                    "count": int(counts.get(class_name, 0)),
                }
            )
        class_rows.append(
            {
                "analysis_set": label,
                "classification": "lost_fdr",
                "count": int(frame.lost_fdr_significance.sum()),
            }
        )
        class_rows.append(
            {
                "analysis_set": label,
                "classification": "gained_fdr",
                "count": int(
                    (
                        ~frame.unadjusted_fdr_significant
                        & frame.adjusted_fdr_significant
                    ).sum()
                ),
            }
        )
    class_counts = pd.DataFrame(class_rows)
    class_counts.to_csv(OUT / "p16_ocr_classification_counts.csv", index=False)

    tcga_m3 = nested[nested.model.eq("m3_microenvironment")].set_index(
        "pmid"
    )["coef"]
    external_rows = []
    for label, frame in (("full_200", full), ("exclude_42493447_199", excluded)):
        significant_ids = frame.loc[frame.adjusted_fdr_significant, "pmid"]
        summary = external_summary(
            significant_ids, gse13507, gse48075, tcga_m3
        )
        summary["analysis_set"] = label
        external_rows.append(summary)
    external = pd.concat(external_rows, ignore_index=True)
    external.to_csv(OUT / "p16_ocr_external_summary.csv", index=False)

    print("\nQ counts")
    print(q_counts.pivot(index="model", columns="analysis_set", values="fdr_significant"))
    print("\nClassification and transitions")
    print(class_counts.pivot(index="classification", columns="analysis_set", values="count"))
    print("\nExternal validation")
    print(external.round(4).to_string(index=False))

    sheets = {
        "S69_ocr_q_counts": q_counts,
        "S70_ocr_class_counts": class_counts,
        "S71_ocr_external": external,
    }
    with pd.ExcelWriter(
        WORKBOOK,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\nAppended {len(sheets)} sheets to {WORKBOOK.name}")


if __name__ == "__main__":
    main()
