from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"
WORKBOOK = Path(
    "supplementary/Supplementary_Tables_S1_S72_UroOnc_revised.xlsx"
)
COLUMNS = [
    "age",
    "sex_female",
    "stage_num",
    "tumor_epithelial_fraction",
    "normal_urothelial_fraction",
    "immune_score",
    "stromal_score",
]


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def build_design():
    clinical = pd.read_csv(
        PROCESSED / "tcga_blca_clinical.csv",
        index_col=0,
        low_memory=False,
    )
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    clinical["event"] = (
        clinical["patient.vital_status"].str.lower().eq("dead").astype(int)
    )
    clinical["time_days"] = clinical["patient.days_to_death"].where(
        clinical["event"].eq(1),
        clinical["patient.days_to_last_followup"],
    )
    clinical["time_months"] = (
        pd.to_numeric(clinical["time_days"], errors="coerce") / 30.44
    )
    stage_map = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}
    clinical["stage_num"] = (
        clinical["patient.stage_event.pathologic_stage"]
        .str.lower()
        .map(stage_map)
    )
    clinical["sex_female"] = (
        clinical["patient.gender"].str.lower().eq("female").astype(int)
    )
    clinical["age"] = pd.to_numeric(
        clinical["patient.age_at_initial_pathologic_diagnosis"],
        errors="coerce",
    )
    clinical = clinical.set_index("patient_id")

    purity = pd.read_csv(PROCESSED / "purity_and_microenvironment_estimates.csv")
    purity = purity[purity["dataset"] == "TCGA"].copy()
    purity["patient_id"] = purity["sample"].map(patient_of)
    purity = purity[purity["sample"].str.contains("-01A-", regex=False)]
    purity = purity.groupby("patient_id")[
        [
            "tumor_epithelial_fraction",
            "normal_urothelial_fraction",
            "immune_score",
            "stromal_score",
        ]
    ].mean()
    design = clinical.join(purity, how="inner").dropna(
        subset=[
            "event",
            "time_months",
            "age",
            "stage_num",
            "tumor_epithelial_fraction",
            "normal_urothelial_fraction",
            "immune_score",
            "stromal_score",
        ]
    )
    return design[COLUMNS]


def main():
    design = build_design()
    scale_rows = []
    for column in COLUMNS:
        scale_rows.append(
            {
                "covariate": column,
                "n": int(design[column].notna().sum()),
                "mean": float(design[column].mean()),
                "sd": float(design[column].std(ddof=1)),
                "min": float(design[column].min()),
                "q25": float(design[column].quantile(0.25)),
                "median": float(design[column].median()),
                "q75": float(design[column].quantile(0.75)),
                "max": float(design[column].max()),
            }
        )
    scale = pd.DataFrame(scale_rows)
    scale.to_csv(OUT / "p15_covariate_scale_summary.csv", index=False)

    corr_rows = []
    for i, left in enumerate(COLUMNS):
        for right in COLUMNS[i + 1 :]:
            frame = design[[left, right]].dropna()
            pearson_r, pearson_p = pearsonr(frame[left], frame[right])
            spearman_r, spearman_p = spearmanr(frame[left], frame[right])
            corr_rows.append(
                {
                    "covariate_1": left,
                    "covariate_2": right,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(OUT / "p15_covariate_correlations.csv", index=False)

    centered = design[COLUMNS].astype(float)
    centered = (centered - centered.mean()) / centered.std(ddof=1)
    condition_number = float(
        np.linalg.cond(
            np.column_stack([np.ones(len(centered)), centered.to_numpy()])
        )
    )
    vif_rows = []
    constant_design = add_constant(centered)
    for idx, column in enumerate(COLUMNS, start=1):
        vif_rows.append(
            {
                "covariate": column,
                "vif": float(
                    variance_inflation_factor(constant_design.to_numpy(), idx)
                ),
            }
        )
    vif = pd.DataFrame(vif_rows)
    vif.loc[len(vif)] = {
        "covariate": "condition_number_intercept_plus_7",
        "vif": condition_number,
    }
    vif.to_csv(OUT / "p15_covariate_vif_condition.csv", index=False)

    penalty = pd.read_csv(OUT / "penalty_sensitivity.csv")
    penalty = penalty.dropna(subset=["pvalue_0", "pvalue_001"])
    rank_rho, rank_p = spearmanr(penalty["pvalue_0"], penalty["pvalue_001"])
    ridge = pd.DataFrame(
        [
            {
                "comparison": "penalty_0_vs_0.01",
                "tested_lists": len(penalty),
                "nominal_p05_penalty_0": int((penalty["pvalue_0"] < 0.05).sum()),
                "nominal_p05_penalty_0.01": int(
                    (penalty["pvalue_001"] < 0.05).sum()
                ),
                "pvalue_spearman_rho": rank_rho,
                "pvalue_spearman_p": rank_p,
            }
        ]
    )
    ridge.to_csv(OUT / "p15_ridge_summary.csv", index=False)

    print("\nScale summary")
    print(scale.round(4).to_string(index=False))
    print("\nLargest absolute covariate correlations")
    print(
        correlations.assign(
            max_abs=lambda frame: frame[
                ["pearson_r", "spearman_rho"]
            ].abs().max(axis=1)
        )
        .sort_values("max_abs", ascending=False)
        .head(8)
        .round(4)
        .to_string(index=False)
    )
    print("\nVIF and condition number")
    print(vif.round(4).to_string(index=False))
    print("\nRidge sensitivity")
    print(ridge.round(4).to_string(index=False))

    sheets = {
        "S65_covariate_scales": scale,
        "S66_covariate_corr": correlations,
        "S67_covariate_vif": vif,
        "S68_ridge_sensitivity": ridge,
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
