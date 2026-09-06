from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def prepare_data():
    clinical = pd.read_csv(
        "data/processed/tcga_blca_clinical.csv",
        index_col=0,
        low_memory=False,
    )
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    clinical["event"] = clinical["patient.vital_status"].str.lower().eq("dead").astype(int)
    clinical["time_days"] = clinical["patient.days_to_death"].where(
        clinical["event"].eq(1),
        clinical["patient.days_to_last_followup"],
    )
    clinical["time_months"] = pd.to_numeric(clinical["time_days"], errors="coerce") / 30.44
    stage_map = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}
    clinical["stage_num"] = clinical["patient.stage_event.pathologic_stage"].str.lower().map(stage_map)
    clinical["sex_female"] = clinical["patient.gender"].str.lower().eq("female").astype(int)
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
    base = clinical.join(purity, how="inner").dropna(
        subset=["event", "time_months", "age", "stage_num", "tumor_epithelial_fraction"]
    )
    scores = pd.read_csv(OUT / "signature_scores_tcga.csv.gz")
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()
    return scores, base


MODELS = {
    "m1_clinical": ["score", "age", "sex_female", "stage_num"],
    "m1_plus_tumor": [
        "score",
        "age",
        "sex_female",
        "stage_num",
        "tumor_epithelial_fraction",
    ],
    "m1_plus_normal": [
        "score",
        "age",
        "sex_female",
        "stage_num",
        "normal_urothelial_fraction",
    ],
    "m1_plus_both": [
        "score",
        "age",
        "sex_female",
        "stage_num",
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
    ],
}


def main():
    scores, base = prepare_data()
    rows = []
    for pmid, group in scores.groupby("pmid"):
        data = base.join(group.set_index("patient_id")["score"], how="inner")
        for model_name, columns in MODELS.items():
            model_data = data.dropna(
                subset=columns + ["time_months", "event"]
            )[columns + ["time_months", "event"]]
            try:
                cph = CoxPHFitter(penalizer=0.0)
                cph.fit(
                    model_data,
                    duration_col="time_months",
                    event_col="event",
                )
                summary = cph.summary.loc["score"]
                rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "coef": float(summary["coef"]),
                        "hr": float(summary["exp(coef)"]),
                        "pvalue": float(summary["p"]),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "coef": np.nan,
                        "hr": np.nan,
                        "pvalue": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result["qvalue_global"] = multipletests(result["pvalue"], method="fdr_bh")[1]
    result.to_csv(OUT / "decomposition_models.csv", index=False)

    pivot = result.pivot(index="pmid", columns="model", values="qvalue_global")
    counts = pivot.apply(lambda s: (s < 0.05).sum())
    print(counts.to_string())
    gained = pivot[
        (pivot["m1_clinical"] >= 0.05)
        & (pivot["m1_plus_both"] < 0.05)
    ]
    print("M1 -> both gained", len(gained), gained.index.tolist())

    coef = result.pivot(index="pmid", columns="model", values="coef")
    corr = pd.read_csv(OUT / "microenvironment_correlations_tcga.csv")
    tumor_corr = corr[
        corr["covariate"] == "tumor_epithelial_fraction"
    ].set_index("pmid")["spearman_rho"]
    normal_corr = corr[
        corr["covariate"] == "normal_urothelial_fraction"
    ].set_index("pmid")["spearman_rho"]
    detail = gained.copy()
    detail = detail.join(coef, rsuffix="_coef")
    detail["tumor_rho"] = tumor_corr.reindex(detail.index)
    detail["normal_rho"] = normal_corr.reindex(detail.index)
    detail = detail[
        [
            "tumor_rho",
            "normal_rho",
            "m1_clinical_coef",
            "m1_plus_tumor_coef",
            "m1_plus_normal_coef",
            "m1_plus_both_coef",
        ]
    ]
    detail.to_csv(OUT / "gained_decomposition_detail.csv")
    print(detail.to_string())


if __name__ == "__main__":
    main()
