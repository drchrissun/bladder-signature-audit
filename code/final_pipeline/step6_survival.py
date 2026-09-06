from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def prepare_clinical():
    clinical = pd.read_csv("data/processed/tcga_blca_clinical.csv", index_col=0, low_memory=False)
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    vital = clinical["patient.vital_status"].str.lower()
    event = vital.eq("dead").astype(int)
    time = clinical["patient.days_to_death"].where(
        event.eq(1),
        clinical["patient.days_to_last_followup"],
    )
    stage_map = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}
    clinical["stage_num"] = clinical["patient.stage_event.pathologic_stage"].str.lower().map(stage_map)
    clinical["sex_female"] = clinical["patient.gender"].str.lower().eq("female").astype(int)
    clinical["age"] = pd.to_numeric(
        clinical["patient.age_at_initial_pathologic_diagnosis"],
        errors="coerce",
    )
    clinical["event"] = event
    clinical["time_days"] = pd.to_numeric(time, errors="coerce")
    clinical["time_months"] = clinical["time_days"] / 30.44
    keep = clinical[
        [
            "patient_id",
            "event",
            "time_months",
            "age",
            "sex_female",
            "stage_num",
        ]
    ].dropna(subset=["event", "time_months"])
    return keep.set_index("patient_id")


def sample_covariates():
    purity = pd.read_csv("data/processed/purity_and_microenvironment_estimates.csv")
    tcga = purity[purity["dataset"] == "TCGA"].copy()
    tcga["patient_id"] = tcga["sample"].map(patient_of)
    tcga = tcga[tcga["sample"].str.contains("-01A-", regex=False)]
    return (
        tcga.groupby("patient_id")[
            [
                "tumor_epithelial_fraction",
                "normal_urothelial_fraction",
                "immune_score",
                "stromal_score",
            ]
        ]
        .mean()
    )


def main():
    processed = Path("data/processed")
    scores = pd.read_csv("data/processed/signature_scores_tcga.csv.gz", compression="gzip")
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[
        scores["sample"].str.contains("-01A-", regex=False)
    ]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()
    clinical = prepare_clinical()
    covariates = sample_covariates()
    base = clinical.join(covariates, how="inner")

    rows = []
    for pmid, group in scores.groupby("pmid"):
        data = base.join(group.set_index("patient_id")["score"], how="inner").dropna(
            subset=["score", "age", "stage_num", "tumor_epithelial_fraction"]
        )
        if len(data) < 50:
            continue
        data["score"] = data["score"].astype(float)
        for model_name, columns in [
            ("unadjusted", ["score"]),
            (
                "adjusted",
                [
                    "score",
                    "age",
                    "sex_female",
                    "stage_num",
                    "tumor_epithelial_fraction",
                    "normal_urothelial_fraction",
                    "immune_score",
                    "stromal_score",
                ],
            ),
        ]:
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(
                    data[columns + ["time_months", "event"]],
                    duration_col="time_months",
                    event_col="event",
                )
                summary = cph.summary.loc["score"]
                rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "n": len(data),
                        "events": int(data["event"].sum()),
                        "coef": float(summary["coef"]),
                        "hr": float(summary["exp(coef)"]),
                        "ci_low": float(summary["exp(coef) lower 95%"]),
                        "ci_high": float(summary["exp(coef) upper 95%"]),
                        "pvalue": float(summary["p"]),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "n": len(data),
                        "events": int(data["event"].sum()),
                        "coef": np.nan,
                        "hr": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "pvalue": np.nan,
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(processed / "signature_survival_results_tcga.csv", index=False)
    print("survival result rows", len(result))
    unadj = result[result["model"] == "unadjusted"]
    adj = result[result["model"] == "adjusted"]
    print("unadjusted significant p<0.05", (unadj["pvalue"] < 0.05).sum())
    print("adjusted significant p<0.05", (adj["pvalue"] < 0.05).sum())
    print("median n", result.groupby("model")["n"].median().to_dict())


if __name__ == "__main__":
    main()
