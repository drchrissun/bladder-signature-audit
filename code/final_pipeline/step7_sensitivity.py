from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def prepare_base():
    scores = pd.read_csv("data/processed/signature_scores_tcga.csv.gz", compression="gzip")
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()

    clinical = pd.read_csv("data/processed/tcga_blca_clinical.csv", index_col=0, low_memory=False)
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    vital = clinical["patient.vital_status"].str.lower()
    clinical["event"] = vital.eq("dead").astype(int)
    clinical["time_days"] = clinical["patient.days_to_death"].where(
        clinical["event"].eq(1),
        clinical["patient.days_to_last_followup"],
    )
    clinical["time_months"] = pd.to_numeric(clinical["time_days"], errors="coerce") / 30.44
    stage_map = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}
    clinical["stage_num"] = clinical["patient.stage_event.pathologic_stage"].str.lower().map(stage_map)
    clinical["sex_female"] = clinical["patient.gender"].str.lower().eq("female").astype(int)
    clinical["age"] = pd.to_numeric(clinical["patient.age_at_initial_pathologic_diagnosis"], errors="coerce")

    purity = pd.read_csv("data/processed/purity_and_microenvironment_estimates.csv")
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

    raw_scores = pd.read_csv("data/processed/deconvolution_scores_tcga.csv", index_col=0)
    raw_scores["patient_id"] = raw_scores.index.map(patient_of)
    raw_scores = raw_scores[raw_scores.index.str.contains("-01A-", regex=False)]
    raw_scores = raw_scores.groupby("patient_id").mean(numeric_only=True)

    base = clinical.set_index("patient_id").join(purity, how="inner").join(raw_scores, how="inner")
    base = base.dropna(subset=["event", "time_months", "age", "stage_num"])
    return scores, base


def main():
    processed = Path("data/processed")
    scores, base = prepare_base()
    survival = pd.read_csv("data/processed/signature_survival_results_tcga.csv")
    significant = survival[
        (survival["model"] == "adjusted") & (survival["pvalue"] < 0.05)
    ]["pmid"].tolist()

    adjusted_cols = [
        "age",
        "sex_female",
        "stage_num",
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
        "immune_score",
        "stromal_score",
    ]
    bootstrap_rows = []
    for pmid in significant:
        data = base.join(scores[scores["pmid"] == pmid].set_index("patient_id")["score"], how="inner")
        data = data.dropna(subset=["score"] + adjusted_cols)
        if len(data) < 50:
            continue
        for iteration in range(100):
            sample = data.sample(frac=1.0, replace=True, random_state=iteration)
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(sample[["score"] + adjusted_cols + ["time_months", "event"]], duration_col="time_months", event_col="event")
                summary = cph.summary.loc["score"]
                bootstrap_rows.append(
                    {
                        "pmid": pmid,
                        "iteration": iteration,
                        "hr": float(summary["exp(coef)"]),
                        "pvalue": float(summary["p"]),
                    }
                )
            except Exception:
                pass
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(processed / "signature_sensitivity_bootstrap.csv", index=False)
    bootstrap_summary = (
        bootstrap.groupby("pmid")
        .agg(median_hr=("hr", "median"), significant_rate=("pvalue", lambda s: (s < 0.05).mean()))
        .reset_index()
    )
    bootstrap_summary.to_csv(processed / "signature_sensitivity_bootstrap_summary.csv", index=False)

    raw_cell_cols = [
        "Normal_urothelial",
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
    alt_rows = []
    for pmid in significant:
        data = base.join(scores[scores["pmid"] == pmid].set_index("patient_id")["score"], how="inner")
        data = data.dropna(subset=["score", "age", "sex_female", "stage_num"] + raw_cell_cols)
        if len(data) < 50:
            continue
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(data[["score", "age", "sex_female", "stage_num"] + raw_cell_cols + ["time_months", "event"]], duration_col="time_months", event_col="event")
            summary = cph.summary.loc["score"]
            alt_rows.append(
                {
                    "pmid": pmid,
                    "hr": float(summary["exp(coef)"]),
                    "ci_low": float(summary["exp(coef) lower 95%"]),
                    "ci_high": float(summary["exp(coef) upper 95%"]),
                    "pvalue": float(summary["p"]),
                }
            )
        except Exception:
            pass
    alt = pd.DataFrame(alt_rows)
    alt.to_csv(processed / "signature_sensitivity_raw_scores_covariates.csv", index=False)

    print("significant signatures", len(significant))
    print("bootstrap rows", len(bootstrap))
    print("bootstrap median significant rate", bootstrap_summary["significant_rate"].median() if not bootstrap_summary.empty else np.nan)
    print("alternative covariates significant", (alt["pvalue"] < 0.05).sum(), "/", len(alt))


if __name__ == "__main__":
    main()
