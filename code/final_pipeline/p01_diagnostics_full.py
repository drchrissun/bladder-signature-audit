from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import spearmanr


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


COLUMNS = [
    "age",
    "sex_female",
    "stage_num",
    "tumor_epithelial_fraction",
    "normal_urothelial_fraction",
    "immune_score",
    "stromal_score",
]


def main():
    scores, base = prepare_data()
    nested = pd.read_csv(OUT / "nested_cox_standard.csv")
    m3_q = nested[
        nested["model"] == "m3_microenvironment"
    ].set_index("pmid")["qvalue_global"]
    ids = m3_q[m3_q < 0.05].index.tolist()

    penalty_rows = []
    for pmid, group in scores.groupby("pmid"):
        data = base.join(group.set_index("patient_id")["score"], how="inner").dropna(
            subset=["score"] + COLUMNS + ["time_months", "event"]
        )
        model_data = data[["score"] + COLUMNS + ["time_months", "event"]]
        for penalizer in [0.0, 0.01]:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(model_data, duration_col="time_months", event_col="event")
            summary = cph.summary.loc["score"]
            penalty_rows.append(
                {
                    "pmid": pmid,
                    "penalizer": penalizer,
                    "pvalue": float(summary["p"]),
                }
            )
    penalty = pd.DataFrame(penalty_rows).pivot(
        index="pmid", columns="penalizer", values="pvalue"
    )
    penalty.columns = ["pvalue_0", "pvalue_001"]
    penalty.to_csv(OUT / "penalty_sensitivity.csv")
    print(
        "penalty significant",
        int((penalty["pvalue_0"] < 0.05).sum()),
        int((penalty["pvalue_001"] < 0.05).sum()),
        "spearman",
        spearmanr(penalty["pvalue_0"], penalty["pvalue_001"]).statistic.round(4),
    )

    ph_rows = []
    for pmid in ids:
        data = base.join(
            scores[scores["pmid"] == pmid].set_index("patient_id")["score"],
            how="inner",
        ).dropna(subset=["score"] + COLUMNS + ["time_months", "event"])
        model_data = data[["score"] + COLUMNS + ["time_months", "event"]]
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(model_data, duration_col="time_months", event_col="event")
        residuals = cph.compute_residuals(model_data, kind="scaled_schoenfeld")
        event_times = data.loc[residuals.index, "time_months"]
        pvalues = {
            column: spearmanr(residuals[column], event_times).pvalue
            for column in residuals.columns
        }
        violated = [column for column, pvalue in pvalues.items() if pvalue < 0.05]
        if violated:
            ph_rows.append(
                {
                    "pmid": pmid,
                    "violated_covariates": ";".join(violated),
                    "min_ph_pvalue": min(pvalues.values()),
                }
            )
    ph = pd.DataFrame(ph_rows)
    ph.to_csv(OUT / "ph_complete.csv", index=False)
    print("PH warnings", len(ph), "of", len(ids))


if __name__ == "__main__":
    main()
