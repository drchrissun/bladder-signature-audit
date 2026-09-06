from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


PROCESSED = Path("data/processed")
REVISION_DATA = Path("outputs/consensus")


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def modal(series):
    return series.sort_index().value_counts().idxmax()


def main():
    calls = pd.read_csv(REVISION_DATA / "consensus_mibc_calls.csv")
    calls["patient_id"] = calls["sample"].map(patient_of)
    calls = calls.drop_duplicates("patient_id")

    scores = pd.read_csv(PROCESSED / "p01_full/signature_scores_tcga.csv.gz")
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()

    clinical = pd.read_csv(
        PROCESSED / "tcga_blca_clinical.csv",
        index_col=0,
        low_memory=False,
    )
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    clinical["age"] = pd.to_numeric(
        clinical["patient.age_at_initial_pathologic_diagnosis"],
        errors="coerce",
    )
    clinical["event"] = clinical["patient.vital_status"].str.lower().eq("dead").astype(int)
    clinical["time_days"] = pd.to_numeric(
        clinical["patient.days_to_death"].where(
            clinical["event"].eq(1),
            clinical["patient.days_to_last_followup"],
        ),
        errors="coerce",
    )
    stage_map = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}
    clinical["stage_num"] = clinical["patient.stage_event.pathologic_stage"].str.lower().map(stage_map)
    clinical = clinical.set_index("patient_id")
    purity = pd.read_csv(PROCESSED / "purity_and_microenvironment_estimates.csv")
    purity = purity[purity["dataset"] == "TCGA"].copy()
    purity["patient_id"] = purity["sample"].map(patient_of)
    purity = purity[purity["sample"].str.contains("-01A-", regex=False)]
    purity = purity.groupby("patient_id")[["tumor_epithelial_fraction"]].mean()
    analysis_ids = set(
        clinical.join(purity, how="inner")
        .dropna(
            subset=[
                "age",
                "stage_num",
                "tumor_epithelial_fraction",
                "event",
                "time_days",
            ]
        )
        .index
    )
    coverage = (
        scores.groupby("patient_id")["pmid"]
        .nunique()
        .eq(scores["pmid"].nunique())
    )
    analysis_ids.intersection_update(set(coverage[coverage].index))
    calls = calls[calls["patient_id"].isin(analysis_ids)]
    scores = scores[scores["patient_id"].isin(calls["patient_id"])]

    classification = pd.read_csv(
        PROCESSED / "p01_full/classification_v6_standard.csv"
    )[["pmid", "classification_v6_standard"]]

    summary_rows = []
    for pmid, group in scores.groupby("pmid"):
        data = group.merge(
            calls[["patient_id", "consensusClass"]],
            on="patient_id",
            how="inner",
        )
        median = data["score"].median()
        data["high_risk"] = data["score"] > median
        high = data[data["high_risk"]]
        summary_rows.append(
            {
                "pmid": pmid,
                "n_patients": len(data),
                "n_high": int(data["high_risk"].sum()),
                "modal_consensus_high": modal(high["consensusClass"]),
                "modal_consensus_all": modal(data["consensusClass"]),
                "ba_sq_high_rate": float(high["consensusClass"].eq("Ba/Sq").mean()),
                "ba_sq_low_rate": float(
                    data.loc[~data["high_risk"], "consensusClass"]
                    .eq("Ba/Sq")
                    .mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).merge(classification, on="pmid")
    summary.to_csv(REVISION_DATA / "p01_consensus_signature_summary.csv", index=False)
    cross = pd.crosstab(
        summary["classification_v6_standard"],
        summary["modal_consensus_high"],
    )
    cross.to_csv(REVISION_DATA / "p01_consensus_cross_tab_classification.csv")

    tests = []
    for name, flag in [
        (
            "persistent",
            summary["classification_v6_standard"].eq(
                "composition_adjustment_persistent_standard"
            ),
        ),
        (
            "attenuation",
            summary["classification_v6_standard"].eq(
                "microenvironment_associated_attenuation_standard"
            ),
        ),
        (
            "correlated",
            summary["classification_v6_standard"].eq(
                "microenvironment_correlated_standard"
            ),
        ),
        (
            "unclassified",
            summary["classification_v6_standard"].eq(
                "unclassified_standard"
            ),
        ),
    ]:
        tab = pd.crosstab(flag, summary["modal_consensus_high"].eq("Ba/Sq"))
        odds, p = fisher_exact(tab.to_numpy(), alternative="two-sided")
        tests.append(
            {
                "classification_flag": name,
                "flag_signatures": int(flag.sum()),
                "flag_ba_sq": int((flag & summary["modal_consensus_high"].eq("Ba/Sq")).sum()),
                "other_ba_sq": int((~flag & summary["modal_consensus_high"].eq("Ba/Sq")).sum()),
                "odds_ratio": odds,
                "p_fisher": p,
            }
        )
    tests = pd.DataFrame(tests)
    tests.to_csv(REVISION_DATA / "p01_consensus_classification_tests.csv", index=False)

    print("patients", calls.patient_id.nunique(), "signatures", len(summary))
    print(cross.to_string())
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
