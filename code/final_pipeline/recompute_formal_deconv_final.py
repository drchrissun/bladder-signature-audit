# -*- coding: utf-8 -*-
"""Recompute formal-deconvolution sensitivity for the final 31 q-significant lists."""

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
P01 = PROCESSED / "p01_full"


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def prepare_clinical():
    clinical = pd.read_csv(PROCESSED / "tcga_blca_clinical.csv", index_col=0, low_memory=False)
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    vital = clinical["patient.vital_status"].str.lower()
    clinical["event"] = vital.eq("dead").astype(int)
    clinical["time_days"] = clinical["patient.days_to_death"].where(
        clinical["event"].eq(1), clinical["patient.days_to_last_followup"]
    )
    clinical["time_months"] = pd.to_numeric(clinical["time_days"], errors="coerce") / 30.44
    stage_map = {"stage i": 1, "stage ii": 2, "stage iii": 3, "stage iv": 4}
    clinical["stage_num"] = clinical["patient.stage_event.pathologic_stage"].str.lower().map(stage_map)
    clinical["sex_female"] = clinical["patient.gender"].str.lower().eq("female").astype(int)
    clinical["age"] = pd.to_numeric(clinical["patient.age_at_initial_pathologic_diagnosis"], errors="coerce")
    keep = clinical[["patient_id", "event", "time_months", "age", "sex_female", "stage_num"]].dropna(subset=["event", "time_months"])
    return keep.set_index("patient_id")


def prepare_fractions():
    music = pd.read_csv(PROCESSED / "r_music_weighted_fractions.csv", index_col=0)
    music["patient_id"] = music.index.map(patient_of)
    music = music[music.index.str.contains("-01A-", regex=False)]
    music = music.groupby("patient_id").mean(numeric_only=True)
    bp = pd.read_csv(PROCESSED / "r_bayesprism_theta.csv", index_col=0)
    bp["patient_id"] = bp.index.map(patient_of)
    bp = bp[bp.index.str.contains("-01A-", regex=False)]
    bp = bp.groupby("patient_id").mean(numeric_only=True)
    music_summary = pd.DataFrame(
        {
            "music_normal_urothelial_fraction": music["Normal_urothelial"],
            "music_tumor_epithelial_fraction": music["Tumor_epithelial"],
            "music_immune_fraction": music[["Myeloid", "T_cell", "NK_cell", "B_plasma"]].sum(axis=1),
            "music_stromal_fraction": music[["Fibroblast", "Myofibroblast", "Endothelial"]].sum(axis=1),
        }
    )
    bp_summary = pd.DataFrame(
        {
            "bp_normal_urothelial_fraction": bp["Normal_urothelial"],
            "bp_tumor_epithelial_fraction": bp["Tumor_epithelial"],
            "bp_immune_fraction": bp[["Myeloid", "T_cell", "NK_cell", "B_plasma", "Mast_cell"]].sum(axis=1),
            "bp_stromal_fraction": bp[["Fibroblast", "Myofibroblast", "Endothelial"]].sum(axis=1),
        }
    )
    return music_summary, bp_summary


def fit_model(data, columns):
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(data[columns + ["time_months", "event"]], duration_col="time_months", event_col="event")
    summary = cph.summary.loc["score"]
    return {
        "n": len(data),
        "events": int(data["event"].sum()),
        "coef": float(summary["coef"]),
        "hr": float(summary["exp(coef)"]),
        "ci_low": float(summary["exp(coef) lower 95%"]),
        "ci_high": float(summary["exp(coef) upper 95%"]),
        "pvalue": float(summary["p"]),
    }


def main():
    scores = pd.read_csv(P01 / "signature_scores_tcga.csv.gz", compression="gzip")
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()
    nested = pd.read_csv(P01 / "nested_cox_standard.csv")
    primary_significant = nested[
        (nested.model == "m3_microenvironment") & (nested.qvalue_global < 0.05)
    ].pmid.drop_duplicates().tolist()

    clinical = prepare_clinical()
    music_fractions, bayesprism_fractions = prepare_fractions()
    base = clinical.join(music_fractions.join(bayesprism_fractions, how="inner"), how="inner")
    base = base.dropna(subset=["age", "stage_num"])
    model_definitions = {
        "music_formal": ["age", "sex_female", "stage_num", "music_tumor_epithelial_fraction", "music_normal_urothelial_fraction", "music_immune_fraction", "music_stromal_fraction"],
        "bayesprism_formal": ["age", "sex_female", "stage_num", "bp_tumor_epithelial_fraction", "bp_normal_urothelial_fraction", "bp_immune_fraction", "bp_stromal_fraction"],
    }
    rows = []
    for pmid in primary_significant:
        data = base.join(scores[scores["pmid"] == pmid].set_index("patient_id")["score"], how="inner")
        for model_name, covariates in model_definitions.items():
            model_data = data.dropna(subset=["score"] + covariates)
            if len(model_data) < 50:
                continue
            try:
                result = fit_model(model_data, ["score"] + covariates)
            except Exception:
                result = {"n": len(model_data), "events": int(model_data["event"].sum()), "coef": np.nan, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "pvalue": np.nan}
            rows.append({"pmid": pmid, "model": model_name, **result})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(P01 / "final_signature_sensitivity_formal_deconvolution.csv", index=False)
    print("tested", len(primary_significant), "rows", len(sensitivity))
    for model_name, group in sensitivity.groupby("model"):
        print(model_name, "tested", group.pmid.nunique(), "significant", int((group.pvalue < 0.05).sum()), "failed", int(group.pvalue.isna().sum()))


if __name__ == "__main__":
    main()
