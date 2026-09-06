from pathlib import Path

import pandas as pd


PROCESSED = Path("data/processed")
OUT = Path("outputs/mediation")


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def prepare_base():
    clinical = pd.read_csv(
        "data/processed/tcga_blca_clinical.csv",
        index_col=0,
        low_memory=False,
    )
    clinical["patient_id"] = clinical["patient.bcr_patient_barcode"].str.upper()
    clinical = clinical.drop_duplicates("patient_id", keep="first")
    clinical["event"] = clinical["patient.vital_status"].str.lower().eq("dead").astype(int)
    clinical["time_months"] = (
        pd.to_numeric(
            clinical["patient.days_to_death"].where(
                clinical["event"].eq(1),
                clinical["patient.days_to_last_followup"],
            ),
            errors="coerce",
        )
        / 30.44
    )
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
        ["immune_score", "stromal_score"]
    ].mean()
    base = clinical.join(purity, how="inner")
    for column in ["immune_score", "stromal_score"]:
        base[f"{column}_z"] = (base[column] - base[column].mean()) / base[column].std()
    combined = base["immune_score_z"] + base["stromal_score_z"]
    base["microenv_z"] = (combined - combined.mean()) / combined.std()
    base["age_c"] = base["age"] - base["age"].mean()
    base["stage_c2"] = base["stage_num"] - 2
    return base


def lost_pmids():
    nested = pd.read_csv(PROCESSED / "p01_full/nested_cox_standard.csv")
    wide = nested.pivot(index="pmid", columns="model", values="qvalue_global")
    return wide[
        (wide["m0_score"] < 0.05) & (wide["m3_microenvironment"] >= 0.05)
    ].index.tolist()


def main():
    base = prepare_base()
    scores = pd.read_csv(PROCESSED / "p01_full/signature_scores_tcga.csv.gz")
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()
    pmids = lost_pmids()
    frames = []
    metadata = []
    for pmid in pmids:
        group = scores[scores["pmid"] == pmid]
        data = base.join(group.set_index("patient_id")["score"], how="inner")
        data = data.dropna(
            subset=[
                "score",
                "microenv_z",
                "time_months",
                "event",
                "age_c",
                "sex_female",
                "stage_c2",
            ]
        )
        data = data[data["time_months"] > 0]
        mean = data["score"].mean()
        sd = data["score"].std()
        data["score_z"] = (data["score"] - mean) / sd
        data = data.reset_index()[
            [
                "patient_id",
                "time_months",
                "event",
                "score_z",
                "microenv_z",
                "age_c",
                "sex_female",
                "stage_c2",
            ]
        ].assign(pmid=pmid)
        frames.append(data)
        metadata.append(
            {
                "pmid": pmid,
                "n": len(data),
                "events": int(data["event"].sum()),
                "score_mean": mean,
                "score_sd": sd,
            }
        )
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(OUT / "p01_regmedint_input.csv", index=False)
    pd.DataFrame(metadata).to_csv(OUT / "p01_regmedint_metadata.csv", index=False)
    print("lost signatures", len(pmids))
    print(pd.DataFrame(metadata).to_string(index=False))


if __name__ == "__main__":
    main()
