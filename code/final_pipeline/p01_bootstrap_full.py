from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"
PARTS = OUT / "bootstrap_parts"
SCORE_FILE = OUT / "signature_scores_tcga.csv.gz"
N_BOOTSTRAP = 1000
WORKERS = 12


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
        clinical["patient.stage_event.pathologic_stage"].str.lower().map(stage_map)
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

    scores = pd.read_csv(SCORE_FILE)
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()
    base = clinical.join(purity, how="inner").dropna(
        subset=["age", "stage_num", "tumor_epithelial_fraction"]
    )
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


def bootstrap_signature(args):
    pmid, group, base = args
    data = base.join(group.set_index("patient_id")["score"], how="inner")
    data = data.dropna(subset=["score"] + COLUMNS + ["time_months", "event"])
    if len(data) < 50:
        return pmid, []
    rows = []
    for iteration in range(N_BOOTSTRAP):
        sample = data.sample(frac=1.0, replace=True, random_state=iteration)
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(
                sample[["score"] + COLUMNS + ["time_months", "event"]],
                duration_col="time_months",
                event_col="event",
            )
            summary = cph.summary.loc["score"]
            rows.append(
                {
                    "pmid": pmid,
                    "iteration": iteration,
                    "hr": float(summary["exp(coef)"]),
                    "pvalue": float(summary["p"]),
                    "converged": True,
                }
            )
        except Exception:
            rows.append(
                {
                    "pmid": pmid,
                    "iteration": iteration,
                    "hr": np.nan,
                    "pvalue": np.nan,
                    "converged": False,
                }
            )
    return pmid, rows


def main():
    PARTS.mkdir(parents=True, exist_ok=True)
    scores, base = prepare_data()
    tasks = [
        (pmid, group, base.copy())
        for pmid, group in scores.groupby("pmid")
    ]
    completed = {path.stem for path in PARTS.glob("*.csv")}
    tasks = [task for task in tasks if task[0] not in completed]

    print("signatures to run", len(tasks), "already checkpointed", len(completed))
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(bootstrap_signature, task): task[0]
            for task in tasks
        }
        done = 0
        for future in as_completed(futures):
            pmid = futures[future]
            result_pmid, rows = future.result()
            if rows:
                pd.DataFrame(rows).to_csv(
                    PARTS / f"{result_pmid}.csv",
                    index=False,
                )
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"checkpointed {done}/{len(tasks)} PMID={result_pmid}", flush=True)

    rows = []
    for path in PARTS.glob("*.csv"):
        rows.append(pd.read_csv(path))
    if rows:
        bootstrap = pd.concat(rows, ignore_index=True)
    else:
        bootstrap = pd.DataFrame(
            columns=["pmid", "iteration", "hr", "pvalue", "converged"]
        )
    bootstrap.to_csv(OUT / "revision_bootstrap_all_1000.csv", index=False)
    summary = (
        bootstrap.groupby("pmid")
        .agg(
            n_bootstrap=("iteration", "size"),
            converged=("converged", "sum"),
            median_hr=("hr", "median"),
            significant_rate=("pvalue", lambda s: (s < 0.05).mean()),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "revision_bootstrap_1000_summary.csv", index=False)
    print("bootstrap rows", len(bootstrap))
    print("signatures", bootstrap["pmid"].nunique())
    print("median significant rate", summary["significant_rate"].median())
    print(">=50%", int((summary["significant_rate"] >= 0.5).sum()))


if __name__ == "__main__":
    main()
