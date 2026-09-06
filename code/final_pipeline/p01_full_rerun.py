import csv
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from step6_signature_scores import comparable_matrix, load_alias_map


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"
CONSENSUS = Path(
    "outputs/screening/screening_full_consensus_decisions.csv"
)


def patient_of(sample):
    return "-".join(sample.split("-")[:3])


def build_full_gene_table():
    raw = pd.read_csv(
        "signature_inventory/gene_extraction_raw.csv",
        dtype={"pmid": str},
    ).fillna("")
    additional = pd.read_csv(
        "signature_inventory/gene_extraction_raw_additional.csv",
        dtype={"pmid": str},
    ).fillna("")
    consensus = pd.read_csv(CONSENSUS, dtype={"pmid": str})
    include = set(
        consensus.loc[
            consensus["reviewer1_decision"] == "include", "pmid"
        ]
    )
    table = pd.concat([raw, additional], ignore_index=True)
    table = table[table["pmid"].isin(include)].copy()
    table["consensus_decision"] = "include"
    return table


def build_scores(table):
    alias_map = load_alias_map()
    tcga = pd.read_csv(
        PROCESSED / "tcga_blca_expr_gene.csv.gz",
        index_col=0,
    )
    gse = pd.read_csv(
        PROCESSED / "gse13507_expr_gene.csv.gz",
        index_col=0,
    )
    tcga_pos = {gene: idx for idx, gene in enumerate(tcga.index)}
    gse_pos = {gene: idx for idx, gene in enumerate(gse.index)}
    tcga_log = comparable_matrix(tcga, "TCGA")
    gse_log = comparable_matrix(gse, "GSE13507")
    tcga_z = (
        tcga_log - tcga_log.mean(axis=1, keepdims=True)
    ) / tcga_log.std(axis=1, keepdims=True, ddof=1)
    gse_z = (
        gse_log - gse_log.mean(axis=1, keepdims=True)
    ) / gse_log.std(axis=1, keepdims=True, ddof=1)

    tcga_rows = []
    gse_rows = []
    coverage = []
    for row in table[table["extraction_status"] == "extracted"].itertuples():
        mapped = []
        for gene in row.genes.split("|"):
            gene = gene.strip()
            if not gene:
                continue
            canonical = alias_map.get(gene, gene)
            if canonical not in mapped:
                mapped.append(canonical)
        tcga_idx = [tcga_pos[gene] for gene in mapped if gene in tcga_pos]
        gse_idx = [gse_pos[gene] for gene in mapped if gene in gse_pos]
        coverage.append(
            {
                "pmid": row.pmid,
                "raw_gene_count": len(mapped),
                "tcga_mapped_count": len(tcga_idx),
                "gse_mapped_count": len(gse_idx),
            }
        )
        if len(tcga_idx) >= 2:
            tcga_rows.append(
                pd.DataFrame(
                    {
                        "pmid": row.pmid,
                        "sample": tcga.columns,
                        "score": tcga_z[tcga_idx, :].mean(axis=0),
                    }
                )
            )
        if len(gse_idx) >= 2:
            gse_rows.append(
                pd.DataFrame(
                    {
                        "pmid": row.pmid,
                        "sample": gse.columns,
                        "score": gse_z[gse_idx, :].mean(axis=0),
                    }
                )
            )

    tcga_scores = pd.concat(tcga_rows, ignore_index=True)
    gse_scores = pd.concat(gse_rows, ignore_index=True)
    tcga_scores.to_csv(
        OUT / "signature_scores_tcga.csv.gz",
        index=False,
        compression="gzip",
    )
    gse_scores.to_csv(
        OUT / "signature_scores_gse13507.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(coverage).to_csv(OUT / "signature_coverage.csv", index=False)
    return tcga_scores, gse_scores, pd.DataFrame(coverage)


def build_correlations(scores):
    wide = scores.pivot(index="sample", columns="pmid", values="score")
    purity = pd.read_csv(PROCESSED / "purity_and_microenvironment_estimates.csv")
    tcga_purity = purity[purity["dataset"] == "TCGA"].set_index("sample")
    covariates = [
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
        "immune_score",
        "stromal_score",
    ]
    rows = []
    for pmid in wide.columns:
        for covariate in covariates:
            frame = pd.concat(
                [
                    wide[pmid].rename("score"),
                    tcga_purity[covariate].rename("covariate"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            rho, pvalue = spearmanr(frame["score"], frame["covariate"])
            rows.append(
                {
                    "pmid": pmid,
                    "covariate": covariate,
                    "n": len(frame),
                    "spearman_rho": rho,
                    "pvalue": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "microenvironment_correlations_tcga.csv", index=False)
    return result


def normal_mucosa_summary(scores):
    with gzip.open(
        "data/raw/GSE13507_series_matrix.txt.gz",
        "rt",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        titles = []
        for line in handle:
            if line.startswith("!Sample_title"):
                titles = [
                    value.strip().strip('"')
                    for value in re.findall(r'"(.*?)"', line)
                ]
                break

    def group_of(title):
        if title.startswith("Primary bladder cancer"):
            return "primary"
        if title.startswith("Surrounding"):
            return "normal_looking"
        return "other"

    meta = pd.DataFrame(
        {"sample": titles, "group": [group_of(title) for title in titles]}
    )
    scores = scores.merge(meta, on="sample", how="inner")
    rows = []
    for pmid, group in scores.groupby("pmid"):
        primary = group.loc[group["group"] == "primary", "score"]
        normal = group.loc[group["group"] == "normal_looking", "score"]
        if len(primary) < 5 or len(normal) < 5:
            continue
        stat, pvalue = mannwhitneyu(
            primary,
            normal,
            alternative="two-sided",
        )
        rows.append(
            {
                "pmid": pmid,
                "primary_mean": primary.mean(),
                "normal_looking_mucosa_mean": normal.mean(),
                "normal_looking_minus_primary": normal.mean() - primary.mean(),
                "mannwhitney_pvalue": pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "normal_mucosa_signature_summary.csv", index=False)
    return result


def run_nested_cox(scores):
    clinical = pd.read_csv(
        PROCESSED / "tcga_blca_clinical.csv",
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

    scores = scores.copy()
    scores["patient_id"] = scores["sample"].map(patient_of)
    scores = scores[scores["sample"].str.contains("-01A-", regex=False)]
    scores = scores.groupby(["patient_id", "pmid"])["score"].mean().reset_index()

    columns = {
        "m0_score": ["score"],
        "m1_clinical": ["score", "age", "sex_female", "stage_num"],
        "m2_purity": [
            "score",
            "age",
            "sex_female",
            "stage_num",
            "tumor_epithelial_fraction",
            "normal_urothelial_fraction",
        ],
        "m3_microenvironment": [
            "score",
            "age",
            "sex_female",
            "stage_num",
            "tumor_epithelial_fraction",
            "normal_urothelial_fraction",
            "immune_score",
            "stromal_score",
        ],
    }
    rows = []
    for pmid, group in scores.groupby("pmid"):
        data = base.join(group.set_index("patient_id")["score"], how="inner")
        for model_name, model_columns in columns.items():
            model_data = data.dropna(
                subset=model_columns + ["time_months", "event"]
            )[model_columns + ["time_months", "event"]]
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
                        "n": len(model_data),
                        "events": int(model_data["event"].sum()),
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
                        "n": len(model_data),
                        "events": int(model_data["event"].sum()),
                        "coef": np.nan,
                        "hr": np.nan,
                        "pvalue": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result["qvalue_global"] = multipletests(result["pvalue"], method="fdr_bh")[1]
    result.to_csv(OUT / "nested_cox_standard.csv", index=False)
    return result


def classify(nested, corr, mucosa):
    m0 = nested[nested["model"] == "m0_score"].set_index("pmid")["qvalue_global"]
    m3 = nested[nested["model"] == "m3_microenvironment"].set_index("pmid")["qvalue_global"]
    rho = corr.pivot(index="pmid", columns="covariate", values="spearman_rho")
    max_micro = rho[["immune_score", "stromal_score"]].abs().max(axis=1)
    strong_micro = max_micro >= 0.4
    mucosa = mucosa.set_index("pmid")
    normal_higher = (
        (mucosa["normal_looking_minus_primary"] > 0)
        & (mucosa["mannwhitney_pvalue"] < 0.05)
    )
    frame = pd.DataFrame(
        {
            "max_micro_abs_rho": max_micro,
            "strong_microenvironment_correlation": strong_micro,
            "normal_mucosa_elevation_significant": normal_higher,
            "m0_global_q": m0,
            "m3_global_q": m3,
        }
    ).dropna(subset=["m0_global_q", "m3_global_q"])
    frame["unadjusted_fdr_significant"] = frame["m0_global_q"] < 0.05
    frame["adjusted_fdr_significant"] = frame["m3_global_q"] < 0.05
    frame["lost_fdr_significance"] = (
        frame["unadjusted_fdr_significant"]
        & ~frame["adjusted_fdr_significant"]
    )

    def label(row):
        if row["adjusted_fdr_significant"] and not (
            row["strong_microenvironment_correlation"]
            or row["normal_mucosa_elevation_significant"]
        ):
            return "composition_adjustment_persistent_standard"
        if row["lost_fdr_significance"]:
            return "microenvironment_associated_attenuation_standard"
        if (
            row["strong_microenvironment_correlation"]
            or row["normal_mucosa_elevation_significant"]
        ):
            return "microenvironment_correlated_standard"
        return "unclassified_standard"

    frame["classification_v6_standard"] = frame.apply(label, axis=1)
    frame.index.name = "pmid"
    frame = frame.reset_index()
    frame.to_csv(OUT / "classification_v6_standard.csv", index=False)
    return frame


def external_validation(scores, nested):
    clinical = pd.read_csv(PROCESSED / "gse13507_clinical.csv")
    clinical = clinical.rename(columns={"Sample name": "sample_id"})
    clinical["event"] = clinical["overall survival"].astype(float).eq(2).astype(int)
    clinical["time_months"] = pd.to_numeric(clinical["survivalMonth"], errors="coerce")
    clinical["sex_female"] = clinical["SEX"].astype(str).str.upper().eq("F").astype(int)
    clinical["age"] = pd.to_numeric(clinical["AGE"], errors="coerce")
    clinical["invasiveness"] = pd.to_numeric(clinical["invasiveness"], errors="coerce")
    clinical["grade"] = pd.to_numeric(clinical["Grade"], errors="coerce")

    scores = scores[scores["sample"].str.startswith("Primary bladder cancer")]
    scores["sample_id"] = scores["sample"].str.extract(r"BT(\d+)$")[0].map(
        lambda value: f"BT{value}"
    )
    purity = pd.read_csv(PROCESSED / "purity_and_microenvironment_estimates.csv")
    purity = purity[purity["dataset"] == "GSE13507"].copy()
    purity = purity[
        ["sample"]
        + [
            "tumor_epithelial_fraction",
            "normal_urothelial_fraction",
            "immune_score",
            "stromal_score",
        ]
    ].rename(columns={"sample": "sample_title"})
    scores = scores.merge(
        purity,
        left_on="sample",
        right_on="sample_title",
        how="left",
    )
    base = clinical.merge(
        scores.drop_duplicates("sample_id")[
            [
                "sample_id",
                "tumor_epithelial_fraction",
                "normal_urothelial_fraction",
                "immune_score",
                "stromal_score",
            ]
        ],
        on="sample_id",
        how="inner",
    )
    base = base.dropna(
        subset=[
            "event",
            "time_months",
            "age",
            "invasiveness",
            "grade",
            "tumor_epithelial_fraction",
        ]
    )
    base = base[base["time_months"] > 0]
    ids = nested[
        (nested["model"] == "m3_microenvironment")
        & (nested["qvalue_global"] < 0.05)
    ]["pmid"].tolist()
    columns = [
        "score",
        "age",
        "sex_female",
        "invasiveness",
        "grade",
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
        "immune_score",
        "stromal_score",
    ]
    rows = []
    for pmid, group in scores.groupby("pmid"):
        data = base.merge(
            group[["sample_id", "score"]],
            on="sample_id",
            how="inner",
        ).dropna(subset=columns)
        if len(data) < 50:
            continue
        for model_name, model_columns in [
            ("unadjusted", ["score"]),
            ("adjusted", columns),
        ]:
            try:
                cph = CoxPHFitter(penalizer=0.0)
                cph.fit(
                    data[model_columns + ["time_months", "event"]],
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
                        "pvalue": float(summary["p"]),
                    }
                )
            except Exception:
                rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "n": len(data),
                        "events": int(data["event"].sum()),
                        "coef": np.nan,
                        "hr": np.nan,
                        "pvalue": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "gse13507_survival_validation_standard.csv", index=False)
    ext = result[(result["model"] == "adjusted") & (result["pmid"].isin(ids))]
    tcga = nested[nested["model"] == "m3_microenvironment"].set_index("pmid")["coef"]
    merged = ext.merge(tcga.rename("tcga_coef"), on="pmid")
    print("external ids", len(ids), "n", len(ext), "P<0.05", (ext.pvalue < 0.05).sum())
    print("direction", int((np.sign(merged.coef) == np.sign(merged.tcga_coef)).sum()))
    print("spearman", merged.coef.rank().corr(merged.tcga_coef.rank()).round(4))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_full_gene_table()
    table.to_csv(OUT / "gene_extraction_raw_full.csv", index=False)
    print("full rows", len(table), "extracted", (table.extraction_status == "extracted").sum())

    tcga_scores, gse_scores, coverage = build_scores(table)
    print("scores TCGA", tcga_scores.pmid.nunique(), "GSE", gse_scores.pmid.nunique())

    corr = build_correlations(tcga_scores)
    mucosa = normal_mucosa_summary(gse_scores)
    nested = run_nested_cox(tcga_scores)
    classification = classify(nested, corr, mucosa)
    print("\nclassification counts")
    print(classification["classification_v6_standard"].value_counts().to_string())

    pivot = nested.pivot(index="pmid", columns="model", values="qvalue_global")
    print("\nFDR counts")
    print(pivot.apply(lambda s: (s < 0.05).sum()).to_frame("q_significant").to_string())
    print(
        "lost",
        int(((pivot["m0_score"] < 0.05) & (pivot["m3_microenvironment"] >= 0.05)).sum()),
        "gained",
        int(((pivot["m0_score"] >= 0.05) & (pivot["m3_microenvironment"] < 0.05)).sum()),
    )
    print(
        "M1->M2 gained",
        int(((pivot["m1_clinical"] >= 0.05) & (pivot["m2_purity"] < 0.05)).sum()),
    )
    external_validation(gse_scores, nested)


if __name__ == "__main__":
    main()
