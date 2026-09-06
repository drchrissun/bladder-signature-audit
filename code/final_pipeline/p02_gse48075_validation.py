from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import power_under_cph

from step5_deconvolution import (
    MARKERS,
    comparable_matrix,
    deconvolve_scores,
    estimate_scores,
)
from step6_signature_scores import load_alias_map


PROCESSED = Path("data/processed")
OUT = PROCESSED / "p01_full"


def patient_score_z(expr):
    values = comparable_matrix(expr, "GSE13507")
    return (values - values.mean(axis=1, keepdims=True)) / values.std(
        axis=1,
        keepdims=True,
        ddof=1,
    )


def main():
    expr = pd.read_csv(OUT / "gse48075_expr_gene.csv.gz", index_col=0)
    clinical = pd.read_csv(OUT / "gse48075_clinical.csv")
    gene_table = pd.read_csv(
        OUT / "gene_extraction_raw_full.csv",
        dtype={"pmid": str},
    )
    gene_table = gene_table[gene_table.extraction_status == "extracted"]
    nested = pd.read_csv(OUT / "nested_cox_standard.csv")
    ids = [
        str(pmid)
        for pmid in nested[
            (nested.model == "m3_microenvironment")
            & (nested.qvalue_global < 0.05)
        ]["pmid"]
    ]

    alias_map = load_alias_map()
    expr_pos = {gene: idx for idx, gene in enumerate(expr.index)}
    z = patient_score_z(expr)

    rows = []
    for row in gene_table[gene_table.pmid.isin(ids)].itertuples():
        mapped = []
        for gene in row.genes.split("|"):
            gene = gene.strip()
            if not gene:
                continue
            canonical = alias_map.get(gene, gene)
            if canonical not in mapped:
                mapped.append(canonical)
        indices = [expr_pos[gene] for gene in mapped if gene in expr_pos]
        if len(indices) >= 2:
            score = z[indices, :].mean(axis=0)
            rows.append(
                pd.DataFrame(
                    {
                        "pmid": row.pmid,
                        "sample_title": expr.columns,
                        "score": score,
                    }
                )
            )
    scores = pd.concat(rows, ignore_index=True)

    scores_frame, fractions = deconvolve_scores(expr, "GSE13507")
    estimate = estimate_scores(expr, "GSE13507")
    fractions = fractions.reset_index().rename(columns={"index": "sample_title"})
    estimate = estimate.reset_index().rename(columns={"index": "sample_title"})
    fractions["tumor_epithelial_fraction"] = fractions["Tumor_epithelial"]
    fractions["normal_urothelial_fraction"] = fractions["Normal_urothelial"]
    fractions["immune_score"] = estimate["immune"]
    fractions["stromal_score"] = estimate["stromal"]

    clinical["stage_num"] = (
        clinical["stage"]
        .str.replace(r"[cp]", "", regex=True)
        .str.extract(r"T(\d)", expand=False)
        .astype(float)
    )
    base = clinical.merge(
        fractions[
            [
                "sample_title",
                "tumor_epithelial_fraction",
                "normal_urothelial_fraction",
                "immune_score",
                "stromal_score",
            ]
        ],
        on="sample_title",
        how="inner",
    )
    base = base.dropna(
        subset=[
            "os_event",
            "time_months",
            "age",
            "stage_num",
            "tumor_epithelial_fraction",
        ]
    )
    base = base[base.time_months > 0]

    columns = [
        "score",
        "age",
        "stage_num",
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
        "immune_score",
        "stromal_score",
    ]
    validation_rows = []
    for pmid, group in scores.groupby("pmid"):
        data = base.merge(
            group[["sample_title", "score"]],
            on="sample_title",
            how="inner",
        ).dropna(subset=columns)
        for model_name, model_columns in [
            ("unadjusted", ["score"]),
            ("adjusted", columns),
        ]:
            try:
                cph = CoxPHFitter(penalizer=0.0)
                cph.fit(
                    data[model_columns + ["time_months", "os_event"]],
                    duration_col="time_months",
                    event_col="os_event",
                )
                summary = cph.summary.loc["score"]
                validation_rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "n": len(data),
                        "events": int(data.os_event.sum()),
                        "coef": float(summary["coef"]),
                        "hr": float(summary["exp(coef)"]),
                        "pvalue": float(summary["p"]),
                    }
                )
            except Exception:
                validation_rows.append(
                    {
                        "pmid": pmid,
                        "model": model_name,
                        "n": len(data),
                        "events": int(data.os_event.sum()),
                        "coef": np.nan,
                        "hr": np.nan,
                        "pvalue": np.nan,
                    }
                )

    result = pd.DataFrame(validation_rows)
    result.to_csv(OUT / "gse48075_survival_validation.csv", index=False)
    adjusted = result[(result.model == "adjusted") & result.pmid.isin(ids)]
    tcga = nested[
        nested.model == "m3_microenvironment"
    ].copy()
    tcga["pmid"] = tcga["pmid"].astype(str)
    tcga = tcga.set_index("pmid")["coef"]
    merged = adjusted.merge(tcga.rename("tcga_coef"), on="pmid")
    print("n", len(adjusted), "events", adjusted.events.max())
    print("P<0.05", int((adjusted.pvalue < 0.05).sum()))
    print("direction", int((np.sign(merged.coef) == np.sign(merged.tcga_coef)).sum()))
    spearman = merged.coef.rank().corr(merged.tcga_coef.rank())
    print("spearman", round(float(spearman), 4))
    power = power_under_cph(
        len(adjusted) // 2,
        len(adjusted) - len(adjusted) // 2,
        45 / 73,
        45 / 73,
        1.5,
        alpha=0.05,
    )
    print("HR1.5 power", power, "expected", power * len(adjusted))


if __name__ == "__main__":
    main()
