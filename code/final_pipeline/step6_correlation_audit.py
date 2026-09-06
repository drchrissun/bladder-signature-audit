from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def main():
    processed = Path("data/processed")
    scores = pd.read_csv("data/processed/signature_scores_tcga.csv.gz", compression="gzip")
    scores = scores.pivot(index="sample", columns="pmid", values="score")
    purity = pd.read_csv("data/processed/purity_and_microenvironment_estimates.csv")
    tcga_purity = purity[purity["dataset"] == "TCGA"].set_index("sample")
    covariates = [
        "tumor_epithelial_fraction",
        "normal_urothelial_fraction",
        "immune_score",
        "stromal_score",
    ]
    rows = []
    for pmid in scores.columns:
        for covariate in covariates:
            frame = pd.concat(
                [scores[pmid].rename("score"), tcga_purity[covariate].rename("covariate")],
                axis=1,
                join="inner",
            ).dropna()
            if len(frame) < 10:
                continue
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
    result.to_csv(processed / "signature_microenvironment_correlations_tcga.csv", index=False)
    print("correlation rows", len(result))
    print("significant FDR<0.05", (result.groupby("pmid")["pvalue"].min() < 0.05).sum())
    print(result.groupby("covariate")["spearman_rho"].describe()[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
