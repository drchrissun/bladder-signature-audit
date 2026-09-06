from pathlib import Path

import pandas as pd


def main():
    processed = Path("data/processed")
    corr = pd.read_csv("data/processed/signature_microenvironment_correlations_tcga.csv")
    rho = corr.pivot(index="pmid", columns="covariate", values="spearman_rho")
    pvalue = corr.pivot(index="pmid", columns="covariate", values="pvalue")
    survival = pd.read_csv("data/processed/signature_survival_results_tcga.csv")
    unadj = survival[survival["model"] == "unadjusted"].set_index("pmid")["pvalue"].rename("unadj_p")
    adj = survival[survival["model"] == "adjusted"].set_index("pmid")["pvalue"].rename("adj_p")
    mucosa = pd.read_csv("data/processed/gse13507_normal_mucosa_signature_summary.csv").set_index("pmid")

    frame = pd.concat(
        [
            rho,
            pvalue.add_suffix("_p"),
            unadj,
            adj,
            mucosa[["normal_looking_minus_primary", "mannwhitney_pvalue"]],
        ],
        axis=1,
    )
    frame["max_micro_abs_rho"] = frame[["immune_score", "stromal_score"]].abs().max(axis=1)

    def classify(row):
        normal_higher = (
            pd.notna(row["normal_looking_minus_primary"])
            and row["normal_looking_minus_primary"] > 0
            and row["mannwhitney_pvalue"] < 0.05
        )
        micro = pd.notna(row["max_micro_abs_rho"]) and row["max_micro_abs_rho"] >= 0.4
        adj_sig = pd.notna(row["adj_p"]) and row["adj_p"] < 0.05
        unadj_sig = pd.notna(row["unadj_p"]) and row["unadj_p"] < 0.05
        normal_rho = row.get("normal_urothelial_fraction", float("nan"))
        if normal_higher and (pd.notna(normal_rho) and normal_rho > 0.3):
            return "normal_urothelium_driven"
        if micro and unadj_sig and not adj_sig:
            return "microenvironment_driven"
        if adj_sig and not micro and not normal_higher:
            return "tumor_cell_autonomous"
        return "ambiguous_or_insufficient"

    frame["classification"] = frame.apply(classify, axis=1)
    frame = frame.reset_index().rename(columns={"index": "pmid"})
    frame.to_csv(processed / "signature_classification.csv", index=False)
    print(frame["classification"].value_counts().to_string())


if __name__ == "__main__":
    main()
