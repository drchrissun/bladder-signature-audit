# -*- coding: utf-8 -*-
"""Recompute classification bootstrap stability for the final 200-list set."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score


BASE = Path("data/processed/p01_full")
OUT = Path("data/processed/p01_full")


def classify(row: pd.Series) -> str:
    if row["adjusted_fdr_significant"] and not (
        row["strong_microenvironment_correlation"]
        or row["normal_mucosa_elevation_significant"]
    ):
        return "persistent"
    if row["lost_significance"]:
        return "attenuation"
    if (
        row["strong_microenvironment_correlation"]
        or row["normal_mucosa_elevation_significant"]
    ):
        return "correlated"
    return "unclassified"


def main() -> None:
    base = pd.read_csv(BASE / "classification_v6_standard.csv").copy()
    base["lost_significance"] = (
        base["unadjusted_fdr_significant"] & ~base["adjusted_fdr_significant"]
    )
    base_labels = base.apply(classify, axis=1)

    bootstrap = pd.read_csv(BASE / "revision_bootstrap_all_1000.csv")
    ari_values = []
    kappa_values = []
    for iteration, frame in bootstrap.groupby("iteration"):
        p = frame.set_index("pmid")["pvalue"].reindex(base.pmid)
        adjusted = base.copy()
        adjusted["adjusted_fdr_significant"] = p.values < 0.05
        adjusted["lost_significance"] = (
            adjusted["unadjusted_fdr_significant"]
            & ~adjusted["adjusted_fdr_significant"]
        )
        labels = adjusted.apply(classify, axis=1)
        ari_values.append(adjusted_rand_score(base_labels, labels))
        kappa_values.append(cohen_kappa_score(base_labels, labels))

    result = pd.DataFrame(
        {
            "iteration": range(len(ari_values)),
            "adjusted_rand_index": ari_values,
            "cohens_kappa": kappa_values,
        }
    )
    result.to_csv(OUT / "final_classification_bootstrap_stability.csv", index=False)
    print("iterations", len(result))
    print("ARI median", round(np.median(ari_values), 3))
    print("ARI min", round(np.min(ari_values), 3))
    print("ARI IQR", [round(x, 3) for x in np.quantile(ari_values, [0.25, 0.75])])
    print("Kappa median", round(np.median(kappa_values), 3))
    print("Kappa min", round(np.min(kappa_values), 3))
    print("Kappa IQR", [round(x, 3) for x in np.quantile(kappa_values, [0.25, 0.75])])


if __name__ == "__main__":
    main()
