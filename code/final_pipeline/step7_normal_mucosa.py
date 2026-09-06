import gzip
import re
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


def sample_titles():
    with gzip.open("data/raw/GSE13507_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                return [value.strip().strip('"') for value in re.findall(r'"(.*?)"', line)]
    return []


def group_of(title):
    if title.startswith("Control"):
        return "normal"
    if title.startswith("Surrounding"):
        return "normal_looking_mucosa"
    if title.startswith("Primary bladder cancer"):
        return "primary_tumor"
    return "recurrent_or_other"


def main():
    processed = Path("data/processed")
    titles = sample_titles()
    sample_meta = pd.DataFrame(
        {
            "sample": [f"sample_{idx}" for idx in range(len(titles))],
            "title": titles,
            "group": [group_of(title) for title in titles],
        }
    )
    # The saved GSE expression columns are titles, not sample_N.
    scores = pd.read_csv("data/processed/signature_scores_gse13507.csv.gz", compression="gzip")
    meta_by_title = pd.DataFrame({"sample": titles, "group": [group_of(title) for title in titles]})
    scores = scores.merge(meta_by_title, on="sample", how="inner")
    rows = []
    for pmid, group in scores.groupby("pmid"):
        primary = group.loc[group["group"] == "primary_tumor", "score"]
        normal_looking = group.loc[group["group"] == "normal_looking_mucosa", "score"]
        normal = group.loc[group["group"] == "normal", "score"]
        recurrent = group.loc[group["group"] == "recurrent_or_other", "score"]
        if len(primary) >= 5 and len(normal_looking) >= 5:
            stat, pvalue = mannwhitneyu(primary, normal_looking, alternative="two-sided")
            rows.append(
                {
                    "pmid": pmid,
                    "primary_mean": primary.mean(),
                    "normal_looking_mucosa_mean": normal_looking.mean(),
                    "normal_mean": normal.mean(),
                    "recurrent_mean": recurrent.mean(),
                    "normal_looking_minus_primary": normal_looking.mean() - primary.mean(),
                    "mannwhitney_pvalue": pvalue,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(processed / "gse13507_normal_mucosa_signature_summary.csv", index=False)
    print("validation rows", len(result))
    print("normal-looking significantly different", (result["mannwhitney_pvalue"] < 0.05).sum())
    print("normal-looking higher than primary", (result["normal_looking_minus_primary"] > 0).sum())


if __name__ == "__main__":
    main()
