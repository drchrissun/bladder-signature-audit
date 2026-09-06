import gzip
import re
from pathlib import Path

import pandas as pd


def parse_series_metadata(path):
    samples = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                samples = [value.strip().strip('"') for value in re.findall(r'"(.*?)"', line)]
                break
    return samples


def main():
    processed = Path("data/processed")
    processed.mkdir(exist_ok=True)

    series_path = "data/raw/GSE13507_series_matrix.txt.gz"
    samples = parse_series_metadata(series_path)
    expr = pd.read_csv(
        series_path,
        sep="\t",
        comment="!",
        index_col=0,
        low_memory=False,
    )
    expr.columns = samples
    expr.to_csv(processed / "gse13507_expr.csv.gz", compression="gzip")

    clinical = pd.read_excel("data/raw/GSE13507_clinical_info.xls")
    clinical.to_csv(processed / "gse13507_clinical.csv", index=False)

    print("expression shape", expr.shape)
    print("clinical shape", clinical.shape)
    print(clinical.head().to_string(index=False))


if __name__ == "__main__":
    main()
