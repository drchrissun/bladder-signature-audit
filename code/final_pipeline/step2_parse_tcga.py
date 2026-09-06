from pathlib import Path

import pandas as pd


def main():
    processed = Path("data/processed")
    processed.mkdir(exist_ok=True)

    sample_header = pd.read_csv(
        "data/processed/tcga_blca_rnaseqv2_rsem.txt",
        sep="\t",
        nrows=1,
        header=None,
        low_memory=False,
    ).iloc[0].tolist()
    expr = pd.read_csv(
        "data/processed/tcga_blca_rnaseqv2_rsem.txt",
        sep="\t",
        skiprows=2,
        header=None,
        index_col=0,
        low_memory=False,
    )
    expr.columns = sample_header[1:]
    expr.index.name = "gene_id"
    expr.to_csv(processed / "tcga_blca_expr.csv.gz", compression="gzip")

    clinical_wide = pd.read_csv(
        "data/processed/tcga_blca_clinical_firehose.txt",
        sep="\t",
        index_col=0,
        low_memory=False,
    )
    clinical = clinical_wide.T
    clinical.index.name = "sample"
    clinical.to_csv(processed / "tcga_blca_clinical.csv", index=True)

    print("expression shape", expr.shape)
    print("clinical shape", clinical.shape)
    print("expression samples", len(expr.columns))


if __name__ == "__main__":
    main()
