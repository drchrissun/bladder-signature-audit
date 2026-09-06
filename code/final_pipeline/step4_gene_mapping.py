import re
from pathlib import Path

import pandas as pd


def main():
    processed = Path("data/processed")
    hgnc = pd.read_csv("reference/hgnc_complete_set.tsv", sep="\t", dtype=str, low_memory=False)
    hgnc = hgnc[hgnc["status"].str.lower() == "approved"]
    ensembl_map = (
        hgnc.dropna(subset=["ensembl_gene_id"])
        .drop_duplicates("ensembl_gene_id")
        .set_index("ensembl_gene_id")["symbol"]
        .to_dict()
    )
    entrez_map = (
        hgnc.dropna(subset=["entrez_id"])
        .drop_duplicates("entrez_id")
        .set_index("entrez_id")["symbol"]
        .to_dict()
    )

    tcga = pd.read_csv(
        "data/processed/tcga_blca_expr.csv.gz",
        compression="gzip",
        index_col=0,
    )
    tcga_ids = tcga.index.astype(str)
    tcga_symbols = []
    for gene_id in tcga_ids:
        if "|" in gene_id:
            symbol_part, numeric_part = gene_id.split("|", 1)
            if symbol_part and symbol_part != "?" and not symbol_part.isdigit():
                tcga_symbols.append(symbol_part)
                continue
            tcga_symbols.append(entrez_map.get(numeric_part, symbol_part))
            continue
        if gene_id in ensembl_map:
            tcga_symbols.append(ensembl_map[gene_id])
            continue
        match = re.search(r"(\d+)", gene_id)
        entrez = match.group(1) if match else ""
        tcga_symbols.append(entrez_map.get(entrez, gene_id))
    tcga["symbol"] = tcga_symbols
    tcga = tcga[tcga["symbol"] != ""]
    tcga = tcga.groupby("symbol").mean(numeric_only=True)
    tcga.to_csv(processed / "tcga_blca_expr_gene.csv.gz", compression="gzip")
    print("tcga mapped genes", tcga.shape[0], "samples", tcga.shape[1])

    gse = pd.read_csv(
        "data/processed/gse13507_expr.csv.gz",
        compression="gzip",
        index_col=0,
    )
    probe_map = pd.read_csv("data/processed/gpl6102_probe_gene.csv.gz", compression="gzip")
    probe_map = probe_map.drop_duplicates("probe_id")
    probe_map = probe_map[probe_map["gene_symbol"].notna() & (probe_map["gene_symbol"] != "---")]
    gse = gse.join(
        probe_map.set_index("probe_id")["gene_symbol"],
        how="inner",
    )
    gse = gse.groupby("gene_symbol").mean(numeric_only=True)
    gse.to_csv(processed / "gse13507_expr_gene.csv.gz", compression="gzip")
    print("gse mapped genes", gse.shape[0], "samples", gse.shape[1])

    common = sorted(set(tcga.index) & set(gse.index))
    print("common genes", len(common))
    pd.Series(common, name="symbol").to_csv(
        processed / "common_gene_symbols.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
