from pathlib import Path

import numpy as np
import pandas as pd


def load_alias_map():
    hgnc = pd.read_csv("reference/hgnc_complete_set.tsv", sep="\t", dtype=str, low_memory=False)
    mapping = {}
    for row in hgnc.itertuples(index=False):
        symbol = row.symbol
        mapping[symbol] = symbol
        for field in ("alias_symbol", "prev_symbol"):
            value = getattr(row, field, "")
            if pd.isna(value):
                continue
            for alias in str(value).split("|"):
                if alias:
                    mapping.setdefault(alias, symbol)
    return mapping


def comparable_matrix(frame, dataset):
    values = frame.values.astype(float)
    if dataset == "TCGA":
        return np.log1p(values)
    linear = np.expm1(values * np.log(2))
    linear[linear < 0] = 0
    return np.log1p(linear)


def main():
    processed = Path("data/processed")
    signatures = pd.read_csv("signature_inventory/gene_extraction_raw.csv", dtype=str).fillna("")
    signatures = signatures[signatures["extraction_status"] == "extracted"].copy()
    alias_map = load_alias_map()

    tcga = pd.read_csv("data/processed/tcga_blca_expr_gene.csv.gz", index_col=0)
    gse = pd.read_csv("data/processed/gse13507_expr_gene.csv.gz", index_col=0)
    tcga_pos = {gene: idx for idx, gene in enumerate(tcga.index)}
    gse_pos = {gene: idx for idx, gene in enumerate(gse.index)}
    tcga_log = comparable_matrix(tcga, "TCGA")
    gse_log = comparable_matrix(gse, "GSE13507")
    tcga_z = (tcga_log - tcga_log.mean(axis=1, keepdims=True)) / tcga_log.std(axis=1, keepdims=True, ddof=1)
    gse_z = (gse_log - gse_log.mean(axis=1, keepdims=True)) / gse_log.std(axis=1, keepdims=True, ddof=1)

    tcga_rows = []
    gse_rows = []
    coverage = []
    for row in signatures.itertuples():
        raw_genes = [gene.strip() for gene in row.genes.split("|") if gene.strip()]
        mapped = []
        for gene in raw_genes:
            canonical = alias_map.get(gene, gene)
            if canonical not in mapped:
                mapped.append(canonical)

        tcga_indices = [tcga_pos[gene] for gene in mapped if gene in tcga_pos]
        gse_indices = [gse_pos[gene] for gene in mapped if gene in gse_pos]
        coverage.append(
            {
                "pmid": row.pmid,
                "raw_gene_count": len(raw_genes),
                "tcga_mapped_count": len(tcga_indices),
                "gse_mapped_count": len(gse_indices),
            }
        )

        if len(tcga_indices) >= 2:
            tcga_score = tcga_z[tcga_indices, :].mean(axis=0)
            tcga_rows.append(
                pd.DataFrame(
                    {"pmid": row.pmid, "sample": tcga.columns, "score": tcga_score}
                )
            )
        if len(gse_indices) >= 2:
            gse_score = gse_z[gse_indices, :].mean(axis=0)
            gse_rows.append(
                pd.DataFrame(
                    {"pmid": row.pmid, "sample": gse.columns, "score": gse_score}
                )
            )

    tcga_scores = pd.concat(tcga_rows, ignore_index=True)
    gse_scores = pd.concat(gse_rows, ignore_index=True)
    tcga_scores.to_csv(processed / "signature_scores_tcga.csv.gz", index=False, compression="gzip")
    gse_scores.to_csv(processed / "signature_scores_gse13507.csv.gz", index=False, compression="gzip")
    pd.DataFrame(coverage).to_csv(processed / "signature_coverage.csv", index=False)

    print("signatures", len(signatures))
    print("TCGA scores rows", len(tcga_scores), "signatures scored", tcga_scores["pmid"].nunique())
    print("GSE scores rows", len(gse_scores), "signatures scored", gse_scores["pmid"].nunique())
    print("coverage min genes", pd.DataFrame(coverage).iloc[:, 2:].min().to_dict())


if __name__ == "__main__":
    main()
