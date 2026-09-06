from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse


MARKERS = {
    "Normal_urothelial": ["UPK1A", "UPK1B", "UPK2", "UPK3A", "KRT20", "KRT18", "KRT8"],
    "Tumor_epithelial": ["KRT5", "KRT14", "TP63", "KRT17", "KRT6A", "KRT19", "KRT8", "KRT18"],
    "Fibroblast": ["DCN", "LUM", "COL1A1", "COL1A2", "PDGFRA", "FBLN1", "VIM"],
    "Myofibroblast": ["ACTA2", "TAGLN", "MYH11", "PDGFRB", "RGS5"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "FLT1", "KDR"],
    "T_cell": ["CD3D", "CD3E", "CD3G", "IL7R", "TRAC", "CD8A", "CD8B", "CD4"],
    "NK_cell": ["NKG7", "KLRD1", "KLRB1", "NCAM1", "GNLY"],
    "B_plasma": ["CD79A", "MS4A1", "CD19", "JCHAIN", "MZB1", "IGHG1"],
    "Myeloid": ["CD68", "CD163", "LYZ", "CD14", "FCGR3A", "ITGAX", "CSF1R"],
    "Mast_cell": ["TPSAB1", "TPSB2", "CPA3", "KIT", "HDC"],
}


def main():
    x = sparse.load_npz("data/processed/gse135337_scRNA_merged_umi.npz").tocsr()
    genes = pd.read_csv("data/processed/gse135337_scRNA_gene_symbols.csv")["Symbol"].tolist()
    cells = pd.read_csv("data/processed/gse135337_scRNA_cell_barcodes.csv")["cell_barcode"]
    gene_pos = {gene: idx for idx, gene in enumerate(genes)}

    totals = np.asarray(x.sum(axis=0)).ravel()
    cpm = x.multiply(1e4 / np.maximum(totals[None, :], 1)).tocsr()
    log_cpm = cpm.copy()
    log_cpm.data = np.log1p(log_cpm.data)

    cell_scores = {}
    for cell_type, marker_list in MARKERS.items():
        indices = [gene_pos[gene] for gene in marker_list if gene in gene_pos]
        if not indices:
            cell_scores[cell_type] = np.zeros(x.shape[1])
            continue
        subset = log_cpm[indices, :]
        cell_scores[cell_type] = np.asarray(subset.mean(axis=0)).ravel()

    score_frame = pd.DataFrame(cell_scores, index=cells)
    assigned = score_frame.idxmax(axis=1)
    max_score = score_frame.max(axis=1)
    assigned = assigned.where(max_score >= 0.05, "Low_quality")

    sample = cells.str.split("_", n=1).str[0]
    annotation = pd.DataFrame(
        {
            "cell_barcode": cells.to_numpy(),
            "sample": sample.to_numpy(),
            "assigned_cell_type": assigned.to_numpy(),
            "max_marker_score": max_score.to_numpy(),
        }
    )
    annotation.to_csv("data/processed/gse135337_cell_annotations.csv", index=False)

    reference = log_cpm @ sparse.csc_matrix(
        (np.ones(x.shape[1]), (range(x.shape[1]), assigned.astype("category").cat.codes)),
        shape=(x.shape[1], assigned.nunique()),
    )
    reference = reference.toarray() / np.maximum(
        np.bincount(assigned.astype("category").cat.codes), 1
    )[None, :]
    reference_frame = pd.DataFrame(
        reference,
        index=genes,
        columns=sorted(assigned.unique()),
    )
    reference_frame.index.name = "Symbol"
    reference_frame.columns.name = "cell_type"
    reference_frame.to_csv("data/processed/gse135337_cell_type_reference_log1p_cpm.csv.gz", compression="gzip")

    print(annotation["assigned_cell_type"].value_counts().to_string())
    print("reference shape", reference_frame.shape)


if __name__ == "__main__":
    main()
