from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls


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


def bulk_log_expression(frame, dataset):
    values = frame.values.astype(float)
    if dataset == "TCGA":
        return np.log1p(values)
    linear = np.expm1(values * np.log(2))
    linear[linear < 0] = 0
    return np.log1p(linear)


def deconvolve(frame, reference, dataset):
    marker_genes = sorted({gene for genes in MARKERS.values() for gene in genes} & set(frame.index) & set(reference.index))
    bulk = bulk_log_expression(frame.loc[marker_genes], dataset)
    ref = reference.loc[marker_genes].values.astype(float)
    combined = np.column_stack([ref, bulk])
    mean = combined.mean(axis=1, keepdims=True)
    std = combined.std(axis=1, keepdims=True)
    std[std == 0] = 1
    ref_std = (ref - mean) / std
    bulk_std = (bulk - mean) / std

    fractions = []
    raw = []
    for index in range(bulk_std.shape[1]):
        coef, residual = nnls(ref_std, bulk_std[:, index], maxiter=10000)
        fractions.append(coef / np.maximum(coef.sum(), 1e-9))
        raw.append(coef)
    fractions = pd.DataFrame(fractions, index=frame.columns, columns=reference.columns)
    raw = pd.DataFrame(raw, index=frame.columns, columns=reference.columns)
    return fractions, raw


def main():
    processed = Path("data/processed")
    reference = pd.read_csv(
        "data/processed/gse135337_cell_type_reference_log1p_cpm.csv.gz",
        index_col=0,
    )
    tcga = pd.read_csv("data/processed/tcga_blca_expr_gene.csv.gz", index_col=0)
    gse = pd.read_csv("data/processed/gse13507_expr_gene.csv.gz", index_col=0)

    for name, frame in [("tcga", tcga), ("gse13507", gse)]:
        frac, raw = deconvolve(frame, reference, "TCGA" if name == "tcga" else "GSE13507")
        frac.to_csv(f"data/processed/nnls_fractions_{name}.csv")
        raw.to_csv(f"data/processed/nnls_raw_{name}.csv")
        print(name, frac.describe().T[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
