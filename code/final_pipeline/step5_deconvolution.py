from pathlib import Path

import numpy as np
import pandas as pd


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

ESTIMATE_STROMAL = ["ACTA2", "COL1A1", "COL1A2", "DCN", "LUM", "VIM", "FN1", "MMP2", "SPARC", "TAGLN"]
ESTIMATE_IMMUNE = ["CD3D", "CD3E", "CD8A", "CD79A", "MS4A1", "CD68", "CD163", "LYZ", "NKG7", "GZMA"]


def comparable_matrix(frame, dataset):
    values = frame.values.astype(float)
    if dataset == "TCGA":
        return np.log1p(values)
    # GSE13507 values are log2; approximate linear CPM then log1p.
    linear = np.expm1(values * np.log(2))
    linear[linear < 0] = 0
    return np.log1p(linear)


def deconvolve_scores(frame, dataset):
    matrix = comparable_matrix(frame, dataset)
    genes = frame.index.tolist()
    pos = {gene: idx for idx, gene in enumerate(genes)}
    scores = {}
    for cell_type, marker_list in MARKERS.items():
        indices = [pos[gene] for gene in marker_list if gene in pos]
        if not indices:
            scores[cell_type] = np.zeros(matrix.shape[1])
            continue
        scores[cell_type] = matrix[indices, :].mean(axis=0)
    score_frame = pd.DataFrame(scores, index=frame.columns)
    fractions = score_frame.div(score_frame.sum(axis=1), axis=0)
    return score_frame, fractions


def estimate_scores(frame, dataset):
    matrix = comparable_matrix(frame, dataset)
    genes = frame.index.tolist()
    pos = {gene: idx for idx, gene in enumerate(genes)}
    out = {}
    for name, marker_list in [("stromal", ESTIMATE_STROMAL), ("immune", ESTIMATE_IMMUNE)]:
        indices = [pos[gene] for gene in marker_list if gene in pos]
        out[name] = matrix[indices, :].mean(axis=0) if indices else np.zeros(matrix.shape[1])
    return pd.DataFrame(out, index=frame.columns)


def main():
    processed = Path("data/processed")
    tcga = pd.read_csv("data/processed/tcga_blca_expr_gene.csv.gz", index_col=0)
    gse = pd.read_csv("data/processed/gse13507_expr_gene.csv.gz", index_col=0)

    tcga_scores, tcga_frac = deconvolve_scores(tcga, "TCGA")
    gse_scores, gse_frac = deconvolve_scores(gse, "GSE13507")
    tcga_est = estimate_scores(tcga, "TCGA")
    gse_est = estimate_scores(gse, "GSE13507")

    tcga_scores.to_csv(processed / "deconvolution_scores_tcga.csv")
    tcga_frac.to_csv(processed / "deconvolution_fractions_tcga.csv")
    gse_scores.to_csv(processed / "deconvolution_scores_gse13507.csv")
    gse_frac.to_csv(processed / "deconvolution_fractions_gse13507.csv")

    purity = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": "TCGA",
                    "sample": tcga.columns,
                    "tumor_epithelial_fraction": tcga_frac["Tumor_epithelial"].values,
                    "normal_urothelial_fraction": tcga_frac["Normal_urothelial"].values,
                    "immune_score": tcga_est["immune"].values,
                    "stromal_score": tcga_est["stromal"].values,
                }
            ),
            pd.DataFrame(
                {
                    "dataset": "GSE13507",
                    "sample": gse.columns,
                    "tumor_epithelial_fraction": gse_frac["Tumor_epithelial"].values,
                    "normal_urothelial_fraction": gse_frac["Normal_urothelial"].values,
                    "immune_score": gse_est["immune"].values,
                    "stromal_score": gse_est["stromal"].values,
                }
            ),
        ],
        ignore_index=True,
    )
    purity.to_csv(processed / "purity_and_microenvironment_estimates.csv", index=False)

    print("TCGA fractions\n", tcga_frac.describe().T[["mean", "min", "max"]].to_string())
    print("GSE fractions\n", gse_frac.describe().T[["mean", "min", "max"]].to_string())
    print("purity rows", len(purity))


if __name__ == "__main__":
    main()
