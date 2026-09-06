import gzip
from pathlib import Path

import pandas as pd
import scipy.sparse as sparse


def read_matrix(path, sample_name):
    frame = pd.read_csv(path, sep="\t", index_col=1, low_memory=False)
    frame = frame.drop(columns=["Gene_ID"], errors="ignore")
    frame = frame[~frame.index.duplicated(keep="first")]
    frame.index.name = "Symbol"
    frame.columns = [f"{sample_name}_{cell}" for cell in frame.columns]
    return frame


def main():
    files = {
        path.name: path
        for path in sorted(Path("data/raw/GSE135337_files").glob("*.gz"))
    }
    frames = []
    shapes = []
    all_genes = set()
    for filename, path in files.items():
        sample_name = filename.split("_")[0]
        frame = read_matrix(path, sample_name)
        shapes.append((sample_name, frame.shape))
        frames.append(frame)
        all_genes.update(frame.index)
        print(sample_name, frame.shape, flush=True)

    genes = sorted(all_genes)
    sparse_frames = []
    for frame in frames:
        aligned = frame.reindex(genes, fill_value=0)
        sparse_frames.append(sparse.csr_matrix(aligned.values.astype("float32")))

    merged = sparse.hstack(sparse_frames, format="csr")
    sparse.save_npz("data/processed/gse135337_scRNA_merged_umi.npz", merged)
    pd.Series(genes, name="Symbol").to_csv(
        "data/processed/gse135337_scRNA_gene_symbols.csv",
        index=False,
    )
    pd.Series(
        [col for frame in frames for col in frame.columns],
        name="cell_barcode",
    ).to_csv("data/processed/gse135337_scRNA_cell_barcodes.csv", index=False)

    shape_frame = pd.DataFrame(shapes, columns=["sample", "shape"])
    shape_frame[["genes", "cells"]] = pd.DataFrame(shape_frame["shape"].tolist(), index=shape_frame.index)
    shape_frame.drop(columns=["shape"]).to_csv("data/processed/gse135337_sample_shapes.csv", index=False)

    print("merged shape", merged.shape)
    print("total cells", merged.shape[1])
    print("stored elements", merged.nnz)


if __name__ == "__main__":
    main()
