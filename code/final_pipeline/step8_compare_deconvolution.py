from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED = Path("data/processed")


def load_frame(name):
    return pd.read_csv(PROCESSED / name, index_col=0)


def agreement(left, right, common, cell_types):
    left = left.loc[common, cell_types]
    right = right.loc[common, cell_types]
    records = []
    for cell_type in cell_types:
        x = left[cell_type]
        y = right[cell_type]
        records.append(
            {
                "mean_absolute_difference": float((x - y).abs().mean()),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "spearman_rho": float(x.rank().corr(y.rank())),
            }
        )
    return pd.DataFrame(records, index=cell_types)


def main():
    bayesprism = load_frame("r_bayesprism_theta.csv")
    music = load_frame("r_music_weighted_fractions.csv")
    nnls = load_frame("deconvolution_fractions_tcga.csv")

    samples = bayesprism.index.intersection(music.index).intersection(nnls.index)
    cell_types = [
        cell_type
        for cell_type in bayesprism.columns
        if cell_type in music.columns and cell_type in nnls.columns
    ]

    music_agreement = agreement(bayesprism, music, samples, cell_types).add_prefix(
        "bp_music_"
    )
    nnls_agreement = agreement(bayesprism, nnls, samples, cell_types).add_prefix(
        "bp_nnls_"
    )
    music_nnls_agreement = agreement(
        music, nnls, samples, cell_types
    ).add_prefix("music_nnls_")

    means = pd.DataFrame(
        {
            "bayesprism_mean": bayesprism.loc[samples, cell_types].mean(),
            "music_mean": music.loc[samples, cell_types].mean(),
            "nnls_mean": nnls.loc[samples, cell_types].mean(),
        }
    )

    summary = pd.concat(
        [music_agreement, nnls_agreement, music_nnls_agreement, means],
        axis=1,
    )
    summary.index.name = "cell_type"
    summary.to_csv(PROCESSED / "deconvolution_method_agreement_tcga.csv")

    print(f"Samples compared: {len(samples)}")
    print(f"Cell types compared: {len(cell_types)}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
