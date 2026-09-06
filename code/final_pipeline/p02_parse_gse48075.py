import csv
import gzip
import io
import re
from pathlib import Path

import pandas as pd

from step6_signature_scores import load_alias_map


RAW = Path("data/raw/revision_geo/GSE48075_series_matrix_clean.txt.gz")
ANNOT = Path("data/raw/revision_geo/GPL6947.annot.gz")
OUT = Path("data/processed/p01_full")


def parse_metadata():
    with gzip.open(RAW, "rt", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    samples = None
    geo = None
    characteristics = []
    for line in lines:
        if line.startswith("!Sample_title"):
            samples = next(csv.reader(io.StringIO(line), delimiter="\t"))[1:]
        elif line.startswith("!Sample_geo_accession"):
            geo = next(csv.reader(io.StringIO(line), delimiter="\t"))[1:]
        elif line.startswith("!Sample_characteristics_ch1"):
            characteristics.append(
                next(csv.reader(io.StringIO(line), delimiter="\t"))[1:]
            )
        elif line.startswith("!series_matrix_table_begin"):
            break

    geo_to_title = dict(zip(geo, samples))
    collected = {title: [] for title in samples}
    for row in characteristics:
        if len(row) != len(samples):
            continue
        for title, value in zip(samples, row):
            if value:
                collected[title].append(value)

    rows = []
    for title, values in collected.items():
        text = "; ".join(values)
        stage_match = re.search(r"(?:cstage|pstage):\s*([^;]+)", text, re.I)
        os_match = re.search(r"os censor:\s*(censored|uncensored)", text, re.I)
        dss_match = re.search(r"dss censor:\s*(censored|uncensored)", text, re.I)
        survival_match = re.search(r"survival \(mo\):\s*([0-9.]+)", text, re.I)
        age_match = re.search(r"age \(at specimen collection\):\s*([0-9.]+)", text, re.I)
        sex_match = re.search(r"gender:\s*(male|female)", text, re.I)
        stage = stage_match.group(1).strip() if stage_match else ""
        rows.append(
            {
                "sample_title": title,
                "geo_accession": geo_to_title and [
                    key for key, value in geo_to_title.items() if value == title
                ][0],
                "stage": stage,
                "muscle_invasive": bool(re.search(r"T[234]", stage, re.I)),
                "os_event": int(os_match.group(1).lower() == "uncensored") if os_match else None,
                "dss_event": int(dss_match.group(1).lower() == "uncensored") if dss_match else None,
                "time_months": float(survival_match.group(1)) if survival_match else None,
                "age": float(age_match.group(1)) if age_match else None,
                "sex_female": int(sex_match.group(1).lower() == "female") if sex_match else None,
                "all_characteristics": text,
            }
        )
    return pd.DataFrame(rows)


def parse_expression(samples):
    expr = pd.read_csv(
        RAW,
        compression="gzip",
        sep="\t",
        skiprows=67,
        nrows=48804,
        index_col=0,
    )
    expr.index = expr.index.astype(str)
    expr = expr[list(samples.keys())]
    expr.columns = [samples[col] for col in expr.columns]

    annot = pd.read_csv(
        ANNOT,
        compression="gzip",
        sep="\t",
        skiprows=28,
        usecols=["ID", "Gene symbol"],
        na_values=[""],
    )
    annot = annot.dropna(subset=["Gene symbol"])
    alias_map = load_alias_map()
    annot["symbol"] = annot["Gene symbol"].map(lambda gene: alias_map.get(gene, gene))
    probe_map = (
        annot.drop_duplicates("ID")
        .set_index("ID")["symbol"]
        .to_dict()
    )
    expr["symbol"] = expr.index.map(lambda probe: probe_map.get(probe, ""))
    expr = expr[expr["symbol"] != ""]
    expr = expr.groupby("symbol").mean(numeric_only=True)
    return expr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = parse_metadata()
    mibc = meta[
        meta["muscle_invasive"]
        & meta["os_event"].notna()
        & meta["time_months"].notna()
    ].copy()
    samples = {
        row["geo_accession"]: row["sample_title"]
        for row in mibc.to_dict("records")
    }
    expr = parse_expression(samples)
    expr.to_csv(OUT / "gse48075_expr_gene.csv.gz", compression="gzip")
    mibc.to_csv(OUT / "gse48075_clinical.csv", index=False)
    print("meta", len(meta), "MIBC complete", len(mibc), "events", mibc.os_event.sum())
    print("stage counts", mibc.stage.value_counts().to_dict())
    print("expression", expr.shape)


if __name__ == "__main__":
    main()
