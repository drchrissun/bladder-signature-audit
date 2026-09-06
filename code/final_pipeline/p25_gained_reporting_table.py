from pathlib import Path

import pandas as pd


OUT = Path("data/processed/p01_full")
WORKBOOK = Path(
    "supplementary/Supplementary_Tables_S1_S72_UroOnc_revised.xlsx"
)

ROWS = [
    {
        "pmid": "30846479",
        "short_title": "Metabolic multi-omics signature",
        "evidence_source": "Extracted full-text window",
        "reported_main_model_adjustment": "Sex, age, and tumor stage",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The extracted text states that multivariable models controlled for sex, age, and tumor stage; no immune, stromal, tumor-fraction, or normal-like composition covariate was stated.",
    },
    {
        "pmid": "34329197",
        "short_title": "Immune-related gene signature",
        "evidence_source": "Public abstract",
        "reported_main_model_adjustment": "Not explicitly stated",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The abstract reports correlations between risk score and immune-cell infiltration but does not state that immune infiltration was included as a covariate in the main multivariable Cox model.",
    },
    {
        "pmid": "37968767",
        "short_title": "Sialyltransferase-related signature",
        "evidence_source": "Public abstract",
        "reported_main_model_adjustment": "Not explicitly stated",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The abstract reports more abundant immune infiltration and higher immune-checkpoint expression in the high-risk group but does not state that these compositional measures entered the reported prognostic model as covariates.",
    },
    {
        "pmid": "38356551",
        "short_title": "Disulfidptosis and tumor microenvironment signature",
        "evidence_source": "Public abstract",
        "reported_main_model_adjustment": "Not explicitly stated",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The abstract reports immune-cell differences between risk groups but does not state that immune-cell fractions were included as covariates in the multivariable prognostic model.",
    },
    {
        "pmid": "40319103",
        "short_title": "Fibroblast signature and T-cell infiltration",
        "evidence_source": "Public abstract",
        "reported_main_model_adjustment": "Not explicitly stated",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The abstract reports that activated CAF signatures correlate with stage and survival but does not state that a fibroblast or bulk-composition covariate was included in a multivariable survival model.",
    },
    {
        "pmid": "41533176",
        "short_title": "Basement-membrane/metastasis signature",
        "evidence_source": "Public abstract",
        "reported_main_model_adjustment": "Not explicitly stated",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The methods describe CIBERSORT and ssGSEA immune-profile estimation and report immune differences by risk group, but the abstract does not state that these profiles entered the main multivariable model as covariates.",
    },
    {
        "pmid": "41727459",
        "short_title": "Calcium-signaling signature",
        "evidence_source": "Public abstract",
        "reported_main_model_adjustment": "Univariate and multivariable Cox described; covariate list not fully stated",
        "compositional_adjustment_stated": "No",
        "public_evidence": "The abstract reports multivariable Cox analysis and examines immune-cell infiltration and TMB, but it does not state that immune infiltration or other compositional scores were included as covariates in the main model.",
    },
]


def main():
    table = pd.DataFrame(ROWS)
    table.to_csv(OUT / "p25_gained_reporting_table.csv", index=False)
    with pd.ExcelWriter(
        WORKBOOK,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        table.to_excel(writer, sheet_name="S72_gained_adjustment", index=False)
    print(table.to_string(index=False))
    print("Appended S72_gained_adjustment.")


if __name__ == "__main__":
    main()
