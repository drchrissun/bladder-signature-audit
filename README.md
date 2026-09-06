# Bladder Cancer Gene-Signature Re-Scoring Audit

This repository contains the analysis code, processed result tables, screening
records, and publication figures for a dual-screened scoping audit of published
bladder cancer mRNA gene signatures.

The audit re-scores published gene lists without their original model weights
or cutoffs and tests whether survival associations remain after adjustment for
clinical, compositional, and microenvironment covariates.

The repository does not contain the manuscript under review, downloaded
full-text PDFs, or raw protected-access patient data.

## Citation

The archived release DOI will be added here after publication:

```text
DOI: [Zenodo DOI]
```

## Public input data

The original public datasets should be obtained from their providers:

- TCGA-BLCA: Broad GDAC Firehose RNA-seq v2 RSEM data.
- GSE13507: NCBI GEO, Illumina human-6 v2.0 expression beadchip.
- GSE48075: NCBI GEO, Illumina HumanHT-12 V3.0 expression beadchip.
- GSE135337: NCBI GEO, single-cell RNA-seq data.
- HGNC symbol and alias files: HUGO Gene Nomenclature Committee.
- HALLMARK gene sets: MSigDB `h.all.v2024.1.Hs.symbols.gmt`.
- `consensusMIBC`: the R package described by Kamoun et al.

A processed gene-symbol expression matrix used for consensusMIBC annotation is
included in `data/processed/tcga_blca_expr_gene.csv.gz`; the original Firehose
archive and raw sequencing files are not included.

PubMed was searched on 23 August 2026. Final consensus screening retained 213
records, of which 203 had complete extractable gene lists. Scoring was possible
for 200 lists in TCGA-BLCA and 201 in GSE13507.

## Repository layout

- `code/final_pipeline/`: final Python and R analysis scripts.
- `data/processed/p01_full/`: final 200/201-list scoring, survival,
  classification, bootstrap, mediation input, diagnostics, and sensitivity
  result tables.
- `data/processed/`: additional public inputs needed by the released scripts,
  including clinical metadata, marker comparisons, and deconvolution inputs.
- `signature_inventory/`: screening inputs, gene extraction tables, and manual
  overrides.
- `outputs/screening/`: independent reviewer inputs, answer key, consensus
  decisions, and adjudication.
- `outputs/mediation/`: survival mediation input and effect estimates.
- `outputs/consensus/`: consensusMIBC calls and signature-level
  cross-tabulations.
- `supplementary/`: final workbook `Supplementary_Tables_S1_S72_UroOnc_revised.xlsx`.
- `supplementary/archive/`: earlier workbook versions retained for provenance.
- `figures/`: final Figure 1, Figure 2, and Figures S1-S6 in PNG and PDF.

## Main analysis order

Run scripts from the repository root with
`PYTHONPATH=code/final_pipeline`.

1. `step1_screen_revision.py`: dual-review screening preparation.
2. `p01_full_rerun.py`: full 203-list extraction, scoring, nested Cox models,
   correlations, and classification.
3. `p01_bootstrap_full.py`: 1000-resample bootstrap.
4. `p01_diagnostics_full.py`: VIF, penalty, and proportional-hazards checks.
5. `p02_parse_gse48075.py` and `p02_gse48075_validation.py`: second external
   cohort.
6. `p03_decomposition.py`: M1-to-M2 decomposition.
7. `p13_classification_sensitivity.py`: rho-threshold sensitivity.
8. `p15_mixed_covariate_diagnostics.py`: covariate scale, correlation, VIF,
   and ridge checks.
9. `p16_ocr_sensitivity.py`: OCR-derived list exclusion sensitivity.
10. `p25_gained_reporting_table.py`: reporting audit of gained lists.
11. `recompute_final_classification_stability.py`: final 1000-iteration
    classification stability.
12. `recompute_formal_deconv_final.py`: final MuSiC/BayesPrism sensitivity.
13. `run_p01_regmedint_survival.R` and `run_consensus_mibc.R`: survival
    mediation and Consensus subtype annotation.
14. `build_final_supplementary_tables.py`: regenerate the final S1-S72 workbook.
15. `p01_full_figures.py`, `prepare_revised_figures.py`, and
    `make_prisma_integrated_figure.py`: regenerate final figures.

The released result tables and figures are the exact versions used for the
manuscript. Bootstrap or model-fitting reruns may produce negligible Monte
Carlo variation.

## Python environment

Install the packages listed in `requirements.txt`:

```text
numpy
pandas
scipy
statsmodels
lifelines
scikit-learn
openpyxl
matplotlib
seaborn
```

The R analyses require R 4.x with `readr`, `regmedint`, and `consensusMIBC`.

## License

- Analysis code: MIT License (`LICENSE-CODE`).
- Data, processed tables, and figures: CC BY 4.0 (`LICENSE-DATA`).
