import csv
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent
INPUT = BASE / "signature_inventory" / "primary_candidates.csv"
OUT_DIR = BASE / "signature_inventory" / "screen_revision"

PUBLICATION_TYPES = re.compile(
    r"\b(?:review|meta-analysis|systematic review|letter|comment|editorial|"
    r"retracted|retraction of publication|published erratum)\b",
    re.IGNORECASE,
)

BLADDER = re.compile(
    r"bladder cancer|bladder urothelial|bladder carcinoma|bladder tumor|"
    r"urinary bladder|urothelial cancer|urothelial carcinoma|"
    r"carcinoma of the bladder|\bBLCA\b|\bMIBC\b|\bNMIBC\b|\bUTUC\b",
    re.IGNORECASE,
)

MODEL = re.compile(
    r"\bgene signature\b|\bprognostic signature\b|\bpredictive signature\b|"
    r"\bexpression signature\b|\bmRNA signature\b|\btranscriptomic signature\b|"
    r"\brisk model\b|\brisk score\b|\brisk signature\b|"
    r"\bprognostic model\b|\bprognostic index\b|\bpredictive model\b|"
    r"\bgene panel\b|\bmulti[- ]?gene\b|\bmultigene\b|"
    r"\bmolecular subtype\b|\bmolecular classifier\b|"
    r"\bgene[- ]based score\b|"
    r"\b\d+\s*[-–]?\s*(?:mRNA|gene)(?:s)?\b",
    re.IGNORECASE,
)

ENDPOINT = re.compile(
    r"overall survival|progression[- ]free survival|disease[- ]free survival|"
    r"recurrence[- ]free survival|cancer[- ]specific survival|"
    r"\bprognos\w*|patient survival|\brecurrence\b|"
    r"disease progression|tumor progression|tumour progression|"
    r"immunotherapy response|response to immunotherapy|"
    r"chemotherapy response|treatment response|clinical outcome|"
    r"\bresponder\w*",
    re.IGNORECASE,
)

NON_MRNA_TITLE = re.compile(
    r"\blncrnas?\b|long non[- ]?coding rna|\bcircrnas?\b|"
    r"\bmirnas?\b|\bmicroRNAs?\b|alternative splicing|"
    r"radiomics|pathomics|whole[- ]slide|"
    r"proteomic signature|metabolomic signature|serum metabolite|"
    r"\bpiRNA\b",
    re.IGNORECASE,
)

METHYLATION_MODEL = re.compile(
    r"methylation site signature|methylation[- ]based signature|"
    r"DNA methylation signature|methylation signature",
    re.IGNORECASE,
)

IMAGING = re.compile(
    r"radiomics|magnetic resonance|\bMRI\b|computed tomography|\bCT\b|"
    r"pathomics|histopatholog|whole[- ]slide|deep learning",
    re.IGNORECASE,
)

SINGLE_GENE_MECHANISM = re.compile(
    r"\b(?:promotes|stabilizes|inhibits|suppresses|mediates|drives|"
    r"associated with prognosis|targeting|exhibits potential|"
    r"identified .{0,40} as .{0,40} biomarker)\b",
    re.IGNORECASE,
)


def classify(row):
    title = row["title"]
    abstract = row["abstract"]
    pubtypes = row["pubtypes"]
    text = f"{title} {abstract}"
    reasons = []

    if PUBLICATION_TYPES.search(pubtypes):
        reasons.append("publication_type")

    if not BLADDER.search(title):
        reasons.append("not_bladder")

    if NON_MRNA_TITLE.search(title):
        reasons.append("non_mrna_feature")

    if METHYLATION_MODEL.search(text):
        reasons.append("methylation_model")

    if IMAGING.search(title):
        reasons.append("imaging_or_pathology")

    if not MODEL.search(text):
        reasons.append("not_signature_model")

    if not ENDPOINT.search(text):
        reasons.append("no_clinical_endpoint")

    if SINGLE_GENE_MECHANISM.search(title) and not MODEL.search(title):
        reasons.append("single_gene_mechanistic")

    return reasons


def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    eligible = []
    excluded = []
    for row in rows:
        reasons = classify(row)
        row["revision_screening_reasons"] = "; ".join(reasons)
        (excluded if reasons else eligible).append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for name, records in [
        ("screened_revision_eligible.csv", eligible),
        ("screened_revision_excluded.csv", excluded),
    ]:
        with (OUT_DIR / name).open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"eligible {len(eligible)} / {len(rows)}")
    print(f"excluded {len(excluded)}")
    for reason in [
        "publication_type",
        "not_bladder",
        "non_mrna_feature",
        "methylation_model",
        "imaging_or_pathology",
        "not_signature_model",
        "no_clinical_endpoint",
        "single_gene_mechanistic",
    ]:
        count = sum(
            1
            for row in excluded
            if reason in row["revision_screening_reasons"]
        )
        print(reason, count)


if __name__ == "__main__":
    main()
