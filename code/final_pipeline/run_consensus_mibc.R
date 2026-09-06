if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}
if (!requireNamespace("consensusMIBC", quietly = TRUE)) {
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
  }
  remotes::install_github(
    "cit-bioinfo/consensusMIBC",
    build_vignettes = FALSE,
    upgrade = "never"
  )
}

library(consensusMIBC)
library(readr)

OUT_DIR <- "outputs/consensus"
dir.create(OUT_DIR, showWarnings = FALSE)

expr <- read.csv(
  gzfile(
    "data/processed/tcga_blca_expr_gene.csv.gz"
  ),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
rownames(expr) <- expr[["symbol"]]
expr <- expr[, setdiff(names(expr), "symbol")]

tcga_samples <- readr::read_csv(
  "data/processed/purity_and_microenvironment_estimates.csv",
  show_col_types = FALSE
)
primary_samples <- unique(
  tcga_samples$sample[
    tcga_samples$dataset == "TCGA" &
    grepl("-01A-", tcga_samples$sample, fixed = TRUE)
  ]
)
primary_samples <- intersect(primary_samples, colnames(expr))

log_expr <- log2(as.matrix(expr[, primary_samples, drop = FALSE]) + 1)
log_expr <- log_expr[rownames(log_expr) != "?", , drop = FALSE]

calls <- getConsensusClass(
  log_expr,
  minCor = 0.2,
  gene_id = "hgnc_symbol"
)
calls <- as.data.frame(calls)
calls$sample <- rownames(calls)
calls <- calls[
  ,
  c(
    "sample",
    "consensusClass",
    "cor_pval",
    "separationLevel",
    "LumP",
    "LumNS",
    "LumU",
    "Stroma-rich",
    "Ba/Sq",
    "NE-like"
  )
]

write.csv(
  calls,
  file.path(OUT_DIR, "consensus_mibc_calls.csv"),
  row.names = FALSE
)

cat("samples classified:", nrow(calls), "\n")
print(table(calls$consensusClass, useNA = "ifany"))
