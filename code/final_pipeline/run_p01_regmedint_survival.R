library(readr)
library(regmedint)

OUT_DIR <- "data/processed"
d <- as.data.frame(read_csv(
  file.path(OUT_DIR, "p01_regmedint_input.csv"),
  show_col_types = FALSE
))

pmids <- sort(unique(d$pmid))
rows <- list()

for (pmid in pmids) {
  x <- d[d$pmid == pmid, ]
  fit <- tryCatch(
    regmedint(
      data = x,
      yvar = "time_months",
      eventvar = "event",
      avar = "score_z",
      mvar = "microenv_z",
      cvar = c("age_c", "sex_female", "stage_c2"),
      a0 = 0,
      a1 = 1,
      m_cde = 0,
      c_cond = c(0, 0, 0),
      mreg = "linear",
      yreg = "survAFT_weibull",
      interaction = TRUE,
      casecontrol = FALSE
    ),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    next
  }
  mat <- summary(fit)$summary_myreg
  for (effect in rownames(mat)) {
    rows[[paste(pmid, effect, sep = "_")]] <- data.frame(
      pmid = pmid,
      effect = effect,
      estimate = mat[effect, "est"],
      se = mat[effect, "se"],
      z = mat[effect, "Z"],
      pvalue = mat[effect, "p"],
      lower = mat[effect, "lower"],
      upper = mat[effect, "upper"],
      stringsAsFactors = FALSE
    )
  }
}

result <- do.call(rbind, rows)
write.csv(
  result,
  file.path(OUT_DIR, "p01_regmedint_survival_results.csv"),
  row.names = FALSE
)
print(result[result$effect %in% c("tnie", "pnie", "pm"), ])
