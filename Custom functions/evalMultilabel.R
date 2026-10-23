evalMultilabel <- function(truth, preds, 
                            labs = c("shame", "guilt", "auth", "hubris"),
                            prefix = "pred_") {
  
  d <- truth %>%
    select(id, starts_with("truth_")) %>%
    inner_join(preds %>% select(id, starts_with(prefix)), by = "id") %>%
    drop_na()
  
  if (nrow(truth) != nrow(d)) {
    warning("rows don't match! truth = ", nrow(truth), ", evaluated = ", nrow(d))
  }
  
  Y <- as.matrix(d[paste0("truth_", labs)])
  P <- as.matrix(d[paste0(prefix, labs)])
  
  tp <- colSums(Y == 1 & P == 1); fp <- colSums(Y == 0 & P == 1)
  fn <- colSums(Y == 1 & P == 0); tn <- colSums(Y == 0 & P == 0)
  
  per_label <- tibble(
    label = labs, n_rows = nrow(d),
    precision   = tp / (tp + fp), # Share of correct positive cases
    recall      = tp / (tp + fn), #
    f1          = 2 * tp / (2 * tp + fp + fn)
  ) 
  
  overall <- tibble(
    n_rows = nrow(d),
    macro_precision = mean(per_label$precision),
    macro_recall    = mean(per_label$recall),
    macro_f1        = mean(per_label$f1),
    micro_precision = sum(tp) / (sum(tp) + sum(fp)),
    micro_recall    = sum(tp) / (sum(tp) + sum(fn)),
    micro_f1        = 2 * sum(tp) / (2 * sum(tp) + sum(fp) + sum(fn)),
  )
  list(per_label = per_label, overall = overall)
}

