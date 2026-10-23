library(MASS)
library(tidyverse)
library(readxl)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Setup the Environment ========================================================
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Automatically sets the directory to where this active script is saved
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# custom functions
customfunc <- list.files(
  path = paste0(dirname(getwd()), "/Custom Functions/"), 
  pattern = ".R")
map(paste0(dirname(getwd()), "/Custom Functions/", customfunc), source)
rm(customfunc)

# load data
essay <- read_xlsx("essay.xlsx")

dictionary <- read_xlsx(paste0(dirname(getwd()),
                               '/Korean Self-Conscious Emotion Dictionary.xlsx'))
dict <- list(
  shame = paste(na.omit(dictionary$수치심),  collapse = "|"),
  guilt = paste(na.omit(dictionary$죄책감),  collapse = "|"),
  auth = paste(na.omit(dictionary$자부심),  collapse = "|"),
  hubris = paste(na.omit(dictionary$오만),  collapse = "|")
)

essay <- essay %>% 
  mutate(
    shame  = str_count(pp_text, dict$shame),
    guilt  = str_count(pp_text, dict$guilt),
    auth  = str_count(pp_text, dict$auth),
    hubris = str_count(pp_text, dict$hubris),
    og_wordcount = str_count(og_text, "\\w+"),
    pp_wordcount = str_count(pp_text, "\\w+")
  )
rm(dict, dictionary)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Descriptives =================================================================
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# age, education, sses
essay %>% 
  showDesc() %>% 
  t()
#sex
essay$sex %>% table()
# groups
essay$group %>% table()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Analyses =====================================================================
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

essay$group <- essay$group %>% factor(levels = c("Shame", "Guilt", "Auth", "Hubris"))

#tot
showResult("Shame", essay)
showResult("Guilt", essay)
showResult("Auth", essay)
showResult("Hubris", essay)

#use poisson for hubris
essay$group <- relevel(essay$group, ref = "Hubris")
model <- glm(hubris ~ group + offset(log(pp_wordcount)),
             essay,
             family = poisson) 

model %>% summary()
abs(model$coefficients) %>% exp() %>% round(2)


#plotting
essay_mean <- essay %>% 
  mutate(across(c(shame, guilt, auth, hubris), ~ ./pp_wordcount)) %>%
  dplyr::select(shame:hubris, group) %>% 
  group_by(group) %>% 
  summarize_all(~mean(.,na.rm = T)*100) %>% 
  pivot_longer(cols = shame:hubris,
               names_to = "emotion",
               values_to = "mean") %>% 
  mutate(emotion = factor(emotion, 
                          levels = c("shame", "guilt", "auth", "hubris"),
                          labels = c("Shame words", "Guilt words", "Authentic pride words", "Hubristic pride words")),
         group = factor(group,
                        levels = c("Shame", "Guilt", "Auth", "Hubris"),
                        labels = c("Shame\nessay", "Guilt\nessay", "Authentic\npride\nessay", "Hubristic\npride\nessay")))


plot <- essay %>%
  mutate(across(c(shame, guilt, auth, hubris), ~ ./pp_wordcount)) %>%
  dplyr::select(shame:hubris, group) %>% 
  pivot_longer(cols = shame:hubris,
               names_to = "emotion",
               values_to = "freq") %>% 
  mutate(emotion = factor(emotion, 
                          levels = c("shame", "guilt", "auth", "hubris"),
                          labels = c("Shame words", "Guilt words", "Authentic pride words", "Hubristic pride words")),
         group = factor(group,
                        levels = c("Shame", "Guilt", "Auth", "Hubris"),
                        labels = c("Shame\nessay", "Guilt\nessay", "Authentic\npride\nessay", "Hubristic\npride\nessay"))) %>% 
  ggplot(aes(group, freq, fill = group)) +
  geom_text(data = essay_mean, aes(group, mean/100+0.01, label = sprintf(fmt = "%.2f%%", mean)), size = 5) + 
  stat_summary(geom = "bar", fun = mean, col = "black") +
  stat_summary(geom = "errorbar", fun.data = mean_cl_normal, width = .2) +
  facet_wrap(~emotion, scales = "free_x") + 
  jtools::theme_apa(x.font.size = 20,
                    y.font.size = 20,
                    facet.title.size = 20) + 
  scale_y_continuous(breaks = seq(0, 0.08, 0.01),
                     labels = scales::percent_format()) +
  scale_fill_manual(values = c("#D55E00", "#009E73", "#0072B2", "#F0E442")) +
  labs(x = NULL, y = "Percentage appeared") + 
  theme(legend.position = "none",
        text = element_text(size = 20))
plot
ggsave("essay.png", plot,
       units = "in", height = 8, width = 10)


#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Comparisons =====================================================================
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

## tibble for the ground truth -------------------------------------------------
truth <- essay %>% 
  select(id, og_text, group) %>% 
  mutate(group = tolower(group)) %>% 
  mutate(truth_shame = ifelse(group == "shame", 1, 0),
         truth_guilt = ifelse(group == "guilt", 1, 0),
         truth_auth = ifelse(group == "auth", 1, 0),
         truth_hubris = ifelse(group == "hubris", 1, 0))

# save the ground trugh file for BERT and GPT
truth %>% 
  mutate(pred_shame = "",
         pred_guilt = "",
         pred_auth = "",
         pred_hubris = "") %>% 
  write_excel_csv(., "essay_truth.csv")

# save examples for few-shot GPT: exclude these for analyzing few-shot GPT
set.seed(20261001)
examples <- truth %>% 
  group_by(group) %>% 
  slice_sample(n = 1) %>% 
  ungroup()

write_excel_csv(examples, "examples.csv")

## KSED ------------------------------------------------------------------------
ksed <- essay %>%
  mutate(
    pred_shame = as.integer(
      shame > 0 & shame > guilt & shame > auth & shame > hubris
    ),
    pred_guilt = as.integer(
      guilt > 0 & guilt > shame & guilt > auth & guilt > hubris
    ),
    pred_auth = as.integer(
      auth > 0 & auth > shame & auth > guilt & auth > hubris
    ),
    pred_hubris = as.integer(
      hubris > 0 & hubris > shame & hubris > guilt & hubris > auth
    )
  ) %>%
  select(id, og_text, pred_shame:pred_hubris)

## BERT ------------------------------------------------------------------------
kcbert <- read_csv('Validation 1 - ESSAY OOF/bert_kcbert_oof_predictions.csv')
kluebert <- read_csv('Validation 1 - ESSAY OOF/bert_klue-bert_oof_predictions.csv')

## GPT ------------------------------------------------------------------------
zeroshot <- read_csv('essay_GPT_zeroshot.csv')
fewshot <- read_csv('essay_GPT_fewshot.csv')

# Exclude ground truth examples
fewshot <- fewshot %>% 
  filter(!id %in% examples$id)


## Comparison ------------------------------------------------------------------------
bind_cols(
  model = c("KSED", "KCBERT", "KLUEBERT", "GPTZERO", "GPTFEW"),
  bind_rows(
    evalMultilabel(truth, ksed)$overall,
    evalMultilabel(truth, kcbert)$overall,
    evalMultilabel(truth, kluebert)$overall,
    evalMultilabel(truth, zeroshot)$overall,
    evalMultilabel(truth, fewshot)$overall
  )
)

bind_cols(
  model = rep(c("KSED", "KCBERT","KLUEBERT", "GPTZERO", "GPTFEW"), each = 4),
  bind_rows(
    evalMultilabel(truth, ksed)$per_label,
    evalMultilabel(truth, kcbert)$per_label,
    evalMultilabel(truth, kluebert)$per_label,
    evalMultilabel(truth, zeroshot)$per_label,
    evalMultilabel(truth, fewshot)$per_label
  )
) %>% print(n = Inf)

# Co-occurence
cooc <- left_join(truth, ksed, by = "id") %>%
  select(starts_with("truth"), starts_with("pred")) %>%
  pivot_longer(starts_with("truth"), names_to = "true_emo",
               names_prefix = "truth_", values_to = "t") %>%
  pivot_longer(starts_with("pred"), names_to = "pred_emo",
               names_prefix = "pred_", values_to = "p") %>%
  group_by(true_emo, pred_emo) %>%
  summarise(n = sum(t == 1 & p == 1, na.rm = TRUE),
            prop = n / sum(t == 1, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(across(.cols = c(true_emo, pred_emo),
                .fns = ~factor(., 
                               levels = c("shame", "guilt", "auth", "hubris"),
                               labels = c("Shame", "Guilt", "Authentic\nPride", "Hubristic\nPride")))) %>% 
  ggplot(aes(x = pred_emo, y = true_emo, fill = prop)) +
  geom_tile(color = "white", linewidth = 1.2) +
  geom_text(aes(label = paste0(n, "\n(", scales::percent(prop, accuracy = 1), ")")),
            size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue", limits = c(0, 1),
                      labels = scales::percent, name = "Predicted | True (%)\n") +
  scale_y_discrete(limits = rev) +
  coord_equal() +
  labs(x = "Predicted emotion", y = "True emotion") +
  theme_minimal(base_size = 13) +
  theme(panel.grid = element_blank())

ggsave("cooc.png", cooc, width = 8, height = 4)

