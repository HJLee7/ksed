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
movie <- read_xlsx("movie.xlsx")
dictionary <- read_xlsx(paste0(dirname(getwd()),
                               '/Korean Self-Conscious Emotion Dictionary.xlsx'))
dict <- list(
  shame = paste(na.omit(dictionary$수치심),  collapse = "|"),
  guilt = paste(na.omit(dictionary$죄책감),  collapse = "|"),
  auth = paste(na.omit(dictionary$자부심),  collapse = "|"),
  hubris = paste(na.omit(dictionary$오만),  collapse = "|")
)

movie <- movie %>% 
  mutate(
    shame  = str_count(pp_comment, dict$shame),
    guilt  = str_count(pp_comment, dict$guilt),
    auth  = str_count(pp_comment, dict$auth),
    hubris = str_count(pp_comment, dict$hubris),
    og_wordcount = str_count(og_comment, "\\w+"),
    pp_wordcount = str_count(pp_comment, "\\w+")
  )

movie <- movie %>%
  filter(movie_en %in% c("shame", "manchester", "idiots", "wolf"))

movie <- movie %>% distinct() %>% mutate(id = row_number()) 
rm(dictionary)

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Descriptives =================================================================
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# counts
movie %>% 
  showDesc() %>% 
  t()

movie %>% 
  group_by(group) %>% 
  count()

#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=
# Analyses =====================================================================
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=

movie$group <- movie$group %>% factor(levels = c("Shame", "Guilt", "Auth", "Hubris"))

#tot
showResult("Shame", movie)
showResult("Guilt", movie)
showResult("Auth", movie)
showResult("Hubris", movie)

#plotting
movie_mean <- movie %>% 
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
                        labels = c("Shame\nmovie", "Guilt\nmovie", "Authentic\npride\nmovie", "Hubristic\npride\nmovie")))



plot <- movie %>%
  mutate(across(c(shame, guilt, auth, hubris), ~ ./pp_wordcount)) %>%
  dplyr::select(shame:hubris, group) %>% 
  pivot_longer(cols = shame:hubris,
               names_to = "emotion",
               values_to = "freq") %>% 
  mutate(emotion = factor(emotion, 
                          levels = c("shame", "guilt", "auth", "hubris"),
                          labels = c("Shame words", "Guilt words", "Authentic pride words", "Hubristic pride words")),
         group = factor(group,
                        labels = c("Shame\nmovie", "Guilt\nmovie", "Authentic\npride\nmovie", "Hubristic\npride\nmovie"))) %>% 
  ggplot(aes(group, freq, fill = group)) +
  geom_text(data = movie_mean, aes(group, mean/100+0.005, label = sprintf(fmt = "%.2f%%", mean)), size = 5) + 
  stat_summary(geom = "bar", fun = mean, col = "black") +
  stat_summary(geom = "errorbar", fun.data = mean_cl_normal, width = .2) +
  facet_wrap(~emotion, scales = "free_x") + 
  jtools::theme_apa(x.font.size = 20,
                    y.font.size = 20,
                    facet.title.size = 20) + 
  scale_y_continuous(breaks = seq(0, 0.06, 0.01),
                     labels = scales::percent_format()) +
  scale_fill_manual(values = c("#D55E00", "#009E73", "#0072B2", "#F0E442")) +
  labs(x = NULL, y = "Percentage appeared") + 
  theme(legend.position = "none",
        text = element_text(size = 20))
plot
ggsave("movie.png", plot,
       units = "in", height = 8, width = 10)
