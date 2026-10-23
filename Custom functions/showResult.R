showResult <- function(dv, df){
  frm = paste0(tolower(dv), "~ group + offset(log(pp_wordcount))")
  
  #poisson regression
  df[["group"]] <- relevel(df[["group"]], ref = dv)
  model.ps <- glm(formula = as.formula(frm),
                  data = df,
                  family = poisson)
  
  #use negative binomial regression
  #(vs poisson, accounts for overdispersion cuz poisson assumes mean = variance)
  #(vs quasi-poisson, smaller values are given more weight)
  #variance = mean + (mean^2)/k
  
  model.nb <- glm.nb(formula = as.formula(frm),
                     data = df)
  
  pchisq(model.nb$null.deviance-model.nb$deviance, 
         model.nb$df.null-model.nb$df.residual, lower.tail = F)
  
  #model comparisons===
  #goodness of fit
  
  #model comparison - boundary likelihood test (Hilbe, 2011), if sig, larger model's better
  chi <- pchisq(
    -2*(as.numeric(logLik(model.ps)) - as.numeric(logLik(model.nb))),
    1,lower.tail = F)/2
  
  #compare the models
  aic <- AIC(model.ps,model.nb) %>% rownames_to_column("model")
  bic <- BIC(model.ps,model.nb) %>% rownames_to_column("model")
  
  #model comparisons tot
  mcomps <- bind_cols(merge(aic, bic, sort = F),
                      tribble(~chi,
                              (-2*(as.numeric(logLik(model.ps)) - as.numeric(logLik(model.nb)))) %>% 
                                round(2) %>% as.character() %>% paste0("x^2 = ",.), 
                                       sprintf(chi,fmt = "%.3f")%>% paste0("p = ",. )))
  
  #model summary
  sum <- summary(model.nb)
  
  print(list(`model summary` = sum,
             `irrs` = abs(model.nb$coefficients) %>% exp() %>% .[2:4] %>% round(2),
             `model comparisons` = mcomps))
}



