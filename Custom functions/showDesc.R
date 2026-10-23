showDesc <- function(df, group = NULL) {
  if (is.null(group)) {
    df <- df %>% summarise(across(where(is.numeric), 
                                  ~sprintf("%.2f (%.2f)", 
                                           mean(., na.rm = TRUE), 
                                           sd(., na.rm = TRUE)
                                  )
    )
    )
  } else {
    df <- df %>% 
      group_by(across(all_of(group))) %>% 
      summarise(across(where(is.numeric), 
                       ~sprintf("%.2f (%.2f)", 
                                mean(., na.rm = TRUE), 
                                sd(., na.rm = TRUE)
                       )
      )
      )
  }
  return(df)
}
