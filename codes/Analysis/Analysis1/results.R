# setwd("~/Desktop/PgdThesis/codes/Analysis/Analysis1")
setwd("D:/Projects/PgdThesis/codes/Analysis/Analysis1")

libs <- c("dplyr", "magrittr", "ggplot2", "readr", "caret", "tidyr", "GGally",
          "survival", "data.table",  "plotly", "lubridate", "tidyverse")

load_and_install <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages)) install.packages(new_packages)
  lapply(packages, library, character.only = TRUE)
}
load_and_install(libs)


################################################################################

parse_metrics_to_df <- function(metrics_string, model_label) {
  # 1. Split by pipe to isolate each metric pair
  pairs <- unlist(strsplit(metrics_string, "\\|"))
  
  # 2. Extract names (before colon) and values (numbers)
  metric_names <- str_trim(sub(":.*", "", pairs))
  # Standardize names immediately to prevent merge mismatches
  metric_names <- case_when(
    metric_names == "Balanced Acc" ~ "Balanced_Acc",
    metric_names == "Log Loss"     ~ "Log_Loss",
    metric_names == "F1"           ~ "F1",
    TRUE ~ metric_names
  )
  
  numeric_values <- as.numeric(str_extract(pairs, "\\d+\\.\\d+"))
  
  # 3. Create Long Dataframe
  metrics_df_long <- data.frame(
    Metric = metric_names,
    Value = numeric_values
  )
  
  # Rename Value column to the model label
  colnames(metrics_df_long)[2] <- model_label
  
  return(metrics_df_long)
}



plot_classification_metrics <- function(df, model_name) {
  
  # 1. Reshape the data from wide to long format
  # Check if column name uses dash or dot (R often converts '-' to '.')
  data_long <- df %>%
    pivot_longer(
      cols = any_of(c("Precision", "Recall", "F1-Score")), 
      names_to = "Metric", 
      values_to = "Value"
    )%>%
  arrange(User)
  
  # 2. Create the grouped line graph
  plot <- ggplot(data_long, aes(x = User, y = Value, color = Metric, group = Metric)) +
    geom_line(linewidth = 1) +  # 'size' is deprecated in newer ggplot2 versions, use 'linewidth'
    geom_point(size = 2) +
    theme_minimal() +
    labs(
      title = paste("Classification Performance of The", model_name, "Model"),
      subtitle = "Comparison of Precision, Recall, and F1-Score",
      x = "User",
      y = "Score Value"
    ) +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
      legend.position = "bottom"
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_brewer(palette = "Set1")
  
  return(plot)
}
################################################################################
Analysis1_Cycks_with_padding <- read_csv("Results3/Cycks_With_Padding.csv")
Analysis1_raw_input_1 <- "Accuracy: 0.0298 | Balanced Acc: 0.0308 | Log Loss: 3.8865 | Precision: 0.0202 | Recall: 0.0298 | F1: 0.0156 | ROC-AUC: 0.5120"

Analysis1_df1 <- parse_metrics_to_df(Analysis1_raw_input_1,
                                     "Analysis1_Cycks_with_padding")
Analysis1_p1 <- plot_classification_metrics(Analysis1_Cycks_with_padding,
                                            "Analysis1_Cycks_with_padding")
Analysis1_p1


################################################################################
Analysis1_Cycks_without_padding <- read_csv("Results3/Cycks_Without_Padding.csv")
#View(Cycks_without_padding)
Analysis1_raw_input_2 <- "Accuracy: 0.9135 | Balanced Acc: 0.8947 | Log Loss: 0.3991 | Precision: 0.9223 | Recall: 0.9135 | F1: 0.9110 | ROC-AUC: 0.9988"
# Using your existing function
Analysis1_df2 <- parse_metrics_to_df(Analysis1_raw_input_2,
                                     "Analysis1_Cycks_without_padding")

Analysis1_p2 <- plot_classification_metrics(Analysis1_Cycks_without_padding,
                                            "Analysis1_Cycks_without_padding")
Analysis1_p2


################################################################################
Analysis1_Vgg_without_padding <- read_csv("Results3/Vgg_without_padding.csv")

Analysis1_raw_input_3 <- "Accuracy: 0.9511 | Balanced Acc: 0.9456 | Log Loss: 0.2339 | Precision: 0.9528 | Recall: 0.9511 | F1: 0.9512 | ROC-AUC: 0.9992"


# Using your existing function
Analysis1_df3 <- parse_metrics_to_df(Analysis1_raw_input_3,
                                     "Analysis1_Vgg_without_padding")
print(Analysis1_df3)
Analysis1_p3 <- plot_classification_metrics(Analysis1_Vgg_without_padding,
                                            "Analysis1_Vgg_without_padding")
Analysis1_p3

################################################################################
Analysis1_Vgg_with_padding <- read_csv("Results3/Vgg_with_padding.csv")
# View(Vgg_with_padding)

Analysis1_raw_input_4 <- "Accuracy: 0.9578 | Balanced Acc: 0.9518 | Log Loss: 0.2025 | Precision: 0.9592 | Recall: 0.9578 | F1: 0.9579 | ROC-AUC: 0.9994"


# Using your existing function
Analysis1_df4 <- parse_metrics_to_df(Analysis1_raw_input_4,
                                     "Analysis1_Vgg_with_padding")
print(Analysis1_df4)
Analysis1_p4 <- plot_classification_metrics(Analysis1_Vgg_without_padding,
                                            "Analysis1_Vgg_with_padding")
Analysis1_p4
################################################################################
Analysis1_red_cycks_without_padding <- read_csv("Results3/red_cycks_without_padding.csv")
# View(red_cycks_without_padding)

Analysis1_raw_input_5 <- "Accuracy: 0.9578 | Balanced Acc: 0.9518 | Log Loss: 0.2025 |Precision: 0.9592 | Recall: 0.9578 | F1: 0.9579 | ROC-AUC: 0.9994"


# Using your existing function
Analysis1_df5 <- parse_metrics_to_df(Analysis1_raw_input_5,
                                     "Analysis1_red_cycks_without_padding")
print(Analysis1_df5)
Analysis1_p5 <- plot_classification_metrics(Analysis1_red_cycks_without_padding,
                                            "Analysis1_red_cycks_without_padding")
Analysis1_p5
################################################################################
Analysis1_red_cycks_with_padding <- read_csv("Results3/red_cycks_with_padding.csv")

Analysis1_raw_input_6 <- "Accuracy: 0.9726 | Balanced Acc: 0.9607 | Log Loss: 0.0808 | Precision: 0.9744 | Recall: 0.9726 | F1: 0.9724 | ROC-AUC: 0.9999"


# Using your existing function
Analysis1_df6 <- parse_metrics_to_df(Analysis1_raw_input_6,
                                     "Analysis1_red_cycks_with_padding")
print(Analysis1_df6)
Analysis1_p6 <- plot_classification_metrics(Analysis1_red_cycks_with_padding,
                                            "Analysis1_red_cycks_with_padding")
Analysis1_p6

################################################################################
list_of_dfs <- list(
  Analysis1_df1,
  Analysis1_df2,
  Analysis1_df3,
  Analysis1_df4,
  Analysis1_df5,
  Analysis1_df6
)

# 2. Merge all data frames by the "Metric" column
Analysis1_combined_metrics <- Reduce(function(x, y) merge(x, y, by = "Metric",
                                                all = TRUE), list_of_dfs)


Analysis1_Final_metrics <- Analysis1_combined_metrics %>%
  pivot_longer(
    cols = -Metric, 
    names_to = "Model_Config", 
    values_to = "Score"
  )%>%
  mutate(Experiment_Number = "One") %>%
  relocate(Experiment_Number) #



View(Analysis1_Final_metrics)
output_file <- "D:/Projects/PgdThesis/codes/Analysis/CombinedAnalysis/Analysis1_Final_metrics.csv"
write_csv(Analysis1_Final_metrics, output_file)






