setwd("~/Desktop/PgdThesis/codes/Analysis/Analysis1")
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
  # 1. Extract numbers
  numeric_values <- as.numeric(unlist(str_extract_all(metrics_string, "\\d+\\.\\d+")))
  
  # 2. Extract and clean names (handles hyphens like ROC-AUC)
  metric_names <- unlist(str_extract_all(metrics_string, "[A-Za-z0-9\\s\\-]+(?=:)")) %>%
    str_trim() %>%
    str_replace_all("\\s+", "_")
  
  # 3. Create Wide Dataframe
  metrics_df_wide <- as.data.frame(t(numeric_values))
  colnames(metrics_df_wide) <- metric_names
  
  # 4. Pivot to Long and use model_label as the value column name to allow merging
  metrics_df_long <- metrics_df_wide %>%
    as_tibble(.name_repair = "minimal") %>%
    pivot_longer(cols = everything(), names_to = "Metric", values_to = model_label)
  
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
Analysis1_raw_input_1 <- "Accuracy: 0.0298 | Balanced Acc: 0.0308 | Log Loss: 3.8865
Precision: 0.0202 | Recall: 0.0298 | F1: 0.0156 | ROC-AUC: 0.5120"

Analysis1_df1 <- parse_metrics_to_df(Analysis1_raw_input_1,
                                     "Analysis1_Cycks_with_padding")
Analysis1_p1 <- plot_classification_metrics(Analysis1_Cycks_with_padding,
                                            "Analysis1_Cycks_with_padding")
Analysis1_p1


################################################################################
Analysis1_Cycks_without_padding <- read_csv("Results3/Cycks_Without_Padding.csv")
#View(Cycks_without_padding)
Analysis1_raw_input_2 <- "Accuracy: 0.9135 | Balanced Acc: 0.8947 | Log Loss: 0.3991
Precision: 0.9223 | Recall: 0.9135 | F1: 0.9110 | ROC-AUC: 0.9988"

# Using your existing function
Analysis1_df2 <- parse_metrics_to_df(Analysis1_raw_input_2,
                                     "Analysis1_Cycks_without_padding")

Analysis1_p2 <- plot_classification_metrics(Analysis1_Cycks_without_padding,
                                            "Analysis1_Cycks_without_padding")
Analysis1_p2


################################################################################
Analysis1_Vgg_without_padding <- read_csv("Results3/Vgg_without_padding.csv")

Analysis1_raw_input_3 <- "Accuracy: 0.9511 | Balanced Acc: 0.9456 | Log Loss: 0.2339
Precision: 0.9528 | Recall: 0.9511 | F1: 0.9512 | ROC-AUC: 0.9992"


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

Analysis1_raw_input_4 <- "Accuracy: 0.9578 | Balanced Acc: 0.9518 | Log Loss: 0.2025
Precision: 0.9592 | Recall: 0.9578 | F1: 0.9579 | ROC-AUC: 0.9994"


# Using your existing function
Analysis1_df4 <- parse_metrics_to_df(Analysis1_raw_input_4,
                                     "Analysis1_Vgg_with_padding")
print(Analysis1_df4)
Analysis1_p4 <- plot_classification_metrics(Analysis1_Vgg_without_padding,
                                            "Analysis1_Vgg_without_padding")
Analysis1_p4
################################################################################
Analysis1_red_cycks_without_padding <- read_csv("Results3/red_cycks_without_padding.csv")
# View(red_cycks_without_padding)

red_cycks_without_padding_Plot <- plot_classification_metrics(red_cycks_without_padding,
                                                     "red_cycks_without_padding")
red_cycks_without_padding_Plot

red_cycks_without_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  red_cycks_without_padding = c(0.9726, 0.9607, 0.0808, 0.9744, 0.9726, 0.9724,
                          0.9999)
)

# 2. View the data set
print(red_cycks_without_padding_metrics)
################################################################################
red_cycks_with_padding <- read_csv("Results3/red_cycks_with_padding.csv")
# View(red_cycks_with_padding)
red_cycks_with_padding_Plot <- plot_classification_metrics(red_cycks_with_padding,
                                                                  "red_cycks_with_padding")
red_cycks_with_padding_Plot

red_cycks_with_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  red_cycks_with_padding = c(0.9726, 0.9607, 0.0808, 0.9744, 0.9726, 0.9724,
                          0.9999)
)

# 2. View the data set
print(red_cycks_with_padding_metrics)


################################################################################
list_of_dfs <- list(
  red_cycks_with_padding_metrics,
  red_cycks_without_padding_metrics,
  Vgg_with_padding_metrics,
  Vgg_without_padding_metrics,
  Cycks_with_padding_metrics,
  Cycks_without_padding_metrics
)

# 2. Merge all data frames by the "Metric" column
combined_metrics <- Reduce(function(x, y) merge(x, y, by = "Metric",
                                                all = TRUE), list_of_dfs)

# 3. View the final dataset
print(combined_metrics)
View(combined_metrics)
# Optional: Save for external use
# write.csv(combined_metrics, "all_model_comparison.csv", row.names = FALSE

plot_data <- combined_metrics %>%
  pivot_longer(
    cols = -Metric, 
    names_to = "Model", 
    values_to = "Value"
  )

# 2. Filter for metrics on the 0-1 scale for better visualization
performance_metrics <- plot_data %>% 
  filter(Metric != "Log Loss")

# 3. Create the grouped bar plot
ggplot(performance_metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  geom_text(aes(label = sprintf("%.2f", Value)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5, size = 3, angle = 90, hjust = -0.1) +
  theme_minimal() +
  labs(
    title = "Comparison of Model Performance Metrics",
    subtitle = "Accuracy, Precision, Recall, F1, and AUC (Excluding Log Loss)",
    x = "Performance Metric",
    y = "Score (0 to 1)",
    fill = "Model Architecture"
  ) +
  scale_y_continuous(limits = c(0, 1.1)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")

# 4. Optional: Plot Log Loss separately (since lower is better and scale is different)
log_loss_data <- plot_data %>% filter(Metric == "Log Loss")

ggplot(log_loss_data, aes(x = reorder(Model, Value), y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison: Log Loss", subtitle = "Lower values indicate better probabilistic predictions", x = "Model", y = "Log Loss") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = "none")
