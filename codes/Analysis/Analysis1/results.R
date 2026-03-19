setwd("~/Desktop/PgdThesis/codes/Analysis")
libs <- c("dplyr", "magrittr", "ggplot2", "readr", "caret", "tidyr", "GGally",
          "survival", "data.table",  "plotly", "lubridate", "mice")

install_or_load_pack <- function(pack){
  create.pkg <- pack[!(pack %in% installed.packages()[, "Package"])]
  if (length(create.pkg))
    install.packages(create.pkg, dependencies = TRUE)
  lapply(libs, require, character.only = T, warn.conflicts=T, quietly=T)
  #I know I should be using purr here, but this is before the Tidyverse is loaded. I know you Tidyverse trend setters will have me here.
}
install_or_load_pack(libs)


################################################################################
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
Cycks_without_padding <- read_csv("Results3/Cycks_Without_Padding.csv")
#View(Cycks_without_padding)

Cycks_without_padding_plot <- plot_classification_metrics(Cycks_without_padding,
                                                     "Cycks_without_padding")
Cycks_without_padding_plot

Cycks_without_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  Cycks_Without_Padding = c(0.9135, 0.8947, 0.3991, 0.9223, 0.9135, 0.9110, 0.9988)
)
print(Cycks_without_padding_metrics)

################################################################################
Cycks_with_padding <- read_csv("Results3/Cycks_With_Padding.csv")
#View(Cycks_with_padding)

Cycks_with_padding_Plot <- plot_classification_metrics(Cycks_with_padding,
                                                     "Cycks_with_padding")
Cycks_with_padding_Plot

# 1. Create the data frame for the new model results
Cycks_with_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  Cycks_with_padding = c(0.0298, 0.0308, 3.8865, 0.0202, 0.0298, 0.0156, 0.5120)
)

# 2. View the data set
print(Cycks_with_padding_Metrics)
################################################################################
Vgg_without_padding <- read_csv("Results3/Vgg_without_padding.csv")
# View(Vgg_without_padding)

Vgg_without_padding_Plot <- plot_classification_metrics(Vgg_without_padding,
                                                  "Vgg_without_padding")
Vgg_without_padding_Plot

Vgg_without_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  Vgg_without_padding = c(0.9511, 0.9456, 0.2339, 0.9528, 0.9511, 0.9512,
                          0.9992)
)

# 2. View the data set
print(Vgg_without_padding_metrics)

################################################################################
Vgg_with_padding <- read_csv("Results3/Vgg_with_padding.csv")
# View(Vgg_with_padding)

Vgg_with_padding_Plot <- plot_classification_metrics(Vgg_with_padding,
                                                        "Vgg_with_padding")
Vgg_with_padding_Plot

Vgg_with_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  Vgg_with_padding = c(0.9578, 0.9518, 0.2025, 0.9592, 0.9578, 0.9579,
                          0.9994)
)

# 2. View the data set
print(Vgg_with_padding_metrics)
################################################################################
reduced_cycks_without_padding <- read_csv("Results3/reduced_cycks_without_padding.csv")
# View(reduced_cycks_without_padding)

reduced_cycks_without_padding_Plot <- plot_classification_metrics(reduced_cycks_without_padding,
                                                     "reduced_cycks_without_padding")
reduced_cycks_without_padding_Plot

reduced_cycks_without_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  reduced_cycks_without_padding = c(0.9726, 0.9607, 0.0808, 0.9744, 0.9726, 0.9724,
                          0.9999)
)

# 2. View the data set
print(reduced_cycks_without_padding_metrics)
################################################################################
reduced_cycks_with_padding <- read_csv("Results3/reduced_cycks_with_padding.csv")
# View(reduced_cycks_with_padding)
reduced_cycks_with_padding_Plot <- plot_classification_metrics(reduced_cycks_with_padding,
                                                                  "reduced_cycks_with_padding")
reduced_cycks_with_padding_Plot

reduced_cycks_with_padding_metrics <- data.frame(
  Metric = c("Accuracy", "Balanced Accuracy", "Log Loss", 
             "Precision", "Recall", "F1-Score", "ROC-AUC"),
  reduced_cycks_with_padding = c(0.9726, 0.9607, 0.0808, 0.9744, 0.9726, 0.9724,
                          0.9999)
)

# 2. View the data set
print(reduced_cycks_with_padding_metrics)


################################################################################
list_of_dfs <- list(
  reduced_cycks_with_padding_metrics,
  reduced_cycks_without_padding_metrics,
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
