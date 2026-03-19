# ==============================================================================
# THESIS DATA ANALYSIS: VGG & CYCKS MODEL COMPARISON
# ==============================================================================

# 1. Setup Environment
setwd("~/Desktop/PgdThesis/codes/Analysis/Analysis2")

# Define and Load Libraries
libs <- c("dplyr", "magrittr", "ggplot2", "readr", "caret", "tidyverse", 
          "GGally", "survival", "data.table", "plotly", "lubridate")

load_and_install <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages)) install.packages(new_packages)
  lapply(packages, library, character.only = TRUE)
}
load_and_install(libs)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Function to parse metric strings into a long-format dataset for a specific model
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

# Function to plot per-user performance line graphs
plot_classification_metrics <- function(df, model_name) {
  data_long <- df %>%
    pivot_longer(
      cols = any_of(c("Precision", "Recall", "F1-Score")), 
      names_to = "Metric", 
      values_to = "Value"
    ) %>%
    arrange(User)
  
  plot <- ggplot(data_long, aes(x = User, y = Value, color = Metric, group = Metric)) +
    geom_line(linewidth = 1) + 
    geom_point(size = 2) +
    theme_minimal() +
    labs(title = paste("User-wise Performance:", model_name), 
         subtitle = "Per-class Precision, Recall, and F1-Score",
         x = "User ID", y = "Score (0-1)") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
          legend.position = "bottom") +
    scale_y_continuous(limits = c(0, 1)) +
    scale_color_brewer(palette = "Set1")
  
  return(plot)
}

# ==============================================================================
# DATA IMPORT & INDIVIDUAL PROCESSING
# ==============================================================================

# Model 1: Cycks With Padding
Cycks_with_padding <- read_csv("Cycks_With_Padding.csv")
raw_input_1 <- "Accuracy: 0.8816 | Balanced Acc: 0.8818 | Log Loss: 0.4588 | Precision: 0.8879 | Recall: 0.8816 | F1: 0.8818 | ROC-AUC: 0.9969"
df1 <- parse_metrics_to_df(raw_input_1, "Cycks_With_Padding")
p1 <- plot_classification_metrics(Cycks_with_padding, "Cycks_With_Padding")

# Model 2: Cycks Without Padding
Cycks_without_padding <- read_csv("Cycks_Without_Padding.csv")
raw_input_2 <- "Accuracy: 0.9830 | Balanced Acc: 0.9831 | Log Loss: 0.0669 | Precision: 0.9837 | Recall: 0.9830 | F1: 0.9830 | ROC-AUC: 1.0000"
df2 <- parse_metrics_to_df(raw_input_2, "Cycks_Without_Padding")
p2 <- plot_classification_metrics(Cycks_without_padding, "Cycks_Without_Padding")

# Model 3: VGG Without Padding
Vgg_without_padding <- read_csv("Vgg_without_padding.csv")
raw_input_3 <- "Accuracy: 0.9580 | Balanced Acc: 0.9578 | Log Loss: 0.2283 | Precision: 0.9587 | Recall: 0.9580 | F1: 0.9581 | ROC-AUC: 0.9994"
df3 <- parse_metrics_to_df(raw_input_3, "VGG_Without_Padding")
p3 <- plot_classification_metrics(Vgg_without_padding, "VGG_Without_Padding")

# Model 4: VGG With Padding
Vgg_with_padding <- read_csv("Vgg_with_padding.csv")
raw_input_4 <- "Accuracy: 0.9525 | Balanced Acc: 0.9524 | Log Loss: 0.2470 | Precision: 0.9536 | Recall: 0.9525 | F1: 0.9525 | ROC-AUC: 0.9993"
df4 <- parse_metrics_to_df(raw_input_4, "VGG_With_Padding")
p4 <- plot_classification_metrics(Vgg_with_padding, "VGG_With_Padding")

# Model 5: Red Cycks Without Padding
Red_cycks_without_padding <- read_csv("Red_cycks_without_padding.csv")
raw_input_5 <- "Accuracy: 0.9641 | Balanced Acc: 0.9648 | Log Loss: 0.1158 | Precision: 0.9671 | Recall: 0.9641 | F1: 0.9642 | ROC-AUC: 0.9999"
df5 <- parse_metrics_to_df(raw_input_5, "Red_Cycks_Without_Padding")
p5 <- plot_classification_metrics(Red_cycks_without_padding, "Red_Cycks_Without_Padding")

# Model 6: Red Cycks With Padding
Red_cycks_with_padding <- read_csv("Red_cycks_with_padding.csv")
raw_input_6 <- "Accuracy: 0.9736 | Balanced Acc: 0.9738 | Log Loss: 0.0896 | Precision: 0.9760 | Recall: 0.9736 | F1: 0.9739 | ROC-AUC: 0.9999"
df6 <- parse_metrics_to_df(raw_input_6, "Red_Cycks_With_Padding")
p6 <- plot_classification_metrics(Red_cycks_with_padding, "Red_Cycks_With_Padding")

# ==============================================================================
# MERGING & GLOBAL COMPARISON
# ==============================================================================

list_of_dfs <- list(df1, df2, df3, df4, df5, df6)

# Merge all into one table
combined_metrics <- Reduce(function(x, y) merge(x, y, by = "Metric", all = TRUE), list_of_dfs)

# --- FIX FOR PRECISION NA ISSUE ---
# Standardize names to ensure rows collapse correctly
combined_metrics_clean <- combined_metrics %>%
  mutate(Metric = case_when(
    grepl("Precision", Metric) ~ "Precision",
    TRUE ~ Metric
  )) %>%
  group_by(Metric) %>%
  summarise(across(everything(), ~ first(na.omit(.))), .groups = 'drop')

# Pivot to Long for Plotting
plot_data <- combined_metrics_clean %>%
  pivot_longer(cols = -Metric, names_to = "Model", values_to = "Value")

# 1. Performance Comparison Plot (Accuracy, Precision, Recall, F1, AUC)
performance_plot <- plot_data %>% 
  filter(!Metric %in% c("Log_Loss", "Balanced_Acc")) %>%
  ggplot(aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  geom_text(aes(label = sprintf("%.2f", Value)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5, size = 3, angle = 0) +
  theme_minimal() +
  labs(title = "Global Model Performance Comparison", y = "Score (0-1)") +
  scale_y_continuous(limits = c(0, 1.1)) +
  scale_fill_brewer(palette = "Paired")

# 2. Log Loss Plot (Lower is Better)
log_loss_plot <- plot_data %>% 
  filter(Metric == "Log_Loss") %>%
  ggplot(aes(x = reorder(Model, Value), y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison: Log Loss", x = "Model", y = "Log Loss Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================
print(performance_plot)
print(log_loss_plot)
print(combined_metrics_clean)
