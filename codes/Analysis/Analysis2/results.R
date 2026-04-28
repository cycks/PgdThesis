# ==============================================================================
# THESIS DATA ANALYSIS: VGG & CYCKS MODEL COMPARISON
# ==============================================================================

# 1. Setup Environment
# setwd("~/Desktop/PgdThesis/codes/Analysis/Analysis2")
setwd("D:/Projects/PgdThesis/codes/Analysis/Analysis2")

# Define and Load Libraries
libs <- c("dplyr", "magrittr", "ggplot2", "readr", "caret", "tidyverse", 
          "GGally", "survival", "data.table", "plotly", "lubridate",
          "tidyr")

load_and_install <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages)) install.packages(new_packages)
  lapply(packages, library, character.only = TRUE)
}
load_and_install(libs)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# Function to parse metric strings into a long-format data set for a specific
# model
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

plot_model_performance <- function(df, model_name) {
  
  # 1. Sort the data by Score within the function for a cleaner look
  df_sorted <- df %>% 
    arrange(desc(Score)) %>%
    mutate(Metric = factor(Metric, levels = Metric))
  
  # 2. Build the plot
  p <- ggplot(df_sorted, aes(x = Metric, y = Score, fill = Metric)) +
    geom_bar(stat = "identity", color = "black", alpha = 0.8) +
    
    # Add text labels on top of bars
    geom_text(
      aes(label = sprintf("%.4f", Score)), 
      vjust = -0.5, 
      size = 4, 
      fontface = "bold"
    ) +
    
    theme_minimal() +
    # 'mako' is a great professional palette, but 'viridis' or 'plasma' also work well
    scale_fill_viridis_d(option = "mako") + 
    
    labs(
      title = paste("Model Performance:", model_name),
      subtitle = "Overall metrics performance comparison",
      x = "Metric",
      y = "Score Value"
    ) +
    
    # Ensure the Y-axis has room for the labels
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
      plot.title = element_text(face = "bold", size = 16),
      panel.grid.major.x = element_blank() # Cleans up the vertical lines
    )
  
  return(p)
}


# ==============================================================================
# DATA IMPORT & INDIVIDUAL PROCESSING
# ==============================================================================

# Model 1: Cycks With Padding
Analysis2_Cycks_with_padding <- read_csv("Cycks_With_Padding.csv")
Analysis2_raw_input_1 <- "Accuracy: 0.8816 | Balanced Acc: 0.8818 | Log Loss: 0.4588 | Precision: 0.8879 | Recall: 0.8816 | F1: 0.8818 | ROC-AUC: 0.9969"
Analysis2_df1 <- parse_metrics_to_df(Analysis2_raw_input_1,
                                     "Analysis2_Cycks_With_Padding")
Analysis2_p1 <- plot_classification_metrics(Analysis2_Cycks_with_padding,
                                            "Analysis2_Cycks_With_Padding")
Analysis2_p1

Analysis2_df1 <- Analysis2_df1 %>% 
  mutate(Score = Analysis2_Cycks_With_Padding) %>%
  arrange(desc(Score)) %>%
  mutate(Metric = factor(Metric, levels = Metric))



p1 <- plot_model_performance(Analysis2_df1, "Analysis 2 Cycks With Padding")
print(p1)




# Model 2: Cycks Without Padding
Analysis2_Cycks_without_padding <- read_csv("Cycks_Without_Padding.csv")
Analysis2_raw_input_2 <- "Accuracy: 0.9830 | Balanced Acc: 0.9831 | Log Loss: 0.0669 | Precision: 0.9837 | Recall: 0.9830 | F1: 0.9830 | ROC-AUC: 1.0000"
Analysis2_df2 <- parse_metrics_to_df(Analysis2_raw_input_2,
                                     "Analysis2_Cycks_Without_Padding")
Analysis2_p2 <- plot_classification_metrics(Analysis2_Cycks_without_padding,
                                            "Analysis2_Cycks_Without_Padding")
Analysis2_p2


Analysis2_df2 <- Analysis2_df2 %>% 
  mutate(Score = Analysis2_Cycks_Without_Padding) %>%
  arrange(desc(Score)) %>%
  mutate(Metric = factor(Metric, levels = Metric))



p2 <- plot_model_performance(Analysis2_df2, "Analysis 2 Cycks Without Padding")
print(p2)





############
# 1. Prepare your data
Cycks_Comparison <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
  Without_Padding = c(0.9830, 0.9837, 0.9830, 0.9830, 1.0000),
  With_Padding    = c(0.8816, 0.8879, 0.8816, 0.8818, 0.9969)
)

# 2. Reshape to long format
df_long <- Cycks_Comparison %>%
  pivot_longer(
    cols = c(Without_Padding, With_Padding),
    names_to = "Condition",
    values_to = "Score"
  ) %>%
  mutate(Condition = factor(Condition, levels = c("Without_Padding", "With_Padding")))

# 3. Create the Comparison Plot
ggplot(df_long, aes(x = Metric, y = Score, fill = Condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black") +
  geom_text(
    aes(label = sprintf("%.4f", Score)),
    position = position_dodge(width = 0.8),
    vjust = -0.5, size = 3.5, fontface = "bold"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("#3498db", "#2ecc71")) + # Blue vs Red
  labs(
    title = "Performance Degradation: Impact of Padding on Cycks",
    subtitle = "Analysis showing ~10% accuracy loss due to padding",
    y = "Score Value"
  ) +
  coord_cartesian(ylim = c(0.7, 1.05)) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold", size = 16))



################################################################################
# Model 3: VGG Without Padding
Analysis2_Vgg_without_padding <- read_csv("Vgg_without_padding.csv")
Analysis2_raw_input_3 <- "Accuracy: 0.9580 | Balanced Acc: 0.9578 | Log Loss: 0.2283 | Precision: 0.9587 | Recall: 0.9580 | F1: 0.9581 | ROC-AUC: 0.9994"
Analysis2_df3 <- parse_metrics_to_df(Analysis2_raw_input_3,
                                     "Analysis2_VGG_Without_Padding")
Analysis2_p3 <- plot_classification_metrics(Analysis2_Vgg_without_padding,
                                            "Analysis2_VGG_Without_Padding")
Analysis2_p3

Analysis2_df3 <- Analysis2_df3 %>% 
  mutate(Score = Analysis2_VGG_Without_Padding) %>%
  arrange(desc(Score)) %>%
  mutate(Metric = factor(Metric, levels = Metric))



p3 <- plot_model_performance(Analysis2_df3, "Analysis 2 VGG Without Padding")
print(p3)



# Model 4: VGG With Padding
Analysis2_Vgg_with_padding <- read_csv("Vgg_with_padding.csv")
Analysis2_raw_input_4 <- "Accuracy: 0.9525 | Balanced Acc: 0.9524 | Log Loss: 0.2470 | Precision: 0.9536 | Recall: 0.9525 | F1: 0.9525 | ROC-AUC: 0.9993"
Analysis2_df4 <- parse_metrics_to_df(Analysis2_raw_input_4, 
                                     "Analysis2_VGG_With_Padding")
Analysis2_p4 <- plot_classification_metrics(Analysis2_Vgg_with_padding,
                                            "Analysis2_VGG_With_Padding")
Analysis2_p4


Analysis2_df4 <- Analysis2_df4 %>% 
  mutate(Score = Analysis2_VGG_With_Padding) %>%
  arrange(desc(Score)) %>%
  mutate(Metric = factor(Metric, levels = Metric))



p4 <- plot_model_performance(Analysis2_df4, "Analysis 2 VGG With Padding")
print(p4)


# 1. Create the data frame
VGG_Comp <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
  No_Padding = c(0.9580, 0.9587, 0.9580, 0.9581, 0.9994),
  With_Padding = c(0.9525, 0.9536, 0.9525, 0.9525, 0.9993)
)

# 2. Reshape to long
VGG_Comp_Long <- VGG_Comp %>%
  pivot_longer(cols = -Metric, names_to = "Condition", values_to = "Score")

# 3. Plot
ggplot(VGG_Comp_Long, aes(x = Metric, y = Score, fill = Condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black") +
  geom_text(
    aes(label = sprintf("%.4f", Score)),
    position = position_dodge(width = 0.8),
    vjust = -0.5, size = 3.5, fontface = "bold"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("#3498db", "#2ecc71")) + # Blue vs Red
  labs(
    title = "Modified VGG16: Impact of Padding on Performance",
    subtitle = "Significant perfromance imporvement observed without padding",
    y = "Metric Score"
  ) +
  coord_cartesian(ylim = c(0.8, 1.05)) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))
################################################################################

# Model 5: Red Cycks Without Padding
Analysis2_Red_cycks_without_padding <- read_csv("Red_cycks_without_padding.csv")
Analysis2_raw_input_5 <- "Accuracy: 0.9641 | Balanced Acc: 0.9648 | Log Loss: 0.1158 | Precision: 0.9671 | Recall: 0.9641 | F1: 0.9642 | ROC-AUC: 0.9999"
Analysis2_df5 <- parse_metrics_to_df(Analysis2_raw_input_5,
                           "Analysis2_Red_Cycks_Without_Padding")
Analysis2_p5 <- plot_classification_metrics(Analysis2_Red_cycks_without_padding,
                                  "Analysis2_Red_Cycks_Without_Padding")
Analysis2_p5
# Model 6: Red Cycks With Padding
Analysis2_Red_cycks_with_padding <- read_csv("Red_cycks_with_padding.csv")
Analysis2_raw_input_6 <- "Accuracy: 0.9736 | Balanced Acc: 0.9738 | Log Loss: 0.0896 | Precision: 0.9760 | Recall: 0.9736 | F1: 0.9739 | ROC-AUC: 0.9999"
Analysis2_df6 <- parse_metrics_to_df(Analysis2_raw_input_6,
                                     "Analysis2_Red_Cycks_With_Padding")
Analysis2_p6 <- plot_classification_metrics(Analysis2_Red_cycks_with_padding,
                                            "Analysis2_Red_Cycks_With_Padding")
Analysis2_p6


# 1. Prepare the comparison data
Red_Cycks_Comparison <- data.frame(
  Metric = c("Accuracy", "Balanced_Acc", "Precision", "Recall", "F1-Score", "ROC-AUC"),
  Without_Padding = c(0.9641, 0.9648, 0.9671, 0.9641, 0.9642, 0.9999),
  With_Padding    = c(0.9736, 0.9738, 0.9760, 0.9736, 0.9739, 0.9999)
)

# 2. Reshape to long format
Red_Cycks_long <- Red_Cycks_Comparison %>%
  pivot_longer(
    cols = c(Without_Padding, With_Padding),
    names_to = "Condition",
    values_to = "Score"
  ) %>%
  mutate(Condition = gsub("_", " ", Condition))

# 3. Create the Side-by-Side Bar Plot
ggplot(Red_Cycks_long, aes(x = Metric, y = Score, fill = Condition)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black") +
  
  # Add labels above the bars
  geom_text(
    aes(label = sprintf("%.4f", Score)),
    position = position_dodge(width = 0.8),
    vjust = -0.5,
    size = 3.5,
    fontface = "bold"
  ) +
  
  theme_minimal() +
  scale_fill_manual(values = c("#3498db", "#2ecc71")) + # Blue for No Padding, Green for With Padding
  labs(
    title = "Performance: Red Cycks Padding Comparison",
    subtitle = "Higher scores observed with padding in this specific dataset",
    x = "Performance Metric",
    y = "Score Value",
    fill = "Condition"
  ) +
  
  # IMPORTANT FIX: Use coord_cartesian to zoom without deleting the bars
  coord_cartesian(ylim = c(0.9, 1.02)) +
  
  # Keep scale_y_continuous only for the tick labels, NOT the limits
  scale_y_continuous(breaks = seq(0.9, 1, 0.02)) +
  
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
    plot.title = element_text(face = "bold", size = 16)
  )


# ==============================================================================
# MERGING & GLOBAL COMPARISON
# ==============================================================================

list_of_dfs <- list(Analysis2_df1, 
                    Analysis2_df2,
                    Analysis2_df3, 
                    Analysis2_df4,
                    Analysis2_df5,
                    Analysis2_df6)

# Merge all into one table
Analysis2_combined_metrics <- Reduce(function(x, y) merge(x, y, by = "Metric",
                                                          all = TRUE),
                                     list_of_dfs)

# 2. Convert to Long
# We use -Metric to say "pivot everything EXCEPT the Metric column"
Analysis2_Final_metrics <- Analysis2_combined_metrics %>%
  pivot_longer(
    cols = -Metric, 
    names_to = "Model_Config", 
    values_to = "Score"
  )%>%
  mutate(Experiment_Number = "Two") %>%
  relocate(Experiment_Number)  %>%
  mutate(Metric = case_when(
    grepl("Precision", Metric) ~ "Precision",
    TRUE ~ Metric
  ))
View(Analysis2_Final_metrics)


output_file <- "D:/Projects/PgdThesis/codes/Analysis/CombinedAnalysis/Analysis2_Final_metrics.csv"
write_csv(Analysis2_Final_metrics, output_file)

# --- FIX FOR PRECISION NA ISSUE ---
# Standardize names to ensure rows collapse correctly
Analysis2_combined_metrics_clean <- Analysis2_combined_metrics %>%
  mutate(Metric = case_when(
    grepl("Precision", Metric) ~ "Precision",
    TRUE ~ Metric
  )) %>%
  group_by(Metric) %>%
  summarise(across(everything(), ~ first(na.omit(.))), .groups = 'drop')


View(Analysis2_Final_metrics)
