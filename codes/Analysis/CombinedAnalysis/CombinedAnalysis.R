setwd("D:/Projects/PgdThesis/codes/Analysis/CombinedAnalysis")


libs <- c("dplyr", "magrittr", "ggplot2", "readr", "caret", "tidyverse", 
          "GGally", "survival", "data.table", "plotly", "lubridate", "stringr")

load_and_install <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages)) install.packages(new_packages)
  lapply(packages, library, character.only = TRUE)
}
load_and_install(libs)


Analysis2_Final_metrics <- read_csv("Analysis2_Final_metrics.csv") %>%
  mutate(Model_Config = str_remove(Model_Config, "Analysis2_")) %>%
  # 2. Replace underscores with spaces
  mutate(Model_Config = str_replace_all(Model_Config, "_", " ")) %>%
  # 3. Convert to Title Case
  mutate(Model_Config = str_to_title(Model_Config))
View(Analysis2_Final_metrics)

Analysis1_Final_metrics <- read_csv("Analysis1_Final_metrics.csv") %>%
  mutate(Model_Config = str_remove(Model_Config, "Analysis1_"))  %>%
  # 2. Replace underscores with spaces
  mutate(Model_Config = str_replace_all(Model_Config, "_", " ")) %>%
  # 3. Convert to Title Case
  mutate(Model_Config = str_to_title(Model_Config))
View(Analysis1_Final_metrics)


# Append Analysis1 to Analysis2
# Re-combine and create the 'Source_Batch' column automatically
Combined_Analysis_Metrics <- bind_rows(
  "Experiment 1" = Analysis1_Final_metrics, 
  "Experiment 2" = Analysis2_Final_metrics, 
  .id = "Experiment"
) %>%
  mutate(Metric = if_else(Metric == "F1", "F1-Score", Metric))

colnames(Combined_Analysis_Metrics)
# View the combined result
View(Combined_Analysis_Metrics)

# Optional: Check how many rows you have now compared to before
nrow(Analysis2_Final_metrics) + nrow(Analysis1_Final_metrics) == nrow(Combined_Analysis_Metrics)


generate_metric_plots <- function(data) {
  
  purrr::walk(unique(data$Metric), function(metric_name) {
    
    # 1. Filter for the specific metric
    plot_data <- data %>% filter(Metric == metric_name)
    
    # 2. Build the plot grouped by Model_Config
    p <- ggplot(plot_data, aes(x = Model_Config, y = Score, fill = Experiment)) +
      # position_dodge(width = 0.9) ensures the bars for Exp 0 and Exp 1 are side-by-side per Model
      geom_bar(stat = "identity", position = position_dodge(width = 0.9), color = "black", size = 0.3) +
      
      # 3. Add Score labels on top of each bar
      geom_text(
        aes(label = round(Score, 4)), 
        position = position_dodge(width = 0.9), 
        vjust = -0.5, 
        size = 3.2,
        fontface = "bold"
      ) +
      
      # 4. Styling and Labels
      theme_minimal() +
      # Using a clean color palette for Experiment 0 vs 1
      scale_fill_brewer(palette = "Paired") + 
      labs(
        title = paste("Model Performance Comparison:", metric_name),
        subtitle = "Grouped by Model Configuration",
        x = "Model Name",
        y = "Score Value",
        fill = "Experiment Phase"
      ) +
      
      # Extra space at the top so labels don't hit the ceiling
      scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
      
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
        legend.position = "top",
        panel.grid.major.x = element_blank() # Removes vertical lines for a cleaner look
      )
    
    print(p)
  })
}
generate_metric_plots(Combined_Analysis_Metrics)
