# Install and load the necessary libraries
if (!require(smotefamily)) install.packages("smotefamily")
if (!require(h2o)) install.packages("h2o")
if (!require(corrplot)) install.packages("corrplot")
if (!require(ggplot2)) install.packages("ggplot2")
library(smotefamily)
library(h2o)
library(dplyr)
library(reshape2)
library(corrplot)
library(ggplot2)


# Load required libraries
library(httr)
library(readr)

# Define the API endpoint and API key
url <- "https://h50pmkmeb6.execute-api.us-east-1.amazonaws.com/dev/quantgov/"
api_key <- "5ntHOMzpYo5T8FoXe7GIq8g9Qx51awhV1tJ2zOPH"

# Make the GET request with the API key
response <- GET(url, add_headers(`X-Api-Key` = api_key), verbose())

# Check if the request was successful
if (status_code(response) == 200) {
  # Read the content as text, assuming it's a CSV
  csv_content <- content(response, as = "text")
  
  # Parse the CSV content into a data frame
  df <- suppressWarnings(read_csv(csv_content))
  print(head(df))
} else {
  print(paste("Request failed with status:", status_code(response)))
  print(content(response, as = "text"))
}

# Assume 'df' is your dataframe
output_file <- "Project-Group6-ForecastingComplianceCost.csv"

# Save the dataframe as a CSV
write_csv(df, output_file)

print(paste("Dataframe saved as:", output_file))


# Step 1: Read the unprocessed data file
data <- df

# Step 2: Analyze and preprocess the data
# Check for missing values and handle them
missing_values <- colSums(is.na(data))
print("Missing Values per Column:")
print(missing_values)

# Check unique values in each column
unique_values <- sapply(data, function(col) length(unique(col)))
print("Unique Values per Column:")
print(unique_values)

# Handle missing values by replacing them with median (for numerical columns)
data <- data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Ensure wordcount and other numerical columns are non-zero to avoid division errors
data <- data %>%
  mutate(wordcount = ifelse(wordcount == 0, 1, wordcount))

# Feature Engineering: Add Score and Compliance columns
significant_weight <- 200
nonsignificant_weight <- 100

data <- data %>%
  mutate(
    Score = (
      significant_weight * ((shall + required) / wordcount) +
        nonsignificant_weight * ((may_not + prohibited) / wordcount)
    ),
    Compliance = ifelse(Score > mean(Score, na.rm = TRUE), "Compliant", "Non-Compliant")
  )

# Drop the Score column before saving the processed data
data <- data %>%
  select(-Score)

# Save the processed data to a new CSV file for future use
write.csv(data, "Project-Group6-ForecastingComplianceCost-Processed.csv", row.names = FALSE)

# Step 3: Correlation matrix analysis
# Select only numerical columns for correlation analysis
numerical_data <- data %>%
  select(where(is.numeric))

# Compute the correlation matrix
correlation_matrix <- cor(numerical_data, use = "complete.obs")

# Convert the correlation matrix to a long format for ggplot2
correlation_data <- melt(correlation_matrix)

# Visualize the heatmap with ggplot2
ggplot(data = correlation_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +  # Add white borders between tiles
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1)) +  # Color gradient
  theme_minimal() +  # Minimal theme for better aesthetics
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),  # Rotate x-axis labels for readability
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  ) +
  labs(fill = "Correlation", title = "Correlation Heatmap")

# Step 4: Use the processed data for ML modeling
# Reload the processed data
processed_data <- read.csv("Project-Group6-ForecastingComplianceCost-Processed.csv")

# Convert Compliance to a factor for classification
processed_data$Compliance <- as.factor(processed_data$Compliance)

# Count of distinct values in the target variable
distinct_counts <- table(processed_data$Compliance)
print("Count of distinct values in the target variable:")
print(distinct_counts)

# Step 4.1: Balance the target variable using oversampling
library(dplyr)

# Separate the data into Compliant and Non-Compliant
compliant_data <- processed_data %>% filter(Compliance == "Compliant")
non_compliant_data <- processed_data %>% filter(Compliance == "Non-Compliant")

# Oversample the minority class
set.seed(123)
oversampled_compliant <- compliant_data %>% sample_n(nrow(non_compliant_data), replace = TRUE)

# Combine the balanced dataset
balanced_data <- bind_rows(oversampled_compliant, non_compliant_data)

# Shuffle the data
balanced_data <- balanced_data %>% sample_frac(1)

# Check the new class distribution
print("Distribution of Compliance after balancing:")
print(table(balanced_data$Compliance))


# Step 5: Initialize h2o and configure timeout/memory
h2o.init(max_mem_size = "4G")  # Restart with increased memory and timeout

# Convert the balanced data to an H2O frame
h2o_data <- as.h2o(balanced_data)

# Split the dataset into training, validation, and test sets
splits <- h2o.splitFrame(h2o_data, ratios = c(0.7, 0.15), seed = 123)
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

# Define predictors and target
predictors <- setdiff(names(balanced_data), c("Compliance"))
target <- "Compliance"

# Step 6: Train multiple models with cross-validation
# Model 1: Gradient Boosting Machine (GBM)
gbm_model <- h2o.gbm(
  x = predictors,
  y = target,
  training_frame = train,
  validation_frame = valid,
  ntrees = 100,
  max_depth = 5,
  learn_rate = 0.1,
  nfolds = 5, # Cross-validation
  seed = 123
)

# Model 2: Deep Learning
dl_model <- h2o.deeplearning(
  x = predictors,
  y = target,
  training_frame = train,
  validation_frame = valid,
  hidden = c(128, 64, 32),
  epochs = 20,
  nfolds = 5, # Cross-validation
  seed = 123
)

# Model 3: Random Forest
rf_model <- h2o.randomForest(
  x = predictors,
  y = target,
  training_frame = train,
  validation_frame = valid,
  ntrees = 100,
  max_depth = 10,
  nfolds = 5, # Cross-validation
  seed = 123
)

# Step 7: Evaluate models using cross-validation metrics
# Compute cross-validation AUC for each model
gbm_cv_auc <- h2o.auc(h2o.performance(gbm_model, xval = TRUE))
dl_cv_auc <- h2o.auc(h2o.performance(dl_model, xval = TRUE))
rf_cv_auc <- h2o.auc(h2o.performance(rf_model, xval = TRUE))

# Display AUC for all models
print(paste("GBM CV AUC:", gbm_cv_auc))
print(paste("Deep Learning CV AUC:", dl_cv_auc))
print(paste("Random Forest CV AUC:", rf_cv_auc))

# Step 8: Select the best model based on cross-validation AUC
best_model_name <- if (gbm_cv_auc > dl_cv_auc & gbm_cv_auc > rf_cv_auc) {
  "GBM"
} else if (dl_cv_auc > rf_cv_auc) {
  "Deep Learning"
} else {
  "Random Forest"
}

print(paste("The best model is:", best_model_name))

# Step 9: Train and Save the Best Model with a Static Name
save_best_model <- function(best_model_name, predictors, target, train, valid) {
  if (best_model_name == "GBM") {
    best_model <- h2o.gbm(
      x = predictors,
      y = target,
      training_frame = train,
      validation_frame = valid,
      ntrees = 100,
      max_depth = 5,
      learn_rate = 0.1,
      seed = 123
    )
  } else if (best_model_name == "Deep Learning") {
    best_model <- h2o.deeplearning(
      x = predictors,
      y = target,
      training_frame = train,
      validation_frame = valid,
      hidden = c(128, 64, 32),
      epochs = 20,
      seed = 123
    )
  } else if (best_model_name == "Random Forest") {
    best_model <- h2o.randomForest(
      x = predictors,
      y = target,
      training_frame = train,
      validation_frame = valid,
      ntrees = 100,
      max_depth = 10,
      seed = 123
    )
  } else {
    stop("Invalid model name. Please check the input.")
  }
  
  # Save the model with a static name
  model_file_name <- paste0("Project-Group6-ForecastingComplianceCost-", best_model_name, ".h2o")
  h2o.saveModel(best_model, path = ".", force = TRUE, filename = model_file_name)
  print(paste("Best model saved as:", model_file_name))
  
  return(best_model)
}

# Train and save the best model
best_model <- save_best_model(best_model_name, predictors, target, train, valid)


# Shutdown h2o
h2o.shutdown(prompt = FALSE)
