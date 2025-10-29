library(ggplot2)
library(GGally)
library(readr)
library(dplyr)
library(caret)
library(tidyr)
library(ggcorrplot)
library(corrplot)
library(MASS)
library(car)
library(splines)
library(randomForest)
library(gbm)
library(kableExtra)

set.seed(100)
# Dataset
LE <- read_csv("Life-Expectancy-Data-Updated.csv")

LEdata <- LE %>%
  arrange(Country, Year) %>%
  mutate(log_GDP_per_capita = log(GDP_per_capita + 1)) %>% # Transformation
  dplyr::select(-Economy_status_Developing) # Redundant dummy

# EDA
summary(LEdata)

# Generate Histogram with Summary Table
summary_stats <- quantile(LEdata$Life_expectancy, probs = c(0.25, 0.50, 0.75))
summary_text <- paste0(
  "Min:", round(min(LEdata$Life_expectancy), 1), "\n",
  "1st Q: ", round(summary_stats[1], 1), "\n",
  "Median: ", round(summary_stats[2], 1), "\n",
  "Mean: ", round(mean(LEdata$Life_expectancy), 1), "\n",
  "3rd Q: ", round(summary_stats[3], 1), "\n",
  "Max: ", round(max(LEdata$Life_expectancy), 1)
)
ggplot(LEdata, aes(x = Life_expectancy)) +
  geom_histogram(binwidth = 2, fill = "blue", color = "black", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribution of Life Expectancy", x = "Life Expectancy (years)", y = "Frequency (units)") +
  annotate("text", x = 40, y = 250, label = summary_text, hjust = 0, size = 5, color = "black")

# Boxplots of Life Expectancy by Region for 2001 and 2015
LEdata_filtered <- LEdata %>%
  filter(Year %in% c(2001, 2015))
ggplot(LEdata_filtered, aes(x = Region, y = Life_expectancy, fill = Region)) +
  geom_boxplot() +
  facet_wrap(~Year) +
  theme_minimal() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank()) +
  labs(title = "Life Expectancy by Region in 2001 and in 2015", x = NULL, y = "Life Expectancy")

# GDP per capita and Log-GDP per capita vs Life Expectancy
ggplot(LEdata, aes(x = GDP_per_capita, y = Life_expectancy)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "loess", color = "red") +
  theme_minimal() +
  labs(title = "GDP per Capita vs Life Expectancy", x = "GDP per Capita", y = "Life Expectancy")
ggplot(LEdata, aes(x = log_GDP_per_capita, y = Life_expectancy)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "loess", color = "red") +
  theme_minimal() +
  labs(title = "Log GDP per Capita vs Life Expectancy", x = "Log GDP per Capita", y = "Life Expectancy")

# Schooling vs Life Expectancy
ggplot(LEdata, aes(x = Schooling, y = Life_expectancy)) +
  geom_point(alpha = 0.5, color = "darkgreen") +
  geom_smooth(method = "lm", color = "red") +
  theme_minimal() +
  labs(title = "Schooling vs Life Expectancy", x = "Years of Schooling", y = "Life Expectancy")

# Log-GDP vs Schooling
ggplot(LEdata, aes(x = log_GDP_per_capita, y = Schooling)) +
  geom_point(alpha = 0.5, color = "cyan") +
  geom_smooth(method = "loess", color = "red") +
  theme_minimal() +
  labs(title = "Log GDP per Capita vs Schooling", x = "Log GDP per Capita", y = "Schooling")

# Correlation matrix
LEdata <- LEdata %>%
  dplyr::select(-GDP_per_capita)
LEdata_numeric <- LEdata %>%
  dplyr::select(where(is.numeric))
cor_matrix <- cor(LEdata_numeric, use = "everything")
corrplot(cor_matrix, method = "color", type = "lower", tl.cex = 0.6, tl.col = "black")

# VIF check and plot
vif_values <- vif(lm(Life_expectancy~., data=LEdata_numeric))
vif_df <- data.frame(Variable = names(vif_values), VIF = vif_values)
vif_df <- vif_df %>% arrange(desc(VIF))
ggplot(vif_df, aes(x = reorder(Variable, VIF), y = VIF, fill = VIF)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip for better readability
  theme_minimal() +
  labs(title = "Variance Inflation Factor",
       x = "Predictors",
       y = "VIF") +
  scale_fill_gradient(low = "cyan", high = "red") +
  geom_hline(yintercept = 10, linetype = "solid", color = "red") +
  theme(legend.position = "none")

LEdata_numeric <- LEdata_numeric %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-Infant_deaths, -Polio)
vif(lm(Life_expectancy~., data=LEdata_numeric))

LEdata <- LEdata %>%
  dplyr::select(-Infant_deaths, -Polio)

# Split dataset into training and testing
train_data <- LEdata_numeric %>% filter(Year >= 2001 & Year <= 2012)
test_data  <- LEdata_numeric %>% filter(Year >= 2013 & Year <= 2015)

# 1. Linear Regression
linreg_model <- lm(Life_expectancy ~ ., data=train_data)
pred1 <- predict(linreg_model, test_data)
te1 <- mean((pred1 - test_data$Life_expectancy)^2)
te1

# 2. Regression using AIC
aic_filter <- stepAIC(linreg_model, direction = "both", trace = TRUE)
aic_model <- lm(Life_expectancy ~ Year + Under_five_deaths + Adult_mortality + 
                  Alcohol_consumption + Hepatitis_B + BMI + Diphtheria + Incidents_HIV + 
                  Thinness_ten_nineteen_years + Schooling + Economy_status_Developed + 
                  log_GDP_per_capita, data=train_data)
pred2 <- predict(aic_model, test_data)
te2 <- mean((pred2 - test_data$Life_expectancy)^2)
te2

# 3. KNN - finding optimal k
k_test <- numeric(15)
for (i in 1:15) {
  knn_model <- knnreg(train_data %>% dplyr::select(-Life_expectancy), 
                      train_data$Life_expectancy, 
                      i)
  knn_pred <- predict(knn_model, test_data %>% dplyr::select(-Life_expectancy))
  k_test[i] <- mean((knn_pred - test_data$Life_expectancy)^2)
}
print(k_test) 

knn_df <- tibble(
  "K Value (1-8)" = 1:8,
  "Testing Error (1-8)" = k_test[1:8],
  "K Value (9-15)" = c(9:15, ""),
  "Testing Error (9-15)" = c(k_test[9:15], "")
)
knn_df %>%
  kable(caption = "Table 1: KNN Testing Error by K Value", align = "cccc") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = FALSE)

# k = 1 is optimal, but opting to use k = 3
knn_model <- knnreg(train_data %>% dplyr::select(-Life_expectancy), 
                    train_data$Life_expectancy, 
                    k = 3)
pred3 <- predict(knn_model, test_data %>% dplyr::select(-Life_expectancy))
te3 <- mean((pred3 - test_data$Life_expectancy)^2)
te3

# 4. LOESS
# Finding optimal span
span_range <- seq(0.1, 1, 0.05)
loess_errors <- numeric(length(span_range))
for (i in seq_along(span_range)) {
  loess_optimal_span <- loess(Life_expectancy ~ Adult_mortality + Under_five_deaths + 
                         BMI + Alcohol_consumption, 
                       data = train_data, 
                       span = span_range[i], 
                       degree = 2)
  loess_pred <- predict(loess_optimal_span, newdata = test_data)
  loess_errors[i] <- mean((loess_pred - test_data$Life_expectancy)^2, na.rm = TRUE)
}
optimal_span <- span_range[which.min(loess_errors)]
# Optimal span is 0.9. Solve for lowest testing error
loess_model <- loess(Life_expectancy ~ Adult_mortality + Under_five_deaths + 
                       BMI + Alcohol_consumption, 
                     data = train_data, 
                     span = optimal_span, 
                     degree = 2)
pred4 <- predict(loess_model, newdata = test_data)
te4 <- mean((pred4 - test_data$Life_expectancy)^2, na.rm = TRUE)
te4

# 5. Random Forest
# Optimize ntree, nodesize, and mtry

# Define parameter grid
ntree_value <- c(100, 200, 400, 600, 800)
nodesize_value <- seq(1, 9, 2)
mtry_value <- seq(1, 15, 2)

# Store parameter values in grid
rf_grid <- expand.grid(ntree = ntree_value, 
                       mtry = mtry_value, 
                       nodesize = nodesize_value)
rf_grid$Test_Error <- NA

# Test for optimal parameters (this will take time)
for (i in 1:nrow(rf_grid)) {
  rf_optimal_param <- randomForest(Life_expectancy ~ ., 
                           data = train_data, 
                           ntree = rf_grid$ntree[i], 
                           mtry = rf_grid$mtry[i], 
                           importance = TRUE,
                           nodesize = rf_grid$nodesize[i])
  
  rf_pred <- predict(rf_optimal_param, test_data)
  rf_grid$Test_Error[i] <- mean((rf_pred - test_data$Life_expectancy)^2)
}

optimal_rf <- rf_grid[which.min(rf_grid$Test_Error), ]
print(optimal_rf)
# Apply random forest using optimal parameters
rf_model <- randomForest(Life_expectancy ~ ., 
                         data = train_data, 
                         ntree = 600, 
                         mtry = 5, 
                         importance = TRUE,
                         nodesize = 1)
pred5 <- predict(rf_model, test_data)
te5 <- mean((pred5 - test_data$Life_expectancy)^2)
te5

# Most important variables according to Random Forest
importance(rf_model)
varImpPlot(rf_model, type = 1, main = "Variable Importance Plot", cex = 0.8)

# 6. GBM
gbm_grid <- expand.grid(
  interaction.depth = c(1, 3, 5),
  n.trees = c(100, 200, 400, 800, 1600),
  shrinkage = seq(0.01, 0.1, 0.01),
  n.minobsinnode = c(5, 10, 15)
)
gbm_results <- data.frame()

for (i in 1:nrow(gbm_grid)) {
  optimal_gbm_param <- gbm(
    Life_expectancy ~ .,train_data,
    distribution = "gaussian",
    n.trees = gbm_grid$n.trees[i],
    interaction.depth = gbm_grid$interaction.depth[i],
    shrinkage = gbm_grid$shrinkage[i],
    n.minobsinnode = gbm_grid$n.minobsinnode[i],
    cv.folds = 1,
    verbose = FALSE
  )
  
  pred <- predict(optimal_gbm_param, newdata = test_data, n.trees = gbm_grid$n.trees[i])
  test_error <- mean((pred - test_data$Life_expectancy)^2)
  
  # Store results
  gbm_results <- rbind(
    gbm_results,
    data.frame(
      n.trees = gbm_grid$n.trees[i],
      interaction.depth = gbm_grid$interaction.depth[i],
      shrinkage = gbm_grid$shrinkage[i],
      n.minobsinnode = gbm_grid$n.minobsinnode[i],
      Test_Error = test_error
    )
  )
}
# Optimal parameters are found
best_gbm <- gbm_results[which.min(gbm_results$Test_Error), ]
best_gbm

gbm_model <- gbm(
  Life_expectancy ~ ., data = train_data,
  distribution = "gaussian",
  n.trees = best_gbm$n.trees,
  interaction.depth = best_gbm$interaction.depth,
  shrinkage = best_gbm$shrinkage,
  n.minobsinnode = best_gbm$n.minobsinnode,
  verbose = FALSE
)

pred6 <- predict(gbm_model, test_data, n.trees = 1600)
te6 <- mean((pred6 - test_data$Life_expectancy)^2)
te6

# Most important variables according to GBM
summary(gbm_model)

# Combine into table using kableExtra
te_combine <- c(
  "Linear Regression" = te1,
  "AIC Regression" = te2,
  "KNN (k = 3)" = te3,
  "LOESS" = te4,
  "Random Forest" = te5,
  "GBM" = te6
)

te_combine_df <- data.frame(
  Model = names(te_combine),
  "Testing Error" = unname(te_combine)
)

colnames(te_combine_df)[2] <- "Testing Error"
te_min <- min(te_combine_df$`Testing Error`)
te_max <- max(te_combine_df$`Testing Error`)

te_combine_df %>%
  kable(caption = "Table 2: Testing Errors (Single Run)", 
        align = "lc") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), 
                full_width = F) %>%
  row_spec(which(te_combine_df$`Testing Error` == te_min), background = "#b8e895") %>%
  row_spec(which(te_combine_df$`Testing Error` == te_max), background = "#d79283")

# MCCV 100 Iterations
m = 100
test_errors <- matrix(NA, nrow = m, ncol = 6)
colnames(test_errors) <- c("Linear Regression", "AIC Regression", "KNN", "LOESS", "Random Forest", "GBM")

for (i in 1:m) {
  cat("Iteration:", i, "\n")
  
  # Random 80/20 split
  train_index <- createDataPartition(LEdata_numeric$Life_expectancy, p = 0.8, list = FALSE)
  train_data <- LEdata_numeric[train_index, ]
  test_data <- LEdata_numeric[-train_index, ]
  
  ### 1. Linear Regression
  linreg_model <- lm(Life_expectancy ~ ., data = train_data)
  pred1 <- predict(linreg_model, test_data)
  test_errors[i, 1] <- mean((pred1 - test_data$Life_expectancy)^2)
  
  ### 2. AIC Regression
  aic_model <- stepAIC(linreg_model, direction = "both", trace = FALSE)
  pred2 <- predict(aic_model, test_data)
  test_errors[i, 2] <- mean((pred2 - test_data$Life_expectancy)^2)
  
  ### 3. KNN (k = 3)
  knn_model <- knnreg(train_data %>% dplyr::select(-Life_expectancy), train_data$Life_expectancy, k = 3)
  knn_pred <- predict(knn_model, test_data %>% dplyr::select(-Life_expectancy))
  test_errors[i, 3] <- mean((knn_pred - test_data$Life_expectancy)^2)
  
  ### 4. LOESS
  loess_model <- loess(Life_expectancy ~ Adult_mortality + Under_five_deaths + BMI + Alcohol_consumption, 
                       data = train_data, span = 0.9, degree = 2)
  loess_pred <- predict(loess_model, newdata = test_data)
  test_errors[i, 4] <- mean((loess_pred - test_data$Life_expectancy)^2, na.rm = TRUE)
  
  ### 5. Random Forest
  rf_model <- randomForest(Life_expectancy ~ ., data = train_data, ntree = 600, mtry = 5, importance = TRUE, nodesize = 1)
  rf_pred <- predict(rf_model, test_data)
  test_errors[i, 5] <- mean((rf_pred - test_data$Life_expectancy)^2)
  
  ### 6. GBM
  gbm_model <- gbm(
    Life_expectancy ~ ., data = train_data, distribution = "gaussian",
    n.trees = 1600, interaction.depth = 5, shrinkage = 0.09,
    n.minobsinnode = 5, verbose = FALSE
  )
  gbm_pred <- predict(gbm_model, test_data, n.trees = 1600)
  test_errors[i, 6] <- mean((gbm_pred - test_data$Life_expectancy)^2)
}
te_avg <- colMeans(as.data.frame(test_errors))
te_avg
te_var <- apply(test_errors, 2, var)
te_var

te_mccv_combine <- tibble(
  Method = names(te_avg), "Mean Testing Error" = as.numeric(te_avg), "Variance Testing Error" = as.numeric(te_var))
te_mccv_combine %>%
  kable(caption = "Table 3: Mean and Variance of Testing Error (MCCV 100 iterations)") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = FALSE) %>%
  row_spec(which.min(te_mccv_combine$"Mean Testing Error"), background = "#b8e895") %>%
  row_spec(which.max(te_mccv_combine$"Mean Testing Error"), background = "#d79283")

# K-means clustering -- categorizing a country's development
library(factoextra)

LE_2015 <- LEdata %>%
  filter(Year == 2015)
cluster_2015 <- LE_2015 %>%
  dplyr::select(Country, log_GDP_per_capita, Schooling, Alcohol_consumption,
         BMI, Hepatitis_B, Diphtheria, Adult_mortality, Under_five_deaths)
scaled_2015 <- scale(cluster_2015[,-1])
rownames(scaled_2015) <- cluster_2015$Country
fviz_nbclust(scaled_2015, kmeans, method = "silhouette") +
  labs(title = "Silhouette Method (2015 Data)")
# K = 2
k2_2015 <- kmeans(scaled_2015, centers = 2, nstart = 25)
cluster_2015$Cluster_k2 <- as.factor(k2_2015$cluster)

# K = 3
k3_2015 <- kmeans(scaled_2015, centers = 3, nstart = 25)
cluster_2015$Cluster_k3 <- as.factor(k3_2015$cluster)

fviz_cluster(k2_2015, data = scaled_2015, 
             main = "k = 2 in 2015",
             labelsize = 7, palette = "Set2")
fviz_cluster(k3_2015, data = scaled_2015, 
             main = "k = 3 in 2015",
             labelsize = 7, palette = "Set2")
comparison_2015 <- LE_2015 %>%
  dplyr::select(Country, Economy_status_Developed) %>%
  distinct() %>%
  left_join(cluster_2015, by = "Country")

cluster_2015 %>%
  group_by(Cluster_k2) %>%
  summarize(across(c(log_GDP_per_capita, Schooling, Adult_mortality, Under_five_deaths), mean))

# Confusion matrix
table(Cluster_2 = comparison_2015$Cluster_k2, Developed = comparison_2015$Economy_status_Developed)
table(Cluster_3 = comparison_2015$Cluster_k3, Developed = comparison_2015$Economy_status_Developed)

k2_matrix <- table(
  "Cluster (k=2)" = comparison_2015$Cluster_k2,
  "Developed" = comparison_2015$Economy_status_Developed
)
conf_df <- as.data.frame.matrix(k2_matrix)
colnames(conf_df) <- c("Developing (0)", "Developed (1)")
rownames(conf_df) <- c("Cluster 1", "Cluster 2")
kable(conf_df, caption = "Table 4: Confusion Matrix For K=2") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = FALSE)

TP = 37
FP = 79
TN = 63
FN = 0
accuracy <- (TP + TN) / (TP + FP + TN + FN)   # (37 + 63) / 179 = 100 / 179 ≈ 55.9%
precision <- TP / (TP + FP)                  # 37 / (37 + 79) ≈ 31.9%
recall <- TP / (TP + FN)                     # 37 / 37 = 100%
f1 <- 2 * (precision * recall) / (precision + recall)  # ≈ 48.4%

k3_matrix <- table(
  "Cluster (k=3)" = comparison_2015$Cluster_k3,
  "Developed" = comparison_2015$Economy_status_Developed
)
k3_conf_df <- as.data.frame.matrix(k3_matrix)
colnames(k3_conf_df) <- c("Developing (0)", "Developed (1)")
rownames(k3_conf_df) <- c("High Development", "Low Development", "Moderate Development")
kable(k3_conf_df, caption = "Table 5: Confusion Matrix For K=3") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"), full_width = FALSE)
