# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)

# Load Data
data <- read.csv('C:/Users/stefa/OneDrive - Careered - CTU/2025/CS871/Week 6/germancredit.csv')

# Check structure and summary
str(data)
summary(data)

# Convert relevant variables to factors
data$default <- as.factor(data$Default)
data$history <- as.factor(data$history)
data$purpose <- as.factor(data$purpose)
data$rent <- as.factor(data$housing == "A152")

# Subset relevant columns
data <- data %>% select(duration, amount, installment, age, history, purpose, rent, default)

# Visualize distributions
par(mfrow=c(2, 2))
hist(data$duration, main = "Duration Distribution", col = "lightblue")
hist(data$amount, main = "Amount Distribution", col = "lightgreen")
hist(data$installment, main = "Installment Distribution", col = "lightcoral")
hist(data$age, main = "Age Distribution", col = "lightyellow")

# Correlation plot
correlation_matrix <- cor(data %>% select(duration, amount, installment, age))
corrplot::corrplot(correlation_matrix, method = "color")

# Split Data
set.seed(42)
train_indices <- sample(1:nrow(data), 900)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train Decision Tree Model
decision_tree_model <- rpart(default ~ ., data = train_data, method = 'class')

# Visualize the Decision Tree
rpart.plot(decision_tree_model)

# Predictions
pred_tree <- predict(decision_tree_model, test_data, type = 'class')

# Evaluation
conf_matrix_tree <- confusionMatrix(pred_tree, test_data$default)

# Results
cat('Decision Tree Accuracy:', conf_matrix_tree$overall['Accuracy'], '\n')
cat('Decision Tree Precision:', conf_matrix_tree$byClass['Precision'], '\n')
cat('Decision Tree Recall:', conf_matrix_tree$byClass['Recall'], '\n')
cat('Decision Tree F1 Score:', conf_matrix_tree$byClass['F1'], '\n')

# Train Logistic Regression Model
logistic_model <- glm(default ~ ., data = train_data, family = binomial)

# Predictions
pred_probs_logistic <- predict(logistic_model, test_data, type = 'response')
pred_classes_logistic <- ifelse(pred_probs_logistic > 0.5, '1', '0')
pred_classes_logistic <- factor(pred_classes_logistic, levels = levels(data$default))

# Evaluation
conf_matrix_logistic <- confusionMatrix(pred_classes_logistic, test_data$default)

# Results
cat('Logistic Regression Accuracy:', conf_matrix_logistic$overall['Accuracy'], '\n')
cat('Logistic Regression Precision:', conf_matrix_logistic$byClass['Precision'], '\n')
cat('Logistic Regression Recall:', conf_matrix_logistic$byClass['Recall'], '\n')
cat('Logistic Regression F1 Score:', conf_matrix_logistic$byClass['F1'], '\n')

# Hyperparameter Tuning for Logistic Regression
tune_grid <- expand.grid(alpha = c(0, 0.5, 1), lambda = c(0.001, 0.01, 0.1, 1, 10, 100))

tuned_logistic_model <- train(
  default ~ ., 
  data = train_data, 
  method = 'glmnet', 
  family = 'binomial', 
  trControl = trainControl(method = 'cv', number = 5), 
  tuneGrid = tune_grid
)

# Best Model and Parameters
best_params <- tuned_logistic_model$bestTune
cat('Best Hyperparameters:\n')
print(best_params)

# Final Prediction with Best Model
pred_probs_tuned <- predict(tuned_logistic_model, test_data, type = 'prob')[, 2]
pred_classes_tuned <- ifelse(pred_probs_tuned > 0.5, '1', '0')
pred_classes_tuned <- factor(pred_classes_tuned, levels = levels(data$default))

# Evaluation
conf_matrix_tuned <- confusionMatrix(pred_classes_tuned, test_data$default)

# Results
cat('Tuned Logistic Regression Accuracy:', conf_matrix_tuned$overall['Accuracy'], '\n')
cat('Tuned Logistic Regression Precision:', conf_matrix_tuned$byClass['Precision'], '\n')
cat('Tuned Logistic Regression Recall:', conf_matrix_tuned$byClass['Recall'], '\n')
cat('Tuned Logistic Regression F1 Score:', conf_matrix_tuned$byClass['F1'], '\n')


