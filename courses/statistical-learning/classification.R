# load libraries
library(ISLR)
library(MASS)
library(ggplot2)
library(leaps)
library(glmnet)
library(caTools)
library(car)
library(corrplot)
library(class)
library(ROCR)
library(caret)
library(rpart)
library(rpart.plot)

# inspect the data
dim(Caravan)
sum(is.na(Caravan))
str(Caravan)
contrasts(Caravan$Purchase)
table(Caravan$Purchase)

# Null model --> predicting the majority of the class which is "No"
accuracy_baseline <- sum(Caravan$Purchase == "No") / nrow(Caravan)
accuracy_baseline

# standarize the data to use in KNN
stand_X <- scale(Caravan[, -86])
var(stand_X[, 1]); mean(stand_X[, 1])

# split the data into training and test sets
set.seed(1)
splt <- sample.split(Caravan$Purchase, SplitRatio = 0.7)
train <- subset(Caravan, splt == TRUE)
test <- subset(Caravan, splt == FALSE)
trainX <- subset(stand_X, splt == TRUE)
testX <- subset(stand_X, splt == FALSE)
trainY <- Caravan$Purchase[splt == TRUE]
testY <- Caravan$Purchase[splt == FALSE]

# Use CV to tune k
set.seed(1)
k <- 10
folds <- sample(rep(1:10, length = nrow(trainX)))
table(folds)
cv_error_knn <- matrix(0, nrow = 10, ncol = 10, dimnames = list(paste(1:10), paste(1:10)))
# iterate over k in KNN
for(j in 1:10){
  for(i in 1:k){
    pred <- knn(trainX[folds != i, ], trainX[folds == i, ], trainY[folds != i], k = j)
    cv_error_knn[i, j] <- mean(trainY[folds == i] != pred)
  }
}
cv_error_knn <- apply(cv_error_knn, 2, mean)
cv_error_knn
which.min(cv_error_knn)
# k = 9 is the best for knn
knn_pred <- knn(trainX, testX, trainY, k = 9)
knn_accuracy <- mean(testY == knn_pred)
knn_accuracy # baseline is better; however it's not good to use accuracy as measure because it an imbalanced problem
table(knn_pred, testY)
# plot precision-recall
knn_pred_num <- ifelse(knn_pred == "No", 0, 1)
testY_num <- ifelse(testY == "No", 0, 1)
pred_roc <- prediction(knn_pred_num, testY_num)
prec_recall <- performance(pred_roc, "prec", "rec")
fp_tp <- performance(pred_roc, "tpr", "fpr")
plot(prec_recall)
plot(fp_tp)
auc_knn <- as.numeric(performance(pred_roc, "auc")@y.values)
auc_knn

# try logistic regrssion
log_model <- glm(Purchase ~ ., data = train, family = "binomial")
summary(log_model)
pred_log <- predict(log_model, newdata = test, type = "response")
pred_log
class_log <- ifelse(pred_log >= 0.5, "Yes", "No")
table(class_log, testY)
log_accuracy <- mean(class_log == testY)
log_accuracy
# plot the roc curve and get the auc
pred_log_roc <- ifelse(class_log == "No", 0, 1)
predict_roc <- prediction(pred_log_roc, testY_num)
fp_tp_log <- performance(predict_roc, "tpr", "fpr")
auc_log <- as.numeric(performance(predict_roc, "auc")@y.values)
auc_log

# try LDA
lda_mod <- lda(Purchase ~ ., data = train[, -c(60, 81)])
plot(lda_mod)
lda_mod
pred_lda <- predict(lda_mod, test)
lda_accuracy <- mean(pred_lda$class == testY)

pred_lda_roc <- ifelse(pred_lda$class == "No", 0, 1)
predict_roc <- prediction(pred_lda_roc, testY_num)
fp_tp_lda <- performance(predict_roc, "tpr", "fpr")
auc_lda <- as.numeric(performance(predict_roc, "auc")@y.values)
auc_lda

# try qDA
qda_mod <- qda(Purchase ~ ., data = train[, -c(60, 81)])
plot(qda_mod)
qda_mod
pred_qda <- predict(qda_mod, test)
qda_accuracy <- mean(pred_qda$class == testY)

pred_qda_roc <- ifelse(pred_qda$class == "No", 0, 1)
predict_roc <- prediction(pred_qda_roc, testY_num)
fp_tp_qda <- performance(predict_roc, "tpr", "fpr")
auc_qda <- as.numeric(performance(predict_roc, "auc")@y.values)
auc_qda

# try lasso
lambda_grid <- 10 ^ seq(10, -2, length = 100)
x <- model.matrix(Purchase ~ ., data = train)[, -1]
xtest <- model.matrix(Purchase ~ ., data = test)[, -1]
lasso_mod <- glmnet(x, trainY, alpha = 1, family = "binomial", lambda = lambda_grid)
cv_out <- cv.glmnet(x, trainY, alpha = 1, family = "binomial")
plot(cv_out)
best_lambda_lasso <- cv_out$lambda.min
pred_lasso <- predict(lasso_mod, xtest, s = best_lambda_lasso, type = "class")
lasso_accuracy <- mean(testY == pred_lasso)
pred_lasso_roc <- ifelse(pred_lasso == "No", 0, 1)
predict_roc <- prediction(pred_lasso_roc, testY_num)
fp_tp_lasso <- performance(predict_roc, "tpr", "fpr")
auc_lasso <- as.numeric(performance(predict_roc, "auc")@y.values)
auc_lasso

# try ridge
ridge_mod <- glmnet(x, trainY, alpha = 0, family = "binomial", lambda = lambda_grid)
cv_out <- cv.glmnet(x, trainY, alpha = 0, family = "binomial")
plot(cv_out)
best_lambda_ridge <- cv_out$lambda.min
pred_ridge <- predict(ridge_mod, xtest, s = best_lambda_ridge, type = "class")
ridge_accuracy <- mean(testY == pred_ridge)
pred_ridge_roc <- ifelse(pred_ridge == "No", 0, 1)
predict_roc <- prediction(pred_ridge_roc, testY_num)
fp_tp_ridge <- performance(predict_roc, "tpr", "fpr")
auc_ridge <- as.numeric(performance(predict_roc, "auc")@y.values)
auc_ridge

# try tree
tree_mod <- tree(Purchase ~ ., data = train)
pred_tree <- predict(tree_mod, test, type = "class")
mean(pred_tree == testY)
cv_tree <- cv.tree(tree_mod, FUN = prune.misclass)
plot(cv_tree$size, cv_tree$dev, type = "b")
plot(cv_tree$k, cv_tree$dev, type = "b")
tree_mod_pruned <- prune.misclass(tree_mod, best = 0)
pred_tree_pruned <- predict(tree_mod_pruned, test, type = "class")
mean(pred_tree == testY)

cp.grid = expand.grid( .cp = (0:10)*0.001)
tr.control <- trainControl(method = "cv", number = 10)
tr <- caret::train(Purchase ~ ., data = train, method = "rpart", trControl = tr.control, tuneGrid = cp.grid)
tr$finalModel
prp(tr$finalModel)
mean(predict(tr$finalModel, test, type = "class") == testY)
