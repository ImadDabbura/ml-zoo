# load libraries
library(ISLR)
library(MASS)
library(ggplot2)
library(leaps)
library(glmnet)
library(caTools)
library(car)
library(corrplot)

# Look at the structure of the data
str(Boston)
# convert chas variable into non-ordered categorial variable
Boston$chas <- factor(Boston$chas, labels = c("Don't tract bounds river", "tract bounds river"))
str(Boston)

# write prediction function to be used in selection methods
predict.regsubsets <- function(object, newdata, id, ...){
  #get model formula
  mod_form <- as.formula(object$call[[2]])
  #create feature matrix
  mod_mat <- model.matrix(mod_form, newdata)
  #get the coefficients
  coeff <- coef(object, id = id)
  #get the names of the features from the coeff
  var_names <- names(coeff)
  #Use coeff to predict new data using matrix multiplication
  mod_mat[, var_names] %*% coeff
}

#split the data into training and test: 70 30
Boston$medv <- log(Boston$medv)
set.seed(1)
splt <- sample.split(Boston$medv, 0.7)
train <- subset(Boston, splt == TRUE)
test <- subset(Boston, splt == FALSE)

# number of folds
set.seed(1)
k <- 10
folds <- sample(rep(1:10, length = nrow(train)))
# number of features
p <- dim(train)[2] - 1
# cross validation matrix
cv_error <- matrix(0, nrow = k, ncol = p, dimnames = list(NULL, paste(1:p)))
# write for loop
for(i in 1:k){
  reg_fit <- regsubsets(medv ~ ., data = train[folds != i, ], nvmax = p)
  for(j in 1:p){
    pred <- predict(reg_fit, train[folds ==i, ], id = j)
    cv_error[i, j] <- mean((pred - train$medv[folds == i]) ^ 2)
  }
}
cv_error_mean <- apply(cv_error, 2, mean)
cv_error_mean
which.min(cv_error_mean)

reg_fit <- regsubsets(medv ~ ., data = train, nvmax = p)
x_var <- names(coef(reg_fit, 12))
x_var
x_var[4] <- "chas"
x_var
pred_test_bestsubset <- predict(reg_fit, test, 12)
test_error_bestsubset <- mean((test$medv - pred_test_bestsubset) ^ 2)

lm_fit <- lm(medv ~ . - indus, data = Boston)
par(mfrow = c(2, 2))
plot(lm_fit)
hist(log(Boston$medv))

par(mfrow = c(1, 1))
pairs(Boston)

corrplot(cor(Boston[, -4]), method = "number", type = "lower")
vif(lm_fit)

#===================#
# create lambda grid for all possible values
lambda_grid <- 10 ^ seq(10, -2, length = 100)
# create feature matrix without the intercept
x <- model.matrix(medv ~ ., data = Boston)[, -1]
y <- Boston$medv
xtrain <- model.matrix(medv ~ ., data = Boston[splt == TRUE, ])[, -1]
ytrain <- Boston$medv[splt == TRUE]
xtest <- model.matrix(medv ~ ., data = Boston[splt == FALSE, ])[, -1]
ytest <- Boston$medv[splt == FALSE]
 
# create ridge model on training data
ridge_mod <- glmnet(xtrain, ytrain, alpha = 0, lambda = lambda_grid)
# create cv ridge model
cv_out <- cv.glmnet(xtrain, ytrain, alpha = 0)
plot(cv_out)
best_ridge_lambda <- cv_out$lambda.min
# estimate test error using best lambda
pred_ridge <- predict(ridge_mod, newx = xtest, s = best_ridge_lambda)
mean((pred_ridge - ytest) ^ 2)
# Rebuild the ridge model using best lambda on full dataset
out <- glmnet(x, y, alpha = 0)
ridge_coeff <- predict(out, type = "coefficients", s = best_ridge_lambda)
ridge_coeff


# create lasso model on training data
lasso_mod <- glmnet(xtrain, ytrain, alpha = 1, lambda = lambda_grid)
# create cv lasso model
cv_out <- cv.glmnet(xtrain, ytrain, alpha = 1)
plot(cv_out)
best_lasso_lambda <- cv_out$lambda.1se
# estimate test error using best lambda
pred_lasso <- predict(lasso_mod, newx = xtest, s = best_lasso_lambda)
mean((pred_lasso - ytest) ^ 2)
# Rebuild the lasso model using best lambda on full dataset
out <- glmnet(x, y, alpha = 1)
lasso_coeff <- predict(out, type = "coefficients", s = best_lasso_lambda)
lasso_coeff
